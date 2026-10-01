import mlx.core as mx
import mlx.nn as nn

ACTS = ["relu", "gelu", "silu", "relu2", "swiglu", "groupgate", "rankrelu2", "smoothrelu2", "crosssquare"]
# ablation only: cross-square's recipe with a separate gate matrix, width cut to match params
EXTRA_ACTS = ["crosssquare_untied"]


def relu2(x):
    return mx.square(nn.relu(x))


def make_smooth_relu2(beta: float):
    # forward is relu^2, backward is the slope of softplus_beta(x)^2
    @mx.custom_function
    def smooth_relu2(x):
        return mx.square(nn.relu(x))

    @smooth_relu2.vjp
    def smooth_relu2_vjp(x, cotangent, output):
        # with one input, mlx passes the input array itself, not a tuple
        sp = nn.softplus(beta * x) / beta
        return cotangent * 2 * sp * mx.sigmoid(beta * x)

    return smooth_relu2


def cross_square(h):
    u, v = mx.split(h, 2, axis=-1)
    return mx.concatenate([relu2(u) * mx.sigmoid(v), relu2(v) * mx.sigmoid(u)], axis=-1)


class GroupGate(nn.Module):
    def __init__(self, d_ff: int, g: int):
        super().__init__()
        if d_ff % g:
            raise ValueError(f"d_ff={d_ff} not divisible by g={g}")
        self.g = g
        self.A = mx.repeat(mx.eye(g)[None], d_ff // g, axis=0)

    def __call__(self, h):
        shape = h.shape
        n_groups = shape[-1] // self.g
        # (groups, tokens, g) @ (groups, g, g)^T gives gate_i = sum_j A_ij h_j per group
        hg = h.reshape(-1, n_groups, self.g).transpose(1, 0, 2)
        gate = (hg @ self.A.transpose(0, 2, 1)).transpose(1, 0, 2).reshape(shape)
        return h * mx.sigmoid(gate)

    def stats(self):
        eye = mx.eye(self.g)
        off = mx.abs(self.A * (1 - eye)).sum() / (self.A.shape[0] * self.g * (self.g - 1))
        diag = (self.A * eye).sum() / (self.A.shape[0] * self.g)
        return {"diag_mean": diag.item(), "offdiag_abs_mean": off.item()}


class RankReLU2(nn.Module):
    def __init__(self, tau: float = 0.0, stop_grad: bool = False, eps: float = 1e-5):
        super().__init__()
        self.tau = mx.array([tau])
        self.stop_grad = stop_grad
        self.eps = eps

    def __call__(self, h):
        mu = h.mean(axis=-1, keepdims=True)
        s = mx.sqrt(h.var(axis=-1, keepdims=True) + self.eps)
        if self.stop_grad:
            mu, s = mx.stop_gradient(mu), mx.stop_gradient(s)
        return mx.square(nn.relu(h - mu - self.tau * s)) / s

    def stats(self):
        return {"tau": self.tau.item()}


def swiglu_width(d: int) -> int:
    # 3 matrices at 8d/3 hold the same params as 2 at 4d; round to a multiple of 8
    return int(round(8 * d / 3 / 8)) * 8


class MLP(nn.Module):
    def __init__(self, d: int, act: str, g: int = 8, beta: float = 1.5, rank_stop_grad: bool = False):
        super().__init__()
        if act not in ACTS + EXTRA_ACTS:
            raise ValueError(f"unknown activation {act}, pick from {ACTS + EXTRA_ACTS}")
        self.kind = act
        gated3 = act in ("swiglu", "crosssquare_untied")
        d_ff = swiglu_width(d) if gated3 else 4 * d
        self.d_ff = d_ff
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)
        if gated3:
            self.gate = nn.Linear(d, d_ff, bias=False)
        elif act == "groupgate":
            self.act = GroupGate(d_ff, g)
        elif act == "rankrelu2":
            self.act = RankReLU2(stop_grad=rank_stop_grad)
        else:
            self.fn = {
                "relu": nn.relu,
                "gelu": nn.gelu,
                "silu": nn.silu,
                "relu2": relu2,
                "smoothrelu2": make_smooth_relu2(beta),
                "crosssquare": cross_square,
            }[act]

    def hidden(self, x):
        if self.kind == "swiglu":
            return nn.silu(self.gate(x)) * self.up(x)
        if self.kind == "crosssquare_untied":
            return relu2(self.up(x)) * mx.sigmoid(self.gate(x))
        h = self.up(x)
        return self.act(h) if self.kind in ("groupgate", "rankrelu2") else self.fn(h)

    def __call__(self, x):
        return self.down(self.hidden(x))
