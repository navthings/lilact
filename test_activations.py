import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from activations import ACTS, EXTRA_ACTS, MLP, GroupGate, RankReLU2, cross_square, make_smooth_relu2, relu2
from model import Model, loss_fn


def close(a, b, tol=1e-3):
    # gpu float math differs from cpu by ~4e-4 on these sizes
    return np.allclose(np.array(a), np.array(b), atol=tol)


def test_groupgate_starts_as_silu():
    h = mx.random.normal((2, 5, 64))
    assert close(GroupGate(64, 8)(h), nn.silu(h))


def test_groupgate_mixes_within_group():
    gg = GroupGate(4, 2)
    gg.A = mx.array([[[1.0, 0.5], [-1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    h = mx.array([[2.0, -1.0, 2.0, -1.0]])
    # worked example from the paper notes: 2*sigmoid(1.5), -1*sigmoid(-3), then plain silu
    want = [2 * mx.sigmoid(mx.array(1.5)).item(), -1 * mx.sigmoid(mx.array(-3.0)).item(),
            nn.silu(mx.array(2.0)).item(), nn.silu(mx.array(-1.0)).item()]
    assert close(gg(h)[0], want)


def test_rankrelu2_is_scale_equivariant():
    r = RankReLU2(tau=0.5)
    h = mx.random.normal((3, 64))
    assert close(r(2 * h), 2 * r(h), tol=1e-3)


def test_rankrelu2_worked_example():
    out = RankReLU2(tau=0.5, eps=0.0)(mx.array([[3.0, 1.0, 0.0, -2.0]]))
    s = np.sqrt(3.25)
    assert close(out[0], [(3 - 0.5 - 0.5 * s) ** 2 / s, 0, 0, 0])


def test_smooth_relu2_forward_and_backward():
    f = make_smooth_relu2(beta=1.0)
    x = mx.array([-1.0, 2.0])
    assert close(f(x), relu2(x))
    g = mx.grad(lambda t: f(t).sum())(x)
    assert close(g, [0.1685, 3.7467], tol=1e-3)


def test_smooth_relu2_large_beta_matches_true_grad():
    f = make_smooth_relu2(beta=50.0)
    x = mx.array([-1.0, 0.5, 2.0])
    assert close(mx.grad(lambda t: f(t).sum())(x), mx.grad(lambda t: relu2(t).sum())(x), tol=1e-2)


def test_cross_square_worked_example():
    out = cross_square(mx.array([[2.0, -1.0]]))
    assert close(out[0], [4 * mx.sigmoid(mx.array(-1.0)).item(), 0.0])


def test_param_counts_match():
    d = 384
    plain = sum(v.size for _, v in tree_flatten(MLP(d, "gelu").parameters()))
    for act in ACTS + EXTRA_ACTS:
        n = sum(v.size for _, v in tree_flatten(MLP(d, act).parameters()))
        # group gate adds d_ff * g, about 1% at d=384, g=8
        assert abs(n - plain) / plain < 0.02, (act, n, plain)


def test_every_act_trains_one_step():
    x = mx.random.randint(0, 100, (2, 16))
    y = mx.random.randint(0, 100, (2, 16))
    for act in ACTS + EXTRA_ACTS:
        m = Model(100, d=32, layers=2, heads=2, act=act)
        loss, grads = nn.value_and_grad(m, loss_fn)(m, x, y)
        assert np.isfinite(loss.item()), act
        assert all(np.isfinite(np.array(g)).all() for _, g in tree_flatten(grads)), act


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
