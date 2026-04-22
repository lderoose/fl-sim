import torch

from src.lib.variant import Diana, ErrorFeedback, ErrorFeedback21, ProjFL

N_LAYERS = 2
LAYER_SHAPES = {0: (3, 4), 1: (5,)}


# ErrorFeedback
def test_error_feedback_gtild_at_init() -> None:
    ef = ErrorFeedback(n_layers=N_LAYERS, alpha=0.5)
    grad = torch.tensor([1.0, 2.0, 3.0])
    result = ef.compute_gtild(grad=grad, layer_number=0)
    assert torch.equal(result, grad)  # errors initialised to 0


def test_error_feedback_update_error() -> None:
    ef = ErrorFeedback(n_layers=N_LAYERS, alpha=1.0)
    gtild = torch.tensor([2.0, 3.0])
    compressed = torch.tensor([1.0, 2.0])
    ef.update_error(gtild=gtild, gtild_sparse=compressed, layer_number=0)
    assert torch.equal(ef.errors[0], gtild - compressed)


def test_error_feedback_gtild_after_update() -> None:
    alpha = 0.5
    ef = ErrorFeedback(n_layers=N_LAYERS, alpha=alpha)
    grad = torch.tensor([1.0, 2.0])
    sparse = torch.tensor([0.5, 1.5])
    ef.update_error(gtild=grad, gtild_sparse=sparse, layer_number=0)
    result = ef.compute_gtild(grad=grad, layer_number=0)
    expected = grad + alpha * (grad - sparse)
    assert torch.allclose(result, expected)


def test_error_feedback_independent_layers() -> None:
    ef = ErrorFeedback(n_layers=N_LAYERS, alpha=1.0)
    g0 = torch.tensor([1.0])
    g1 = torch.tensor([2.0])
    ef.update_error(gtild=g0, gtild_sparse=torch.zeros(1), layer_number=0)
    ef.update_error(gtild=g1, gtild_sparse=torch.zeros(1), layer_number=1)
    assert torch.equal(ef.errors[0], g0)
    assert torch.equal(ef.errors[1], g1)


# ProjFL
def test_projfl_init_no_ef() -> None:
    proj = ProjFL(n_layers=N_LAYERS, layer_gradient_shapes=LAYER_SHAPES, K=3)
    assert proj.K == 3
    assert proj.EF is False
    assert not hasattr(proj, "errors")
    assert len(proj.desc_dir) == N_LAYERS


def test_projfl_init_with_ef() -> None:
    proj = ProjFL(n_layers=N_LAYERS, layer_gradient_shapes=LAYER_SHAPES, K=2, EF=True)
    assert proj.EF is True
    assert len(proj.errors) == N_LAYERS


def test_projfl_compute_proj_before_history() -> None:
    proj = ProjFL(n_layers=N_LAYERS, layer_gradient_shapes=LAYER_SHAPES, K=1)
    grad = torch.randn(3, 4)
    alpha, grad_orth = proj.compute_proj(grad=grad, layer_number=0)
    assert alpha == 0
    assert torch.equal(grad_orth, grad)


def test_projfl_update_dbar_before_history() -> None:
    proj = ProjFL(n_layers=N_LAYERS, layer_gradient_shapes=LAYER_SHAPES, K=1)
    proj.update_dbar(layer_number=0)
    expected_shape = torch.Size(LAYER_SHAPES[0])
    assert proj.dbar[0].shape == expected_shape
    assert torch.equal(proj.dbar[0], torch.zeros(LAYER_SHAPES[0]))


def test_projfl_update_desc_dir_first_call() -> None:
    proj = ProjFL(n_layers=N_LAYERS, layer_gradient_shapes=LAYER_SHAPES, K=2)
    proj.update_dbar(layer_number=0)
    grad = torch.randn(3, 4)
    proj.update_desc_dir(sparse_grad_orth=grad, alpha=0, layer_number=0)
    assert proj.desc_dir[0][0] is not None


def test_projfl_update_dbar_after_history() -> None:
    proj = ProjFL(n_layers=N_LAYERS, layer_gradient_shapes=LAYER_SHAPES, K=2)
    g = torch.ones(3, 4) * 2.0
    proj.update_dbar(layer_number=0)
    proj.update_desc_dir(sparse_grad_orth=g, alpha=0, layer_number=0)
    proj.update_dbar(layer_number=0)
    proj.update_desc_dir(sparse_grad_orth=g, alpha=0, layer_number=0)
    proj.update_dbar(layer_number=0)
    assert proj.dbar[0].shape == torch.Size(LAYER_SHAPES[0])
    assert torch.allclose(proj.dbar[0], g)


# ErrorFeedback21
def test_ef21_desc_dir_none_at_init() -> None:
    ef21 = ErrorFeedback21(n_layers=N_LAYERS)
    assert all(d is None for d in ef21.desc_dir)


def test_ef21_first_update_stores_gradient() -> None:
    ef21 = ErrorFeedback21(n_layers=N_LAYERS)
    grad = torch.tensor([1.0, 2.0])
    ef21.update_desc_dir(sparse_grad_diff=grad, layer_number=0)
    assert torch.equal(ef21.desc_dir[0], grad)


def test_ef21_subsequent_update_applies_momentum() -> None:
    gamma = 0.75
    ef21 = ErrorFeedback21(n_layers=N_LAYERS, gamma=gamma)
    g1 = torch.tensor([2.0, 4.0])
    g2 = torch.tensor([1.0, 1.0])
    ef21.update_desc_dir(sparse_grad_diff=g1, layer_number=0)
    ef21.update_desc_dir(sparse_grad_diff=g2, layer_number=0)
    expected = gamma * g1 + g2
    assert torch.allclose(ef21.desc_dir[0], expected)


def test_ef21_independent_layers() -> None:
    ef21 = ErrorFeedback21(n_layers=N_LAYERS)
    g0 = torch.tensor([1.0])
    g1 = torch.tensor([2.0])
    ef21.update_desc_dir(sparse_grad_diff=g0, layer_number=0)
    ef21.update_desc_dir(sparse_grad_diff=g1, layer_number=1)
    assert torch.equal(ef21.desc_dir[0], g0)
    assert torch.equal(ef21.desc_dir[1], g1)


# Diana
def test_diana_h_none_at_init() -> None:
    diana = Diana(n_layers=N_LAYERS, alpha=0.5, beta=0.9)
    assert all(h is None for h in diana.h)


def test_diana_desc_dir_none_at_init() -> None:
    diana = Diana(n_layers=N_LAYERS, alpha=0.5, beta=0.9)
    assert all(d is None for d in diana.desc_dir)


def test_diana_update_h_first_call() -> None:
    alpha = 0.5
    diana = Diana(n_layers=N_LAYERS, alpha=alpha, beta=0.9)
    m = torch.tensor([2.0, 4.0])
    diana.update_h(m=m, layer_number=0)
    assert torch.allclose(diana.h[0], alpha * m)


def test_diana_update_h_subsequent() -> None:
    alpha, gamma = 0.5, 0.75
    diana = Diana(n_layers=N_LAYERS, alpha=alpha, beta=0.9, gamma=gamma)
    m1 = torch.tensor([2.0, 4.0])
    m2 = torch.tensor([1.0, 1.0])
    diana.update_h(m=m1, layer_number=0)
    diana.update_h(m=m2, layer_number=0)
    h1 = alpha * m1
    expected = gamma * h1 + alpha * m2
    assert torch.allclose(diana.h[0], expected)


def test_diana_update_desc_dir_first_call() -> None:
    diana = Diana(n_layers=N_LAYERS, alpha=0.5, beta=0.9)
    sparse_grad = torch.tensor([1.0, 2.0])
    diana.update_h(m=sparse_grad, layer_number=0)
    diana.update_desc_dir(sparse_grad=sparse_grad, layer_number=0)
    assert torch.equal(diana.desc_dir[0], sparse_grad)


def test_diana_update_desc_dir_subsequent() -> None:
    alpha, beta, gamma = 0.5, 0.9, 0.75
    diana = Diana(n_layers=N_LAYERS, alpha=alpha, beta=beta, gamma=gamma)
    g1 = torch.tensor([1.0, 0.0])
    g2 = torch.tensor([0.0, 1.0])

    diana.update_h(m=g1, layer_number=0)
    diana.update_desc_dir(sparse_grad=g1, layer_number=0)

    h1 = alpha * g1
    diana.update_h(m=g2, layer_number=0)
    diana.update_desc_dir(sparse_grad=g2, layer_number=0)

    h2 = gamma * h1 + alpha * g2
    expected = beta * g1 + gamma * h2 + g2
    assert torch.allclose(diana.desc_dir[0], expected)
