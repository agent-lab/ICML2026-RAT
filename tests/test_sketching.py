import torch

from utils.sketching import normalize_score_kernel, score_kernel, solve_score_kernel_system


def test_exact_score_kernel_matches_matmul():
    H = torch.randn(8, 31)
    K = score_kernel(H, kernel="exact")
    torch.testing.assert_close(K, H @ H.t() / H.shape[0])


def test_random_countsketch_kernel_is_psd_and_reproducible():
    H = torch.randn(16, 127)
    K1 = score_kernel(H, kernel="countsketch_random", sketch_dim=32, sketch_seed=123)
    K2 = score_kernel(H, kernel="countsketch_random", sketch_dim=32, sketch_seed=123)

    torch.testing.assert_close(K1, K2)
    torch.testing.assert_close(K1, K1.t())
    min_eig = torch.linalg.eigvalsh(K1).min()
    assert min_eig > -1e-5


def test_row_rescale_preserves_exact_diagonal_without_breaking_psd():
    H = torch.randn(16, 127)
    K_exact = score_kernel(H, kernel="exact")
    K_rescale = score_kernel(
        H,
        kernel="countsketch_random",
        sketch_dim=32,
        sketch_seed=123,
        diagonal_mode="row_rescale",
    )

    torch.testing.assert_close(K_rescale.diag(), K_exact.diag(), rtol=1e-5, atol=1e-6)
    min_eig = torch.linalg.eigvalsh(K_rescale).min()
    assert min_eig > -1e-5


def test_kernel_normalization_returns_correlation_scaled_matrix_and_row_scale():
    H = torch.randn(16, 127)
    K_norm, row_scale = normalize_score_kernel(score_kernel(H), normalization="correlation")

    assert row_scale is not None
    torch.testing.assert_close(K_norm.diag(), torch.ones_like(K_norm.diag()), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(K_norm, K_norm.t())


def test_normalized_solve_scales_solution_back_internally():
    H = torch.randn(16, 127)
    rhs = torch.randn(16)
    ratio = torch.rand(16) + 0.5
    damping = 0.1

    K, row_scale = normalize_score_kernel(score_kernel(H), normalization="correlation")
    manual = torch.linalg.solve(
        K * ratio.unsqueeze(0) + damping * torch.eye(K.shape[0]),
        rhs,
    )
    manual = row_scale * manual

    solved = solve_score_kernel_system(
        H,
        rhs,
        damping,
        ratio=ratio,
        normalization="correlation",
    )
    torch.testing.assert_close(solved, manual)


def test_exact_solve_matches_original_rat_formula():
    H = torch.randn(16, 127)
    rhs = torch.randn(16)
    ratio = torch.rand(16) + 0.5
    previous_projection = torch.randn(16)
    damping = 0.1

    K = H @ H.t() / H.shape[0]
    manual = torch.linalg.solve(
        K * ratio.unsqueeze(0) + damping * torch.eye(K.shape[0]),
        rhs - previous_projection,
    )
    solved = solve_score_kernel_system(
        H,
        rhs,
        damping,
        ratio=ratio,
        previous_projection=previous_projection,
    )
    torch.testing.assert_close(solved, manual)


def test_exact_matrix_rhs_solve_matches_original_shared_rat_formula():
    H = torch.randn(16, 127)
    rhs = torch.randn(16, 2)
    ratio = torch.rand(16) + 0.5
    previous_projection = torch.randn(16)
    damping = 0.1

    K = H @ H.t() / H.shape[0]
    manual = torch.linalg.solve(
        K * ratio.unsqueeze(0) + damping * torch.eye(K.shape[0]),
        rhs - previous_projection.unsqueeze(1),
    )
    solved = solve_score_kernel_system(
        H,
        rhs,
        damping,
        ratio=ratio,
        previous_projection=previous_projection,
    )
    torch.testing.assert_close(solved, manual)
