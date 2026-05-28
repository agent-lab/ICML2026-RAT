import torch


_COUNT_SKETCH_MAP_CACHE = {}


def _device_key(device):
    if device.type == "cuda":
        return (device.type, device.index if device.index is not None else torch.cuda.current_device())
    return (device.type, device.index)


def _countsketch_maps(p, r, device, seed):
    key = (p, r, _device_key(device), int(seed))
    cached = _COUNT_SKETCH_MAP_CACHE.get(key)
    if cached is not None:
        return cached

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    bucket = torch.randint(0, r, (p,), device=device, generator=gen, dtype=torch.long)
    sign = torch.randint(0, 2, (p,), device=device, generator=gen, dtype=torch.long)
    sign = sign.mul_(2).sub_(1)
    _COUNT_SKETCH_MAP_CACHE[key] = (bucket, sign)
    return bucket, sign


def countsketch_scores(H, sketch_dim, sketch_seed):
    if H.dim() != 2:
        raise ValueError(f"H must be 2D, got shape {tuple(H.shape)}")

    m, p = H.shape
    bucket, sign = _countsketch_maps(p, sketch_dim, H.device, sketch_seed)
    Z = torch.zeros(m, sketch_dim, device=H.device, dtype=H.dtype)
    Z.scatter_add_(
        dim=1,
        index=bucket.unsqueeze(0).expand(m, p),
        src=H * sign.to(dtype=H.dtype).unsqueeze(0),
    )
    return Z


def score_kernel(
    H,
    kernel="exact",
    sketch_dim=None,
    sketch_seed=0,
    diagonal_mode="z_only",
    eps=1e-12,
):
    """Return a sample-space score Gram matrix K = H H^T / m or a sketch of it."""
    if H.dim() != 2:
        raise ValueError(f"H must be 2D, got shape {tuple(H.shape)}")

    kernel = str(kernel or "exact").lower()
    diagonal_mode = str(diagonal_mode or "z_only").lower()

    m = H.shape[0]
    if kernel == "exact":
        return H @ H.t() / m

    if kernel in ("countsketch", "countsketch_random", "random_countsketch"):
        if sketch_dim is None or sketch_dim <= 0:
            raise ValueError("sketch_dim must be positive for CountSketch kernels")
        Z = countsketch_scores(H, int(sketch_dim), sketch_seed)

        if diagonal_mode == "z_only":
            pass
        elif diagonal_mode == "row_rescale":
            h_norm = H.norm(dim=1)
            z_norm = Z.norm(dim=1).clamp_min(eps)
            Z = Z * (h_norm / z_norm).unsqueeze(1)
        else:
            raise ValueError(f"unknown sketch diagonal_mode: {diagonal_mode}")

        return Z @ Z.t() / m

    raise ValueError(f"unknown Fisher kernel: {kernel}")


def normalize_score_kernel(K, normalization="none", eps=1e-12):
    norm_mode = str(normalization or "none").lower()
    if norm_mode not in {"diag", "diagonal", "correlation", "row", "row_norm", "sample", "sample_norm"}:
        return K, None

    row_scale = K.diag().clamp_min(eps).rsqrt()
    K = K * row_scale[:, None] * row_scale[None, :]
    return 0.5 * (K + K.t()), row_scale


def solve_score_kernel_system(
    H,
    rhs,
    damping,
    ratio=None,
    kernel="exact",
    sketch_dim=None,
    sketch_seed=0,
    diagonal_mode="z_only",
    normalization="none",
    normalization_eps=1e-12,
    previous_projection=None,
):
    """Build the score kernel and solve the RAT sample-space system.

    When normalization is enabled, the solve is performed with K <- S K S and
    the returned solution is scaled back by S. This keeps the trainer decoupled
    from the normalization details.
    """
    K = score_kernel(
        H,
        kernel=kernel,
        sketch_dim=sketch_dim,
        sketch_seed=sketch_seed,
        diagonal_mode=diagonal_mode,
    )
    K, row_scale = normalize_score_kernel(K, normalization=normalization, eps=normalization_eps)

    rhs_eff = rhs.to(dtype=K.dtype)
    if previous_projection is not None:
        projection = previous_projection.to(dtype=K.dtype)
        if row_scale is not None:
            projection = row_scale.to(dtype=K.dtype) * projection
        if rhs_eff.dim() == 1:
            rhs_eff = rhs_eff - projection
        else:
            rhs_eff = rhs_eff - projection.unsqueeze(1)

    if ratio is not None:
        K = K * ratio.to(dtype=K.dtype).unsqueeze(0)

    eye = torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
    sol = torch.linalg.solve(K + damping * eye, rhs_eff)

    if row_scale is None:
        return sol
    if sol.dim() == 1:
        return row_scale.to(dtype=sol.dtype) * sol
    return row_scale.to(dtype=sol.dtype).unsqueeze(1) * sol
