"""
Script containing utility functions for the main notebook. Made by Ines Marques and Pablo Bakker.
"""

# Importing libraries
import numpy as np


# Function to build the controlability matrix
def build_C(A, T):
    return np.hstack([np.linalg.matrix_power(A, T - 1 - k) for k in range(T)])


# Function to get matrix diagnostics
def matrix_diagnostics(M, name):
    svals = np.linalg.svd(M, compute_uv=False)

    print(name)
    print("-" * len(name))
    print("Rank:", np.linalg.matrix_rank(M))
    print("Largest singular value :", svals[0])
    print("Smallest singular value:", svals[-1])
    print("Condition number:", svals[0] / svals[-1])
    print()


# Least squares sanity check function
def ls_sanity_check(C, C_tilde, safe_col_norms, Y, num_tests=10):
    raw_errors = []
    normalized_errors = []

    for i in range(num_tests):
        y = Y[i]

        u_raw, *_ = np.linalg.lstsq(C, y, rcond=None)
        y_hat_raw = C @ u_raw
        raw_errors.append(np.linalg.norm(y - y_hat_raw) / np.linalg.norm(y))

        z, *_ = np.linalg.lstsq(C_tilde, y, rcond=None)
        u = z / safe_col_norms
        y_hat = C @ u
        normalized_errors.append(np.linalg.norm(y - y_hat) / np.linalg.norm(y))
    print("Least-squares sanity check")
    print("--------------------------")
    print("Raw C mean error:", np.mean(raw_errors))
    print("Normalized mean error:", np.mean(normalized_errors))
    print("Raw C max error:", np.max(raw_errors))
    print("Normalized max error:", np.max(normalized_errors))


# OMP implementation
def omp(Phi, y, K):
    """
    Standard OMP with fixed global sparsity K.
    Used as a relaxed baseline with K = T*s.
    """
    m, n = Phi.shape

    residual = y.copy()
    support = []
    residual_norms = [np.linalg.norm(residual)]

    z_hat = np.zeros(n)

    for _ in range(K):
        corr = Phi.T @ residual

        if support:
            corr[support] = 0

        idx = np.argmax(np.abs(corr))
        support.append(idx)

        Phi_S = Phi[:, support]
        z_S, *_ = np.linalg.lstsq(Phi_S, y, rcond=None)

        residual = y - Phi_S @ z_S
        residual_norms.append(np.linalg.norm(residual))

    z_hat[support] = z_S

    return z_hat, support, residual_norms


# Function for evaluating OMP over all target states
def evaluate_omp(Xf, C, C_tilde, safe_col_norms, s, N=25, T=25):
    """
    OMP uses the same total budget as the piecewise problem:
        K = T*s
    but does not enforce s nonzeros per block.
    """
    K = T * s

    errors = []
    energies = []
    U_hat = []

    for y in Xf:
        z_hat, _, _ = omp(C_tilde, y, K)

        u_hat = z_hat / safe_col_norms
        x_hat = C @ u_hat

        errors.append(np.linalg.norm(y - x_hat) / np.linalg.norm(y))
        energies.append(np.linalg.norm(u_hat))
        U_hat.append(u_hat)

    return {
        "errors": np.array(errors),
        "energies": np.array(energies),
        "U_hat": np.array(U_hat),
    }


# Function for checking block sparsity
def block_sparsity_counts(U_hat, N=25, T=25, threshold=1e-10):
    """
    Returns block sparsity counts for each experiment.
    Output shape: num_experiments x T
    """
    counts = []

    for u in U_hat:
        U = u.reshape(T, N)
        block_counts = np.count_nonzero(np.abs(U) > threshold, axis=1)
        counts.append(block_counts)

    return np.array(counts)


# Getting heatmap of the support
def support_frequency_from_U(U_hat, N=25, T=25, threshold=1e-10):
    counts = np.zeros((T, N))

    for u in U_hat:
        U = u.reshape(T, N)
        counts += (np.abs(U) > threshold)

    frequency = counts / len(U_hat)

    return counts, frequency


# Function to get the largest k indices
def top_k_abs_indices(v, k):
    """Indices of the k largest absolute entries."""
    k = min(k, len(v))
    if k <= 0:
        return np.array([], dtype=int)
    return np.argpartition(np.abs(v), -k)[-k:]


# Function to threshold the top s entires of a block
def piecewise_prune(z, s, N=25, T=25):
    """Keep at most s entries inside each time block."""
    z_pruned = np.zeros_like(z)

    for t in range(T):
        start, end = t*N, (t+1)*N
        local_idx = top_k_abs_indices(z[start:end], s)
        z_pruned[start + local_idx] = z[start + local_idx]

    return z_pruned


# POMP Implementation
def pomp(Phi, y, s, N=25, T=25, max_iter=None, tol=1e-12):
    """
    Piecewise OMP.

    Enforces:
        ||u_t||_0 <= s  for every time block t.

    The best iterate is returned, since pruning can occasionally increase
    the residual after later iterations.
    """
    if max_iter is None:
        max_iter = s

    n = N * T
    z = np.zeros(n)
    support = np.array([], dtype=int)

    residual = y.copy()
    residual_norms = [np.linalg.norm(residual)]

    best_z = z.copy()
    best_residual_norm = residual_norms[0]

    for _ in range(max_iter):
        proxy = Phi.T @ residual

        new_support = []
        for t in range(T):
            start, end = t*N, (t+1)*N
            local_idx = top_k_abs_indices(proxy[start:end], s)
            new_support.extend(start + local_idx)

        merged_support = np.union1d(support, np.array(new_support))

        Phi_S = Phi[:, merged_support]
        z_S, *_ = np.linalg.lstsq(Phi_S, y, rcond=None)

        z_temp = np.zeros(n)
        z_temp[merged_support] = z_S

        z = piecewise_prune(z_temp, s=s, N=N, T=T)
        support = np.flatnonzero(np.abs(z) > 1e-12)

        residual = y - Phi @ z
        res_norm = np.linalg.norm(residual)
        residual_norms.append(res_norm)

        if res_norm < best_residual_norm:
            best_residual_norm = res_norm
            best_z = z.copy()

        if res_norm <= tol:
            break

    best_support = np.flatnonzero(np.abs(best_z) > 1e-12)

    return best_z, best_support, residual_norms


# Function to evaluate POMP over all the targets
def evaluate_pomp(Xf, C, C_tilde, safe_col_norms, s, N=25, T=25):
    errors = []
    energies = []
    U_hat = []

    for y in Xf:
        z_hat, _, _ = pomp(
            Phi=C_tilde,
            y=y,
            s=s,
            N=N,
            T=T,
            max_iter=s
        )

        u_hat = z_hat / safe_col_norms
        x_hat = C @ u_hat

        errors.append(np.linalg.norm(y - x_hat) / np.linalg.norm(y))
        energies.append(np.linalg.norm(u_hat))
        U_hat.append(u_hat)

    return {
        "errors": np.array(errors),
        "energies": np.array(energies),
        "U_hat": np.array(U_hat),
    }


# Piecewise SBL implementation
def pcsbl_piecewise(Phi, y, N=25, T=25,
                    beta_couple=1.0,
                    a=0.5, b=1e-4,
                    noise_var=1e-6,
                    max_iter=300,
                    tol=1e-6,
                    alpha_max=1e12,
                    prune_thresh=1e-4,
                    per_block_cap=None):
    """
    Pattern-Coupled SBL (Fang, Shen, Li, Wang, IEEE TSP 2014),
    adapted to PIECEWISE structure: hyperparameter coupling is cut at
    block boundaries so clustering pressure stays within each time block
    (matches U2 piecewise, not U3 block sparsity).

    Prior:  z_j ~ N(0, (alpha_j + beta*alpha_{j-1} + beta*alpha_{j+1})^-1)
            with neighbors restricted to the same length-N block.

    beta_couple = 0  -> reduces to plain SBL.
    beta_couple > 0  -> couples neighbors within each block.

    E-step uses Woodbury (work in m x m measurement space, since m << n).

    Returns
    -------
    z_hat   : (n,) recovered coefficients (normalized-column space)
    support : indices of nonzeros after pruning
    info    : dict (alpha, n_iter, converged, z_dense)
    """
    m, n = Phi.shape
    beta_noise = 1.0 / noise_var          # measurement precision

    # --- Precompute within-block neighbor index lists ---
    # left[j]  = index of left neighbor (or -1 if none / across boundary)
    # right[j] = index of right neighbor (or -1)
    left = np.full(n, -1, dtype=int)
    right = np.full(n, -1, dtype=int)
    for t in range(T):
        start, end = t * N, (t + 1) * N
        idx = np.arange(start, end)
        left[idx[1:]] = idx[:-1]     # each has a left neighbor except block-first
        right[idx[:-1]] = idx[1:]    # each has a right neighbor except block-last

    alpha = np.ones(n)               # hyper-precisions
    z_hat = np.zeros(n)
    delta = np.inf

    for it in range(max_iter):
        alpha_prev = alpha.copy()

        # Effective per-coefficient precision (pattern-coupled) 
        eff_prec = alpha.copy()
        has_left = left >= 0
        has_right = right >= 0
        eff_prec[has_left]  += beta_couple * alpha[left[has_left]]
        eff_prec[has_right] += beta_couple * alpha[right[has_right]]

        gamma = 1.0 / eff_prec # effective prior variances (diag of prior cov)

        # E-step via Woodbury (m x m solve) 
        PhiG = Phi * gamma[None, :]                 # Phi @ diag(gamma)
        B = (1.0 / beta_noise) * np.eye(m) + PhiG @ Phi.T
        Binv = np.linalg.inv(B)

        mu = gamma * (Phi.T @ (Binv @ y))           # posterior mean
        BinvPhi = Binv @ Phi
        diag_quad = np.sum(Phi * BinvPhi, axis=0)   # phi_j^T B^-1 phi_j
        Sigma_diag = gamma - (gamma**2) * diag_quad # posterior variances

        z_hat = mu
        energy = mu**2 + Sigma_diag                 # per-coefficient "energy"

        # M-step: pattern-coupled update for alpha
        # Each coefficient's energy is attributed to itself and (with weight
        # beta) to its within-block neighbors. omega_j aggregates these.
        omega = energy.copy()
        omega[has_left]  += beta_couple * energy[left[has_left]]
        omega[has_right] += beta_couple * energy[right[has_right]]

        alpha = (2.0 * a + 1.0) / (2.0 * b + omega)
        alpha = np.minimum(alpha, alpha_max)

        delta = np.linalg.norm(alpha - alpha_prev) / (np.linalg.norm(alpha_prev) + 1e-300)
        if delta < tol:
            break

    # Natural pruning relative to largest coefficient 
    z_pruned = z_hat.copy()
    peak = np.max(np.abs(z_pruned))
    if peak > 0:
        z_pruned[np.abs(z_pruned) < prune_thresh * peak] = 0.0

    support = np.flatnonzero(z_pruned)
    info = {
        "alpha": alpha,
        "n_iter": it + 1,
        "converged": delta < tol,
        "z_dense": z_hat,
    }
    return z_pruned, support, info


# Evaluate PCSBL over all the final states
def evaluate_pcsbl(Xf, C, C_tilde, safe_col_norms, N=25, T=25,
                   beta_couple=1.0, a=0.5, b=1e-4, noise_var=1e-6,
                   prune_thresh=1e-4, per_block_cap=None):
    errors, energies, U_hat, n_iters = [], [], [], []

    for y in Xf:
        z_hat, _, info = pcsbl_piecewise(
            Phi=C_tilde, y=y, N=N, T=T,
            beta_couple=beta_couple, a=a, b=b,
            noise_var=noise_var, prune_thresh=prune_thresh,
            per_block_cap=per_block_cap,
        )
        u_hat = z_hat / safe_col_norms
        x_hat = C @ u_hat

        errors.append(np.linalg.norm(y - x_hat) / np.linalg.norm(y))
        energies.append(np.linalg.norm(u_hat))
        U_hat.append(u_hat)
        n_iters.append(info["n_iter"])

    return {
        "errors": np.array(errors),
        "energies": np.array(energies),
        "U_hat": np.array(U_hat),
        "n_iters": np.array(n_iters),
    }