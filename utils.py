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
def sbl_piecewise(
    Phi, y, s, N=25, T=25,
    max_iter=500,
    tol=1e-8,
    sigma2=1e-8,
    gamma_init=1.0,
    prune_gamma=1e-12,
    refit=True,
):
    """
    Piecewise Sparse Bayesian Learning.

    Learns one variance gamma_{i,t} per coefficient, then enforces
        ||u_t||_0 <= s
    by keeping the s largest posterior variances per time block.

    Returns:
        z              final estimate
        support        active indices
        residual_norms residual history
        gamma          learned SBL hyperparameters
    """

    m, n = Phi.shape
    assert n == N * T

    gamma = gamma_init * np.ones(n)
    residual_norms = []

    z = np.zeros(n)

    for _ in range(max_iter):
        gamma_old = gamma.copy()

        Gamma = np.diag(gamma)

        Sigma_y = Phi @ Gamma @ Phi.T + sigma2 * np.eye(m)

        # Posterior mean
        alpha = np.linalg.solve(Sigma_y, y)
        mu = gamma * (Phi.T @ alpha)

        # Posterior covariance diagonal
        B = np.linalg.solve(Sigma_y, Phi @ Gamma)
        Sigma_diag = gamma - np.sum((Gamma @ Phi.T) * B.T, axis=1)

        # EM update
        gamma = mu**2 + Sigma_diag
        gamma = np.maximum(gamma, 0.0)

        # Numerical pruning
        gamma[gamma < prune_gamma] = 0.0

        z = mu.copy()

        residual = y - Phi @ z
        residual_norms.append(np.linalg.norm(residual))

        rel_change = np.linalg.norm(gamma - gamma_old) / (np.linalg.norm(gamma_old) + 1e-16)

        if rel_change < tol:
            break

    # Enforce piecewise sparsity: keep top s gammas in each time block
    support = []

    for t in range(T):
        start, end = t * N, (t + 1) * N
        local_gamma = gamma[start:end]

        if s >= N:
            local_support = np.arange(N)
        else:
            local_support = np.argsort(np.abs(local_gamma))[-s:]

        local_support = local_support[local_gamma[local_support] > prune_gamma]
        support.extend(start + local_support)

    support = np.array(sorted(support), dtype=int)

    z_final = np.zeros(n)

    if len(support) > 0:
        if refit:
            Phi_S = Phi[:, support]
            z_S, *_ = np.linalg.lstsq(Phi_S, y, rcond=None)
            z_final[support] = z_S
        else:
            z_final[support] = z[support]

    residual = y - Phi @ z_final
    residual_norms.append(np.linalg.norm(residual))

    return z_final, support, residual_norms, gamma




# Evaluate PCSBL over all the final states
def evaluate_sbl_piecewise(Xf, C, C_tilde, safe_col_norms, s, N=25, T=25):
    errors = []
    energies = []
    U_hat = []
    gammas = []

    for y in Xf:
        z_hat, _, _, gamma = sbl_piecewise(
            Phi=C_tilde,
            y=y,
            s=s,
            N=N,
            T=T,
            max_iter=500,
            tol=1e-8,
            sigma2=1e-8,
            refit=True,
        )

        u_hat = z_hat / safe_col_norms
        x_hat = C @ u_hat

        errors.append(np.linalg.norm(y - x_hat) / np.linalg.norm(y))
        energies.append(np.linalg.norm(u_hat))
        U_hat.append(u_hat)
        gammas.append(gamma)

    return {
        "errors": np.array(errors),
        "energies": np.array(energies),
        "U_hat": np.array(U_hat),
        "gammas": np.array(gammas),
    }