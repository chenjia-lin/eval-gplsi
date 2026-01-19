import time
import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import svds
import os
from multiprocessing import Pool
from functools import partial


from .utils import get_folds_disconnected_G, interpolate_X
import pycvxcluster.pycvxcluster



def _subspace_gap(U_old, U_new):
    # U_old, U_new: (n x K) with orthonormal columns (approximately)
    K = U_new.shape[1]
    S = U_old.T @ U_new
    # ||UU^T - VV^T||_F = sqrt(2K - 2||U^T V||_F^2)
    g2 = max(0.0, 2 * K - 2 * (np.linalg.norm(S, "fro") ** 2))
    return np.sqrt(g2)


def graphSVD(
    X,
    N,
    K,
    edge_df,
    weights,
    lamb_start,
    step_size,
    grid_len,
    maxiter,
    eps,
    verbose,
    initialize,
):
    n = X.shape[0]
    # folds is a dictionary with five key-value pairs
    srn, folds, G, mst = get_folds_disconnected_G(edge_df)
    
    lambd_grid = (lamb_start * np.power(step_size, np.arange(grid_len))).tolist()
    lambd_grid.insert(0, 1e-6)

    if initialize:
        if verbose:
            print("Initializing...")
        
        t0 = time.time()
        print("[graphSVD] Computing colsums...")
        colsums = np.sum(X, axis=0)
        print(f"    {time.time() - t0} s"); t0 = time.time()
        print("[graphSVD] Computing cov...")
        cov = X.T @ X - np.diag(colsums / N)
        print(f"    {time.time() - t0} s"); t0 = time.time()
        print("[graphSVD] Running svds(cov, k=K)...")
        U_cov, s_cov, VT_cov = svds(cov, k=K)
        print(f"    {time.time() - t0} s"); t0 = time.time()
        V = VT_cov.T
        L = s_cov  # keep as 1-D
        V_init, L_init = V, L
        print("[graphSVD] Running svds(X, k=K)")
        U, sX, VTX = svds(X, k=K)
        print(f"    {time.time() - t0} s"); t0 = time.time()
        U_init = U
    else:
        U, sX, VTX = svds(X, k=K)
        V = VTX.T
        L = sX  # keep as vector
        U_init = None
        V_init = None
        L_init = None

    score = 1.0
    niter = 0
    while score > eps and niter < maxiter:
        U_old = U
        V_old = V

        # Update U (with CV across folds), then V,L
        #P_U_old = np.dot(U, U.T) 
        #P_V_old = np.dot(V, V.T) 
        #X_hat_old = (P_U_old @ X) @ P_V_old

        t0 = time.time()
        print("[graphSVD] Running update_U_tilde...")
        U, lambd, lambd_errs = update_U_tilde(X, V, L, G, weights, folds, lambd_grid)
        print(f"    {time.time() - t0} s"); t0 = time.time()
        print("[graphSVD] Running update_V_L_tilde...")
        V, L = update_V_L_tilde(X, U)
        print(f"    {time.time() - t0} s"); t0 = time.time()

        # # Subspace convergence (cheap): no multiplies by X
        print("[graphSVD] Performing _subspace_gap(U_old, U)...")
        gapU = _subspace_gap(U_old, U)
        print(f"    {time.time() - t0} s"); t0 = time.time()
        print("[graphSVD] Performing _subspace_gap(V_old, V)...")
        gapV = _subspace_gap(V_old, V)
        print(f"    {time.time() - t0} s"); t0 = time.time()
        print("[graphSVD] Performing (gapU + gapV) / (2 * np.sqrt(K))...")
        score = (gapU + gapV) / (2 * np.sqrt(K))
        print(f"    {time.time() - t0} s"); t0 = time.time()
        # 
        #P_U = np.dot(U, U.T)
        #P_V = np.dot(V, V.T)
        #X_hat = (P_U @ X) @ P_V 
        #score = norm(X_hat-X_hat_old)/n
        niter += 1
        if verbose == 1:
            print(f"Convergence score = {score:.6g}")

    if verbose:
        print(f"Graph-aligned SVD ran for {niter} steps.")

    return U, V, L, U_init, V_init, L_init, lambd, lambd_errs, niter


def lambda_search(j, folds, X, V, L, G, weights, lambd_grid):
    fold = folds[j]
    X_tilde = interpolate_X(X, G, folds, j)
    # print(f"Type X_tilde in graphsvd.py: {type(X_tilde)}")
    # print(f"Type V in graphsvd.py: {type(V)}")
    X_tildeV = X_tilde @ V
    # print(f"Type X_tildeV in graphsvd.py: {type(X_tildeV)}")
    # print(f"Type X in graphsvd.py: {type(X)}")
    X_j = X[fold, :] @ V


    errs = []
    best_err = float("inf")
    U_best = None
    lambd_best = 0.0

    # Cheaper inner settings
    ssnal = pycvxcluster.pycvxcluster.SSNAL(verbose=0, admm_iter=0, maxiter=200, stoptol=1e-5)

    rises = 0
    last_err = float("inf")

    for fitn, lambd in enumerate(lambd_grid):
        ssnal.gamma = lambd
        ssnal.fit(
            X=X_tildeV,
            weight_matrix=weights,
            save_centers=True,
            save_labels=False,
            recalculate_weights= (fitn == 0),
        )
        # warm-start
        ssnal.kwargs["x0"] = ssnal.centers_
        ssnal.kwargs["y0"] = ssnal.y_
        ssnal.kwargs["z0"] = ssnal.z_

        U_tilde = ssnal.centers_.T
        err = norm(X_j - U_tilde[fold, :]) / max(1, len(fold))
        errs.append(err)

        if err < best_err:
            lambd_best = lambd
            U_best = U_tilde
            best_err = err

        rises = rises + 1 if err > last_err else 0
        last_err = err
        if rises >= 3:  # early stop when rising consecutively
            break

    return j, errs, U_best, lambd_best


def update_U_tilde(X, V, L, G, weights, folds, lambd_grid):
    lambds_best = []
    lambd_errs = {"fold_errors": {}, "final_errors": []}

    # Keep "X @ V" (do not build diag(L) or invert unless theory requires)
    XV = X @ V
    print("[update_U_tilde] Computed XV = X @ V")

    # For getting the correct CPU count on the cluster
    try:
        # For cluster runs
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # For local runs
        num_cpus = os.cpu_count()

    print(f"[update_U_tilde] len(folds): {len(folds)}")
    print(f"[update_U_tilde] num_cpus: {num_cpus}")
    print("[update_U_tilde] Pooling...")
    with Pool(min(len(folds), num_cpus)) as p:
        results = p.starmap(
            lambda_search,
            [(j, folds, X, V, L, G, weights, lambd_grid) for j in folds.keys()],
        )
    print("[update_U_tilde] Obtained results...")

    print("[update_U_tilde] Looping over results...")
    for j, errs, _, lambd_best in results:
        lambd_errs["fold_errors"][j] = errs
        lambds_best.append(lambd_best)

    # Aggregate CV errors over available folds
    print("[update_U_tilde] Aggregating CV errors over available folds...")
    common_len = min(len(v) for v in lambd_errs["fold_errors"].values())
    cv_errs = np.sum([np.array(lambd_errs["fold_errors"][i][:common_len]) for i in lambd_errs["fold_errors"]], axis=0)
    lambd_cv = lambd_grid[int(np.argmin(cv_errs))]

    # Final, stricter fit
    ssnal = pycvxcluster.pycvxcluster.SSNAL(gamma=lambd_cv, verbose=0, admm_iter=50, maxiter=1000, stoptol=1e-6)
    ssnal.fit(X=XV, weight_matrix=weights, save_centers=True)
    U_tilde = ssnal.centers_.T

    # Orthonormalize via QR (cheaper than SVD)
    U_hat, _ = np.linalg.qr(U_tilde, mode="reduced")

    if np.isnan(U_hat).any():
        # Fallback to SVD if QR fails (rare)
        U_hat, _, _ = np.linalg.svd(U_tilde, full_matrices=False)

    if isinstance(lambd_cv, np.ndarray):
        lambd_cv = float(lambd_cv)

    if hasattr(ssnal, "centers_"):
        lambd_errs["final_errors"] = []  # placeholder; keep structure

    if isinstance(U_hat, np.matrix):
        U_hat = np.asarray(U_hat)

    if np.any(np.isinf(U_hat)):
        U_hat = np.nan_to_num(U_hat, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Optimal lambda is {lambd_cv}...")
    return U_hat, lambd_cv, lambd_errs


def update_V_L_tilde(X, U_tilde):
    V_mul = X.T @ U_tilde
    V_hat, s_hat, _ = np.linalg.svd(V_mul, full_matrices=False)
    # Keep L as vector
    return V_hat, s_hat
