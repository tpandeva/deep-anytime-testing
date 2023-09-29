import numpy as np
from data.blob import sample_D_blobs
from scipy.spatial.distance import cdist
# rbf kernel
def rbf_kernel(x, y=None, bw=1.0, amp=1.0):
    y = x if y is None else y
    dists = cdist(x, y)
    squared_dists = dists * dists
    k = amp * np.exp( -(1/(2*bw*bw)) * squared_dists )
    return k

def mmd2(X, Y, kernel_func):
    Kxx = kernel_func(X, X)
    Kyy = kernel_func(Y, Y)
    Kxy = kernel_func(X, Y)

    n, m = len(X), len(Y)

    term1 = Kxx.sum()
    term2 = Kyy.sum()
    term3 = 2*Kxy.mean()
    term1 -= np.trace(Kxx)
    term2 -= np.trace(Kyy)
    MMD_squared = (term1/(n*(n-1)) + term2/(m*(m-1)) - term3)

    return MMD_squared

def mmd_test_rbf(X,Y, num_perms):
    Z = np.vstack((X, Y))
    n = X.shape[0]
    m = Y.shape[0]
    perm_statistics = np.zeros((num_perms,))
    stat = mmd2(X, Y, rbf_kernel)
    for i in range(num_perms):
        perm = np.random.permutation(n+m)
        X_, Y_ = Z[perm[:n]], Z[perm[n:]]
        stat_perm = mmd2(X_, Y_, rbf_kernel)
        perm_statistics[i] = stat_perm
    # obtain the threshold

    perm_statistics = np.sort(perm_statistics)
    p_val = np.sum(perm_statistics > stat) / num_perms
    return p_val

