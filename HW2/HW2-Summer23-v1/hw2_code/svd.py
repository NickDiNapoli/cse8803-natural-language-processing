import numpy as np


class SVD(object):
    def __init__(self):
        pass

    def svd(self, data):
        """
        Do SVD. You could use numpy SVD.
        Args:
                data: (N, D) TF-IDF features for the data.

        Return:
                U: (N,N) numpy array
                S: (min(N,D), ) numpy array
                V^T: (D,D) numpy array
        """
        # print(f"data shape = {data.shape}")
        U, S, V_T = np.linalg.svd(data, full_matrices=True)
        # print(U.shape, S.shape, V_T.shape)
        return U, S, V_T

    def rebuild_svd(self, U, S, V, k):
        """
        Rebuild SVD by k componments.

        Args:
                U: (N,N) numpy array
                S: (min(N,D), ) numpy array
                V: (D,D) numpy array
                k: int corresponding to number of components

        Return:
                data_rebuild: (N,D) numpy array

        Hint: numpy.matmul may be helpful for reconstruction.
        """
        # data_rebuild = np.matmul(S[0:k], V.T[0:k, :])
        # data_rebuild = np.matmul(U[:, 0:k], S[0:k])
        # data_rebuild = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
        data_rebuild = U[:, 0:k] @ np.diag(S[0:k]) @ V[0:k, :]
        # print(data_rebuild.shape)
        #data_rebuild = np.matmul(U[:, 0:k], data_rebuild.T)
        return data_rebuild

    def compression_ratio(self, data, k):  # [5pts]
        """
        Compute the compression ratio: (num stored values in compressed)/(num stored values in original)

        Args:
                data: (N, D) TF-IDF features for the data.
                k: int corresponding to number of components

        Return:
                compression_ratio: float of proportion of storage used
        """
        N, D = data.shape
        compression_ratio = k*(1 + N + D) / (N*D)
        return compression_ratio

    def recovered_variance_proportion(self, S, k):  # [5pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
                S: (min(N,D), ) numpy array
                k: int, rank of approximation

        Return:
                recovered_var: float corresponding to proportion of recovered variance
        """
        # recovered_var = np.var(S[0:k]) / np.var(S)
        recovered_var = np.sum(S[:k]**2) / np.sum(S**2)
        return recovered_var
