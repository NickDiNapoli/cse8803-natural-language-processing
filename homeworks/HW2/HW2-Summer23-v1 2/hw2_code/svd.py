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
        raise NotImplementedError

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
        raise NotImplementedError

    def compression_ratio(self, data, k):  # [5pts]
        """
        Compute the compression ratio: (num stored values in compressed)/(num stored values in original)

        Args:
                data: (N, D) TF-IDF features for the data.
                k: int corresponding to number of components

        Return:
                compression_ratio: float of proportion of storage used
        """
        raise NotImplementedError

    def recovered_variance_proportion(self, S, k):  # [5pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
                S: (min(N,D), ) numpy array
                k: int, rank of approximation

        Return:
                recovered_var: float corresponding to proportion of recovered variance
        """
        raise NotImplementedError