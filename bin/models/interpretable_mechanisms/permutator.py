import numpy as np

class Permutator:
    """
    This class provides methods to perform different permutations on time series data
    represented in a 3-dimensional numpy array (X). The third dimension is indexed with k.

    Attributes:
        None

    Methods:
        permutation_time_series(X: np.ndarray, k: int) -> np.ndarray:
            - Permutes time series data for patients with the same length of stay.
            - X is a 3-dimensional array where X[:, :, k] represents the feature to be permuted.
        
        permutation_no_time(X: np.ndarray, k: int) -> np.ndarray:
            - Permutes time series data for all patients, except those with a feature value of 666.
            - X is a 3-dimensional array where X[:, :, k] represents the feature to be permuted.
    """

    def __init__(self):
        """
        Initializes the Permutator class. Currently, no initialization parameters are necessary.
        """
        pass

    @staticmethod
    def permutation_time_series(X: np.ndarray, k: int, flag_nan: int = 666) -> np.ndarray:
        """
        Permutes time series data for patients with the same length of stay.

        Parameters:
            X (np.ndarray): Input 3-dimensional array.
            k (int): Index of the third dimension to be permuted.

        Returns:
            np.ndarray: Permuted array.
        """
        v_feature = X[:, :, k].copy()
        stay_length_array = (v_feature == flag_nan).sum(axis=1)
        for i in range(v_feature.shape[1]):
            v_feature_masked = v_feature[stay_length_array == i]
            np.random.shuffle(v_feature_masked)
            v_feature[stay_length_array == i] = v_feature_masked
        X[:, :, k] = v_feature
        return X

    @staticmethod
    def permutation_no_time(X: np.ndarray, k: int, flag_nan: int = 666) -> np.ndarray:
        """
        Permutes time series data for all patients, except those with a feature value of 666.

        Parameters:
            X (np.ndarray): Input 3-dimensional array.
            k (int): Index of the third dimension to be permuted.

        Returns:
            np.ndarray: Permuted array.
        """
        v_feature = X[:, :, k].copy()
        v_feature_masked = v_feature[v_feature != flag_nan]
        np.random.shuffle(v_feature_masked.T)
        np.random.shuffle(v_feature_masked)
        v_feature[v_feature != 666] = v_feature_masked
        X[:, :, k] = v_feature
        return X
