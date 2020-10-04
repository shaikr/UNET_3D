import glob
import os

from collections import defaultdict

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing.data import _handle_zeros_in_scale
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES, check_array

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_progress_graph(training_dir):
    exp = os.path.basename(training_dir)
    training_fpath = os.path.join(training_dir, 'training')
    if not os.path.exists(training_fpath):
        return False
    try:
        df = pd.read_csv(training_fpath)
        if any(np.isinf(df).all()):
            print(f'found column all infs in {exp}')
        elif not all(np.isfinite(df).all()):
            print(f'found column some infs in {exp}')
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
        x = list(range(len(df)))
        axs[0].plot(x, df['loss'], label='Training loss')
        axs[0].plot(x, df['val_loss'], label='Validation loss')
        axs[0].legend(prop={'size': 14})
        axs[0].tick_params(axis="x", labelsize=12)
        axs[0].tick_params(axis="y", labelsize=12)
        axs[1].plot(x, df['dice_coefficient'], label='DICE')
        axs[1].plot(x, df['val_dice_coefficient'], label='Validation DICE')
        axs[1].legend(prop={'size': 14})
        axs[1].tick_params(axis="x", labelsize=12)
        axs[1].tick_params(axis="y", labelsize=12)
        fig.savefig(os.path.join(training_dir, 'progress_graph.png'))
        plt.close()
    except:
        print(f'Something went wrong in {exp}')
    return True


def get_last_model_path(model_file_path):
    return sorted(glob.glob(model_file_path + '*.h5'), key=os.path.getmtime)[-1]


class AttributeDict(defaultdict):
    def __init__(self, **kwargs):
        super(AttributeDict, self).__init__(AttributeDict, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

class MinMaxScaler(BaseEstimator, TransformerMixin):
    """Transforms features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, i.e. between
    zero and one.

    The transformation is given by::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    copy : boolean, optional, default True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).

    Attributes
    ----------
    min_ : ndarray, shape (n_features,)
        Per feature adjustment for minimum.

    scale_ : ndarray, shape (n_features,)
        Per feature relative scaling of the data.

        .. versionadded:: 0.17
           *scale_* attribute.

    data_min_ : ndarray, shape (n_features,)
        Per feature minimum seen in the data

        .. versionadded:: 0.17
           *data_min_*

    data_max_ : ndarray, shape (n_features,)
        Per feature maximum seen in the data

        .. versionadded:: 0.17
           *data_max_*

    data_range_ : ndarray, shape (n_features,)
        Per feature range ``(data_max_ - data_min_)`` seen in the data

        .. versionadded:: 0.17
           *data_range_*

    Examples
    --------
    >>> from sklearn.preprocessing import MinMaxScaler
    >>>
    >>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    >>> scaler = MinMaxScaler()
    >>> print(scaler.fit(data))
    MinMaxScaler(copy=True, feature_range=(0, 1))
    >>> print(scaler.data_max_)
    [  1.  18.]
    >>> print(scaler.transform(data))
    [[ 0.    0.  ]
     [ 0.25  0.25]
     [ 0.5   0.5 ]
     [ 1.    1.  ]]
    >>> print(scaler.transform([[2, 2]]))
    [[ 1.5  0. ]]

    See also
    --------
    minmax_scale: Equivalent function without the estimator API.

    Notes
    -----
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    """

    def __init__(self, feature_range=(0, 1), copy=True):
        self.feature_range = feature_range
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """Online computation of min and max on X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when `fit` is not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : Passthrough for ``Pipeline`` compatibility.
        """
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(feature_range))

        X = check_array(X, copy=self.copy, # warn_on_dtype=True,
                        estimator=self, dtype=FLOAT_DTYPES, ensure_2d=False, allow_nd=True)

        data_min = np.min(X)
        data_max = np.max(X)

        # First pass
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = X.shape[0]
        # Next steps
        else:
            data_min = np.minimum(self.data_min_, data_min)
            data_max = np.maximum(self.data_max_, data_max)
            self.n_samples_seen_ += X.shape[0]

        data_range = data_max - data_min
        self.scale_ = ((feature_range[1] - feature_range[0]) /
                       _handle_zeros_in_scale(data_range))
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        """Scaling features of X according to feature_range.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed.
        """
        check_is_fitted(self, 'scale_')

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES, ensure_2d=False, allow_nd=True)

        X *= self.scale_
        X += self.min_

        X = np.minimum(X, self.feature_range[1])
        X = np.maximum(X, self.feature_range[0])

        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed. It cannot be sparse.
        """
        check_is_fitted(self, 'scale_')

        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES, ensure_2d=False, allow_nd=True)

        X -= self.min_
        X /= self.scale_
        return X
