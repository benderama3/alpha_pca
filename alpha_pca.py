from copy import deepcopy
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.decomposition._pca import *
from sklearn.decomposition._base import *
from sklearn.utils import check_random_state, check_scalar
from sklearn.utils.extmath import fast_logdet, svd_flip
from math import log
import numbers

from scipy.sparse import issparse
from scipy import linalg


class AlphaPCA(BaseEstimator, ClassifierMixin):

    def __init__(
        self,
        n_components=None,
        *,
        copy=True,
        center=True,
        precenter=False,
        prereduce=False,
        do_inverse=False,
        alpha=1.,
        n_oversamples=10,
        random_state=None,
    ):
        self.n_components = n_components
        self.copy = copy
        self.center = center
        self.precenter = precenter
        self.prereduce = prereduce
        self.do_inverse = do_inverse
        self.alpha = alpha
        self.n_oversamples = n_oversamples
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model with X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Ignored.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        check_scalar(
            self.n_oversamples,
            "n_oversamples",
            min_val=1,
            target_type=numbers.Integral,
        )

        self._fit(X)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : Ignored
            Ignored.
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed values.
        Notes
        -----
        This method returns a Fortran-ordered array. To convert it to a
        C-ordered array, use 'np.ascontiguousarray'.
        """

        
        X_ = self._fit(X)
        X_ = X_[:, :self.n_components_]

        return X_

    def _fit(self, X):

        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        if issparse(X):
            raise TypeError(
                "PCAAlpha does not support sparse input. See "
                "TruncatedSVD for a possible alternative."
            )
        
        if torch.is_tensor(X):
            self.is_torch = True
            self.phi_function = self._phi_alpha_torch
            if self.copy:
                assert len(X.size()) == 2, "X must be 2d tensor"
                X = deepcopy(X)
        else:
            self.is_torch = False
            self.phi_function = self._phi_alpha

            # Validate data
            X = self._validate_data(
                X, dtype=[np.float64, np.float32], ensure_2d=True, copy=self.copy
            )

        # Handle n_components==None
        if self.n_components is None:
            n_components = min(X.shape)
        else:
            n_components = self.n_components

        return self._fit_model(X, n_components)

    def _phi_alpha(self, x, is_inverse=False):
        '''
        :param x input matrix
        :param is_inverse (boolean) function _phi_alpha or _phi_alpha^-1 ?
        '''
        alpha = 1/self.alpha if is_inverse else self.alpha
        return np.sign(x) * np.abs(x)**alpha
    
    def _phi_alpha_torch(self, x, is_inverse=False):
        '''
        :param x input matrix
        :param is_inverse (boolean) function _phi_alpha or _phi_alpha^-1 ?
        '''
        alpha = 1/self.alpha if is_inverse else self.alpha
        return torch.sign(x) * torch.abs(x)**alpha
    
    def _numpy_decomposition(self, X, n_samples):
        
        n, d = X.shape

        # Center data
        X = self.phi_function(X)
        self.mean_ = X.mean(axis=0) if self.center else 0
        X = X - self.mean_

        if d <= n:
            U, S, Vt = linalg.svd(X.T @ X / n, full_matrices=False)
            # flip eigenvectors' sign to enforce deterministic output
            U, Vt = svd_flip(U, Vt)
            X_ = self.phi_function(X @ Vt, is_inverse=True)
        else:
            U, S, Vt = linalg.svd(X @ X.T / d, full_matrices=False)
            # flip eigenvectors' sign to enforce deterministic output
            U, Vt = svd_flip(U, Vt)
            Vt = Vt @ X
            Vt = Vt / np.linalg.norm(Vt, 2, axis=-1)[:,np.newaxis]
            X_ = self.phi_function(X @ Vt.T, is_inverse=True)
        return X_, Vt

    def _torch_decomposition(self, X, n_samples):
        
        n, d = X.size()

        # Center data
        X = self.phi_function(X)
        self.mean_ = X.mean(dim=0, keepdim=True) if self.center else 0
        X = X - self.mean_

        if d <= n:
            U, S, Vt = torch.svd(X.T @ X / n)
            #Vt = torch.real(Vt).T
            Vt = Vt.T
            # flip eigenvectors' sign to enforce deterministic output
            U, Vt = self._torch_svd_flip(U, Vt, u_based_decision=False)
            X_ = self.phi_function(X @ Vt, is_inverse=True)
        else:
            
            U, S, Vt = torch.svd(X @ X.T / d)
            #Vt = torch.real(Vt).T
            Vt = Vt.T
            U, Vt = self._torch_svd_flip(U, Vt, u_based_decision=False)
            Vt = Vt @ X
            Vt = Vt / Vt.norm(dim=-1, keepdim=True)
            X_ = self.phi_function(X @ Vt.T, is_inverse=True)

        return X_, Vt

    def _fit_model(self, X, n_components):
        
        """Fit the model."""
        n_samples, n_features = X.shape

        if not 0 <= n_components <= min(n_samples, n_features):
            raise ValueError(
                "n_components=%r must be between 0 and "
                "min(n_samples, n_features)=%r" % (n_components, min(n_samples, n_features))
            )
        elif n_components >= 1:
            if not isinstance(n_components, numbers.Integral):
                raise ValueError(
                    "n_components=%r must be of type int "
                    "when greater than or equal to 1, "
                    "was of type=%r" % (n_components, type(n_components))
                )

        # Projection & components
        if self.is_torch:
            self.premean_ = X.mean(dim=0, keepdim=True) if self.precenter else 0
            self.prestd_ = X.std(dim=0, keepdim=True) if self.prereduce else 1
            X = (X - self.premean_) / self.prestd_
            X_, components_ = self._torch_decomposition(X, n_samples)
        else:
            self.premean_ = X.mean(axis=0) if self.precenter else 0
            self.prestd_ = X.std(axis=0) if self.prereduce else 1
            X = (X - self.premean_) / self.prestd_
            X_, components_ = self._numpy_decomposition(X, n_samples)

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components

        return X_
 
    def transform(self, X):
        """Apply dimensionality reduction to X.
        X is projected on the first principal components previously extracted
        from a training set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Projection of X in the first principal components, where `n_samples`
            is the number of samples and `n_components` is the number of the components.
        """
        check_is_fitted(self)

        if not self.is_torch:
            X = self._validate_data(X, dtype=[np.float64, np.float32], reset=False)
        else:
            assert len(X.size()) == 2, "X must be 2d tensor"

        X = X - self.premean_
        X = X / self.prestd_
        X = self.phi_function(X) - self.mean_
        X_transformed = self.phi_function(X @ self.components_.T, is_inverse=True)
        return X_transformed

    def inverse_transform(self, X):
        
        X = self.phi_function(X)
        if not self.do_inverse:
            return self.phi_function(X @ self.components_ + self.mean_, is_inverse=True)*self.prestd_ + self.premean_

        pinv = torch.linalg.pinv if self.is_torch else np.linalg.pinv
        return self.phi_function(X @ pinv(self.components_.T) + self.mean_, is_inverse=True)*self.prestd_ + self.premean_

    def approximate(self, X):
        return self.inverse_transform(self.transform(X))

    def score_samples(self, X):
        """Return the log-likelihood of each sample.
        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.
        Returns
        -------
        ll : ndarray of shape (n_samples,)
            Log-likelihood of each sample under the current model.
        """
        check_is_fitted(self)
        if self.is_torch:
            raise "Not compatible with torch.tensor, use numpy instead"

        X = self._validate_data(X, dtype=[np.float64, np.float32], reset=False)
        Xr = X - self.mean_
        n_features = X.shape[1]
        precision = self.get_precision()
        log_like = -0.5 * (Xr * (np.dot(Xr, precision))).sum(axis=1)
        log_like -= 0.5 * (n_features * log(2.0 * np.pi) - fast_logdet(precision))
        return log_like

    def score(self, X, y=None):
        """Return the average log-likelihood of all samples.
        See. "Pattern Recognition and Machine Learning"
        by C. Bishop, 12.2.1 p. 574
        or http://www.miketipping.com/papers/met-mppca.pdf
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.
        y : Ignored
            Ignored.
        Returns
        -------
        ll : float
            Average log-likelihood of the samples under the current model.
        """
        return np.mean(self.score_samples(X))

    def _more_tags(self):
        return {"preserves_dtype": [np.float64, np.float32]}

    def _torch_svd_flip(self, u, v, u_based_decision=True):
        if u_based_decision:
            # columns of u, rows of v
            max_abs_cols = torch.argmax(u.abs(), dim=0)
            signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
            u *= signs
            v *= signs.unsqueeze(-1)
        else:
            # rows of v, columns of u
            max_abs_rows = torch.argmax(v.abs(), dim=-1)
            signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
            u *= signs
            v *= signs.unsqueeze(-1)
        return u, v

    def compute_optimal_alpha(self, X, alphas=None, n_components=None, degree=3, reconstruction_loss="mae"):
        """Return the optimal alpha from data points.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data.
        alphas : a list of alpha values (default: [0.6, 0.8, 1.2, 1.4])
        n_components : number of components, (default: self.n_components)
        degree : polynomial degree fit (2 or 3, default: 3)
        reconstruction_loss : reconstruction loss function in ['mae', 'mse', 'std'] (default: 'mae')
        Returns
        -------
        best_alpha : float
            Approximated optimal alpha.
        """

        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        import math 

        def mae(X, X_, is_torch):
            if is_torch:
                X = X.cpu().numpy()
                X_ = X_.cpu().numpy()
            return np.abs(X - X_).mean()
        
        def mse(X, X_, is_torch):
            if is_torch:
                X = X.cpu().numpy()
                X_ = X_.cpu().numpy()
            return np.power(X - X_, 2).mean()
        
        def std(X, X_, is_torch):
            if is_torch:
                X = X.cpu().numpy()
                X_ = X_.cpu().numpy()
            return (X - X_).std()

        if n_components is None:
            n_components = self.n_components
        if alphas is None:
            if degree >= 3:
                alphas = [0.6, 0.8, 1.2, 1.4]
            elif degree == 2:
                alphas = [0.6, 1.0, 1.4]

        assert degree in [2, 3], "Degree must be 2 or 3"
        assert isinstance(alphas, list), "Variable alphas must be a list"
        assert len(alphas), "len(alphas) should be >= 3"
        assert reconstruction_loss in ["mae", "mse", "std"], "reconstruction_loss must in ['mae', 'mse', 'std']"
        
        reconstruction_losses = []
        for alpha in alphas:
            pca = AlphaPCA(
                n_components=n_components, 
                copy=self.copy, 
                center=self.center, 
                precenter=self.precenter, 
                prereduce=self.prereduce, 
                do_inverse=self.do_inverse,
                alpha=alpha,
                n_oversamples=self.n_oversamples,
                random_state=self.random_state
                )
            pca.fit(X)
            is_torch = pca.is_torch

            if reconstruction_loss == "mae":
                reconstruction_losses.append(mae(X, pca.approximate(X), is_torch))
            elif reconstruction_loss == "mse":
                reconstruction_losses.append(mse(X, pca.approximate(X), is_torch))
            elif reconstruction_loss == "std":
                reconstruction_losses.append(std(X, pca.approximate(X), is_torch))

        poly = PolynomialFeatures(degree=degree, include_bias=False)

        #reshape data to work properly with sklearn
        poly_features = poly.fit_transform(np.array(alphas).reshape(-1, 1))

        #fit polynomial regression model
        poly_reg_model = LinearRegression()
        poly_reg_model.fit(poly_features, np.array(reconstruction_losses).reshape(-1, 1))

        if degree == 2:
            b, a = poly_reg_model.coef_[0]
            return -b / (2*a)
        elif degree == 3:
            c, b, a = poly_reg_model.coef_[0]
            delta = 4 * b**2 - 12 * a * c 
            assert delta >= 0, "Failed to find the optimal alpha, consider using more data points"
            return (-2*b + math.sqrt(delta)) / (6*a)