"""Implementation of the Probabilistic‑gap PU‑Classification algorithm
---------------------------------------------------------------------
This module provides a scikit‑learn–style class ``ProbabilisticGapPU`` that
implements Algorithm 1 from the *Probabilistic‑gap PU Classification* paper.

The implementation follows the two‑stage procedure described in the paper:
    1. **Conditional Bayesian optimal relabelling.**
       We fit a probabilistic classifier to distinguish *labelled positives*
       (``s = 1``) from *unlabeled* examples (``s = 0``).  From the predicted
       probabilities we compute the *probabilistic gap* ΔP̃(x) and relabel a
       subset of the data as confident positive or confident negative.
    2. **Weighted SVM learning.**
       A weighted SVM is fit on the confidently relabelled set.  The
       importance weight β(x) = P_D(x) / P_{D*}(x) corrects the sample‑selection
       bias introduced by Step 1.  In practice this ratio is unknown, so the
       default implementation sets β(x) = 1.  You can plug in any density‑ratio
       estimator by overriding ``_estimate_beta``.

Example
-------
>>> from probabilistic_gap_pu import ProbabilisticGapPU
>>> pgpu = ProbabilisticGapPU()
>>> pgpu.fit(X_train, s_train)      # s = 1 for labelled‑positive, 0 otherwise
>>> y_pred = pgpu.predict(X_test)   # returns −1 / +1 labels

References
----------
* Yasunori Fujikawa and Jihoon Shin, *Probabilistic‑gap PU Classification*,
  2020.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class ProbabilisticGapPU(BaseEstimator, ClassifierMixin):
    """Probabilistic‑gap Positive‑Unlabeled classifier.

    Parameters
    ----------
    base_estimator : scikit‑learn estimator, default = ``LogisticRegression``
        Probabilistic classifier used in Step 1 to produce *P̃₊(x)*.
    svm_kwargs : dict, optional
        Extra keyword arguments forwarded to ``sklearn.svm.SVC`` in Step 2.
    density_ratio : str or callable, default = "uniform"
        How to compute the importance weight β(x).  If "uniform", β(x) = 1.
        If a callable is supplied it must have the signature
        ``beta = f(X_selected, y_selected)`` and return a 1‑D array of weights.
    random_state : int or None, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        base_estimator=None,
        svm_kwargs: Optional[dict] = None,
        density_ratio: str | callable = "uniform",
        random_state: int | None = None,
    ) -> None:
        self.base_estimator = (
            base_estimator
            if base_estimator is not None
            else make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000, n_jobs="auto")
            )
        )
        self.svm_kwargs = {"kernel": "rbf", "C": 1.0, "class_weight": "balanced"}
        if svm_kwargs:
            self.svm_kwargs.update(svm_kwargs)
        self.density_ratio = density_ratio
        self.random_state = random_state

        # Attributes set during ``fit``
        self._delta_: Optional[np.ndarray] = None
        self._threshold_L_: Optional[float] = None
        self._svm_: Optional[SVC] = None

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _check_input(X, s):
        X = np.asarray(X)
        s = np.asarray(s).ravel()
        if X.shape[0] != s.shape[0]:
            raise ValueError("X and s must have the same number of samples.")
        if not set(np.unique(s)).issubset({0, 1, -1, +1}):
            raise ValueError("s must contain only 0/1 or −1/+1.")
        # Normalize labels to 0 (unlabeled) and 1 (labelled‑positive).
        s = (s > 0).astype(int)
        return X, s

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------
    def _estimate_delta(self, X: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Compute ΔP̃(x) = P̃₊(x) − P̃₋(x) using the base estimator."""
        self.base_estimator.fit(X, s)
        proba_pos = self.base_estimator.predict_proba(X)[:, 1]
        proba_neg = 1.0 - proba_pos
        return proba_pos - proba_neg  # in [−1, 1]

    def _select_confident(self, X: np.ndarray, s: np.ndarray, delta: np.ndarray):
        """Relabel confidently positive/negative instances."""
        # Lower threshold among *original* labelled positives.
        threshold_L = delta[s == 1].min()
        idx_pos = delta > 0
        idx_neg = delta < threshold_L
        X_sel = np.concatenate([X[idx_pos], X[idx_neg]], axis=0)
        y_sel = np.concatenate([np.ones(idx_pos.sum()), -np.ones(idx_neg.sum())])
        return X_sel, y_sel, threshold_L

    def _estimate_beta(self, X_sel: np.ndarray, y_sel: np.ndarray):
        """Importance weights β(x).

        The default returns uniform weights.  Override to plug in your own
        density‑ratio estimator or pass a callable via ``density_ratio``.
        """
        if self.density_ratio == "uniform":
            return np.ones(X_sel.shape[0])
        if callable(self.density_ratio):
            beta = self.density_ratio(X_sel, y_sel)
            if beta.shape != (X_sel.shape[0],):
                raise ValueError("custom density_ratio must return 1‑D array")
            return beta
        raise ValueError("density_ratio must be 'uniform' or a callable")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X, s):
        """Fit the classifier.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            Feature matrix.
        s : array‑like of shape (n_samples,)
            PU indicator.  ``s = 1`` for labelled‑positive, ``s = 0`` otherwise.
        """
        X, s = self._check_input(X, s)
        # Step 1 ─ relabelling
        delta = self._estimate_delta(X, s)
        X_sel, y_sel, threshold_L = self._select_confident(X, s, delta)

        # Step 2 ─ weighted SVM
        beta = self._estimate_beta(X_sel, y_sel)
        self._svm_ = SVC(**self.svm_kwargs, random_state=self.random_state)
        self._svm_.fit(X_sel, y_sel, sample_weight=beta)

        # Remember for inspection
        self._delta_ = delta
        self._threshold_L_ = threshold_L
        return self

    # ------------------------------------------------------------------
    # scikit‑learn compatibility
    # ------------------------------------------------------------------
    def predict(self, X):
        """Predict class labels ±1 for *unseen* examples."""
        if self._svm_ is None:
            raise RuntimeError("Must call fit() before predict().")
        return self._svm_.predict(X)

    def decision_function(self, X):
        """Signed distance to the separating hyperplane (like SVC.decision_function)."""
        if self._svm_ is None:
            raise RuntimeError("Must call fit() before decision_function().")
        return self._svm_.decision_function(X)

    # Convenience accessors
    @property
    def delta(self):
        """ΔP̃(x) values for the *training* data (set after ``fit``)."""
        return self._delta_

    @property
    def threshold_L(self):
        """Lower gap threshold ∆P̃_L used for confident negatives."""
        return self._threshold_L_

    @property
    def svm(self):
        """Underlying weighted SVM (set after ``fit``)."""
        return self._svm_