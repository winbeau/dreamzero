"""Reusable classifier components for dynamic sparse-attention routing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture


BUDGET_BUCKETS = np.asarray((0.10, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00))


@dataclass(frozen=True)
class RoutePolicy:
    """Calibrated deployment rule for budget promotion and Dense fallback."""

    confidence_threshold: float
    promotion_buckets: int


class MappedGMMClassifier(ClassifierMixin, BaseEstimator):
    """Original unsupervised GMM baseline with clusters mapped to budgets.

    The Gaussian mixture itself never sees the Oracle label.  Labels are used
    only after fitting to map each component's responsibility-weighted mass to
    the seven fixed budget buckets, which makes the baseline expose the same
    ``predict_proba`` interface as supervised candidates.
    """

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "diag",
        reg_covar: float = 1e-5,
        max_iter: int = 200,
        random_state: int = 20260830,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        labels = np.asarray(y, dtype=np.int64)
        self.classes_ = np.arange(len(BUDGET_BUCKETS), dtype=np.int64)
        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            random_state=self.random_state,
        ).fit(X)
        responsibilities = self.gmm_.predict_proba(X)
        component_classes = np.ones(
            (self.n_components, len(self.classes_)), dtype=np.float64
        )
        for class_index in self.classes_:
            component_classes[:, class_index] += responsibilities[
                labels == class_index
            ].sum(axis=0)
        self.component_class_probability_ = component_classes / component_classes.sum(
            axis=1, keepdims=True
        )
        return self

    def predict_proba(self, X):
        return self.gmm_.predict_proba(X) @ self.component_class_probability_

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
