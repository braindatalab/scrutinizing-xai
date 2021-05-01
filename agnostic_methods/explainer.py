import copy
from abc import ABCMeta
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Any, Callable, Tuple

import numpy as np
from alibi.explainers import AnchorTabular
from joblib import Parallel, delayed
from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss
from anchor.anchor_tabular import AnchorTabularExplainer
from shap import LinearExplainer, maskers
from agnostic_methods.utils import assign_model

SEED = 1337

Explanation = namedtuple('Explanation', ['importance'])


@dataclass
class FeatureImportanceTypes:
    pfi: str = 'pfi'
    mr: str = 'mr'
    mr_empirical: str = 'mr_empirical'
    anchors: str = 'anchors'
    lime: str = 'lime'
    shap_linear: str = 'shap_linear'
    pattern: str = 'pattern'
    corr: str = 'correlation'
    firm: str = 'firm'
    impurity: str = 'impurity'

    # name_map: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.name_map = {
            'pfi': 'PFI',
            'mr': 'PFIO',
            'mr_empirical': 'PFIE',
            'anchors': 'Anchors',
            'lime': 'LIME',
            'shap_linear': 'SHAP_linear',
            'pattern': 'Pattern',
            'firm': 'FIRM',
            'tree_fi': 'Impurity'
        }


class FeatureImportance(metaclass=ABCMeta):
    def __init__(self, seed: int):
        self._seed = seed
        self._explanation = None

    @classmethod
    def __subclasshook(cls, subclass):
        return (hasattr(subclass, 'fit') and
                callable(subclass.fit) and
                hasattr(subclass, 'explain') and
                callable(subclass.explain) or
                NotImplemented)


class PFI(FeatureImportance):
    """
    Feature importance method by [BRE]_.

    References
    ----------
    .. [BRE] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32,
             2001. https://doi.org/10.1023/A:1010933404324

    """

    def __init__(self, model: Any, seed: int = SEED):
        super().__init__(seed=seed)
        self._model = assign_model(model=model)
        self._importance_result = None

    def fit(self, X: np.ndarray, y: np.ndarray, num_jobs: int = 2):
        self._importance_result = permutation_importance(
            estimator=self._model, X=X,
            y=y, n_repeats=1, n_jobs=num_jobs,
            random_state=self._seed)

    def explain(self) -> np.ndarray:
        return self._importance_result.importances[:, 0]


class ModelReliance(FeatureImportance):
    """
    Feature importance method by [FIS]_.

    References
    ----------
    .. [FIS] Fisher, et al, "All Models are Wrong, but Many are Useful:
    Learning a Variable's Importance by Studying an Entire Class of Prediction
    Models Simultaneously", Journal of Machine Learning Research, Volume 20, 1-81,
             2019. http://jmlr.org/papers/v20/18-760.html

    """

    def __init__(self, model: Any, loss: Callable = log_loss, seed: int = SEED):
        super().__init__(seed=seed)
        self._loss = loss
        self._model = assign_model(model=model)
        self._original_error = None
        self._switch_errors = None

    def _calculate_errors_of_permutations(self, X: np.ndarray,
                                          y: np.ndarray,
                                          feature_idx: int) -> float:
        num_instances = X.shape[0]
        mask = np.ones(num_instances, dtype=bool)
        swapped_X = copy.deepcopy(X)
        sum_of_errors = 0.0
        for k in range(num_instances):
            swapped_X[:, feature_idx] = np.array(num_instances * [X[k, feature_idx]])
            mask[k] = False
            predictions = self._model.predict_proba(X=swapped_X[mask])
            sum_of_errors += self._loss(y_true=y[mask], y_pred=predictions)
            mask[k] = True
        return sum_of_errors / num_instances

    def _calculate_original_model_error(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self._model.predict_proba(X=X)
        return self._loss(y_true=y, y_pred=predictions)

    def fit(self, X: np.ndarray, y: np.ndarray, num_jobs: int = 2):
        self._original_error = self._calculate_original_model_error(X=X, y=y)
        self._switch_errors = Parallel(n_jobs=num_jobs)(
            delayed(self._calculate_errors_of_permutations)(X, y, feature_idx)
            for feature_idx in range(X.shape[1]))

    def _calculate_importance(self, use_subtraction: bool = False) -> np.ndarray:
        if use_subtraction:
            importance = self._switch_errors - self._original_error
        else:
            importance = self._switch_errors / self._original_error
        return importance

    def explain(self, use_subtraction: bool = False) -> np.ndarray:
        return self._calculate_importance(use_subtraction=use_subtraction)


class EmpiricalModelReliance(ModelReliance):
    """
    Feature importance method by [FIS]_.

    References
    ----------
    .. [FIS] Fisher, et al, "All Models are Wrong, but Many are Useful:
    Learning a Variable's Importance by Studying an Entire Class of Prediction
    Models Simultaneously", Journal of Machine Learning Research, Volume 20, 1-81,
             2019. http://jmlr.org/papers/v20/18-760.html

    """

    def __init__(self, model: Any, loss: Callable = log_loss, seed: int = SEED):
        super().__init__(model=model, loss=loss, seed=seed)

    def _calculate_errors_of_permutations(self, X: np.ndarray,
                                          y: np.ndarray,
                                          feature_idx: int) -> float:
        swapped_X = self._swap_data(X=X, feature_idx=feature_idx)
        predictions = self._model.predict_proba(X=swapped_X)
        errors = self._loss(y_true=y, y_pred=predictions)
        return errors

    def _swap_data(self, X: np.ndarray, feature_idx: int) -> np.ndarray:
        first_half = X.shape[0] // 2
        second_half = X.shape[0] // 2
        permuted = np.zeros(shape=(2 * first_half, *X.shape[1:]))
        permuted[:first_half] = X[:first_half].copy()
        permuted[second_half:] = X[second_half:(2 * second_half)].copy()
        permuted[:first_half, feature_idx] = X[second_half:(2 * second_half), feature_idx].copy()
        permuted[second_half:, feature_idx] = X[:first_half, feature_idx].copy()
        return permuted


class Anchors(FeatureImportance):
    """
    Feature importance method by [RIB]_.

    References
    ----------
    .. [RIB] Ribeiro, et al, "Anchors: High-precision model-agnostic explanations",
     Proceedings of the AAAI Conference on Artificial Intelligence, Volume 32, 2018.

    """

    def __init__(self, model: Any, seed: int = SEED):
        super().__init__(seed=seed)
        self._model = assign_model(model=model)
        self._explainer = None

    def fit(self, X: Any) -> None:
        self._explainer = AnchorTabular(
            predictor=self._model.predict_proba,
            feature_names=list(range(X.shape[1])), seed=self._seed)
        self._explainer.fit(train_data=X)
        # disc_perc=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
        # disc_perc=(0.1, 0.3, 0.5, 0.7, 0.9))
        # disc_perc=(0.2, 0.4, 0.6, 0.8))

    def _compute_anchors_per_sample(self, X: np.ndarray, idx: int) -> List:
        result = self._explainer.explain(X=X[idx, :])
        return result.data['raw']['feature']

    @staticmethod
    def _calculate_importance(anchors: List, output_shape: Tuple) -> np.ndarray:
        importance = np.zeros(shape=output_shape)
        for k, anchor in enumerate(anchors):
            if isinstance(anchor, list):
                importance[k, anchor] = 1
            else:
                importance[anchor] = 1
        return importance

    def _compute_anchors(self, X: np.ndarray, num_jobs: int) -> List:
        return Parallel(n_jobs=num_jobs)(
            delayed(self._compute_anchors_per_sample)(X, sample_idx)
            for sample_idx in range(X.shape[0]))

    def explain(self, X: np.ndarray, sample_idx: int) -> np.ndarray:
        anchors = self._compute_anchors_per_sample(X=X, idx=sample_idx)
        return self._calculate_importance(anchors=anchors, output_shape=(X.shape[1],))

    def explain_batch(self, X: np.ndarray, num_jobs: int = 2) -> np.ndarray:
        anchors = self._compute_anchors(X=X, num_jobs=num_jobs)
        return self._calculate_importance(anchors=anchors, output_shape=X.shape)


class Anchors2(FeatureImportance):
    """
    Feature importance method by [RIB]_.

    References
    ----------
    .. [RIB] Ribeiro, et al, "Anchors: High-precision model-agnostic explanations",
     Proceedings of the AAAI Conference on Artificial Intelligence, Volume 32, 2018.

    """

    def __init__(self, model: Any, seed: int = SEED):
        super().__init__(seed=seed)
        self._model = assign_model(model=model)
        self._explainer = None

    def fit(self, X: Any) -> None:
        self._explainer = AnchorTabularExplainer(
            class_names=['0', '1'],
            feature_names=list(range(X.shape[1])),
            train_data=X,
            # discretizer='quartile')
            discretizer='decile')

    def _compute_anchors_per_sample(self, X: np.ndarray, idx: int) -> List:
        result = self._explainer.explain_instance(
            data_row=X[idx, :], classifier_fn=self._model.predict)
        return result.exp_map['feature']

    @staticmethod
    def _calculate_importance(anchors: List, output_shape: Tuple) -> np.ndarray:
        importance = np.zeros(shape=output_shape)
        for k, anchor in enumerate(anchors):
            if isinstance(anchor, list):
                importance[k, anchor] = 1
            else:
                importance[anchor] = 1
        return importance

    def _compute_anchors(self, X: np.ndarray, num_jobs: int) -> List:
        return Parallel(n_jobs=num_jobs)(
            delayed(self._compute_anchors_per_sample)(X, sample_idx)
            for sample_idx in range(X.shape[0]))

    def explain(self, X: np.ndarray, sample_idx: int) -> np.ndarray:
        anchors = self._compute_anchors_per_sample(X=X, idx=sample_idx)
        return self._calculate_importance(anchors=anchors, output_shape=(X.shape[1],))

    def explain_batch(self, X: np.ndarray, num_jobs: int = 2) -> np.ndarray:
        np.random.seed(self._seed)
        anchors = self._compute_anchors(X=X, num_jobs=num_jobs)
        return self._calculate_importance(anchors=anchors, output_shape=X.shape)


class Lime(FeatureImportance):
    """
    Feature importance method by [RIB]_.

    References
    ----------
    .. [RIB] Ribeiro, et al, "" Why should I trust you?" Explaining the
    predictions of any classifier", Proceedings of the 22nd ACM SIGKDD
    international conference on knowledge discovery and data_generation mining, 1136-1144, 2016.

    """

    def __init__(self, model: Any, seed: int = SEED):
        super().__init__(seed=seed)
        self._model = assign_model(model=model)
        self._explainer = None

    def fit(self, X: Any, class_names: List[str] = None) -> None:
        if class_names is None:
            class_names = ['0', '1']
        self._explainer = LimeTabularExplainer(
            training_data=X, feature_names=list(range(X.shape[1])),
            class_names=class_names, discretize_continuous=False,
            random_state=self._seed)

    @staticmethod
    def _calculate_importance(explanation_matrix: np.ndarray) -> np.ndarray:
        w_abs = np.abs(explanation_matrix)
        sums_of_columns = np.sum(w_abs, axis=0)
        return np.sqrt(sums_of_columns)

    def _compute_local_explanations(self, X: np.ndarray, idx: int) -> np.ndarray:
        W = np.zeros(X.shape[1])
        exp = self._explainer.explain_instance(
            data_row=X[idx], top_labels=0,
            predict_fn=self._model.predict_proba, num_features=X.shape[1])
        for key, value in exp.local_exp[1]:
            W[key] = value
        return W

    def _compute_explanation_matrix(self, X: np.ndarray, num_jobs: int) -> np.ndarray:
        weights = Parallel(n_jobs=num_jobs)(
            delayed(self._compute_local_explanations)(X, sample_idx)
            for sample_idx in range(X.shape[0]))
        return np.vstack(weights)

    def explain_batch(self, X: np.ndarray, num_jobs: int = 2) -> np.ndarray:
        return self._compute_explanation_matrix(X=X, num_jobs=num_jobs)

    def explain(self, X: np.ndarray, sample_idx: int) -> np.ndarray:
        return self._compute_local_explanations(X=X, idx=sample_idx)


class Shap(FeatureImportance):
    """
    Feature importance method by [LUN]_.

    References
    ----------
    .. [LUN] Lundberg, et al, "A Unified Approach to Interpreting Model Predictions",
     Advances in Neural Information Processing Systems 30, 4765–4774, 2017.

    """

    def __init__(self, model: Any, seed: int = SEED):
        super().__init__(seed=seed)
        self._model = assign_model(model=model)
        self._explainer = None

    def fit(self, X: Any) -> None:
        masker = maskers.Impute(data=X)
        self._explainer = LinearExplainer(
            model=self._model, data=X, seed=self._seed,
            feature_perturbation='correlation_dependent', masker=masker)

    @staticmethod
    def _calculate_importance(explanation_matrix: np.ndarray) -> np.ndarray:
        w_abs = np.abs(explanation_matrix)
        sums_of_columns = np.sum(w_abs, axis=0)
        return np.sqrt(sums_of_columns)

    def explain_batch(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self._explainer.shap_values(X=X)

    def explain(self, X: np.ndarray, sample_idx: int, **kwargs) -> np.ndarray:
        return self._explainer.shap_values(X=X[sample_idx, :])


class Firm(FeatureImportance):
    """
    Feature importance method by [ZIE]_.

    References
    ----------
    .. [ZIE] Zien, et al, "The Feature Importance Ranking Measure",
     Machine Learning and Knowledge Discovery in Databases, 694-709, 2009.

    """

    def __init__(self, model: Any, seed: int = SEED):
        super().__init__(seed=seed)
        self._model = assign_model(model=model)
        self._explainer = None

    def fit(self, X: Any) -> None:
        covariance_matrix = np.cov(X.T)
        D = np.diag(1 / np.sqrt(np.diag(covariance_matrix)))
        self._explainer = np.matmul(D, covariance_matrix)

    def _calculate_importance(self) -> np.ndarray:
        return np.dot(self._explainer, self._model.coef_.T).flatten()

    def explain(self, **kwargs) -> np.ndarray:
        return self._calculate_importance()


class Pattern(FeatureImportance):
    """
    Feature importance method by [HAU]_.

    References
    ----------
    .. [HAU] Haufe, et al, "On the interpretation of weight vectors of linear
    models in multivariate neuroimaging", NeuroImage, Volume 87, 96-110, 2014.

    """

    def __init__(self, model: Any, seed: int = SEED):
        super().__init__(seed=seed)
        self._model = assign_model(model=model)
        self._explainer = None

    def fit(self, X: Any) -> None:
        self._explainer = np.cov(X.T)  # / np.var(np.dot(self._model.coef_, X.T))

    def _calculate_importance(self) -> np.ndarray:
        return np.dot(self._explainer, self._model.coef_.T).flatten()

    def explain(self, **kwargs) -> np.ndarray:
        return self._calculate_importance()


class Correlation(FeatureImportance):
    def __init__(self, model: Any, seed: int = SEED):
        super().__init__(seed=seed)
        self._model = assign_model(model=model)
        self._explainer = None

    def fit(self, X: Any) -> None:
        self._explainer = np.corrcoef(X.T)  # / np.var(np.dot(self._model.coef_, X.T))

    def _calculate_importance(self) -> np.ndarray:
        return np.dot(self._explainer, self._model.coef_.T).flatten()

    def explain(self, **kwargs) -> np.ndarray:
        return self._calculate_importance()


class TreeFI(FeatureImportance):
    """
    Feature importance method by [BRE]_.

    References
    ----------
    .. [BRE] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32,
             2001. https://doi.org/10.1023/A:1010933404324

    """

    def __init__(self, model: Any, seed: int = SEED):
        super().__init__(seed=seed)
        self._model = assign_model(model=model)
        self._explainer = None

    def fit(self, **kwargs) -> None:
        self._explainer = self._model.feature_importances_

    def explain(self, **kwargs) -> np.ndarray:
        return self._explainer


feature_importance_methods_lookup = {
    FeatureImportanceTypes.mr_empirical: EmpiricalModelReliance,
    FeatureImportanceTypes.mr: ModelReliance,
    FeatureImportanceTypes.pfi: PFI,
    FeatureImportanceTypes.anchors: Anchors,
    FeatureImportanceTypes.lime: Lime,
    FeatureImportanceTypes.shap_linear: Shap,
    FeatureImportanceTypes.firm: Firm,
    FeatureImportanceTypes.pattern: Pattern,
    FeatureImportanceTypes.corr: Correlation,
    FeatureImportanceTypes.impurity: TreeFI}
