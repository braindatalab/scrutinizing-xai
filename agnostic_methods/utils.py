import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
NUM_JOBS = 2


def generate_empty_results_dict():
    return {
        'results': dict(),
        'method_names': list()
    }


def _has_predict_method(model: Callable) -> bool:
    return hasattr(model, 'predict') and callable(getattr(model, 'predict'))


def _has_predict_proba_method(model: Callable) -> bool:
    return hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba'))


def is_scikit_learn_model(model: Any) -> bool:
    return _has_predict_method(model=model) or _has_predict_proba_method(model=model)


def assign_model(model: Callable) -> Callable:
    if not is_scikit_learn_model(model):
        raise TypeError('Sorry, at the moment we only support models from scikit-learn!')
    return model
