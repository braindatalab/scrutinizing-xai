import json
import os
import pickle
from dataclasses import dataclass, field
from os.path import join
from typing import Dict, Any, ClassVar


@dataclass
class ScoresAttributes:
    global_based: str
    sample_based: str
    explanations: str
    method_names: str
    data_weights: str
    model_weights: str
    model_accuracies: str
    logistic_regression: str
    neural_net: str
    instance: ClassVar = field(default=None)

    @staticmethod
    def default_conf() -> dict:
        return {
            'global_based': 'global',
            'sample_based': 'sample',
            'explanations': 'expl',
            'method_names': 'names',
            'data_weights': 'd_weights',
            'model_weights': 'm_weights',
            'model_accuracies': 'm_accuracy',
            'logistic_regression': 'Logistic Regression',
            'neural_net': 'Single-Layer Neural Net'
        }

    @classmethod
    def get(cls):
        if not cls.instance:
            conf_dict = cls.default_conf()
            cls.instance = cls(**conf_dict)
        return cls.instance


def load_json_file(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        file = json.load(f)
    return file


def load_pickle(file_path: str) -> Any:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def to_pickle(output_dir: str, data: Any, suffix: str) -> str:
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_path = join(output_dir, f'data_{suffix}.pkl')
    print(f'Output path: {output_path}')
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    return output_path


def extract_pattern_type(data_path: str) -> str:
    return data_path.split('.')[0].split('pattern_type_')[-1]
