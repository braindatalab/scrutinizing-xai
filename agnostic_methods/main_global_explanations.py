import argparse
import math
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from common import load_pickle, load_json_file, to_pickle, extract_pattern_type
from agnostic_methods.utils import generate_empty_results_dict
from agnostic_methods.configuration import Config
from agnostic_methods.explainer import (feature_importance_methods_lookup,
                                        FeatureImportanceTypes)


def is_permutation_feature_importance(method: str) -> bool:
    return (FeatureImportanceTypes.pfi == method or
            FeatureImportanceTypes.mr_empirical == method or
            FeatureImportanceTypes.mr == method)


def is_tree(method: str) -> bool:
    return FeatureImportanceTypes.impurity == method


def generate_explanations(config: Config, model: Any, data: Dict) -> Dict:
    explanations = dict()
    for method in config.method_names:
        explainer = feature_importance_methods_lookup[method](model=model, seed=config.seed)
        if is_permutation_feature_importance(method=method):
            explainer.fit(X=data['train']['x'], y=data['train']['y'], num_jobs=1)
        elif is_tree(method=method):
            # model = DecisionTreeClassifier(random_state=config.seed, min_samples_split=10)
            # Adapted from https://github.com/xiyanghu/OSDT
            lamb = 0.02
            y = data['train']['y'].flatten()
            model = DecisionTreeClassifier(
                random_state=config.seed,
                min_samples_split=max(math.ceil(lamb * 2 * len(y)), 2),
                min_samples_leaf=math.ceil(lamb * len(y)),
                max_leaf_nodes=math.floor(1 / (2 * lamb)),
                min_impurity_decrease=lamb)

            model.fit(X=data['train']['x'], y=data['train']['y'].flatten())
            explainer = feature_importance_methods_lookup[method](model=model, seed=config.seed)
            explainer.fit()
        else:
            explainer.fit(X=data['train']['x'])
        explanations[method] = explainer.explain()
    return explanations


def train_model(data: Dict, seed: int, use_tree: bool = False) -> Any:
    if use_tree:
        model = DecisionTreeClassifier(random_state=seed)
    else:
        model = LogisticRegression(penalty='none', fit_intercept=False,
                                   max_iter=1000, random_state=seed)
    model.fit(X=data['x'], y=data['y'].flatten())
    return model


def main_experiment(data: Dict, config: Config) -> Dict:
    results = dict()
    results['model'] = train_model(data=data['train'], seed=config.seed)
    results['explanations'] = generate_explanations(
        config=config, model=results['model'], data=data)

    return results


def main(input_path: str) -> None:
    config = Config.get(input_conf=load_json_file(file_path=input_path))
    data = load_pickle(file_path=config.data_path)
    results = generate_empty_results_dict()
    results['method_names'] = config.method_names
    print(f'Input: {asdict(config)}')
    print('Run experiments!')
    for weights, data_list in data.items():
        results_per_weight = list()
        print(f'Run experiments for weights: {weights}')
        for data in tqdm(data_list):
            results_per_weight += [main_experiment(data=data, config=config)]
        results['results'][weights] = results_per_weight

    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    pattern_type = f'pattern_type_{extract_pattern_type(data_path=config.data_path)}'
    suffix = '_'.join(['results_agnostic_global', date, pattern_type])
    to_pickle(output_dir=config.output_dir, data=results, suffix=suffix)


def get_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', required=True,
                        help='Input file path of json file containing'
                             'input parameter for the experiment!')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_command_line_arguments()
    path = args.path
    try:
        main(input_path=path)
    except KeyboardInterrupt as e:
        print(e)
