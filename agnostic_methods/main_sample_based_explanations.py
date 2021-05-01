import argparse
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Any

from tqdm import tqdm
import numpy as np

from common import load_pickle, load_json_file, to_pickle, extract_pattern_type
from agnostic_methods.main_global_explanations import train_model
from agnostic_methods.utils import generate_empty_results_dict
from agnostic_methods.configuration import Config
from agnostic_methods.explainer import (feature_importance_methods_lookup,
                                        FeatureImportanceTypes)


def is_local_explanation_method(method: str) -> bool:
    return (FeatureImportanceTypes.lime == method or
            FeatureImportanceTypes.shap_linear == method or
            FeatureImportanceTypes.anchors == method)


def generate_explanations(config: Config, model: Any, data: Dict) -> Dict:
    explanations = dict()
    for method in config.method_names:
        explainer = feature_importance_methods_lookup[method](model=model, seed=config.seed)
        if is_local_explanation_method(method=method):
            explainer.fit(X=data['train']['x'])
        else:
            raise Exception('Please select sample based explanation methods!')
        explanations[method] = explainer.explain_batch(X=data['val']['x'], num_jobs=4)
    return explanations


def main_experiment(data: Dict, config: Config) -> Dict:
    results = dict()
    results['model'] = train_model(data=data['train'], seed=config.seed)
    results['explanations'] = generate_explanations(config=config, model=results['model'],
                                                    data=data)

    return results


def main(input_path: str) -> None:
    config = Config.get(input_conf=load_json_file(file_path=input_path))
    data = load_pickle(file_path=config.data_path)
    np.random.seed(seed=config.seed)
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
        suffix = '_'.join(['results_agnostic_sample_based', date, f'_{weights}'])
        to_pickle(output_dir=config.output_dir, data=results_per_weight, suffix=suffix)

    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    pattern_type = f'pattern_type_{extract_pattern_type(data_path=config.data_path)}'
    suffix = '_'.join(['results_agnostic_sample_based', date, pattern_type])
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
