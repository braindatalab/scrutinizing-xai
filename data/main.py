import argparse
from dataclasses import asdict
from datetime import datetime
from typing import Dict

from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from data.configuration import Config
from common import load_json_file, to_pickle
from data.data_generator import (WeightsDataGeneration,
                                 TwoBlobsForwardModelDataset,
                                 generate_data_weights)
from data.patterns import PatternMatrix


def weights_to_str(w: WeightsDataGeneration) -> str:
    return '_'.join([f'{v:0.2f}' for v in asdict(w).values()])


def generate_data(index: int, signal_type: str,
                  weights: WeightsDataGeneration, config: Config) -> Dict:
    pattern_matrix = PatternMatrix(config.pattern_type).matrix
    dataset = TwoBlobsForwardModelDataset(
        num_examples=config.num_examples, pattern_matrix=pattern_matrix,
        seed=config.seed + index, noise_type=config.noise_type,
        weights=weights, signal_type=signal_type)
    x, y = dataset.data.x, dataset.data.y.flatten()

    data_splitter = StratifiedShuffleSplit(n_splits=1, test_size=config.val_size,
                                           random_state=config.seed)
    train_indices, val_indices = list(data_splitter.split(X=x, y=y))[0]
    return {'train': {'x': x[train_indices], 'y': y[train_indices]},
            'val': {'x': x[val_indices], 'y': y[val_indices]}}


def create_datasets(config: Config, experiment_type: str, signal_type: str) -> Dict:
    weights = generate_data_weights(experiment_type=experiment_type,
                                    config=config, scaling=config.signal_scaling)
    all_data = dict()
    for j, w in enumerate(weights):
        data = list()
        print(f'Create data_generation with weights: {w}')
        for k in tqdm(range(config.num_experiments)):
            data += [generate_data(index=k, signal_type=signal_type, weights=w, config=config)]
        all_data[weights_to_str(w=w)] = data
    return all_data


def main(input_path: str) -> None:
    config = Config.get(input_conf=load_json_file(file_path=input_path))
    print(f'Input: {asdict(config)}')
    for type_of_experiment in config.experiment_types:
        print(f'Type of experiment: {type_of_experiment}.')
        for type_of_signal in config.signal_types:
            print(f'Type of signal: {type_of_signal}.')
            data = create_datasets(config=config, signal_type=type_of_signal,
                                   experiment_type=type_of_experiment)
            date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            pattern_type = f'pattern_type_{config.pattern_type}'
            suffix = '_'.join([type_of_experiment, type_of_signal, date, pattern_type])
            to_pickle(output_dir=config.output_dir, data=data, suffix=suffix)


def get_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', required=True,
                        help='Input file path of json file containing'
                             'configuration file for the data_generation generation process!')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_command_line_arguments()
    path = args.path
    try:
        main(input_path=path)
    except KeyboardInterrupt as e:
        print(e)
