from collections import namedtuple
from dataclasses import dataclass
from typing import Tuple, Union, Dict, List

import numpy as np
from numpy.random import Generator
from scipy.linalg import expm

from data.configuration import Config

BASE_POWER_FACTOR = 3.0

Signals = namedtuple('SignalTypes', ['exact', 'noisy'])._make(['exact', 'noisy'])

NoiseTypes = namedtuple(
    'NoiseTypes',
    ['white_gaussian', 'zero_mean_gaussian'])._make(
    ['white_gaussian', 'zero_mean_gaussian'])

ExperimentTypes = namedtuple(
    'ExperimentTypes',
    ['vary_signal', 'vary_noise', 'vary_distractor'])._make(
    ['vary_signal', 'vary_noise', 'vary_distractor'])


@dataclass
class WeightsDataGeneration:
    signal: float = 1 / 3
    distractor: float = 1 / 3
    noise: float = 1 / 3


@dataclass
class SimpleDataset:
    x: np.ndarray
    y: np.ndarray


class TwoBlobsForwardModelDataset:
    def __init__(self, num_examples: int, pattern_matrix: np.ndarray,
                 seed: int, noise_type: str, signal_type: str,
                 weights: WeightsDataGeneration = WeightsDataGeneration()):
        super().__init__()
        self._num_examples = num_examples
        self._random_generator = np.random.default_rng(seed=seed)
        self._pattern_matrix = self._check_dimension_of_pattern_matrix(matrix=pattern_matrix)
        self._blob_centers = (-1, 1)
        self._noise_type = noise_type
        self._signal_type = signal_type
        self._weights = weights
        self._cov_matrix = None
        self._signal = None
        self._noise = None
        self._distractor = None

        self._data = self._generate_data()

    def _noise_map(self) -> Dict:
        return {NoiseTypes.white_gaussian: self._univariate_white_gaussian_noise,
                NoiseTypes.zero_mean_gaussian: self._zero_mean_gaussian_noise}

    @staticmethod
    def _check_dimension_of_pattern_matrix(matrix: np.ndarray) -> np.ndarray:
        if 2 != matrix.shape[1]:
            raise ValueError('The pattern matrix has to be of the form Mx2, '
                             'at the moment!')
        else:
            return matrix

    def _assemble_data(self, signal: np.ndarray, distractor: np.ndarray, noise: np.ndarray):
        signal_term = np.dot(np.expand_dims(self._pattern_matrix[:, 0], axis=1), signal.T).T
        distractor_term = np.dot(np.expand_dims(self._pattern_matrix[:, 1], axis=1), distractor.T).T
        signal_term /= np.linalg.norm(signal_term, ord='fro')
        d_norm = np.linalg.norm(distractor_term, ord='fro')
        distractor_term = distractor_term if 0 == d_norm else distractor_term / d_norm
        noise /= np.linalg.norm(noise, ord='fro')
        return (self._weights.signal * signal_term +
                self._weights.distractor * distractor_term +
                self._weights.noise * noise)

    def _generate_data(self):
        self._signal, labels = self._generate_signal_of_interest_and_labels()
        self._distractor = self._random_generator.normal(loc=0, scale=1., size=self._signal.shape)
        self._noise = self._generate_noise(dimension=self._pattern_matrix.shape[0],
                                           size=self._num_examples)
        x = self._assemble_data(signal=self._signal, distractor=self._distractor, noise=self._noise)
        return SimpleDataset(x=x, y=labels)

    def _generate_signal_of_interest_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        y1_shape = (self._num_examples // 2, 1)
        y2_shape = (self._num_examples - y1_shape[0], 1)
        if Signals.noisy == self._signal_type:
            y1 = self._random_generator.normal(loc=self._blob_centers[0], scale=1, size=y1_shape)
            y2 = self._random_generator.normal(loc=self._blob_centers[1], scale=1, size=y2_shape)
        else:
            y1 = self._blob_centers[0] * np.ones(shape=y1_shape)
            y2 = self._blob_centers[1] * np.ones(shape=y2_shape)
        labels = np.concatenate([np.zeros(shape=y1_shape), np.ones(shape=y2_shape)], axis=0)
        return np.concatenate([y1, y2], axis=0), labels

    def _random_orthonormal_matrix(self, dim: int) -> np.ndarray:
        A = np.triu(2 * np.pi * self._random_generator.random(size=(dim, dim)) - np.pi)
        B = expm(A - A.T)
        return B

    def _random_spd_matrix(self, dim: int, power_factor: float = 3.0) -> np.ndarray:
        U = self._random_orthonormal_matrix(dim=dim)
        d = self._random_generator.random(size=dim)
        d += 0.01 * np.max(d)
        D = np.diag(d)
        return np.matmul(np.matmul(U, D), U.T)

    def _univariate_white_gaussian_noise(self, dimension: int, size: int) -> np.ndarray:
        samples = self._random_generator.normal(
            loc=0.0, scale=1, size=(size, 1))
        return samples * np.ones(shape=(size, dimension))

    def _zero_mean_gaussian_noise(self, dimension: int, size: Union[int, Tuple]) -> np.ndarray:
        self._cov_matrix = self._random_spd_matrix(dim=dimension)
        return self._random_generator.multivariate_normal(
            mean=np.zeros(dimension), cov=self._cov_matrix, size=size)

    def _generate_noise(self, dimension: int, size: Union[int, Tuple]) -> np.ndarray:
        return self._noise_map().get(self._noise_type)(dimension=dimension, size=size)

    @property
    def data(self) -> SimpleDataset:
        return self._data

    @property
    def cov_matrix(self) -> np.ndarray:
        return self._cov_matrix

    @property
    def signal(self) -> np.ndarray:
        return self._signal

    @property
    def noise(self) -> np.ndarray:
        return self._noise

    @property
    def distractor(self) -> np.ndarray:
        return self._distractor


def initialize_weights(pivot_weight: float, experiment_type: str) -> WeightsDataGeneration:
    rest = (1 - pivot_weight) / 2
    if ExperimentTypes.vary_signal == experiment_type:
        weight = WeightsDataGeneration(signal=pivot_weight, distractor=rest, noise=rest)
    elif ExperimentTypes.vary_noise == experiment_type:
        weight = WeightsDataGeneration(signal=rest, distractor=rest, noise=pivot_weight)
    else:
        weight = WeightsDataGeneration(signal=rest, distractor=pivot_weight, noise=rest)
    return weight


def generate_data_weights(experiment_type: str,
                          config: Config, scaling: float = 0.1) -> List:
    weights = list()
    n = config.num_data_weight_combinations
    for k in range(n):
        tmp = scaling * k / n
        weights += [initialize_weights(pivot_weight=tmp, experiment_type=experiment_type)]
    return weights
