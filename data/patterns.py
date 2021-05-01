from collections import namedtuple
from typing import Dict

import numpy as np

PATTERN_BLOCK = np.array([[0, 0.5, 0.5, 0],
                          [0.5, 1, 1, 0.5],
                          [0.5, 1, 1, 0.5],
                          [0, 0.5, 0.5, 0]])

PatternType = namedtuple('PatternMap',
                         ['sixty_four_dimensional',
                          'two_dimensional',
                          'sixteen_dimensional',
                          'sixty_four_dimensional_one_blob',
                          'sixty_four_dimensional_one_blob_each'])._make([0, 1, 2, 3, 4])


class PatternMatrix:
    def __init__(self, pattern_type: int):
        self._pattern_type = pattern_type
        self._matrix = self._generate_pattern()
        self.dim_of_signal = 0

    def _generate_pattern(self) -> np.ndarray:
        return map_of_patterns()[self._pattern_type]

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


def create_pattern_matrix_2x1() -> np.ndarray:
    return np.array([1, 0])


def create_distractor_matrix_2x1() -> np.ndarray:
    return np.array([0, 0])


def create_two_blobs_pattern_matrix_8x8() -> np.ndarray:
    pattern_matrix = np.zeros((8, 8))
    pattern_matrix[0:4, 0:4] = PATTERN_BLOCK
    pattern_matrix[4:, 0:4] = (-1) * PATTERN_BLOCK
    return pattern_matrix


def create_one_blob_pattern_matrix_8x8() -> np.ndarray:
    pattern_matrix = np.zeros((8, 8))
    pattern_matrix[0:4, 0:4] = PATTERN_BLOCK
    return pattern_matrix


def create_distractor_matrix_8x8() -> np.ndarray:
    distractor_matrix = np.zeros((8, 8))
    distractor_matrix[0:4, 0:4] = PATTERN_BLOCK
    distractor_matrix[0:4, 4:] = (-1) * PATTERN_BLOCK
    return distractor_matrix


def create_one_blob_distractor_matrix_8x8() -> np.ndarray:
    distractor_matrix = np.zeros((8, 8))
    distractor_matrix[0:4, 4:] = PATTERN_BLOCK
    return distractor_matrix


def create_two_blobs_pattern_matrix_4x4() -> np.ndarray:
    return np.array([[1, 0.5, 0, 0],
                     [0.5, 0, 0, 0],
                     [0.5, 0, 0, 0],
                     [1, 0.5, 0, 0]])


def create_distractor_matrix_4x4() -> np.ndarray:
    return np.array([[0, 0, 0.5, 1],
                     [0, 0, 0, 0.5],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0]])


def map_of_patterns() -> Dict:
    return {1: np.array([create_pattern_matrix_2x1(),
                         create_distractor_matrix_2x1()]),
            0: np.column_stack((create_two_blobs_pattern_matrix_8x8().flatten(),
                                create_distractor_matrix_8x8().flatten())),
            2: np.column_stack((create_two_blobs_pattern_matrix_4x4().flatten(),
                                create_distractor_matrix_4x4().flatten())),
            3: np.column_stack((create_one_blob_pattern_matrix_8x8().flatten(),
                                create_distractor_matrix_8x8().flatten())),
            4: np.column_stack((create_one_blob_pattern_matrix_8x8().flatten(),
                                create_one_blob_distractor_matrix_8x8().flatten())),
            5: np.column_stack((create_one_blob_pattern_matrix_8x8().flatten(),
                                (-1) * create_one_blob_distractor_matrix_8x8().flatten())),
            6: np.column_stack((create_two_blobs_pattern_matrix_8x8().flatten(),
                                np.zeros((8, 8)).flatten()))}
