from dataclasses import field, dataclass
from typing import Dict, ClassVar, List


@dataclass
class Config:
    num_experiments: int
    num_examples: int
    val_size: float
    signal_scaling: float
    seed: int
    pattern_type: int
    noise_type: str
    num_data_weight_combinations: int
    output_dir: str
    experiment_types: List = field(default_factory=list)
    signal_types: List = field(default_factory=list)
    instance: ClassVar = field(default=None)

    @staticmethod
    def default_conf() -> dict:
        return {
            'num_experiments': 1,
            'experiment_types': ['vary_signal'],
            'num_expamples': 100,
            'val_size': 0.2,
            'signal_scaling': 0.1,
            'seed': 1142,
            'pattern_type': 0,
            'noise_type': 'zero_mean_gaussian',
            'signal_types': ['exact'],
            'num_data_weight_combinations': 1,
            'output_dir': 'data'
        }

    @classmethod
    def get(cls, input_conf: Dict = None):
        if not cls.instance:
            if input_conf is not None:
                conf_dict = input_conf
            else:
                conf_dict = cls.default_conf()
            cls.instance = cls(**conf_dict)
        return cls.instance
