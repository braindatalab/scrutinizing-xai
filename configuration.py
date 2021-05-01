from dataclasses import dataclass, field
from typing import Dict, List, ClassVar


@dataclass
class Config:
    seed: int
    dpi: int
    output_dir_scores: str
    output_dir_plots: str
    data_path: str
    result_paths: List[str]
    instance: ClassVar = field(default=None)

    @staticmethod
    def default_conf() -> dict:
        return {
            'seed': 1142,
            'dpi': 300,
            'data_path': 'data/data_vary_signal_exact_2021-01-18-16-07-37.pkl',
            'output_dir_scores': 'results/scores',
            'output_dir_plots': 'results/plots'
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
