from dataclasses import field, dataclass
from typing import Dict, ClassVar, List


@dataclass
class Config:
    seed: int
    output_dir: str
    data_path: str
    method_names: List = field(default_factory=list)
    instance: ClassVar = field(default=None)

    @staticmethod
    def default_conf() -> dict:
        return {
            'seed': 1142,
            'method_names': ['pfi', 'mr_empirical', 'firm', 'pattern'],
            'data_path': 'data/data_vary_signal_exact.pkl',
            'output_dir': 'results/model_agnostic_methods'
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
