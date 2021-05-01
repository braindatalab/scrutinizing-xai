import argparse

from common import load_json_file
from configuration import Config
from visualization import plot
from evaluation import evaluate


def get_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', required=True,
                        help='Input file path of json file containing'
                             'input parameter for the experiment!')
    return parser.parse_args()


def main(config_path: str):
    config = Config.get(input_conf=load_json_file(file_path=config_path))
    paths_evaluation_results = evaluate(config=config)
    plot(config=config, score_paths=paths_evaluation_results)


if __name__ == '__main__':
    args = get_command_line_arguments()
    path = args.path
    try:
        main(config_path=path)
    except KeyboardInterrupt as e:
        print(e)
