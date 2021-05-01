from scipy.io import savemat
from common import load_pickle


def main():
    # fpath = '../data/data_vary_signal_exact_2021-01-18-16-07-37.pkl'
    fpath = '../data/data_vary_signal_exact_2021-02-23-12-45-08.pkl'
    # fpath = '../data/data_vary_signal_exact_2021-02-01-11-36-15.pkl'
    data = load_pickle(file_path=fpath)
    idx_experiment = 22
    new_data = dict()
    for weight, data_list in data.items():
        new_data['w'+'d'.join(weight.split('.'))] = data_list[idx_experiment]
    savemat(file_name='data_vary_signal_exact_2021-02-23-12-45-08.mat', mdict=new_data)


if __name__ == '__main__':
    main()
