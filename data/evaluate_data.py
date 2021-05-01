from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from common import load_pickle
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    fpath = '../data/data_vary_signal_exact_2021-04-29-14-44-03_pattern_type_6.pkl'
    data = load_pickle(file_path=fpath)
    idx_experiment = 22
    idx_sample = 107
    for weight, data_list in data.items():
        print(weight)
        d = data_list[idx_experiment]
        sample = data_list[idx_experiment]['val']['x'][idx_sample, :]
        model = LogisticRegression(penalty='none', fit_intercept=False, max_iter=10, random_state=123)
        model.fit(X=d['train']['x'], y=d['train']['y'].flatten())
        pred_train = model.predict(d['train']['x'])
        pred_val = model.predict(d['val']['x'])
        print(f"Accuracy train: {accuracy_score(y_true=d['train']['y'].flatten(), y_pred=pred_train)}")
        print(f"Accuracy val: {accuracy_score(y_true=d['val']['y'].flatten(), y_pred=pred_val)}")
        sns.heatmap(sample.reshape((8, 8)), center=0.0)
        plt.show()
        label = data_list[idx_experiment]['val']['y'][idx_sample]
        print(f'Weight: {weight} Prediction: {model.predict(sample.reshape((1, 64)))} Label: {label}')

if __name__ == '__main__':
    main()
