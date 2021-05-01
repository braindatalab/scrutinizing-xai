import argparse
import os
from copy import deepcopy
from os.path import join, split
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns

from common import load_pickle, load_json_file, ScoresAttributes, extract_pattern_type
from configuration import Config
from evaluation import _generate_true_feature_importance

A = ScoresAttributes.get()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NAME_MAPPING = {
    'llr': 'LLR',
    'pfi': 'PFI',
    'mr': 'PFIO',
    'mr_empirical': 'EMR',
    'anchors': 'Anchors',
    'lime': 'LIME',
    'shap_linear': 'SHAP',
    'pattern': 'Pattern',
    'firm': 'FIRM',
    'tree_fi': 'Impurity',
    'model_weights': '$|w_{LLR}|$',
    'model_weights_NN': '$|w_{NN}|$',  # '$|w_{NN, 0} - w_{NN, 1}|$',
    'gradient': '$Grad_{NN}$',
    'deep_taylor': 'DTD',
    'lrp.z': '$LRP_{z}$',
    'sample': 'Sample',
    'lrp.alpha_beta': '$LRP_{\\alpha\\beta}$',
    'pattern.net': 'PatternNet',
    'pattern.attribution': 'PatternAttr.',
    'input_t_gradient': 't_gradient',
    'impurity': 'Impurity',
    'correlation': 'Corr.',
    'binary_mask': 'Ground\nTruth',
    'pattern_distractor': 'Pattern/Distractor'

}

METHOD_BLACK_LIST = ['input_t_gradient', 'impurity']

NAME_MAPPING_SCORES = {
    'pr_auc': 'Precision-Recall AUC',
    # 'max_precision': 'Max Precision',
    # 'max_precision': 'Precision \n for Specificity $\\approx$ 0.9',
    'max_precision': 'PREC90',
    'avg_precision': 'Average Precision',
    'auc': 'AUROC'
}

FONT_SIZES = {
    'ticks': 10,
    'label': 12,
    'legend': 10
}


def save_figure(file_path: str, fig: plt.Figure, dpi: int) -> None:
    output_dir = split(file_path)[0]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    fig.savefig(fname=file_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig=fig)


def create_accuracy_plot(model_accuracies: Dict, config: Config,
                         file_name_suffix: str) -> None:
    data_dict = {'SNR': list(), 'Accuracy': list(), 'data_type': list()}
    for weights, values in model_accuracies.items():
        snr = f'{weights.split("_")[0]}'
        data_dict['SNR'] += [snr]
        data_dict['Accuracy'] += [np.mean(values['train'])]
        data_dict['data_type'] += ['train']
        data_dict['SNR'] += [snr]
        data_dict['Accuracy'] += [np.mean(values['val'])]
        data_dict['data_type'] += ['val']

    sns.set_theme('paper')
    with sns.axes_style("whitegrid"):
        g = sns.lineplot(data=pd.DataFrame(data_dict), x='SNR', y='Accuracy', hue='data_type')
        g.set_ylim(0, 1)
        plt.legend(loc='lower right')
        plt.tight_layout()
    file_name = '_'.join(['accuracy_avg_plot', file_name_suffix, '.png'])
    output_path = join(config.output_dir_plots, file_name)
    fig = g.get_figure()
    save_figure(file_path=output_path, fig=fig, dpi=config.dpi)


def overall_accuracy_plot(scores: Dict, config: Config) -> None:
    data_dict = {'SNR': list(), 'Accuracy': list(),
                 'Dataset': list(), 'Model': list()}

    for weights, values in scores[A.model_accuracies].items():
        snr = f'{weights.split("_")[0]}'
        for model_name, accuracies in values.items():
            data_dict['SNR'] += [snr]
            data_dict['Accuracy'] += [np.mean(accuracies['train'])]
            data_dict['Dataset'] += ['train']
            data_dict['SNR'] += [snr]
            data_dict['Accuracy'] += [np.mean(accuracies['val'])]
            data_dict['Dataset'] += ['val']
            data_dict['Model'] += [model_name]
            data_dict['Model'] += [model_name]

    sns.set_theme('paper')
    with sns.axes_style("whitegrid"):
        g = sns.lineplot(data=pd.DataFrame(data_dict), x='SNR', linewidth=4,
                         y='Accuracy', hue='Dataset', style='Model', markers=False)
        g.set_ylim(0, 1)
        g.set_aspect(aspect=2.5)

        g.set(xlabel='$\lambda_1$')
        g.set_yticklabels(labels=[f'{float(l):.2f}' for l in g.get_yticks()], size=FONT_SIZES['label'])
        g.set_xticklabels(labels=np.unique(data_dict['SNR']), size=FONT_SIZES['label'])
        for item in ([g.xaxis.label, g.yaxis.label]):
            item.set_fontsize(FONT_SIZES['label'])

        plt.legend(loc='lower right', prop={'size': FONT_SIZES['legend']})
        plt.tight_layout()
        fig = g.get_figure()
    file_name = '_'.join(['overall_accuracy_avg_plot', '.png'])
    output_path = join(config.output_dir_plots, file_name)
    save_figure(file_path=output_path, fig=fig, dpi=config.dpi)


def create_rain_cloud_data(data: Dict, metric_name: str) -> pd.DataFrame:
    data_dict = {'$\lambda_1$': list(), 'Method': list(),
                 metric_name: list(), 'Methods': list()}
    for snr, snr_data in data.items():
        for method, method_data in snr_data.items():
            for roc_auc_data in method_data:
                for score in roc_auc_data[metric_name]:
                    if method in METHOD_BLACK_LIST:
                        continue
                    data_dict[metric_name] += [score]
                    data_dict['$\lambda_1$'] += [snr.split('_')[0]]
                    data_dict['Method'] += [NAME_MAPPING.get(method, method)]
                    # data_dict['Methods'] += ['1']
                    data_dict['Methods'] += [NAME_MAPPING.get(method, method)]

    return pd.DataFrame(data_dict)


def create_rain_cloud_plots(data: Dict, config: Config,
                            score_attribute: str, file_name_suffix: str) -> None:
    df = create_rain_cloud_data(data=data, metric_name=score_attribute)
    sigma = .5
    sns.set_theme('paper')
    sns.set(font_scale=1)
    with sns.axes_style("whitegrid"):
        g = sns.FacetGrid(df, col='$\lambda_1$', height=6, ylim=(0, 1.05), )
        g.map_dataframe(pt.RainCloud, x='Method', y=score_attribute, data=df,
                        orient='v', bw=sigma, width_viol=.5, linewidth=1)
        for ax in g.axes.flat:
            labels = ax.get_xticklabels()
            ax.set_xticklabels(labels, rotation=20)
        g.tight_layout()
    file_name = '_'.join(['rain_cloud_plot', file_name_suffix, score_attribute, '.png'])
    output_path = join(config.output_dir_plots, file_name)
    save_figure(file_path=output_path, fig=g.fig, dpi=config.dpi)


def create_violin_plots(data: Dict, config: Config,
                        score_attribute: str, file_name_suffix: str) -> None:
    df = create_rain_cloud_data(data=data, metric_name=score_attribute)
    sigma = .5
    sns.set_theme('paper')
    sns.set(font_scale=1)
    with sns.axes_style("whitegrid"):
        g = sns.FacetGrid(df, col='SNR', height=6, ylim=(0, 1.05), )
        g.map_dataframe(sns.violinplot, x='Method', y=score_attribute, data=df,
                        orient='v', hue='Method', bw=sigma, width_viol=.9,
                        palette='muted', linewidth=1)
        for ax in g.axes.flat:
            labels = ax.get_xticklabels()
            ax.set_xticklabels(labels, rotation=20)
        g.fig.subplots_adjust(bottom=0.15)
        g.tight_layout()
    file_name = '_'.join(['violin_plot', file_name_suffix, score_attribute, '.png'])
    output_path = join(config.output_dir_plots, file_name)
    save_figure(file_path=output_path, fig=g.fig, dpi=config.dpi)


def is_keras_model(data: np.ndarray):
    shape = data.shape
    p = np.prod(shape)
    return not (p == shape[0] or p == shape[1])


def is_sample_based(data: np.ndarray):
    shape = data.shape
    p = np.prod(shape)
    return not (p == shape[0] or p == shape[1])


def get_randomized_heat_map_data(scores: Dict, data: Dict, rnd_idx: int) -> Dict:
    data_dict = dict()
    for weight, data_list in data.items():
        model_weights = scores[A.model_weights][weight]
        data_dict[weight] = {'data': data_list[rnd_idx],
                             'model_weights_nn': model_weights[A.neural_net][rnd_idx],
                             'model_weights_lr': model_weights[A.logistic_regression][rnd_idx],
                             'rnd_experiment_idx': rnd_idx}
    return data_dict


def add_column_for_class_of_explanation_method(data: pd.DataFrame) -> pd.DataFrame:
    num_samples = data.shape[0]
    if any(data['Method'].map(lambda x: 'LIME' == x)):
        data['class'] = ['Agnostic Sample Based'] * num_samples
    elif any(data['Method'].map(lambda x: 'Deep' in x)):
        data['class'] = ['Saliency'] * num_samples
    else:
        data['class'] = ['Agnostic Global'] * num_samples
    return data


def overview_rain_cloud_plot(paths: List[str], config: Config,
                             score_data_key: str, metric_name: str):
    df = pd.DataFrame()
    for score_path in paths:
        scores = load_pickle(file_path=score_path)
        aux_df = create_rain_cloud_data(data=scores[score_data_key], metric_name=metric_name)
        aux_df = add_column_for_class_of_explanation_method(data=aux_df)
        df = df.append(aux_df)

    sigma = .5
    sns.set_theme('paper')
    sns.set(font_scale=1)
    with sns.axes_style("whitegrid"):
        g = sns.FacetGrid(df, row='class', col='SNR', height=6, ylim=(0, 1.05))
        g.map_dataframe(pt.RainCloud, x='Method', y=metric_name, data=df,
                        orient='v', bw=sigma, width_viol=.0)
        for ax in g.axes.flat:
            labels = ax.get_xticklabels()
            ax.set_xticklabels(labels, rotation=20)
        g.fig.subplots_adjust(bottom=0.15)

    file_name = '_'.join(['rain_cloud_plot', 'overview', metric_name, '.png'])
    output_path = join(config.output_dir_plots, file_name)
    save_figure(file_path=output_path, fig=g.fig, dpi=config.dpi)


def rain_clouds(scores: Dict, config: Config,
                score_data_keys: List[Tuple], mode: str = 'sample_based'):
    dfs = list()
    metric_name_plot = '.'.join([item[1] for item in score_data_keys])
    for score_data_key, metric_name in score_data_keys:
        aux_df = create_rain_cloud_data(data=scores[score_data_key], metric_name=metric_name)
        aux_df['class'] = A.sample_based
        aux_df[metric_name_plot] = aux_df[metric_name]
        aux_df['Metric'] = [metric_name] * aux_df.shape[0]
        dfs += [deepcopy(aux_df)]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    sns.set_theme('paper')
    with sns.axes_style('white'):
        f = sns.catplot(x='Method', y=metric_name_plot, hue='Methods',
                        col='$\lambda_1$', row='Metric', legend_out=True, legend=True,
                        data=df, kind='box', width=0.7, seed=config.seed,
                        height=4, aspect=0.7, palette='Set2')
        legend_handles = f.legend.legendHandles
        plt.close(fig=f.fig)

        g = sns.FacetGrid(df, col='$\lambda_1$', row='Metric', height=4,
                          ylim=(0, 1.05), aspect=0.7,
                          legend_out=True, palette='Set2')
        g.map_dataframe(pt.RainCloud, x='Method', y=metric_name_plot, data=df,
                        orient='v', bw=0.45, width_viol=0.7, width_box=0.1)

        ax = g.axes
        configure_axes(ax=ax)
        g.fig.subplots_adjust(bottom=0.15)
        g.fig.subplots_adjust(wspace=0.05, hspace=0.05)

    # plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1.5),
    #            loc='upper left', borderaxespad=0., facecolor='white', framealpha=1)
    plt.legend(handles=legend_handles, bbox_to_anchor=(-1.3, -0.05),
               ncol=int(np.unique(df['Method'].values).shape[0] / 2 + 0.5), fancybox=True,
               loc='upper center', borderaxespad=0., facecolor='white', framealpha=1)
    file_name = '_'.join(['rain_cloud_plot', mode, metric_name_plot, '.png'])
    output_path = join(config.output_dir_plots, file_name)
    save_figure(file_path=output_path, fig=g.fig, dpi=config.dpi)


def configure_axes(ax: Any) -> None:
    for row_idx in range(ax.shape[0]):
        t = ax[row_idx, 0].get_title(loc='center')
        name_of_metric = t.split('|')[0].split('=')[-1].strip()
        ax[row_idx, 0].set_ylabel(ylabel=NAME_MAPPING_SCORES[name_of_metric],
                                  fontdict={'fontsize': 18})
        for col_idx in range(ax.shape[1]):
            if 0 == row_idx:
                t = ax[row_idx, col_idx].get_title(loc='center')
                new_title = t.split('|')[-1].strip()
                ax[row_idx, col_idx].set_title(
                    label=new_title, fontdict={'fontsize': 18})
            else:
                ax[row_idx, col_idx].set_title(label='')
            ax[row_idx, col_idx].set_xlabel(xlabel='',
                                            fontdict={'fontsize': 18})
            labels = ax[row_idx, col_idx].get_xticklabels()
            # ax.set_xticklabels(labels, rotation=45, fontdict={'fontsize': 9})
            ax[row_idx, col_idx].set_xticklabels('')
            ax[row_idx, col_idx].patch.set_edgecolor('black')
            ax[row_idx, col_idx].grid(True)
            sns.despine(ax=ax[row_idx, col_idx],
                        top=False, bottom=False, left=False, right=False)


def box_plot(scores: Dict, config: Config, snrs_of_interest: list,
             score_data_keys: List[Tuple], mode: str = 'sample_based'):
    dfs = list()
    metric_name_plot = '.'.join([item[1] for item in score_data_keys])
    for score_data_key, metric_name in score_data_keys:
        aux_df = create_rain_cloud_data(data=scores[score_data_key], metric_name=metric_name)
        aux_df['class'] = A.sample_based
        aux_df[metric_name_plot] = aux_df[metric_name]
        aux_df['Metric'] = [metric_name] * aux_df.shape[0]
        dfs += [deepcopy(aux_df)]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    snr_filter = df['$\lambda_1$'].map(lambda x: x in snrs_of_interest).values

    # anchors_filter = df['Method'].map(lambda x: 'Anchors' == x)
    # anchors = df.loc[anchors_filter, :]
    # metric_filter = anchors['Metric'].map(lambda x: 'max_precision' == x)
    # anchors_max_precision = anchors.loc[metric_filter, :]

    df = df.loc[snr_filter, :]
    with sns.axes_style('white'):
        f = sns.catplot(x='Method', y=metric_name_plot, hue='Methods',
                        col='$\lambda_1$', row='Metric', legend_out=True, seed=config.seed,
                        data=df, kind='box', width=0.7,
                        height=4, aspect=1, palette='colorblind')
        legend_handles = f.legend.legendHandles
        plt.close(fig=f.fig)

        g = sns.catplot(x='Method', y=metric_name_plot, seed=config.seed,
                        col='$\lambda_1$', legend_out=True, row='Metric',
                        data=df, kind='box', width=0.7,
                        height=4, aspect=1, palette='colorblind')
        ax = g.axes
        configure_axes(ax=ax)
        g.fig.subplots_adjust(bottom=0.15)
        g.fig.subplots_adjust(wspace=0.05, hspace=0.05)

    # plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1.5),
    #            loc='upper left', borderaxespad=0., facecolor='white', framealpha=1)
    plt.legend(handles=legend_handles, bbox_to_anchor=(-0.55, -0.05), prop={'size': 14},
               ncol=int(np.unique(df['Method'].values).shape[0] / 2 + 0.5), fancybox=True,
               loc='upper center', borderaxespad=0., facecolor='white', framealpha=1)
    # loc = 'upper center', bbox_to_anchor = (0.5, -0.05),
    # fancybox = True, shadow = True, ncol = 5
    file_name = '_'.join(['box_plot', mode, metric_name_plot, '.png'])
    output_path = join(config.output_dir_plots, file_name)
    save_figure(file_path=output_path, fig=g.fig, dpi=config.dpi)


def is_saliency(method_names: List) -> bool:
    return True if 'deep_taylor' in method_names else False


def is_sample_based_agnostic(method_names: List) -> bool:
    return True if 'lime' in method_names else False


def global_heat_maps(scores: Dict, config: Config, pattern_type: int,
                     rnd_experiment_idx: int, snrs_of_interest: list) -> None:
    def _heat_map(result: np.ndarray, sub_ax: Any):
        return sns.heatmap(result, vmin=0, ax=sub_ax, square=True,
                           cbar=False, cbar_kws={"shrink": shrinking})

    methods = [item for item in scores[A.global_based][A.method_names] if
               item not in METHOD_BLACK_LIST]
    explanations = dict()
    model_weights = scores[A.model_weights]
    data_weights = deepcopy(scores[A.data_weights])
    for d in data_weights:
        if d.split('_')[0] not in snrs_of_interest:
            data_weights.remove(d)

    for j, w in enumerate(data_weights):
        explanations_per_method = dict()
        for method in methods:
            explanations_per_method[method] = scores[A.global_based][A.explanations][w][method]
        explanations[w] = explanations_per_method

    print(f'Number of methods: {len(methods)}')
    methods.insert(0, 'model_weights')
    # methods.insert(0, 'model_weights_NN')
    methods.insert(0, 'binary_mask')
    num_methods = len(methods)
    num_weights = len(data_weights)
    num_cols = np.maximum(num_methods, 2)
    num_rows = np.maximum(num_weights, 2)
    sns.set_theme('paper')
    shrinking = 1.0
    hspace_values = {5: -885, 3: -0.900}
    fig, ax = plt.subplots(ncols=num_cols, nrows=num_rows, sharex=True, sharey=True,
                           gridspec_kw={'wspace': 0.05, 'hspace': hspace_values[len(data_weights)]})
    for i, weight in enumerate(data_weights):
        print(weight)
        model_weight = model_weights[weight]
        for k, method in enumerate(methods):
            if 0 == i:
                ax[i, k].set_title(NAME_MAPPING[method], rotation=90,
                                   fontdict={'fontsize': 4})

            if 1 == k:
                w = model_weight[A.logistic_regression][rnd_experiment_idx]
                heat_map = np.abs(w)
                g = _heat_map(result=heat_map.reshape(8, 8), sub_ax=ax[i, k])
            # elif 1 == k:
            #     w = model_weight[A.neural_net][rnd_experiment_idx]
            #     heat_map = np.abs(w[:, 0] - w[:, 1])
            #     g = _heat_map(result=heat_map.reshape(8, 8), sub_ax=ax[i, k])
            elif 0 == k:
                b = _generate_true_feature_importance(pattern_type=pattern_type)
                g = _heat_map(result=np.abs(b).reshape((8, 8)), sub_ax=ax[i, k])
                ylabel = f'{weight.split("_")[0]}'
                g.set_ylabel(f'$\lambda_1=${ylabel}', fontdict={'fontsize': 4})
            else:
                explanation = explanations[weight][method][rnd_experiment_idx]
                if is_sample_based(data=explanation):
                    heat_map = np.mean(np.abs(explanation), axis=0)
                else:
                    heat_map = np.abs(explanation)
                g = _heat_map(result=heat_map.reshape((8, 8)), sub_ax=ax[i, k])

            g.set(yticks=[])
            g.set(xticks=[])

            if 0 != k:
                ax[i, k].yaxis.set_visible(False)
            if (num_weights - 1) != i:
                ax[i, k].xaxis.set_visible(False)
            ax[i, k].set_aspect('equal', adjustable='box')

    file_name = '_'.join(['heat_map_global_mean', '.png'])
    output_path = join(config.output_dir_plots, file_name)
    save_figure(file_path=output_path, fig=fig, dpi=config.dpi)
    x = 1


def sample_based_heat_maps(scores: Dict, config: Config, data: Dict, pattern_type: int,
                           rnd_sample_idx: int, snrs_of_interest: list) -> None:
    def _heat_map(result: np.ndarray, sub_ax: Any):
        return sns.heatmap(result, vmin=0, ax=sub_ax, square=True,
                           cbar=False, cbar_kws={"shrink": shrinking})

    local_method_black_list = METHOD_BLACK_LIST + ['gradient']
    methods = [item for item in scores[A.sample_based][A.method_names] if
               item not in local_method_black_list]
    explanations = dict()
    model_weights = scores[A.model_weights]
    data_weights = deepcopy(scores[A.data_weights])
    for d in data_weights:
        if d.split('_')[0] not in snrs_of_interest:
            data_weights.remove(d)

    for j, w in enumerate(data_weights):
        explanations_per_method = dict()
        for method in methods:
            explanations_per_method[method] = scores[A.sample_based][A.explanations][w][method]
        explanations[w] = explanations_per_method

    print(f'Number of methods: {len(methods)}')
    # methods.insert(0, 'model_weights')
    # methods.insert(0, 'model_weights_NN')
    methods.insert(0, 'binary_mask')
    methods.insert(0, 'sample')
    num_methods = len(methods)
    num_weights = len(data_weights)
    num_cols = np.maximum(num_methods, 2)
    num_rows = np.maximum(num_weights, 2)
    sns.set_theme('paper')
    shrinking = 1.0
    hspace_values = {5: -825, 3: -0.815}
    fig, ax = plt.subplots(ncols=num_cols, nrows=num_rows, sharex=True, sharey=True,
                           gridspec_kw={'wspace': 0.05, 'hspace': hspace_values[len(data_weights)]})
    sample = None
    for i, weight in enumerate(data_weights):
        print(weight)
        model_weight = model_weights[weight]
        dataset = data[weight]['data']
        rnd_experiment_idx = data[weight]['rnd_experiment_idx']
        for k, method in enumerate(methods):
            if 0 == i:
                ax[i, k].set_title(NAME_MAPPING[method], rotation=90,
                                   fontdict={'fontsize': 8})

            # if 3 == k:
            #     w = model_weight[A.logistic_regression][rnd_experiment_idx]
            #     heat_map = np.abs(w)
            #     g = _heat_map(result=heat_map.reshape(8, 8), sub_ax=ax[i, k])
            # elif 2 == k:
            #     w = model_weight[A.neural_net][rnd_experiment_idx]
            #     heat_map = np.abs(w[:, 0] - w[:, 1])
            #     g = _heat_map(result=heat_map.reshape(8, 8), sub_ax=ax[i, k])
            if 1 == k:
                b = _generate_true_feature_importance(pattern_type=pattern_type)
                g = _heat_map(result=np.abs(b).reshape((8, 8)), sub_ax=ax[i, k])
            elif 0 == k:
                x = dataset['val']['x'][rnd_sample_idx]
                g = sns.heatmap(x.reshape((8, 8)), ax=ax[i, k], center=0.0,
                                square=True, cbar=False, cbar_kws={"shrink": shrinking})
                ylabel = f'{weight.split("_")[0]}'
                g.set_ylabel(f'$\lambda_1=${ylabel}', fontdict={'fontsize': 7})
            else:
                explanation = explanations[weight][method][rnd_experiment_idx]
                heat_map = np.abs(explanation[rnd_sample_idx, :])
                g = _heat_map(result=heat_map.reshape((8, 8)), sub_ax=ax[i, k])

            g.set(yticks=[])
            g.set(xticks=[])

            if 0 != k:
                ax[i, k].yaxis.set_visible(False)
            if (num_weights - 1) != i:
                ax[i, k].xaxis.set_visible(False)
            ax[i, k].set_aspect('equal', adjustable='box')

    file_name = '_'.join(['heat_map_sample_based', '.png'])
    output_path = join(config.output_dir_plots, file_name)
    save_figure(file_path=output_path, fig=fig, dpi=config.dpi)


def overview_correlation_plot(scores: Dict, config: Config) -> None:
    nn_data = {'Model Weight': list(), 'SNR': list(), 'Model': list()}
    lr_data = {'Model Weight': list(), 'SNR': list(), 'Model': list()}
    for data_weight, model_weights in scores[A.model_weights].items():
        for l in range(len(model_weights[A.neural_net])):
            snr = data_weight.split('_')[0]
            nn_data['Model Weight'] += [
                (model_weights[A.neural_net][l][:, 1] - model_weights[A.neural_net][l][:,
                                                        0]).flatten()]
            nn_data['Model'] += ['Single Layer NN']
            nn_data['SNR'] += [snr]
            lr_data['Model Weight'] += [model_weights[A.logistic_regression][l].flatten()]
            lr_data['Model'] += ['Logistic Regression']
            lr_data['SNR'] += [snr]

    model_weights = pd.merge(left=pd.DataFrame(nn_data), right=pd.DataFrame(lr_data),
                             left_index=True, right_index=True)

    def compute_correlation(x):
        return np.corrcoef(x[0], x[1])[0, 1]

    model_weights['Correlation'] = model_weights[['Model Weight_x', 'Model Weight_y']].apply(
        compute_correlation, axis=1)

    sigma = .5
    sns.set_theme('paper')
    sns.set(font_scale=1)
    with sns.axes_style("whitegrid"):
        g = sns.FacetGrid(model_weights, height=6, ylim=(0, 1.05))
        g.map_dataframe(pt.RainCloud, x='SNR_x', y='Correlation', data=model_weights,
                        orient='v', bw=sigma, width_viol=.3)
        for ax in g.axes.flat:
            labels = ax.get_xticklabels()
            ax.set_xticklabels(labels, rotation=20)
        g.fig.subplots_adjust(bottom=0.15)
        g.fig.subplots_adjust(left=0.15)
        g.set_ylabels(label='$Corr(w_{LLR}, w_{NN, 1} - w_{NN, 0})$')
        g.set_xlabels(label='$\lambda_1$')

    file_name = '_'.join(['correlation_of_model_weights', 'overview', '.png'])
    output_path = join(config.output_dir_plots, file_name)
    save_figure(file_path=output_path, fig=g.fig, dpi=config.dpi)


def plot(config: Config, score_paths: List[str]) -> None:
    rnd_state = np.random.default_rng(config.seed)
    idx = rnd_state.integers(low=0, high=100)
    rnd_sample_idx = 107
    scores = load_pickle(file_path=score_paths[0])
    data = load_pickle(file_path=config.data_path)
    data_dict = get_randomized_heat_map_data(scores=scores, data=data, rnd_idx=idx)

    print(f'Create plots!')
    overview_correlation_plot(scores=scores, config=config)
    overall_accuracy_plot(scores=scores, config=config)

    print(f'Create rain cloud plots!')
    rain_clouds(scores=scores[A.sample_based], config=config, mode='sample_based',
                score_data_keys=[('roc_auc', 'auc'),
                                 # ('precision_based_scores', 'pr_auc'),
                                 ('precision_based_scores', 'max_precision'),
                                 ('precision_based_scores', 'avg_precision')])
    rain_clouds(scores=scores[A.global_based], config=config, mode='global',
                score_data_keys=[('roc_auc', 'auc'),
                                 # ('precision_based_scores', 'pr_auc'),
                                 ('precision_based_scores', 'max_precision'),
                                 ('precision_based_scores', 'avg_precision')])
    print(f'Create box plots!')
    box_plot(scores=scores[A.sample_based], config=config, mode='sample_based',
             snrs_of_interest=['0.00', '0.04', '0.08'],
             score_data_keys=[('roc_auc', 'auc'),
                              # ('precision_based_scores', 'pr_auc'),
                              ('precision_based_scores', 'max_precision'),
                              # ('precision_based_scores', 'avg_precision')
                              ])
    box_plot(scores=scores[A.global_based], config=config, mode='global',
             snrs_of_interest=['0.00', '0.04', '0.08'],
             score_data_keys=[('roc_auc', 'auc'),
                              # ('precision_based_scores', 'pr_auc'),
                              ('precision_based_scores', 'max_precision'),
                              # ('precision_based_scores', 'avg_precision')
                              ])

    print(f'Create heat maps!')
    pattern_type = int(extract_pattern_type(data_path=config.data_path))
    global_heat_maps(scores=scores, config=config, rnd_experiment_idx=idx,
                     pattern_type=pattern_type, snrs_of_interest=['0.00', '0.04', '0.08'])
    sample_based_heat_maps(scores=scores, config=config, data=data_dict,
                           rnd_sample_idx=rnd_sample_idx, pattern_type=pattern_type,
                           snrs_of_interest=['0.00', '0.04', '0.08'])


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
        c = Config.get(input_conf=load_json_file(file_path=path))
        input_paths = ['results/scores/data_evaluation_2021-04-30-12-32-54_pattern_type_0.pkl']
        plot(config=c, score_paths=input_paths)
    except KeyboardInterrupt as e:
        print(e)
