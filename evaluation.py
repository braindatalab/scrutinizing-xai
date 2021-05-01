import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import (accuracy_score, roc_curve, auc, precision_recall_curve,
                             average_precision_score, precision_score)
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.utils.extmath import softmax
from tqdm import tqdm

from common import load_pickle, load_json_file, to_pickle, ScoresAttributes, extract_pattern_type
from data.patterns import PatternMatrix, PatternType
from configuration import Config

NUM_JOBS = 4
A = ScoresAttributes.get()


def _results_dict() -> Dict:
    return {
        # {'SNR_0': {'method_0': [roc_auc_0, roc_auc_1, ...], ...}, ...}
        # roc_auc_k is {'fpr': [], 'tpr': [], 'auc': []}
        'roc_auc': dict(),
        'precision_based_scores': dict(),

        # {'SNR_0': {'method_0': [explanation_0, explanation_1, ...], ...}, ...}
        # explanation_k is either R^{n_validation_data x n_features} or R^{n_features}
        A.explanations: dict(),

        A.method_names: list()
    }


def _generate_empty_evaluation_results_dict() -> Dict:
    return {
        A.global_based: _results_dict(),
        A.sample_based: _results_dict(),

        # {'SNR_0': {'train': [acc_data_set_0, acc_data_set_1, ...], 'val': [...]},
        #  'SNR_1': {'train': [...], ...}
        A.model_accuracies: dict(),

        # {'SNR_0': {'logistic_regression': [model_weight_0, model_weight_1, ...],
        #            'neural_net':[...]},
        #  'SNR_1': {[...]}, ...}
        A.model_weights: dict(),

        A.data_weights: list(),
    }


def _is_keras_model(model: Any) -> bool:
    return isinstance(model, list)


def _calculate_model_accuracies(data: List, results_per_snr: List) -> Dict:
    def predict_with_keras_model(model_weights: List, x: np.ndarray) -> np.ndarray:
        try:
            pred = softmax(np.dot(x, model_weights[0]) + model_weights[1])
        except IndexError as e:
            pred = softmax(np.dot(x, model_weights[0]))
        return np.argmax(pred, axis=1)

    accuracies = {'train': list(), 'val': list()}
    for k in range(len(results_per_snr)):
        model = results_per_snr[k]['model']
        if _is_keras_model(model=model):
            pred_train = predict_with_keras_model(model_weights=model, x=data[k]['train']['x'])
            pred_val = predict_with_keras_model(model_weights=model, x=data[k]['val']['x'])
        else:
            pred_train = model.predict(X=data[k]['train']['x'])
            pred_val = model.predict(X=data[k]['val']['x'])
        accuracies['val'] += [accuracy_score(y_true=data[k]['val']['y'], y_pred=pred_val)]
        accuracies['train'] += [accuracy_score(y_true=data[k]['train']['y'], y_pred=pred_train)]
    return accuracies


def precision_curves(y_true, probas_pred, *, pos_label=None,
                     sample_weight=None):
    """
    Minor adaption of corresponding scikit-learn function
    """
    fps, tps, thresholds = _binary_clf_curve(y_true, probas_pred,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight)

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    specificity = 1 - fps / fps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], np.r_[specificity[sl], 1]


def _assemble_explanations(method_names: List, results_per_snr: List) -> Dict:
    explanations_dict = dict()
    for method in method_names:
        explanations_list = list()
        for k in range(len(results_per_snr)):
            explanations_list += [results_per_snr[k]['explanations'][method]]
        explanations_dict[method] = explanations_list
    return explanations_dict


def _generate_true_feature_importance(pattern_type: int = 0) -> np.ndarray:
    pattern = PatternMatrix(pattern_type=pattern_type)
    binary_mask_signal = np.array(1 * (pattern.matrix[:, pattern.dim_of_signal] > 0))
    binary_mask_signal[(pattern.matrix[:, pattern.dim_of_signal] < 0)] = -1
    return binary_mask_signal


def _compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray,
                     use_abs: bool = True) -> float:
    if use_abs:
        y_true, y_score = np.abs(y_true), np.abs(y_score)
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score)
    auc_value = auc(x=fpr, y=tpr)
    return auc_value


def _assemble_roc_auc_values(y_true: np.ndarray, y_score: List,
                             idx: int, use_absolute_values: bool = True) -> Dict:
    if 1 < len(y_score[idx].shape):
        fpr_list, tpr_list, auc_value_list = list(), list(), list()
        for k in range(y_score[idx].shape[0]):
            auc_value = _compute_roc_auc(
                y_true=y_true, y_score=y_score[idx][k, :], use_abs=use_absolute_values)
            auc_value_list += [auc_value]
        output = {'auc': auc_value_list}
    else:
        auc_value = _compute_roc_auc(
            y_true=y_true, y_score=y_score[idx], use_abs=use_absolute_values)
        output = {'auc': [auc_value]}
    return output


def _precision_specificity_score(precision: np.ndarray, specificity: np.ndarray,
                                 threshold: float = 0.9) -> float:
    score = 0.0
    if any(threshold < specificity[:-1]):
        score = precision[:-1][threshold < specificity[:-1]][0]
    return score


def _compute_precision_based_scores(y_true: np.ndarray, y_score: np.ndarray,
                                    use_abs: bool = True) -> Tuple:
    if use_abs is True:
        y_true, y_score = np.abs(y_true), np.abs(y_score)
    precision, recall, specificity = precision_curves(y_true=y_true, probas_pred=y_score)
    pr_auc = auc(x=recall, y=precision)

    avg_prec_score = average_precision_score(y_true=y_true, y_score=y_score)
    avg_npv_specificity = average_precision_score(y_true=1 - y_true, y_score=1 - y_score)
    prec_spec_score = _precision_specificity_score(precision=precision, specificity=specificity)
    return pr_auc, avg_prec_score, avg_npv_specificity, prec_spec_score


def _assemble_precision_based_scores(y_true: np.ndarray, y_score: List,
                                     idx: int, use_absolute_values: bool = True) -> Dict:
    if 1 < len(y_score[idx].shape):
        pr_auc_list, avg_list, avg_npv_specificity_list, max_prec_list = list(), list(), list(), list()
        for k in range(y_score[idx].shape[0]):
            auc, avg_prec_score, avg_npv_specificity, max_prec = _compute_precision_based_scores(
                y_true=y_true, y_score=y_score[idx][k, :], use_abs=use_absolute_values)
            avg_list += [avg_prec_score]
            pr_auc_list += [auc]
            max_prec_list += [max_prec]
            avg_npv_specificity_list += [avg_npv_specificity]
        output = {'pr_auc': pr_auc_list, 'avg_precision': avg_list,
                  'avg_npv_specificity': avg_npv_specificity_list, 'max_precision': max_prec_list}
    else:
        auc, avg_prec_score, avg_npv_specificity, max_prec = _compute_precision_based_scores(
            y_true=y_true, y_score=y_score[idx], use_abs=use_absolute_values)
        output = {'pr_auc': [auc], 'avg_npv_specificity': [avg_npv_specificity],
                  'avg_precision': [avg_prec_score], 'max_precision': [max_prec]}
    return output


def _roc_analysis(explanations_by_methods: Dict, pattern_type: int) -> Dict:
    roc_auc_values = dict()
    true_importance = _generate_true_feature_importance(pattern_type=pattern_type)
    for method, explanations_list in explanations_by_methods.items():
        roc_auc_values[method] = Parallel(n_jobs=NUM_JOBS)(
            delayed(_assemble_roc_auc_values)(true_importance, explanations_list, idx)
            for idx in range(len(explanations_list)))
    return roc_auc_values


def _precision_based_analysis(explanations_by_methods: Dict, pattern_type: int) -> Dict:
    pr_scores = dict()
    true_importance = _generate_true_feature_importance(pattern_type=pattern_type)
    for method, explanations_list in explanations_by_methods.items():
        pr_scores[method] = Parallel(n_jobs=NUM_JOBS)(
            delayed(_assemble_precision_based_scores)(true_importance, explanations_list, idx)
            for idx in range(len(explanations_list)))
    return pr_scores


def _get_model_weights(results_per_snr: List) -> List:
    model_weights = list()
    for k in range(len(results_per_snr)):
        model = results_per_snr[k]['model']
        if _is_keras_model(model=model):
            model_weights += [model[0]]
        else:
            model_weights += [model.coef_]
    return model_weights


def _is_saliency(method_names: List) -> bool:
    return True if 'deep_taylor' in method_names else False


def _is_sample_based_agnostic(method_names: List) -> bool:
    return True if 'lime' in method_names else False


def _get_data_weights(result: Dict) -> List:
    return list(result['results'].keys())


def _extract_explanations(results: List[Dict], aggregate: bool, use_abs: bool = True) -> Dict:
    output = dict()
    weights = _get_data_weights(result=results[0])
    method_names = list()
    for k, r in enumerate(results):
        method_names += [(m, k) for m in r['method_names']]

    for w in weights:
        results_per_method = dict()
        for m, j in method_names:
            results_per_experiment = list()
            for r in results[j]['results'][w]:
                explanation = np.abs(r['explanations'][m]) if use_abs else r['explanations'][m]
                if aggregate and 1 < len(explanation.shape):
                    results_per_experiment += [np.mean(explanation, axis=0)]
                else:
                    results_per_experiment += [explanation]
            results_per_method[m] = results_per_experiment
        output[w] = results_per_method
    return output


def _is_sample_based_method(names: List[str]) -> bool:
    return _is_saliency(method_names=names) or _is_sample_based_agnostic(method_names=names)


def _get_method_names(results: List, sample_based: bool) -> List[str]:
    output = list()
    for result in results:
        method_names = result['method_names']
        if sample_based and _is_sample_based_method(names=method_names):
            output += method_names
        elif not sample_based:
            output += method_names
    return output


def _assemble_explanations2(results: List, scores: Dict) -> Tuple[Dict, Dict]:
    sample_based_results = list()
    for result in results:
        if _is_sample_based_method(names=result['method_names']):
            sample_based_results += [result]

    global_explanations = _extract_explanations(results=results, aggregate=True)
    sample_based_explanations = _extract_explanations(results=sample_based_results, aggregate=False)
    for key in global_explanations.keys():
        global_explanations[key]['llr'] = get_weights_of_logistic_regression(key=key, scores=scores)
    return global_explanations, sample_based_explanations


def _load_results(result_paths: List[str]) -> List:
    output = list()
    for p in result_paths:
        output += [load_pickle(file_path=p)]
    return output


def _assemble_model_accuracies(results: List, data: Dict) -> Dict:
    output = dict()
    for w, data_list in data.items():
        logistic_regression_counter = 0
        results_per_model_type = dict()
        for result in results:
            if _is_saliency(method_names=result['method_names']):
                results_per_model_type[A.neural_net] = _calculate_model_accuracies(
                    data=data_list, results_per_snr=result['results'][w])
            # Since we have two of the same logistic regression models but just
            # one neural net model.
            elif 0 == logistic_regression_counter:
                logistic_regression_counter += 1
                results_per_model_type[A.logistic_regression] = _calculate_model_accuracies(
                    data=data_list, results_per_snr=result['results'][w])
            else:
                continue
        output[w] = results_per_model_type
    return output


def _assemble_model_weights(results: List, weights: List) -> Dict:
    output = dict()
    for w in weights:
        logistic_regression_counter = 0
        model_weights_per_model_type = dict()
        for result in results:
            if _is_saliency(method_names=result['method_names']):
                model_weights_per_model_type[A.neural_net] = _get_model_weights(
                    results_per_snr=result['results'][w])
            # Since we have two of the same logistic regression models but just
            # one neural net model.
            elif 0 == logistic_regression_counter:
                logistic_regression_counter += 1
                model_weights_per_model_type[A.logistic_regression] = _get_model_weights(
                    results_per_snr=result['results'][w])
            else:
                continue
        output[w] = model_weights_per_model_type
    return output


def _assemble_results_roc_analysis(explanations: Dict, weights: List, pattern_type: int) -> Dict:
    output = dict()
    for w in tqdm(weights):
        output[w] = _roc_analysis(explanations_by_methods=explanations[w],
                                  pattern_type=pattern_type)
    return output


def _assemble_results_precision_analysis(explanations: Dict,
                                         weights: List, pattern_type: int) -> Dict:
    output = dict()
    for w in tqdm(weights):
        output[w] = _precision_based_analysis(explanations_by_methods=explanations[w],
                                              pattern_type=pattern_type)
    return output


def get_weights_of_logistic_regression(key: str, scores: Dict) -> List:
    return [c.flatten() for c in scores[A.model_weights][key]['Logistic Regression']]


def evaluate(config: Config) -> List[str]:
    data = load_pickle(file_path=config.data_path)
    results = _load_results(result_paths=config.result_paths)

    print('Assemble explanations, model weights, ...!')
    scores = _generate_empty_evaluation_results_dict()

    scores[A.global_based][A.method_names] = _get_method_names(results=results, sample_based=False)
    scores[A.sample_based][A.method_names] = _get_method_names(results=results, sample_based=True)
    scores[A.data_weights] = _get_data_weights(result=load_pickle(file_path=config.result_paths[0]))
    scores[A.model_accuracies] = _assemble_model_accuracies(results=results, data=data)
    scores[A.model_weights] = _assemble_model_weights(results=results, weights=scores[A.data_weights])
    g, s = _assemble_explanations2(results=results, scores=scores)
    scores[A.global_based][A.explanations] = g
    scores[A.sample_based][A.explanations] = s

    print('Calculate scores!')
    pattern_type = int(extract_pattern_type(data_path=config.data_path))
    scores[A.global_based]['roc_auc'] = _assemble_results_roc_analysis(
        explanations=scores[A.global_based][A.explanations],
        weights=scores[A.data_weights], pattern_type=pattern_type)
    scores[A.global_based]['precision_based_scores'] = _assemble_results_precision_analysis(
        explanations=scores[A.global_based][A.explanations],
        weights=scores[A.data_weights], pattern_type=pattern_type)
    scores[A.sample_based]['roc_auc'] = _assemble_results_roc_analysis(
        explanations=scores[A.sample_based][A.explanations],
        weights=scores[A.data_weights], pattern_type=pattern_type)
    scores[A.sample_based]['precision_based_scores'] = _assemble_results_precision_analysis(
        explanations=scores[A.sample_based][A.explanations],
        weights=scores[A.data_weights], pattern_type=pattern_type)

    print('Save results!')
    output_paths = list()
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    pattern_type = f'pattern_type_{extract_pattern_type(data_path=config.data_path)}'
    suffix = '_'.join(['evaluation', date, pattern_type])
    output_paths += [to_pickle(output_dir=config.output_dir_scores, data=scores, suffix=suffix)]
    return output_paths


def get_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='path', required=True,
                        help='Input file path of json file containing'
                             'input parameter for the experiment!')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_command_line_arguments()
    try:
        evaluate(config=Config.get(input_conf=load_json_file(file_path=args.path)))
    except KeyboardInterrupt as e:
        print(e)
