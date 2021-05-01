import innvestigate
import innvestigate.utils
import numpy as np
from sklearn.preprocessing import minmax_scale


def run_interpretation_methods(model, methods, data, X_train_blob=None, normalize=False, **kwargs):
    """This function applies all interpretation methods given in methods (as implemented in innvestigate) to the
    trained model.

    Input:
    Model : trained model implemented in keras
    Methods : list of interpretation methods (implemented in innvestigate) to apply to the model
    data : test data, not used for training the model
    X_train_blob : training data, only use for pattern.net and pattern.attribution
    normalize : whether to normalize the heatmaps to a [0, 1] or [-1, 1] (in case of gradient, pattern.net) range

    Output:
    dict_results : dictionary with methods as keys, containing heatmaps for each sample for each method
    """
    model = innvestigate.utils.model_wo_softmax(model)
    data = data.reshape(len(data), 64)

    dict_results = {}

    for method in methods:
        analyzer = innvestigate.create_analyzer(method[0], model, **method[1])
        if method[0] == 'pattern.net' or method[0] == 'pattern.attribution':
            analyzer.fit(X_train_blob)
            heatmaps = analyzer.analyze(data)
        else:
            heatmaps = analyzer.analyze(data)

        if normalize is True:
            if method[0] == 'pattern.net' or method[0] == 'gradient':
                heatmaps = np.array(
                    [minmax_scale(heatmap.flatten(), feature_range=(-1, 1)) for heatmap in
                     heatmaps])
            else:
                heatmaps = np.array(
                    [minmax_scale(heatmap.flatten(), feature_range=(0, 1)) for heatmap in heatmaps])

        dict_results[method[0]] = heatmaps

    return dict_results
