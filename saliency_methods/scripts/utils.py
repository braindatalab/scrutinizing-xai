import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils 
import matplotlib.pyplot as plt

def split_dataset(data, params, random_state = 42):
  """Split dataset into train and test data, including binary and categorical labels.""" 
  X_train, X_test, y_train_bin, y_test_bin = train_test_split(data.data, data.labels, test_size = params['test_size'], random_state = random_state)
  X_train = X_train.reshape(int((1 - params['test_size'])*params['sample_size']), params['input_dim'])
  X_test = X_test.reshape(int((params['test_size'])*params['sample_size']), params['input_dim'])

  y_train_bin = (y_train_bin + 1)/2
  y_test_bin = (y_test_bin + 1)/2

  y_train_cat = np_utils.to_categorical(y_train_bin, num_classes = 2)
  y_test_cat = np_utils.to_categorical(y_test_bin, num_classes = 2)
  return X_train, X_test, y_train_bin, y_test_bin, y_train_cat, y_test_cat
  
def plot_distribution(data_dict, methods, score_type = 'auc'):
  """Plots distribution of a specific metric for each method."""
  fig = plt.figure(figsize = (20, 5))
  for i, method in enumerate(methods): 
    plt.subplot(1, len(methods), i + 1)
    score = data_dict[method][score_type]
    plt.hist(score, bins = 20, density = False, alpha = 0.5)
    plt.title(f'{method}')

def create_binary_map(data, threshold):
  """Creates a binary map for a single observation using a given threshold"""
  binary_map = []
  for element in data.flatten():
    if element > threshold:
      binary_map.append(1)
    else:
      binary_map.append(0)
  return np.array(binary_map)

def get_classwise_data(data_dict):
    """Takes result of the analyzers and returns classwise data (0, 1)
    
    Input: 
    data_dict : result of run_interpretation_methods()
    
    Output: 
    data_classes : dictionary with methods as keys, containing data separated by class 
    """
    labels = data_dict['labels']
    methods = data_dict.keys()

    data_classes = {}

    for method in methods:
        method_class_0 = data_dict[method][np.where(labels == 0)[0]]
        method_class_1 = data_dict[method][np.where(labels == 1)[0]]

        data_classes[method] = {'class 0': method_class_0, 'class 1': method_class_1}

    return data_classes

def binary_map_opt_threshold(auc_dict, data_dict, methods):
    """Creates binary maps for all data based on optimal threshold.
    
    Input: 
    auc_dict : result of plot_roc_curves
    data_dict : result of run_interpretation_methods()
    methods : list of applied methods 
    
    Output: 
    dict_binary_maps : dictionary of binary maps based on optimal threshold 
    """

    dict_binary_maps = {}

    for method in methods:
        binary_maps = []
        data = data_dict[method]
        thresh = auc_dict[method]['opt_threshold']
        for i, obs in enumerate(data):
            binary_map = create_binary_map(np.abs(obs), thresh[i])
            binary_maps.append(binary_map)

        dict_binary_maps[method] = binary_maps

    return dict_binary_maps
