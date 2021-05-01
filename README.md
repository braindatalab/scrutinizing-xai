![Pattern and Distractor](images/pattern_and_distractor_matrices_pattern_type_0.png)

This repository contains all the experiments presented in the corresponding paper: *"Scrutinizing XAI using linear ground-truth data with suppressor variables"*.

We use `Pipfiles` to create Python environments. Since  we use [innvestigate](https://github.com/albermax/innvestigate) to create the saliency maps, and this framework uses particular dependencies, there is one extra Pipfile included in the `saliency_method` folder.

In three steps we can reproduce the results: (i) we generate the ground truth data, (ii) train the linear models and apply the XAI methods, (iii) run the evaluation steps and generate plots.

#### Generate data

Set the parameter `pattern_type=0` to use the signal pattern and suppressor combination analyzed in the paper (see image above). Use `pattern_type=3` to generate the data, used to produce the result in the supplementary material. 

```shell
python -m data.main --path data/config.json 
```



#### Run the experiments of model agnostic XAI methods

Update the `data_path` parameter of the `agnostic_methods/conf.json` with the path to the freshly generated pickle file containing the ground truth data.

```shell
python -m agnostic_methods.main_global_explanations --path agnostic_methods/config.json
```

Run experiment for sample based explanation, which will take a couple hours, depending on your machine. Here update the `data_path` of  the file `agnostic_methods/config_sample_based.json`.

```shell
python -m agnostic_methods.main_sample_based_explanations --path agnostic_methods/config_sample_based.json
```

#### Run experiment of saliency methods

Create a new Python environment, and run the experiments for heat-mapping methods by running through the notebook, change the `file_path` variable in the notebook.

```shell
compute_explanations_heatmapping.ipynb
```



#### Run evaluation and generate plots

Update the parameter `data_path` and `results_paths` of the `config.json`. Add the data path and the paths to the artifacts of the experiments. 

```shell
python run_evaluation_and_visualization.py --path config.json
```
