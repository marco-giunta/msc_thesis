# come sopra ma score personalizzati

import numpy as np
from cosmolime.emulator import Emulator
from cosmolime.preprocessing import DefaultIOTransformer
import optuna
# disable detailed print of optuna results
optuna.logging.disable_default_handler()
# set seeds to ensure reproducibility
NUM_INITIAL_SAMPLES = 1500
SAMPLER_SEED = 1234
TRAIN_VAL_SPLIT_SEED = 1234
# obtain data
X = np.load('./data/background_example_1_labels.npy')
derived_parameters = np.load('./data/background_example_1_data_0.npy')
luminosity_distances = np.load('./data/background_example_1_data_1.npy')
angular_distances = np.load('./data/background_example_1_data_2.npy')
# define generating function
def sample_background_data(size: int, seed, num_initial_samples = NUM_INITIAL_SAMPLES):
    if seed == 0:
        a, b = 0, num_initial_samples
    else:
        a = num_initial_samples + (seed - 1) * size
        b = a + size
    if b > len(X):
        raise ValueError('Run out of potential samples')
    return {
        'X': X[a:b],
        'derived_parameters': derived_parameters[a:b],
        'luminosity_distances': luminosity_distances[a:b],
        'angular_distances': angular_distances[a:b]
    }
# specify optimizable hyperparameters; leave the rest set to sklearn default values
# decision tree parameters (luminosity & angular distances)
DT_PARAMS = {
    'random_state': 1234,
    'criterion': ['squared_error', 'friedman_mse'],
    'splitter': ['best', 'random'],
    'max_depth': [0, 20],
    'min_samples_split': [2, 5]
}
def max_depth_none_param_updater(params):
    if params['max_depth'] == 0:
        params['max_depth'] = None
    return params
# decision tree optimizer arguments (luminosity & angular distances)
opt_args_dt = dict(
    objective_model = 'decision_tree',
    objective_args = dict(params = DT_PARAMS, conditional_params_updater = max_depth_none_param_updater),
    create_study_params = {'sampler': optuna.samplers.TPESampler(seed = SAMPLER_SEED)},
    optimize_study_params = {'n_trials': 20},
    train_val_split_seed = TRAIN_VAL_SPLIT_SEED
)
# random forest parameters (derived parameters)
RF_PARAMS = {
    'random_state': 1234,
    'n_estimators': [50, 100],
    'criterion': ['squared_error', 'friedman_mse'],
    'max_depth': [0, 20],
    'min_samples_split': [2, 5]
}
# random forest optimizer arguments (derived parameters)
opt_args_rf = dict(
    objective_model = 'random_forest',
    objective_args = dict(params = RF_PARAMS, conditional_params_updater = max_depth_none_param_updater),
    create_study_params = {'sampler': optuna.samplers.TPESampler(seed = SAMPLER_SEED)},
    optimize_study_params = {'n_trials': 20},
    train_val_split_seed = TRAIN_VAL_SPLIT_SEED
)
# generator arguments
gen_args = dict(
    generator_fun = sample_background_data,
    generator_fun_args = {'num_initial_samples': NUM_INITIAL_SAMPLES},
    num_initial_samples = NUM_INITIAL_SAMPLES
)
# preprocessing arguments (all components)
preproc_args = [DefaultIOTransformer, {}]
# component parameters
comp_params = {
    'derived_parameters': {'optimizer_args': opt_args_rf, 'preprocessing': preproc_args, 'target': 0.89},
    'luminosity_distances': {'optimizer_args': opt_args_dt, 'preprocessing': preproc_args, 'target': 0.89},
    'angular_distances': {'optimizer_args': opt_args_dt, 'preprocessing': preproc_args, 'target': 0.9}
}
# parameters to be shared between components
common_comp_params = {
    'max_iter': 150,
    'iterations_between_generations': 0
}
# perform sequential training of components
emu = Emulator(
    generator_args = gen_args,
    components_params = comp_params,
    common_components_params = common_comp_params
)
emu.train()
# print and save results
for component_name in emu.components.keys():
    tmp = emu.components[component_name].optimizer.study.trials_dataframe()\
    .drop(['datetime_start', 'datetime_complete', 'duration', 'state', 'number'], axis = 1)\
    .sort_values('value', ascending = False)
    print(f'{component_name}:')
    print(tmp)
    tmp.to_csv(f'd:/tesi magistrale/trials_df/BG_DT_RF_{component_name}.csv')