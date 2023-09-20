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
X = np.load('./data/background_input.npy')
b0 = np.load('./data/background_output_0.npy')
b1 = np.load('./data/background_output_1.npy')
b2 = np.load('./data/background_output_2.npy')
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
        '0': b0[a:b],
        '1': b1[a:b],
        '2': b2[a:b]
    }
# specify optimizable hyperparameters; leave the rest set to sklearn default values
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
# optimizer arguments
opt_args = dict(
    objective_model = 'decision_tree',
    objective_args = dict(params = DT_PARAMS, conditional_params_updater = max_depth_none_param_updater),
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
# parameters to be shared between components
comp_params_same = {
    'optimizer_args': opt_args,
    'preprocessing': [DefaultIOTransformer, {}]
}
comp_params = {str(i): comp_params_same for i in range(3)}
common_comp_params = {
    'max_iter': 150,
    'target': 0.85,
    'iterations_between_generations': 0
}
# perform sequential training of components
emu = Emulator(
    generator_args = gen_args,
    components_params = comp_params,
    component_data_names = 'same',
    common_components_params = common_comp_params
)
emu.train()
# print and save results
for i in range(3):
    tmp = emu.components[i].optimizer.study.trials_dataframe()\
    .drop(['datetime_start', 'datetime_complete', 'duration', 'state', 'number'], axis = 1)\
    .sort_values('value', ascending = False)
    print(f'background {i}:')
    print(tmp)
    tmp.to_csv(f'./trials_results/BG_DT_{i}.csv')