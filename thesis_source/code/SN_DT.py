import numpy as np
import optuna
from cosmolime.optimizer import Optimizer
# set seeds to ensure reproducibility
DT_SEED = 1234
SAMPLER_SEED = 1234
TRAIN_VAL_SPLIT_SEED = 1234
# obtain data
SN_X, SN_y = np.load('./data/SN_example_input.npy'), np.load('./data/SN_example_output.npy')
# specify optimizable hyperparameters; leave the rest set to sklearn default values
DT_PARAMS = {
    'random_state': DT_SEED,
    'criterion': ['squared_error', 'friedman_mse'],
    'splitter': ['best', 'random'],
    'max_depth': [0, 20],
    'min_samples_split': [2, 5]
}
def max_depth_none_param_updater(params):
    if params['max_depth'] == 0:
        params['max_depth'] = None
    return params
# create optimizer instance with decision trees and the above parameters
opt_SN = Optimizer(
    objective_model = 'decision_tree',
    objective_args = {
        'params': DT_PARAMS,
        'conditional_params_updater': max_depth_none_param_updater
    },
    X = SN_X, 
    y = SN_y,
    val_size = 0.2,
    train_val_split_seed = TRAIN_VAL_SPLIT_SEED,
    create_study_params = {'sampler': optuna.samplers.TPESampler(seed = SAMPLER_SEED)}
)
# perform optimization and save results
opt_SN.optimize(n_trials = 30, n_jobs = 6)
print(f'{"-" * 50}\nBest params: {opt_SN.study.best_params}, best score: {opt_SN.study.best_value}')
opt_SN.study.trials_dataframe()\
   .drop(['datetime_start', 'datetime_complete', 'duration', 'state', 'number'], axis = 1)\
   .sort_values('value', ascending = False)\
   .to_csv('./trials_results/SN_DT.csv')