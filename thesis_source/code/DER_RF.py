import numpy as np
import optuna
from cosmolime.optimizer import Optimizer
# set seeds to ensure reproducibility
SAMPLER_SEED = 1234
TRAIN_VAL_SPLIT_SEED = 1234
# obtain data
X_der = np.load('./data/derived_example_input.npy')
y_der = np.load('./data/derived_example_output.npy')
# specify optimizable hyperparameters; leave the rest set to sklearn default values
RF_PARAMS = {
    'random_state': 1234,
    'n_estimators': [50, 100],
    'criterion': ['squared_error', 'friedman_mse'],
    'max_depth': [0, 20],
    'min_samples_split': [2, 5]
}
def max_depth_none_param_updater(params):
    if params['max_depth'] == 0:
        params['max_depth'] = None
    return params
# create optimizer instance with random forest and the above parameters
opt_der = Optimizer(
    objective_model = 'random_forest',
    objective_args = {
        'params': RF_PARAMS,
        'conditional_params_updater': max_depth_none_param_updater
    },
    X = X_der, 
    y = y_der,
    val_size = 0.2,
    train_val_split_seed = TRAIN_VAL_SPLIT_SEED,
    create_study_params = {'sampler': optuna.samplers.TPESampler(seed = SAMPLER_SEED)}
)
# perform optimization and save results
opt_der.optimize(n_trials = 25, n_jobs = 2)
print(f'{"-" * 50}\nBest params: {opt_der.study.best_params}, best score: {opt_der.study.best_value}')
opt_der.study.trials_dataframe()\
   .drop(['datetime_start', 'datetime_complete', 'duration', 'state', 'number'], axis = 1)\
   .sort_values('value', ascending = False)\
   .to_csv('./trials_results/DER_RF.csv')