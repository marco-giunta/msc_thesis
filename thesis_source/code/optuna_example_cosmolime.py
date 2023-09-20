import optuna
from cosmolime.utils import from_param_dict_to_trial_suggest
def f(x):
    return (x - 2) ** 2
user_params = {'x': [-10.0, 10.0]} # x should be real & in [-10, 10]
def objective(trial):
    params = from_param_dict_to_trial_suggest(trial, user_params)
    return f(**params)
study = optuna.create_study()
study.optimize(objective, n_trials=100)
print(study.best_params) # e.g. {'x': 1.9966594189678515}