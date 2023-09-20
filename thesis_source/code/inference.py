import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from cosmolime.emulator import Emulator
from sklearn.tree import DecisionTreeRegressor
import optuna
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import optimize
# import the data:
data = np.loadtxt('./data.txt')
cov = np.loadtxt('./covariance.txt')
inv_cov = np.linalg.inv(cov)
z = data[:,0]
# define the model (notice we take the log and neglect overall distance scaling):
def LCDM_distance_shape(z0, z1, omegam):
    return integrate.romberg(lambda x: 1./np.sqrt(omegam*(1.+x)**3 +1.-omegam), z0, z1, show=False)
def LCDM_magdiff(z, omegam):
    _temp_d = np.array([LCDM_distance_shape(0.0, z[0], omegam)]+[LCDM_distance_shape(z[i], z[i+1], omegam) for i in range(len(z)-1) ] )
    _temp_d = np.cumsum(_temp_d)
    return 5.*np.log10((1.+z)*_temp_d)
# make a couple of theory predictions:
test_prediction_1 = LCDM_magdiff(z, omegam=0.30)
test_prediction_2 = LCDM_magdiff(z, omegam=1.0)
# a plot:
plt.errorbar(z, data[:,1]-data[0,1], yerr=np.sqrt(np.diag(cov)), fmt='.', label = 'experimental data')
plt.plot(z, test_prediction_1 -test_prediction_1[0], lw=2, zorder=999, label = r'predictions with $\Omega_m=0.3$')
plt.plot(z, test_prediction_2 -test_prediction_1[0], lw=2, zorder=999, label = r'predictions with $\Omega_m=1$')
plt.xscale('log');
plt.xlabel('$z$');
plt.ylabel('Relative distance modulus');
plt.title(r'Measured vs Predicted $\mu_{\Omega_m}(z)$')
plt.legend();
plt.savefig('./sn_data.pdf')
# some precomputations:
num_sn = len(z)
temp1 = np.dot(np.dot(np.ones(num_sn),inv_cov),np.ones(num_sn))
temp2 = np.dot(np.ones(num_sn),inv_cov)
# define likelihood:
def likelihood(omegam):
    residual = LCDM_magdiff(z, omegam) - data[:, 1]
    return -0.5*np.dot(np.dot(residual,inv_cov),residual) +0.5*np.dot(temp2,residual)**2/temp1
# define the prior:
def prior(omegam, vmin=0., vmax=1.):
    if omegam >= vmin and omegam <= vmax:
        return np.log(1./(vmax-vmin))
    else:
        return np.log(0.)
# define the posterior:
def posterior(omegam, vmin=0., vmax=1.):
    return likelihood(omegam) + prior(omegam, vmin=vmin, vmax=vmax)
# train emulator:
def generate_loglikelihood(size, seed,):
    rng = np.random.default_rng(seed = seed)
    omega_m_vec = rng.uniform(size = size).reshape((-1, 1))
    return omega_m_vec, np.array([likelihood(omega_m) for omega_m in omega_m_vec])
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

opt_args = dict(
    objective_model = 'decision_tree',
    objective_args = dict(params = DT_PARAMS, conditional_params_updater = max_depth_none_param_updater),
    create_study_params = {'sampler': optuna.samplers.TPESampler(seed = 1234)},
    optimize_study_params = {'n_trials': 50}
)
emu = Emulator(
    dict(generator_fun = generate_loglikelihood, num_initial_samples = 1500, num_generated_samples_per_iter = 500),
    {'loglikelihood':{'optimizer_args': opt_args, 'max_iter': 200, 'target': 0.99998, 'preprocessing': None}},
    'num',
    {'max_iter': 150, 'iterations_between_generations': 0},
)
emu.train()
# compare exact and emulated results
omega_m_vec = np.linspace(0, 1, 1000)#.reshape((-1, 1))
x = omega_m_vec
y = np.array([posterior(omegam) for omegam in omega_m_vec])

best_value = emu.best_models[0]['best_value']
best_params = emu.best_models[0]['best_params']

if best_params['max_depth'] == 0:
    best_params['max_depth'] = None

best = DecisionTreeRegressor(**best_params).fit(emu.generator.X, emu.generator.y)
pred = best.predict(x.reshape((-1, 1)))
# plot log likelihood
plt.plot(x, y, label = 'exact')
plt.plot(x, pred, label = 'decision tree')
plt.xlabel(r'$\Omega_m$')
plt.ylabel('log-likelihood')
plt.title(f'MSE = {mean_squared_error(pred, y):.6f}, $R^2$ = {r2_score(pred, y):.6%}')
plt.legend()
plt.savefig('./loglikelihood_dt.pdf')
# compare posteriors
p_true = np.exp(y - y.max()) # equivalent to np.exp(y)/np.exp(y.max())
p_emulated = np.exp(pred - pred.max())
plt.plot(x, p_true, label = 'exact') 
plt.plot(x, p_emulated, label = 'decision tree')
plt.xlabel(r'$\Omega_m$')
plt.ylabel(r'$P/P_{max}$')
plt.title(f'MSE = {mean_squared_error(p_true, p_emulated):.6f}, $R^2$ = {r2_score(p_true, p_emulated):.6%}')
plt.legend()
plt.savefig('./posterior_dt.pdf');
# compute log evidence
log_evidence_true = np.log(integrate.simps(np.exp(y), x))
log_evidence_emulated = np.log(integrate.simps(np.exp(pred), x))
print(f'{log_evidence_true = }, {log_evidence_emulated = }')
print('MSE:', mean_squared_error([log_evidence_true], [log_evidence_emulated]))
print('MAE:', mean_absolute_error([log_evidence_true], [log_evidence_emulated]))
print('rel. error:', mean_absolute_error([log_evidence_true], [log_evidence_emulated])/log_evidence_true)
# normalize posterior
P_true = np.exp(y - log_evidence_true)
P_emulated = np.exp(y - log_evidence_emulated)
# get summary statistics
mean_true = integrate.simps(x*P_true, x)
mean_emulated = integrate.simps(x*P_emulated, x)
print(f'{mean_true = }, {mean_emulated = }')
print('MSE:', mean_squared_error([mean_true], [mean_emulated]))
print('MAE:', mean_absolute_error([mean_true], [mean_emulated]))
print('rel. error:', mean_absolute_error([mean_true], [mean_emulated])/mean_true)
variance_true = integrate.simps((x - mean_true)**2 * P_true, x)
variance_emulated = integrate.simps((x - mean_emulated)**2 * P_emulated, x)
print(f'{variance_true = }, {variance_emulated = }')
print('MSE:', mean_squared_error([variance_true], [variance_emulated]))
print('MAE:', mean_absolute_error([variance_true], [variance_emulated]))
print('rel. error:', mean_absolute_error([variance_true], [variance_emulated])/variance_true)
std_true = np.sqrt(integrate.simps((x - mean_true)**2 * P_true, x))
std_emulated = np.sqrt(integrate.simps((x - mean_emulated)**2 * P_emulated, x))
print(f'{std_true = }, {std_emulated = }')
print('MSE:', mean_squared_error([std_true], [std_emulated]))
print('MAE:', mean_absolute_error([std_true], [std_emulated]))
print('rel. error:', mean_absolute_error([std_true], [std_emulated])/std_true)
MAP_true = x[np.argmax(P_true)]
MAP_emulated = x[np.argmax(P_emulated)]
print(f'{MAP_true = }, {MAP_emulated = }')
print('MSE:', mean_squared_error([MAP_true], [MAP_emulated]))
print('MAE:', mean_absolute_error([MAP_true], [MAP_emulated]))
# gaussian approximation
def gaussian(x, mu, sig):
    return 1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-((x - mu) / sig) ** 2 / 2)
plt.plot(x, gaussian(x, mean_true, std_true), label = 'gaussian approximation')
plt.plot(x, P_true, label = 'true posterior')
plt.xlabel(r'$\Omega_m$')
plt.title('Gaussian approximation of the posterior')
plt.legend();
plt.savefig('./gaussian_approx.pdf')
# credibility intervals
def credibility_intervals(p_vec, x_vec, prob):
    def temp(alpha): return 1 - integrate.simps(p_vec[p_vec >= alpha], x_vec[p_vec >= alpha])
    iso_probability = optimize.brentq(lambda x: temp(x) - (1.-0.95), p_vec.min(), p_vec.max())
    return x_vec[np.where(p_vec > iso_probability)][[0, -1]]
print(r'68% credibility interval (true and emulator):')
print(credibility_intervals(P_true, x, 0.68), credibility_intervals(P_emulated, x, 0.68))
print(r'68% credibility interval difference:')
print(credibility_intervals(P_true, x, 0.68) - credibility_intervals(P_emulated, x, 0.68))
print(r'95% credibility interval (true and emulator):')
print(credibility_intervals(P_true, x, 0.95), credibility_intervals(P_emulated, x, 0.95))
print(r'95% credibility interval difference:')
print(credibility_intervals(P_true, x, 0.95) - credibility_intervals(P_emulated, x, 0.95))
print(r'99% credibility interval difference:')
print(credibility_intervals(P_true, x, 0.99), credibility_intervals(P_emulated, x, 0.99))
print(r'99% credibility interval difference:')
print(credibility_intervals(P_true, x, 0.99) - credibility_intervals(P_emulated, x, 0.99))
# plot credibility intervals
def temp_true(alpha): return 1 - integrate.simps(P_true[P_true >= alpha], x[P_true >= alpha])
iso_probability_68 = optimize.brentq(lambda x: temp_true(x) - (1.-0.68), np.min(P_true), np.max(P_true))
iso_probability_95 = optimize.brentq(lambda x: temp_true(x) - (1.-0.95), np.min(P_true), np.max(P_true))
iso_probability_99 = optimize.brentq(lambda x: temp_true(x) - (1.-0.99), np.min(P_true), np.max(P_true))
plt.plot(x, P_true > iso_probability_68, label = r'68% cred. int.')
plt.plot(x, P_true > iso_probability_95, label = r'95% cred. int.')
plt.plot(x, P_true > iso_probability_99, label = r'99% cred. int.')
plt.plot(x, P_true/P_true.max())
plt.xlim([0.2,0.4])
plt.xlabel('$\Omega_m$');
plt.ylabel('$P/P_{max}$');
plt.vlines(x[np.argmax(P_true)], 0, 1, color = 'red', linestyle = 'dashed', label = 'MAP est.')
plt.title('Posterior credibility intervals')
plt.legend();
plt.savefig('./cred_intervals.pdf')