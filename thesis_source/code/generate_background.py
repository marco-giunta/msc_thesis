"""
Less easy cosmology example: bad input parameters, several outputs
"""

##################################################################
# Initial imports:

import os
import numpy as np
import camb
import tqdm

##################################################################
# Initial definitions:

priors = {
          'omegabh2': [0.005, 1.0],
          'omegach2': [0.001, 0.99],
          'H0': [40, 100],
          'w': [-3.0, 1.0],
          'wa': [-3.0, 2.0],
          }

# number of samples in the initial training set:
num_samples = 10000

# number of data-points per sample:
num_redshifts = 1000

# name of the data files:
data_name = 'background_example'

# name of the data folder:
data_folder = './data/'
if not os.path.isdir(data_folder):
    os.makedirs(data_folder)

##################################################################
# Example generator:


def data_generator(params, redshift=np.linspace(0.0, 10.0, 1000)):
    """
    Generate distance data on a redshift grid. Can input the grid as a kwarg.
    """
    # digest parameter vector:
    _omegabh2 = params.get('omegabh2')
    _omegach2 = params.get('omegach2')
    _H0 = params.get('H0')
    _w = params.get('w')
    _wa = params.get('wa')
    # process parameters:
    ombh2 = _omegabh2
    omch2 = _omegach2
    H0 = _H0
    # print(params)
    # get the results:
    try:
        # set up CAMB:
        pars = camb.set_params(ombh2=ombh2,
                               omch2=omch2,
                               H0=H0,
                               w=_w,
                               wa=_wa,
                               dark_energy_model='DarkEnergyPPF'
                               )
        # do background cosmology:
        results = camb.get_background(pars, no_thermo=False)
        # get distances and derived parameters:
        luminosity_distance = results.luminosity_distance(redshift)
        angular_distance = results.angular_diameter_distance(redshift)
        derived_parameters = np.fromiter(results.get_derived_params().values(), dtype=float)
    except:
        luminosity_distance = np.nan * np.zeros(len(redshift))
        angular_distance = np.nan * np.zeros(len(redshift))
        derived_parameters = np.nan * np.zeros(13)
    #
    return [derived_parameters, luminosity_distance, angular_distance]

##################################################################
# Sample if called directly:


if __name__ == '__main__':

    # prepare:
    _temp_keys = list(priors.keys())
    _temp_priors = np.array(list(priors.values()))
    _d = len(_temp_priors)
    # create random samples (uniformly distributed):
    _samples = np.random.rand(num_samples * _d).reshape((num_samples, _d))
    _samples = _samples * (_temp_priors[:, 1] - _temp_priors[:, 0]) + _temp_priors[:, 0]
    # evaluate once to get output shape:
    _temp_results = data_generator({_k: _v for _k, _v in zip(_temp_keys, _samples[0])},
                                   redshift=np.linspace(0.0, 10.0, num_redshifts))
    _num_components = len(_temp_results)
    # allocate:
    _data = [np.zeros((num_samples, len(_comp))) for _comp in _temp_results]
    # do the actual samples:
    for i, samp in enumerate(tqdm.tqdm(_samples)):
        _input_params = {_k: _v for _k, _v in zip(_temp_keys, samp)}
        _temp_results = data_generator(_input_params, redshift=np.linspace(0.0, 10.0, num_redshifts))
        for _d, _r in zip(_data, _temp_results):
            _d[i] = _r
    # save out:
    np.save(data_folder+data_name+'_input', _samples)
    for i, _dat in enumerate(_data):
        np.save(data_folder+data_name+'_output_'+str(i), _dat)
