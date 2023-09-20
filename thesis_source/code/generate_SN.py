"""
Easy cosmology example: distance measurements
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
           'omegam': [0.1, 1.0],
           'omegav': [0.1, 1.0],
           'H0': [40, 100],
           }

# number of samples in the initial training set:
num_samples = 10000

# number of data-points per sample:
num_redshifts = 1000

# name of the data files:
data_name = 'SN_example'

# name of the data folder:
data_folder = './data/'
if not os.path.isdir(data_folder):
    os.makedirs(data_folder)

##################################################################
# Example generator:


def SN_generator(params, redshift=np.linspace(0.0, 10.0, 1000)):
    """
    Generate distance data on a redshift grid. Can input the grid as a kwarg.
    """
    # digest parameter vector:
    _H0 = params.get('H0')
    _omegam = params.get('omegam')
    _omegav = params.get('omegav')
    # process parameters:
    ombh2 = 0.1*_omegam*(_H0/100.)**2
    omch2 = 0.9*_omegam*(_H0/100.)**2
    omk = 1. - _omegam - _omegav
    H0 = _H0
    # set up CAMB:
    pars = camb.set_params(ombh2=ombh2,
                           omch2=omch2,
                           omk=omk,
                           H0=H0,)
    # do background cosmology:
    results = camb.get_background(pars, no_thermo=True)
    # get distances:
    distances = results.luminosity_distance(redshift)
    #
    return distances

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
    # allocate:
    _data = np.zeros((num_samples, num_redshifts))
    # do the actual samples:
    for i, samp in enumerate(tqdm.tqdm(_samples)):
        _input_params = {_k: _v for _k, _v in zip(_temp_keys, samp)}
        _data[i] = SN_generator(_input_params, redshift=np.linspace(0.01, 10.0, num_redshifts))
    # save out:
    np.save(data_folder+data_name+'_input', _samples)
    np.save(data_folder+data_name+'_output', _data)
