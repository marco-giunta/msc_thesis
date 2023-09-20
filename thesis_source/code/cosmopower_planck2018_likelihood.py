    [...]
    self.bval, self.X_data, self.X_sig=np.genfromtxt(self.like_file, unpack=True)
    [...]
    self.fisher=self.get_inverse_covmat() # Read and invert covariance matrix
    [...]    
def get_loglkl(self, parameters):
    # creating lookup table with parameters
    parameters_table = self.from_parameters_tensor_to_table(parameters)
    cal = parameters_table.lookup(tf.constant(['A_planck']))
    cosmo_params = tf.transpose(parameters_table.lookup(tf.constant(self.tt_emu_model.parameters)))

    # sourcing C_ells from CosmoPower
    Cltt= self.tt_emu_model.ten_to_predictions_tf(cosmo_params)
    Clte= self.te_emu_model.predictions_tf(cosmo_params)
    Clee= self.ee_emu_model.ten_to_predictions_tf(cosmo_params)

    # units of measure
    Cl = tf.scalar_mul(self.units_factor, tf.concat([Cltt, Clte, Clee], axis=1))

    # window function: batches
    self.window_ttteee = tf.concat([self.windowtt, self.windowte, self.windowee], axis=1)
    self.window_tile = tf.tile(self.window_ttteee, [parameters.shape[0], 1])

    # binning C_ells
    Cl_bin = tf.math.segment_sum( \
    tf.transpose( \
    tf.math.multiply(tf.gather(Cl, self.indices, axis=1), self.window_tile)), \
    self.indices_rep)

    # final theory prediction
    X_model = tf.transpose(tf.divide(Cl_bin, tf.square(cal)))

    # chi2 computation
    diff_vec = tf.subtract(self.X_data, X_model)
    chi2 = tf.matmul(self.fisher, tf.transpose(diff_vec))
    chi2 = tf.matmul(diff_vec, chi2)
    chi2 = tf.linalg.diag_part(chi2)
    loglkl = tf.scalar_mul(-0.5, chi2)
    loglkl = tf.reshape(loglkl, [parameters.shape[0], 1])

    return loglkl