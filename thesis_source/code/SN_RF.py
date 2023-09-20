import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

SN_X, SN_y = np.load('./data/SN_example_input.npy'), np.load('./data/SN_example_output.npy')
SN_X_train, SN_X_test, SN_y_train, SN_y_test = train_test_split(SN_X, SN_y, test_size = 0.2, random_state = 1234)

rf = RandomForestRegressor(n_estimators = 100, random_state = 1234, n_jobs = 6).fit(SN_X_train, SN_y_train)
print(rf.score(SN_X_test, SN_y_test)) # Result: 0.9987787647185189