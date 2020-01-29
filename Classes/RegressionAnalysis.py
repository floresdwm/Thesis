from math import sqrt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def partial_leasts_square_regression(x, y, train_split_percentage):
    # Define X and Y matrix after cleaned by PCA and Mahalanobis distance
    x_df = pd.DataFrame(x)
    y_df = pd.DataFrame(y)

    # split data to train and test
    x_test, x_train, y_test, y_train = train_test_split(x_df, y_df, test_size=train_split_percentage, random_state=0)

    # Train one PLS model for each Y parameter
    parameters = len(y_df.columns)
    models = []
    rmsec = []
    r2cal = []
    rmsecv = []
    r2cv = []
    for i in range(parameters):
        i_model, i_rmsec, i_r2c, i_rmsecv, i_r2cv = do_pls(x_df, y_df.iloc[:, i], train_split_percentage)
        models.append(i_model)
        rmsec.append(i_rmsec)
        r2cal.append(i_r2c)
        rmsecv.append(i_rmsecv)
        r2cv.append(i_r2cv)

    df_models_summary = pd.DataFrame(
        pd.concat([pd.DataFrame(list(y_df.columns)), pd.DataFrame(rmsec), pd.DataFrame(r2cal), pd.DataFrame(rmsecv), pd.DataFrame(r2cv)], axis=1))
    s = pd.Series(['Parameter', 'RMSEC', 'R2CAL', 'RMSECV', 'R2CV'])
    df_models_summary = df_models_summary.transpose().set_index(s)

    return df_models_summary, models, x_train, y_train, x_test, y_test


def do_pls(data_x, data_y, train_split_percentage):
    latent_variables = []

    x_test, x_train, y_test, y_train = train_test_split(data_x, data_y, test_size=train_split_percentage, random_state=0)

    for i in range(20):
        pls = PLSRegression(n_components=(i + 1), scale=True)
        pls.fit(x_train, y_train)
        predicted_cv_y = pls.predict(x_test)
        mean_squared_error_cv = sqrt(mean_squared_error(y_test, predicted_cv_y))
        latent_variables.append(mean_squared_error_cv)

    best_factor = np.argmin(latent_variables)
    pls2 = PLSRegression(n_components=(best_factor + 1), scale=True)
    pls2.fit(x_train, y_train)
    predicted_cal = pls2.predict(x_train)
    rmsec = sqrt(mean_squared_error(y_train, predicted_cal))
    r2c = pls2.score(x_train, y_train)

    predicted_cv_y = pls2.predict(x_test)
    rmsecv = sqrt(mean_squared_error(y_test, predicted_cv_y))
    r2v = pls2.score(x_test, y_test)

    plsfinal = PLSRegression(n_components=(best_factor + 1), scale=True)
    plsfinal.fit(data_x, data_y)

    return plsfinal, rmsec, r2c, rmsecv, r2v


def run_pls(x, model):
    y_hat = model.predict(x)
    return y_hat
