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
    x_test, x_train, y_test, y_train = train_test_split(x_df, y_df, test_size=train_split_percentage, random_state=1)

    # Train one PLS model for each Y parameter
    parameters = len(y_df.columns)
    models = []
    rmsec = []
    r2cal = []
    rmsecv = []
    r2cv = []
    for i in range(parameters):
        i_model, i_rmse, i_r2 = do_pls(x_train, y_train.iloc[:, i], train_split_percentage)
        models.append(i_model)
        rmsec.append(i_rmse)
        r2cal.append(i_r2)
        predicted_cv_y = i_model.predict(x_test)
        rmsecv.append(sqrt(mean_squared_error(y_test.iloc[:, i], predicted_cv_y)))
        r2cv.append(i_model.score(x_test, y_test.iloc[:, i]))

    df_models_summary = pd.DataFrame(
        pd.concat([pd.DataFrame(list(y_df.columns)), pd.DataFrame(rmsec), pd.DataFrame(r2cal), pd.DataFrame(rmsecv), pd.DataFrame(r2cv)], axis=1))
    s = pd.Series(['Parameter', 'RMSEC', 'R2CAL', 'RMSECV', 'R2CV'])
    df_models_summary = df_models_summary.transpose().set_index(s)

    return df_models_summary, models, x_train, y_train, x_test, y_test


def do_pls(data_x, data_y, train_split_percentage):
    latent_variables = []
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=train_split_percentage, random_state=0)
    for i in range(20):
        pls = PLSRegression(n_components=(i + 1), scale=True, max_iter=500, tol=1e-08)
        pls.fit(x_train, y_train)
        predicted_cv_y = pls.predict(x_test)
        mean_squared_error_cv = sqrt(mean_squared_error(y_test, predicted_cv_y))
        latent_variables.append(mean_squared_error_cv)

    best_factor = np.argmin(latent_variables)
    pls = PLSRegression(n_components=(best_factor + 1), scale=True, max_iter=500, tol=1e-08)
    pls.fit(x_train, y_train)
    predicted_cv_y = pls.predict(x_test)
    rmsev = sqrt(mean_squared_error(y_test, predicted_cv_y))
    r2v = pls.score(x_test, y_test)
    pls = PLSRegression(n_components=best_factor + 1, scale=True, max_iter=500, tol=1e-08)
    pls.fit(data_x, data_y)
    return pls, rmsev, r2v


def run_pls(x, model):
    y_hat = model.predict(x)
    return y_hat
