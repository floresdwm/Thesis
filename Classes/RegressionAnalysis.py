from math import sqrt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import Classes.Configurations as cfg
import os
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


def partial_leasts_square_regression(x, y, train_split_percentage, file_name):
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
        if cfg.sigma_detection:
            x_sigma, y_sigma = do_sigma_pls(x_df, y_df.iloc[:, i], train_split_percentage)
            if cfg.polarization_test:
                x_sigma_r, y_sigma_r = polarization_reducer_by_amplitude_groups(x_sigma, y_sigma)
                i_model, i_rmsec, i_r2c, i_rmsecv, i_r2cv = do_pls(x_sigma_r, y_sigma_r, train_split_percentage)
                models.append(i_model)
                rmsec.append(i_rmsec)
                r2cal.append(i_r2c)
                rmsecv.append(i_rmsecv)
                r2cv.append(i_r2cv)
                med_x_pred_polarized(x_sigma_r, y_sigma_r, i_rmsec, i_r2c, i_rmsecv, i_r2cv, i_model,
                                 file_name + '\Figures\sigma_' + y_df.columns[i], i + 200, y_df.columns[i])
                sigma_data_to_excel(file_name + '\SigmaReport_' + y_df.columns[i],
                                    pd.concat([x_sigma_r, y_sigma_r], axis=1, sort=False))
            else:
                i_model, i_rmsec, i_r2c, i_rmsecv, i_r2cv = do_pls(x_sigma, y_sigma, train_split_percentage)
                models.append(i_model)
                rmsec.append(i_rmsec)
                r2cal.append(i_r2c)
                rmsecv.append(i_rmsecv)
                r2cv.append(i_r2cv)
                med_x_pred_sigma(x_sigma, y_sigma, i_rmsec, i_r2c, i_rmsecv, i_r2cv, i_model, file_name + '\Figures\sigma_' + y_df.columns[i], i + 200, y_df.columns[i])
                sigma_data_to_excel(file_name + '\SigmaReport_' + y_df.columns[i], pd.concat([x_sigma, y_sigma], axis=1, sort=False))
        else:
            if cfg.polarization_test:
                x_df_r, y_df_r = polarization_reducer_by_amplitude_groups(x_df, y_df.iloc[:, i])
                i_model, i_rmsec, i_r2c, i_rmsecv, i_r2cv = do_pls(x_df_r, y_df_r, train_split_percentage)
                models.append(i_model)
                rmsec.append(i_rmsec)
                r2cal.append(i_r2c)
                rmsecv.append(i_rmsecv)
                r2cv.append(i_r2cv)
                med_x_pred_polarized(x_df_r, y_df_r, i_rmsec, i_r2c, i_rmsecv, i_r2cv, i_model,
                                     file_name + '\Figures\polarized_' + y_df.columns[i], i + 200, y_df.columns[i])
            else:
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
    df_y_resume = pd.DataFrame(y_df.describe().dropna())
    df_indexes = pd.DataFrame(pd.concat([df_y_resume, df_models_summary], axis=0))

    return df_indexes, df_models_summary, df_y_resume, models, x_train, y_train, x_test, y_test


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


def do_sigma_pls(data_x, data_y, train_split_percentage):
    latent_variables = []

    x_test, x_train, y_test, y_train = train_test_split(data_x, data_y, test_size=train_split_percentage, random_state=0)

    for i in range(20):
        pls = PLSRegression(n_components=(i + 1), scale=True)
        pls.fit(x_train, y_train)
        predicted_cv_y = pls.predict(x_test)
        mean_squared_error_cv = sqrt(mean_squared_error(y_test, predicted_cv_y))
        latent_variables.append(mean_squared_error_cv)

    best_factor = np.argmin(latent_variables)
    pls_sigma = PLSRegression(n_components=(best_factor + 1), scale=True)
    pls_sigma.fit(data_x, data_y)
    predicted_cv_y_sigma = pd.DataFrame(pls_sigma.predict(data_x))
    data_labels = pd.DataFrame(data_y.index)
    data_x = pd.DataFrame(data_x).reset_index(drop=True)
    data_y = pd.DataFrame(data_y).reset_index(drop=True)

    if cfg.sigma_percentage:
        percentual_error = pd.DataFrame(abs(data_y.iloc[:, 0] - predicted_cv_y_sigma.iloc[:, 0]))
        percentual_error = pd.DataFrame((percentual_error.iloc[:, 0] * 100) / data_y.iloc[:, 0])
        df_x = pd.DataFrame(pd.DataFrame(pd.concat([data_x, percentual_error], axis=1)))
        df_x = df_x.drop(df_x[df_x.iloc[:, -1] > cfg.sigma_confidence].index)
        df_x.drop(df_x.columns[len(df_x.columns) - 1], axis=1, inplace=True)
        df_y = pd.DataFrame(pd.DataFrame(pd.concat([data_y, data_labels, percentual_error], axis=1)))
        df_y = df_y.drop(df_y[df_y.iloc[:, -1] > cfg.sigma_confidence].index)

        df_x.set_index(df_y.iloc[:, 1], inplace=True)
        df_y.set_index(df_x.index, inplace=True)
        df_y.drop(df_y.columns[len(df_y.columns) - 1], axis=1, inplace=True)

        return df_x, df_y
    else:
        abs_error = pd.DataFrame(abs(data_y.iloc[:, 0] - predicted_cv_y_sigma.iloc[:, 0]))
        df_x = pd.DataFrame(pd.DataFrame(pd.concat([data_x, abs_error], axis=1)))
        df_x = df_x.drop(df_x[df_x.iloc[:, -1] > cfg.sigma_confidence].index)
        df_x.drop(df_x.columns[len(df_x.columns) - 1], axis=1, inplace=True)
        df_y = pd.DataFrame(pd.DataFrame(pd.concat([data_y, abs_error], axis=1)))
        df_y = df_y.drop(df_y[df_y.iloc[:, -1] > cfg.sigma_confidence].index)

        df_x.set_index(df_y.iloc[:, 1], inplace=True)
        df_y.set_index(df_x.index, inplace=True)
        df_y.drop(df_y.columns[len(df_y.columns) - 1], axis=1, inplace=True)
        return df_x, df_y


def run_pls(x, model):
    y_hat = model.predict(x)
    return y_hat


def remove_rows_with_zeros(x, y):
    df_x = pd.DataFrame(x)
    df_y = pd.DataFrame(y)
    params = df_y.shape[1]

    df = pd.concat([df_x, df_y], axis=1)
    bol = df != 0
    df = df[bol]
    df = df.dropna()

    df_x = pd.DataFrame(df.iloc[:, 0:-params])
    df_x = df_x.set_index(df.iloc[:, 0:-params].index)
    df_y = pd.DataFrame(df.iloc[:, -params:])
    df_y = df_y.set_index(df.iloc[:, -params:].index)

    return df_x, df_y


def med_x_pred_sigma(x_data, y_data, i_rmsec, i_r2c, i_rmsecv, i_r2cv, i_model, file_name, fig, paramname):
    x_test, x_train, y_test, y_train = train_test_split(x_data, y_data, test_size=cfg.train_split_percentage, random_state=0)
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    for i in range(y_train.shape[1]):
        figname = 'sigma ' + str(paramname)
        figname = plt.figure(figname, figsize=(8, 4), dpi=125)
        plt.xlabel('Y measured')
        plt.ylabel('Y predicted')
        yhats = run_pls(x_train, i_model)
        plt.scatter(y_train.iloc[:, i], yhats, label="Train data")
        plt.plot(y_train.iloc[:, i], y_train.iloc[:, i], color='red', linewidth=0.5)
        plt.legend(loc='best', shadow=False, scatterpoints=1)

    for i in range(y_test.shape[1]):
        figname = 'sigma ' + str(paramname)
        figname = plt.figure(figname, figsize=(8, 4), dpi=125)
        plt.xlabel('Y measured')
        plt.ylabel('Y predicted')
        yhats = run_pls(x_test, i_model)
        plt.scatter(y_test.iloc[:, i], yhats, label="Test data")
        plt.plot(y_test.iloc[:, i], y_test.iloc[:, i], color='red', linewidth=0.5)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        figname.text(0, 0, str('RMSEC: ' + str(i_rmsec) + ' ' +'\nR2C: ' + str(i_r2c) + ' '+'\nRMSECV: ' + str(i_rmsecv) + ' '+'\nR2CV: ' + str(i_r2cv) + ' ')+
                     str('\n N Train: ' + str(n_train) + str(' N Test: ' + str(n_test))), color='red', fontsize=7,
                     bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
        path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls', '') + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + str(figname) + str(i) + '_.png')


def med_x_pred_polarized(x_data, y_data, i_rmsec, i_r2c, i_rmsecv, i_r2cv, i_model, file_name, fig, paramname):
    x_test, x_train, y_test, y_train = train_test_split(x_data, y_data, test_size=cfg.train_split_percentage, random_state=0)
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    figname = 'Polarized test ' + str(paramname)
    figname = plt.figure(figname, figsize=(8, 4), dpi=125)
    plt.xlabel('Y measured')
    plt.ylabel('Y predicted')
    yhats = run_pls(x_train, i_model)
    plt.scatter(y_train, yhats, label="Train data")
    plt.plot(y_train, y_train, color='red', linewidth=0.5)
    plt.legend(loc='best', shadow=False, scatterpoints=1)

    figname = 'Polarized test ' + str(paramname)
    figname = plt.figure(figname, figsize=(8, 4), dpi=125)
    plt.xlabel('Y measured')
    plt.ylabel('Y predicted')
    yhats = run_pls(x_test, i_model)
    plt.scatter(y_test, yhats, label="Test data")
    plt.plot(y_test, y_test, color='red', linewidth=0.5)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    figname.text(0, 0, str('RMSEC: ' + str(i_rmsec) + ' ' +'\nR2C: ' + str(i_r2c) + ' '+'\nRMSECV: ' + str(i_rmsecv) + ' '+'\nR2CV: ' + str(i_r2cv) + ' ')+
                 str('\n N Train: ' + str(n_train) + str(' N Test: ' + str(n_test))), color='red', fontsize=7,
                 bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
    path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls', '') + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + str(figname) + '_polarized' + '_.png')


def med_x_pred_sigmab(x_data, y_data, summary_models, models, file_name, fig, paramname):
    x_test, x_train, y_test, y_train = train_test_split(x_data, y_data, test_size=cfg.train_split_percentage, random_state=0)
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    for i in range(y_train.shape[1]):
        figname = 'fig' + str(10+fig)
        figname = plt.figure(10+fig, figsize=(8, 4), dpi=125)
        plt.xlabel('Y measured')
        plt.ylabel('Y predicted')
        yhats = run_pls(x_train, models[i])
        plt.scatter(y_train.iloc[:, i], yhats, label="Train data")
        plt.plot(y_train.iloc[:, i], y_train.iloc[:, i], color='red', linewidth=0.5)
        plt.legend(loc='best', shadow=False, scatterpoints=1)

    for i in range(y_test.shape[1]):
        figname = 'fig' + str(10+fig)
        figname = plt.figure(10+fig, figsize=(8, 4), dpi=125)
        plt.xlabel('Y measured')
        plt.ylabel('Y predicted')
        yhats = run_pls(x_test, models[i])
        plt.scatter(y_test.iloc[:, i], yhats, label="Test data")
        plt.plot(y_test.iloc[:, i], y_test.iloc[:, i], color='red', linewidth=0.5)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        figname.text(0, 0, str(summary_models.iloc[0:5, 0].to_string()) +
                     str(' N Train: ' + str(n_train) + str(' N Test: ' + str(n_test))), color='red', fontsize=7,
                     bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
        path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls', '') + '/Figures' + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + str(figname) + str(i) + paramname + '_.png')


def sigma_data_to_excel(file_name, all_df):
    path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls', '')
    if not os.path.exists(path):
        os.makedirs(path)
    writer = pd.ExcelWriter(path + '\Sigma_report.xlsx')
    all_data = pd.DataFrame(all_df)
    all_data.to_excel(writer, 'Selected data')
    stats_all = pd.DataFrame(all_data.describe())
    stats_all.to_excel(writer, 'Stats selected data')
    writer.save()


def polarization_reducer_by_percentile(df_x, df_y):
    print('-- starting polarization test --')
    df = pd.DataFrame(pd.concat([df_x, df_y], axis=1))
    print('initial shape: ' + str(df.shape))
    median = df.iloc[:, -1].median()
    min = df.iloc[:, -1].min()
    max = df.iloc[:, -1].max()
    df_described = pd.DataFrame(df.iloc[:, -1].describe())
    percentil_25 = df_described.iloc[4, 0]
    percentil_50 = df_described.iloc[5, 0]
    percentil_75 = df_described.iloc[6, 0]
    bins = [min, percentil_25, percentil_50, percentil_75, max]
    print('bins: ' + str(bins[:]))
    groups = df.groupby(pd.cut(df.iloc[:, -1], bins, duplicates='drop'))
    df_balanced = pd.DataFrame(groups.apply(lambda x: x.sample(groups.size().min()).reset_index(drop=True)))
    print('balanced shape: ' + str(df_balanced.shape))
    print('Samples dropped: ' + str(df.shape[0] - df_balanced.shape[0]))
    df_x_balanced = df_balanced.iloc[:, :-1]
    df_y_balanced = df_balanced.iloc[:, -1]
    print('-- ending polarization test --')
    return df_x_balanced, df_y_balanced


def polarization_reducer_by_amplitude_groups(df_x, df_y):
    print('-- starting polarization test --')
    df = pd.DataFrame(pd.concat([df_x, df_y], axis=1))
    print('initial shape: ' + str(df.shape))
    min = df.iloc[:, -1].min()
    max = df.iloc[:, -1].max()
    amp = max - min
    amp_group = amp / cfg.polarization_n_groups

    bins = []
    bins.append(min)
    for i in range(cfg.polarization_n_groups):
        bins.append(bins[i] + amp_group)

    print('bins: ' + str(bins[:]))
    groups = df.groupby(pd.cut(df.iloc[:, -1], bins, duplicates='drop'))
    df_balanced = pd.DataFrame(groups.apply(lambda x: x.sample(groups.size().min()).reset_index(drop=True)))
    print('balanced shape: ' + str(df_balanced.shape))
    print('Samples dropped: ' + str(df.shape[0] - df_balanced.shape[0]))
    df_x_balanced = df_balanced.iloc[:, :-1]
    df_y_balanced = df_balanced.iloc[:, -1]
    print('-- ending polarization test --')
    return df_x_balanced, df_y_balanced