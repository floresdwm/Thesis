import os
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import Classes.RegressionAnalysis as pls
import Classes.Configurations as cfg
from sklearn.model_selection import train_test_split


def x_data(df, file_name, fig):
    plt.figure(fig, figsize=(8, 4), dpi=125)
    plt.subplot(121)
    plt.title('All X Data')
    plt.plot(pd.DataFrame(df).transpose())
    df_std_positive = pd.DataFrame(df).transpose().mean(axis=1) + pd.DataFrame(df).transpose().std(axis=1)
    df_std_negative = pd.DataFrame(df).transpose().mean(axis=1) - pd.DataFrame(df).transpose().std(axis=1)
    plt.subplot(122)
    plt.title('X Data average +/- STD')
    plt.plot(df_std_positive, color='indianred', linestyle='dashed')
    plt.plot(pd.DataFrame(df).transpose().mean(axis=1), color='black')
    plt.plot(df_std_negative, color='indianred', linestyle='dashed')
    path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls',
                                                                                                             '') + '/Figures' + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + 'x_plot_.png')


def x_data_epo(df, file_name, fig, plot_name):
    plt.figure(fig, figsize=(8, 4), dpi=125)
    plt.subplot(121)
    plt.title(plot_name)
    plt.plot(pd.DataFrame(df).transpose())
    df_std_positive = pd.DataFrame(df).transpose().mean(axis=1) + pd.DataFrame(df).transpose().std(axis=1)
    df_std_negative = pd.DataFrame(df).transpose().mean(axis=1) - pd.DataFrame(df).transpose().std(axis=1)
    plt.subplot(122)
    plt.title('X Data average +/- STD')
    plt.plot(df_std_positive, color='indianred', linestyle='dashed')
    plt.plot(pd.DataFrame(df).transpose().mean(axis=1), color='black')
    plt.plot(df_std_negative, color='indianred', linestyle='dashed')
    path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls',
                                                                                                             '') + '/Figures' + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + 'x_plot_.png')

def x_data_outliers(df_clean, df_out, file_name):
    plt.figure(20, figsize=(8, 4), dpi=125)
    plt.subplot(221)
    plt.title('All X Data OUTLIERS')
    plt.plot(pd.DataFrame(df_out).transpose())
    df_std_positive = pd.DataFrame(df_out).transpose().mean(axis=1) + pd.DataFrame(df_out).transpose().std(axis=1)
    df_std_negative = pd.DataFrame(df_out).transpose().mean(axis=1) - pd.DataFrame(df_out).transpose().std(axis=1)
    plt.subplot(222)
    plt.title('X Data OUTLIERS average +/- STD')
    plt.plot(df_std_positive, color='indianred', linestyle='dashed')
    plt.plot(pd.DataFrame(df_out).transpose().mean(axis=1), color='black')
    plt.plot(df_std_negative, color='indianred', linestyle='dashed')
    plt.subplot(223)
    plt.title('All X Data selected')
    plt.plot(pd.DataFrame(df_clean).transpose())
    df_std_positive3 = pd.DataFrame(df_clean).transpose().mean(axis=1) + pd.DataFrame(df_clean).transpose().std(axis=1)
    df_std_negative3 = pd.DataFrame(df_clean).transpose().mean(axis=1) - pd.DataFrame(df_clean).transpose().std(axis=1)
    plt.subplot(224)
    plt.title('X Data selected average +/- STD')
    plt.plot(df_std_positive3, color='indianred', linestyle='dashed')
    plt.plot(pd.DataFrame(df_clean).transpose().mean(axis=1), color='black')
    plt.plot(df_std_negative3, color='indianred', linestyle='dashed')
    path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls',
                                                                                                             '') + '/Figures' + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + 'x_plot_outliers.png')


def correlation_matrix(df, file_name):
    if df.shape[1] > 1:
        correlations = df.corr()
        figname = sns.clustermap(data=correlations, annot=True, cmap='Greens')
        plt.title('Pearson (r)')
        path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls',
                                                                                                                 '') + '/Figures' + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + 'y_correlation_map_.png')


def kde(df):
    fig3 = plt.figure()
    sns.pairplot(df)


def pca_x_data(df, labels, file_name):
    plt.figure()
    df = pd.DataFrame(df)
    p4 = sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], data=df, hue=labels.iloc[:, 0])
    p4.axhline(y=0, color='k', linewidth=1)
    p4.axvline(x=0, color='k', linewidth=1)
    p4.set(title='PCA data X')
    p4 = sns.kdeplot(df.iloc[:, 0], df.iloc[:, 1], linewidth=0.5)
    path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls',
                                                                                                             '') + '/Figures' + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + 'x_plot_.png')


def pca_y_data(df, labels):
    plt.figure()
    p4 = sns.scatterplot(x=df.iloc[:, 0], y=df.iloc[:, 1], data=df, hue=labels.iloc[:, 0])
    p4.axhline(y=0, color='k', linewidth=1)
    p4.axvline(x=0, color='k', linewidth=1)
    p4.set(title='PCA data Y')
    p4 = sns.kdeplot(df.iloc[:, 0], df.iloc[:, 1], linewidth=0.5)


def pca_xy_and_outliers(df_cleaned_x, df_cleaned_xy, df_outliers_x, df_outliers_xy, labels, n_out_x, n_out_y, file_name):
    plt.figure(5, figsize=(10, 5), dpi=125)
    plt.subplot(121)
    f1 = sns.scatterplot(x='PC1', y='PC2', data=df_cleaned_x, label="Selected data")
    f1.set(title='PCA OF (X)' + ' outliers: ' + str(n_out_x))
    f1.axhline(y=0, color='k', linewidth=1)
    f1.axvline(x=0, color='k', linewidth=1)
    confidence_ellipse(df_cleaned_x.PC1, df_cleaned_x.PC2, f1, edgecolor='red', label="Confidence interval")
    f1 = sns.scatterplot(x='PC1', y='PC2', data=df_outliers_x, color='red', label="Outlier data")
    df_cleaned_x = pd.DataFrame(df_cleaned_x)
    # f1 = sns.kdeplot(df_cleaned_x.PC1, df_cleaned_x.PC2, linestyles="--")
    plt.legend(loc='best', shadow=False, scatterpoints=1)

    plt.subplot(122)
    f3 = sns.scatterplot(x='PC1', y='PC2', data=df_cleaned_xy, label="Selected data")
    f3.set(title='PCA OF (Y)' + ' outliers: ' + str(n_out_y))
    f3.axhline(y=0, color='k', linewidth=1)
    f3.axvline(x=0, color='k', linewidth=1)
    confidence_ellipse(df_cleaned_xy.PC1, df_cleaned_xy.PC2, f3, edgecolor='red', label="Confidence interval")
    f3 = sns.scatterplot(x='PC1', y='PC2', data=df_outliers_xy, color='red', label="Outlier data")
    # f1 = sns.kdeplot(df_cleaned_xy.PC1, df_cleaned_xy.PC2, linestyles="--")
    plt.legend(loc='best', shadow=False, scatterpoints=1)

    path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls',
                                                                                                             '') + '/Figures' + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path + 'pca_outliers_plot_.png')


def pca_xy_and_outliersb(df_cleaned_x, df_cleaned_xy, df_outliers_x, df_outliers_xy, labels, n_out_x, n_out_y):
    plt.figure(5, figsize=(8, 4), dpi=125)
    plt.subplot(221)
    f1 = sns.scatterplot(x='PC1', y='PC2',data=df_cleaned_x)
    f1.set(title='CLEANED PCA OF (X)')
    f1.axhline(y=0, color='k', linewidth=1)
    f1.axvline(x=0, color='k', linewidth=1)
    plt.subplot(222)
    f2 = sns.scatterplot(x='PC1', y='PC2', data=df_outliers_x)
    f2.set(title='OUTLIERS PCA OF (X)' + ' outliers: ' + str(n_out_x))
    f2.axhline(y=0, color='k', linewidth=1)
    f2.axvline(x=0, color='k', linewidth=1)

    plt.subplot(223)
    f3 = sns.scatterplot(x='PC1', y='PC2', data=df_cleaned_xy)
    f3.set(title='CLEANED PCA OF (Y)')
    f3.axhline(y=0, color='k', linewidth=1)
    f3.axvline(x=0, color='k', linewidth=1)
    plt.subplot(224)
    f4 = sns.scatterplot(x='PC1', y='PC2', data=df_outliers_xy)
    f4.set(title='OUTLIERS PCA OF (Y)' + ' outliers: ' + str(n_out_y))
    f4.axhline(y=0, color='k', linewidth=1)
    f4.axvline(x=0, color='k', linewidth=1)


def confidence_ellipse(x, y, ax, n_std=cfg.confidence_pca, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def scatter_pred_ref(x, y):
    for i in range(y.shape[1]):
        figname = 'fig' + str(i+10)
        figname = plt.figure(i+10, figsize=(8.27, 11.69), dpi=100)
        plt.xlabel('Y measured')
        plt.ylabel('Y predicted')
        i_model, i_rmse, i_r2 = pls.do_pls(x, y.iloc[:, i], 0.7)
        yhats = pls.run_pls(x, i_model)
        plt.scatter(y.iloc[:, i], pls.run_pls(x, i_model))
        plt.plot(y.iloc[:, i], y.iloc[:, i], color='red', linewidth=0.5)
        labels = list(y.columns)
        plt.title('N: ' + str(len(y)) + ' ' + str(labels[i]) + ' R²: ' + str("%.4f" % i_r2) + ' RMSE: ' + str("%.4f" % i_rmse))


def scatter_x_y(x_train, y_train, x_test, y_test, summary_models, models, file_name):
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    for i in range(y_train.shape[1]):
        figname = 'fig' + str(i+10)
        figname = plt.figure(i+10, figsize=(8, 4), dpi=125)
        plt.xlabel('Y measured')
        plt.ylabel('Y predicted')
        yhats = pls.run_pls(x_train, models[i])
        plt.scatter(y_train.iloc[:, i], yhats, label="Train data")
        plt.plot(y_train.iloc[:, i], y_train.iloc[:, i], color='red', linewidth=0.5)
        plt.legend(loc='best', shadow=False, scatterpoints=1)

    for i in range(y_test.shape[1]):
        figname = 'fig' + str(i+10)
        figname = plt.figure(i+10, figsize=(8, 4), dpi=125)
        plt.xlabel('Y measured')
        plt.ylabel('Y predicted')
        yhats = pls.run_pls(x_test, models[i])
        plt.scatter(y_test.iloc[:, i], yhats, label="Test data")
        plt.plot(y_test.iloc[:, i], y_test.iloc[:, i], color='red', linewidth=0.5)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        figname.text(0, 0, str(summary_models.iloc[:, i].to_string()) +
                     str(' N Train: ' + str(n_train) + str(' N Test: ' + str(n_test))), color='red', fontsize=7,
                     bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
        path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls', '') + '/Figures' + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + str(figname) + str(i) + '_.png')


def scatter_x_y_n(x_train, y_train, x_test, y_test, summary_models, models, file_name, fig, paramname):
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    for i in range(y_train.shape[1]):
        figname = 'fig ' + str(paramname)
        figname = plt.figure(figname, figsize=(8, 4), dpi=125)
        plt.xlabel('Y measured')
        plt.ylabel('Y predicted')
        yhats = pls.run_pls(x_train, models[i])
        plt.scatter(y_train.iloc[:, i], yhats, label="Train data")
        plt.plot(y_train.iloc[:, i], y_train.iloc[:, i], color='red', linewidth=0.5)
        plt.legend(loc='best', shadow=False, scatterpoints=1)

    for i in range(y_test.shape[1]):
        figname = 'fig ' + str(paramname)
        figname = plt.figure(figname, figsize=(8, 4), dpi=125)
        plt.xlabel('Y measured')
        plt.ylabel('Y predicted')
        yhats = pls.run_pls(x_test, models[i])
        plt.scatter(y_test.iloc[:, i], yhats, label="Test data")
        plt.plot(y_test.iloc[:, i], y_test.iloc[:, i], color='red', linewidth=0.5)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        figname.text(0, 0, str(summary_models.iloc[0:5, 0].to_string()) +
                     str(' N Train: ' + str(n_train) + str(' N Test: ' + str(n_test))), color='red', fontsize=7,
                     bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
        path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls', '') + '/Figures' + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + str(figname) + str(i) + '_.png')


def med_x_pred_sigma(x_data, y_data, summary_models, models, file_name, fig, paramname):
    x_test, x_train, y_test, y_train = train_test_split(x_data, y_data, test_size=cfg.train_split_percentage, random_state=0)
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    for i in range(y_train.shape[1]):
        figname = 'fig' + str(10+fig)
        figname = plt.figure(10+fig, figsize=(8, 4), dpi=125)
        plt.xlabel('Y measured')
        plt.ylabel('Y predicted')
        yhats = pls.run_pls(x_train, models[i])
        plt.scatter(y_train.iloc[:, i], yhats, label="Train data")
        plt.plot(y_train.iloc[:, i], y_train.iloc[:, i], color='red', linewidth=0.5)
        plt.legend(loc='best', shadow=False, scatterpoints=1)

    for i in range(y_test.shape[1]):
        figname = 'fig' + str(10+fig)
        figname = plt.figure(10+fig, figsize=(8, 4), dpi=125)
        plt.xlabel('Y measured')
        plt.ylabel('Y predicted')
        yhats = pls.run_pls(x_test, models[i])
        plt.scatter(y_test.iloc[:, i], yhats, label="Test data")
        plt.plot(y_test.iloc[:, i], y_test.iloc[:, i], color='red', linewidth=0.5)
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        figname.text(0, 0, str(summary_models.iloc[0:5, 0].to_string()) +
                     str(' N Train: ' + str(n_train) + str(' N Test: ' + str(n_test))), color='red', fontsize=7,
                     bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
        path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls', '') + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + str(figname) + str(i) + paramname + '_.png')

