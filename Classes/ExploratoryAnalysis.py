import seaborn as sns; sns.set()
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import scipy as sp


def principal_component_analysis(data_to_analyze):
    scale = StandardScaler(copy=True, with_mean=True, with_std=True)
    scale.fit(data_to_analyze)
    df = scale.transform(data_to_analyze)
    n_of_pcs = data_to_analyze.shape[1]-1
    if n_of_pcs > 20:
        n_of_pcs = 20
    pca_model = PCA(n_components=n_of_pcs)
    pcs = pca_model.fit_transform(df)
    pcs_labels = []
    for i in range(n_of_pcs):
        pc_label = ('PC' + str(i+1) + ' ' + "{0:.2f}".format(pca_model.explained_variance_ratio_[i]*100) + '%')
        pcs_labels.append(pc_label)
    pcs = pd.DataFrame(pcs, columns=[pcs_labels])
    return pca_model, pcs


def do_mahalanobis(data_to_analyze):
    df = pd.DataFrame(data_to_analyze)
    model, pcs = extract_pc1_pc2(df, 2)
    pca_df = pd.DataFrame(data=pcs, columns=['PC1', 'PC2'])
    covariance_matrix = pca_df.cov()
    inverse_covariance_matrix = sp.linalg.inv(covariance_matrix)
    xy_mean = pca_df['PC1'].mean(), pca_df['PC2'].mean()
    x_diff = np.array([x_i - xy_mean[0] for x_i in pca_df['PC1']])
    y_diff = np.array([y_i - xy_mean[1] for y_i in pca_df['PC2']])
    diff_xy = np.transpose([x_diff, y_diff])
    md = []
    for i in range(len(diff_xy)):
        md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]), inverse_covariance_matrix), diff_xy[i])))
    mahalanobis_distance = pd.DataFrame(data=md, columns=['Mahalanobis'])
    df = pd.concat([df.reset_index(), pca_df, mahalanobis_distance], axis=1)
    df.index = pd.DataFrame(data_to_analyze).index
    df.drop(['index'], axis=1, inplace=True)
    return df


def drop_mahalanobis_outliers(data_to_analyze, sigma_confidence):
    df = pd.DataFrame(data_to_analyze)
    df_out = df.copy(deep=True)
    md = df['Mahalanobis']
    limit = sigma_confidence
    df_bool = df['Mahalanobis'] < limit
    df = df[df_bool]
    df = df.dropna()
    df_bool_out = df_out['Mahalanobis'] > limit
    df_out = df_out[df_bool_out]
    df_out = df_out.dropna()
    pcs_cleaned = pd.DataFrame(df)
    df.drop(['PC1'], axis=1, inplace=True)
    df.drop(['PC2'], axis=1, inplace=True)
    df.drop(['Mahalanobis'], axis=1, inplace=True)
    pcs_outliers = pd.DataFrame(df_out)
    df_out.drop(['PC1'], axis=1, inplace=True)
    df_out.drop(['PC2'], axis=1, inplace=True)
    df_out.drop(['Mahalanobis'], axis=1, inplace=True)
    return df_out, df, pcs_outliers, pcs_cleaned


def extract_pc1_pc2(data_to_analyze, n_components):
    scale = StandardScaler(copy=True, with_mean=True, with_std=True)
    scale.fit(data_to_analyze)
    df = scale.transform(data_to_analyze)
    pca_model = PCA(n_components=n_components)
    pcs = pca_model.fit_transform(df)
    pcs = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
    return pca_model, pcs


def exclude_outliers(x, y, labels, outlier_confidence_level_x, outlier_confidence_level_y):
    main_df = pd.DataFrame(pd.concat([x, y], axis=1))
    main_df.index = labels
    x_shape = x.shape[1]
    y_start = x_shape

    n_total = x.shape[0]
    mahalanobis_x = pd.DataFrame(do_mahalanobis(main_df.iloc[:, 0:y_start]))
    df_cleaned_x = pd.DataFrame(pd.concat([mahalanobis_x, main_df.iloc[:, y_start:]], axis=1))
    df_outliers_x, df_cleaned_x, pcs_out_x, pcs_cleaned_x = drop_mahalanobis_outliers(df_cleaned_x, outlier_confidence_level_x)
    n_outliers_x = df_outliers_x.shape[0]

    mahalanobis_y = pd.DataFrame(do_mahalanobis(df_cleaned_x.iloc[:, y_start:]))
    df_cleaned_xy = pd.DataFrame(
        pd.concat([df_cleaned_x.iloc[:, 0:y_start], mahalanobis_y], axis=1))
    df_outliers_xy, df_cleaned_xy, pcs_out_y, pcs_cleaned_y = drop_mahalanobis_outliers(df_cleaned_xy, outlier_confidence_level_y)
    n_outliers_y = df_outliers_xy.shape[0]

    x_data = pd.DataFrame(df_cleaned_xy.iloc[:, 0:y_start])
    y_data = pd.DataFrame(df_cleaned_xy.iloc[:, y_start:])

    x_out = pd.DataFrame(df_outliers_x.iloc[:, 0:y_start])
    y_out = pd.DataFrame(df_outliers_xy.iloc[:, y_start:])

    return x_data, y_data, pcs_cleaned_x, pcs_cleaned_y, pcs_out_x, pcs_out_y, n_outliers_x, n_outliers_y, x_out, y_out
