import Classes.ExploratoryAnalysis as Exp
import Classes.Plots as Plot
import pandas as pd


def extract_pcs_and_plot(df_cleaned_x, df_cleaned_xy, df_outliers_x, df_outliers_xy, labels, n_outliers_x, n_outliers_y):
    model1, pcs1 = Exp.extract_pc1_pc2(df_cleaned_x, 2)
    df_cleaned_x = pd.DataFrame(data=pcs1, columns=['PC1', 'PC2'])

    model2, pcs2 = Exp.extract_pc1_pc2(df_cleaned_xy, 2)
    df_cleaned_xy = pd.DataFrame(data=pcs2, columns=['PC1', 'PC2'])

    model3, pcs3 = Exp.extract_pc1_pc2(df_outliers_x, 2)
    df_outliers_x = pd.DataFrame(data=pcs3, columns=['PC1', 'PC2'])

    model4, pcs4 = Exp.extract_pc1_pc2(df_outliers_xy, 2)
    df_outliers_xy = pd.DataFrame(data=pcs4, columns=['PC1', 'PC2'])

    Plot.pca_xy_and_outliers(df_cleaned_x, df_cleaned_xy, df_outliers_x, df_outliers_xy, labels, n_outliers_x, n_outliers_y)


def outliers_pcs_plot(df_cleaned_x, df_cleaned_xy, df_outliers_x, df_outliers_xy, labels, n_outliers_x,
                             n_outliers_y, file_name):
        Plot.pca_xy_and_outliers(df_cleaned_x, df_cleaned_xy, df_outliers_x, df_outliers_xy, labels, n_outliers_x,
                                 n_outliers_y, file_name)



