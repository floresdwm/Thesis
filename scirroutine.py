import Classes.FileIO as Fio
import Classes.MissingData as Miss
import Classes.ExploratoryAnalysis as Exp
import Classes.Helpers as help
import Classes.Plots as Plot
from datetime import datetime
import matplotlib.pyplot as plt
import Classes.RegressionAnalysis as Reg
import pandas as pd
import Classes.Configurations as cfg


# 1. import data
print('-- Routine started --')
outlier_confidence_level_x = cfg.outlier_confidence_level_x
outlier_confidence_level_y = cfg.outlier_confidence_level_y
confidence_pca = cfg.confidence_pca
labels, xor, y, file_name = Fio.import_excel_data()
n_total = xor.shape[0]
parameters_count = y.shape[1]

# 2. check if data has same n of rows
inputs_are_empty = Miss.check_matrix_empty(labels, xor, y)
inputs_has_same_n = Miss.check_matrix_rows(labels, xor, y)

# 2.1 slice x data
x = xor.iloc[:, cfg.Xinit:cfg.Xend]

if True == inputs_are_empty and inputs_has_same_n == False and y.shape[1] <= 1:
    print('Initializing routine failed to start at ' + str(datetime.now()))
    print('imported data is empty ? ' + str(inputs_are_empty))
    print('imported data has same size? ' + str(inputs_has_same_n))
    print('Routine was not executed and finished at ' + str(datetime.now()))
else:
    started_time = datetime.now()
    print('Starting machine learning routine at ' + str(started_time))
    plt.style.use('seaborn')

    # 3. exclude outliers based on Mahalanobis distance
    x_data, y_data, df_cleaned_x, df_cleaned_xy, df_outliers_x, df_outliers_xy, n_outliers_x, n_outliers_y, x_out, y_out = Exp.exclude_outliers(
        x, y, labels, outlier_confidence_level_x, outlier_confidence_level_y)
    if n_outliers_y != 0 and n_outliers_x != 0:
        help.outliers_pcs_plot(df_cleaned_x, df_cleaned_xy, df_outliers_x, df_outliers_xy, labels, n_outliers_x,
                               n_outliers_y, file_name)
        Plot.x_data_outliers(x_data, x_out, file_name)

    summary_outliers = ['N Total: ' + str(n_total),
                        'N Outliers X: ' + str(n_outliers_x) + ' - %: ' + str(((n_outliers_x * 100) / n_total)),
                        'N Outliers Y: ' + str(n_outliers_y) + ' - %: ' + str(
                            ((n_outliers_y * 100) / (n_total - n_outliers_x)))]

    # 4. plot imported data y corr
    Plot.correlation_matrix(y, file_name)

    # 5. build multivariate regression models
    for i in range(y_data.shape[1]):
        x_data_i = x_data
        y_data_i = y_data
        x_data_i, y_data_i = Reg.remove_rows_with_zeros(x_data_i, y_data_i.iloc[:, i])

        summary_models, df_model_summary, df_y_resume, models, x_train, y_train, x_test, y_test = Reg.partial_leasts_square_regression(x_data_i,
                                                                                                        y_data_i,
                                                                                                        cfg.train_split_percentage, file_name)
        Plot.scatter_x_y_n(x_train, y_train, x_test, y_test, df_model_summary, models, file_name, i,
                           str(y.columns[i]))

        # 6. export full report
        Fio.data_to_excel(file_name, df_cleaned_xy, pd.concat([df_outliers_x, df_outliers_xy], axis=0, sort=True))
        Fio.summary_data_to_excel_scir(summary_models, file_name, str(y.columns[i]))
        Fio.summary_outlier_to_excel(pd.DataFrame(summary_outliers), file_name)
        parameters = y.columns[i]
        Fio.save_model_to_pkl(models[0], parameters, file_name)
        Fio.save_model_to_json(models[0], parameters, file_name, df_model_summary, df_y_resume)
        print('Model ' + str(y.columns[i]) + ' done.')

    finished_time = datetime.now()
    print('Routine finalized in ' + str(finished_time - started_time))
    plt.show()