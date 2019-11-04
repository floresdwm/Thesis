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
print('Starting at ' + str(datetime.now()))
confidence = cfg.confidence_level
labels, x, y, file_name = Fio.import_excel_data()
n_total = x.shape[0]

# 2. check if data has same n of rows
inputs_are_empty = Miss.check_matrix_empty(labels, x, y)
inputs_has_same_n = Miss.check_matrix_rows(labels, x, y)

if True == inputs_are_empty and inputs_has_same_n == False:
    print('Initializing routine failed to start at ' + str(datetime.now()))
    print('imported data is empty ? ' + str(inputs_are_empty))
    print('imported data has same size? ' + str(inputs_has_same_n))
    print('Routine was not executed and finished at ' + str(datetime.now()))
else:
    print('Starting machine learning routine at ' + str(datetime.now()))
    plt.style.use('seaborn')

    # 3. plot imported data
    if y.shape[1] >= 1:
        Plot.correlation_matrix(y, file_name)
    Plot.x_data(x, file_name)

    # 4. exclude outliers based on Mahalanobis distance
    x_data, y_data, df_cleaned_x, df_cleaned_xy, df_outliers_x, df_outliers_xy, n_outliers_x, n_outliers_y = Exp.exclude_outliers(
        x, y, labels, confidence)
    if n_outliers_y != 0 and n_outliers_x != 0:
        help.outliers_pcs_plot(df_cleaned_x, df_cleaned_xy, df_outliers_x, df_outliers_xy, labels, n_outliers_x,
                               n_outliers_y, file_name)

    summary_outliers = ['N Total: ' + str(n_total),
                        'N Outliers X: ' + str(n_outliers_x) + ' - %: ' + str(((n_outliers_x * 100) / n_total)),
                        'N Outliers Y: ' + str(n_outliers_y) + ' - %: ' + str(
                            ((n_outliers_y * 100) / (n_total - n_outliers_x)))]

    # 5. build multivariate regression models
    summary_models, models, x_train, y_train, x_test, y_test = Reg.partial_leasts_square_regression(x_data, y_data,
                                                                                                    cfg.train_split_percentage)
    Plot.scatter_x_y(x_train, y_train, x_test, y_test, summary_models, models, file_name)

    # 6. export full report
    Fio.data_to_excel(file_name, df_cleaned_xy, pd.concat([df_outliers_x, df_outliers_xy], axis=0, sort=True))
    Fio.summary_data_to_excel(summary_models, file_name)
    Fio.summary_outlier_to_excel(pd.DataFrame(summary_outliers), file_name)

    print('Done at ' + str(datetime.now()))
    print('Hello Github!')
    plt.show()
