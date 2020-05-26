import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import os
import joblib
import json, codecs
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from datetime import date
import Classes.Configurations as cfg

from Classes import Configurations


def import_excel_data():
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Import data (Excel file)", "Choose your matrix of inputs.")
        file_path = filedialog.askopenfilename()
        file_name = os.path.basename(file_path)
        labels = pd.read_excel(file_path, 0)
        label_df = pd.DataFrame(labels)
        xdata = pd.read_excel(file_path, 1)
        x_df = pd.DataFrame(xdata)
        ydata = pd.read_excel(file_path, 2)
        y_df = pd.DataFrame(ydata)
        return label_df, x_df, y_df, file_name
    except:
        print("An exception occurred while importing excel file.")


def import_excel_data_single():
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Import data (Excel file)", "Choose your matrix of inputs.")
        file_path = filedialog.askopenfilename()
        file_name = os.path.basename(file_path)
        data = pd.read_excel(file_path, 0)
        data_df = pd.DataFrame(data)
        return data_df
    except:
        print("An exception occurred while importing excel file.")


def import_excel_data_epo():
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Import data (Excel file)", "Choose your matrix of inputs.")
        file_path = filedialog.askopenfilename()
        file_name = os.path.basename(file_path)
        labels = pd.read_excel(file_path, 0)
        label_df = pd.DataFrame(labels)
        xdatau = pd.read_excel(file_path, 1)
        x_df_u = pd.DataFrame(xdatau)
        xdatas = pd.read_excel(file_path, 2)
        x_df_s = pd.DataFrame(xdatas)
        ydata = pd.read_excel(file_path, 3)
        y = pd.DataFrame(ydata)
        return label_df, x_df_u, x_df_s, y, file_name
    except:
        print("An exception occurred while importing excel file.")


def data_to_excel(file_name, all_df, outliers_df):
    path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls', '')
    if not os.path.exists(path):
        os.makedirs(path)
    writer = pd.ExcelWriter(path + '\Mahalanobis_report.xlsx')
    all_data = pd.DataFrame(all_df)
    all_data.to_excel(writer, 'Selected data')
    stats_all = pd.DataFrame(all_data.describe())
    stats_all.to_excel(writer, 'Stats selected data')
    outliers_data = pd.DataFrame(outliers_df)
    outliers_data.to_excel(writer, 'Outliers data')
    stats_outliers = pd.DataFrame(outliers_data.describe())
    stats_outliers.to_excel(writer, 'Stats Outliers data')
    writer.save()


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


def summary_data_to_excel(df, file_name):
    path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls', '')
    if not os.path.exists(path):
        os.makedirs(path)

    writer = pd.ExcelWriter(path + '\PLS_Report.xlsx')
    all_data = pd.DataFrame(df)
    all_data.to_excel(writer, 'PLS SUMMARY')
    writer.save()


def summary_outlier_to_excel(df, file_name):
    path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls', '')
    if not os.path.exists(path):
        os.makedirs(path)

    writer = pd.ExcelWriter(path + '\Outliers_Report.xlsx')
    all_data = pd.DataFrame(df)
    all_data.to_excel(writer, 'OUTLIERS SUMMARY')
    writer.save()


def data_describe_to_excel(df, df_outliers, file_name):
    path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls', '')
    if not os.path.exists(path):
        os.makedirs(path)

    writer = pd.ExcelWriter(path + '\PCA_stats_report.xlsx')
    df_describe = pd.DataFrame(df.describe())
    df_describe.to_excel(writer, 'Descriptive statistics inliers')
    df_describe_out = pd.DataFrame(df_outliers.describe())
    df_describe_out.to_excel(writer, 'Descriptive statistics outliers')
    writer.save()


def save_model_to_pkl(model, model_name, file_name):
    path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls', '') + '/PKL'
    if not os.path.exists(path):
        os.makedirs(path)

    file_name_pls_final = os.path.join(path, str(model_name) + '.pkl')
    joblib.dump(model, file_name_pls_final, protocol=2)


def save_model_to_json(model, model_name, file_name, df_model_resume, df_y_resume):
    path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls', '') + '/JSON'
    if not os.path.exists(path):
        os.makedirs(path)

    file_name_jsonfy = os.path.join(path, str(model_name) + '.json')
    coef = np.array(model.coef_).tolist()
    x_mean = np.array(model.x_mean_).tolist()
    x_std = np.array(model.x_std_).tolist()
    y_mean = np.array(model.y_mean_).tolist()[0]
    coef_list = []
    for value in coef:
        coef_list.append(value[0])

    train_resume = []
    df_y_resume = pd.DataFrame(df_y_resume)
    for item in range(df_y_resume.shape[0]):
        if isinstance(df_y_resume.values[item][0], str):
            train_resume.append(str(df_y_resume.index[item]) + ": " + str(df_y_resume.values[item][0]))
        else:
            train_resume.append(str(df_y_resume.index[item]) + ": " + '{0:.4f}'.format(float(df_y_resume.values[item][0])))

    model_resume = []
    df_model_resume = pd.DataFrame(df_model_resume)
    for statistic in range(df_model_resume.shape[0]):
        if isinstance(df_model_resume.values[statistic][0], str):
            model_resume.append(str(df_model_resume.index[statistic]) + ": " + str(df_model_resume.values[statistic][0]))
        else:
            model_resume.append(
                str(df_model_resume.index[statistic]) + ": " + '{0:.4f}'.format(float(df_model_resume.values[statistic][0])))

    model_datetime = date.today()
    train_resume.append("Time stamp: " +str(model_datetime))

    json_model = {
            "Coefficients": coef_list,
            "Xmean": x_mean,
            "Xstd": x_std,
            "Ymean": y_mean,
            "Transformation": Configurations.transformation,
            "ParameterName": model_name,
            "ApplicationName": "",
            "Xinit": cfg.Xinit,
            "Xend": cfg.Xend,
            "ModelSummary": model_resume,
            "Ysummary": train_resume,
            "Scans": cfg.scans
    }

    with open(file_name_jsonfy, 'w') as write_file:
        json.dump(json_model, write_file)


def data_to_excel_scir(file_name, all_df, outliers_df, param_name):
    path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls', '')
    if not os.path.exists(path):
        os.makedirs(path)
    writer = pd.ExcelWriter(path + '\Mahalanobis' + '_' + param_name + '.xlsx')
    all_data = pd.DataFrame(all_df)
    all_data.to_excel(writer, 'Selected data')
    stats_all = pd.DataFrame(all_data.describe())
    stats_all.to_excel(writer, 'Stats selected data')
    outliers_data = pd.DataFrame(outliers_df)
    outliers_data.to_excel(writer, 'Outliers data')
    stats_outliers = pd.DataFrame(outliers_data.describe())
    stats_outliers.to_excel(writer, 'Stats Outliers data')
    writer.save()


def summary_data_to_excel_scir(df, file_name, param_name):
    path = os.path.expanduser("~/Desktop") + '/foodscienceml' + '/' + file_name.replace('.xlsx', '').replace('.xls', '')
    if not os.path.exists(path):
        os.makedirs(path)

    writer = pd.ExcelWriter(path + '\PLS_Report_' + param_name + '.xlsx')
    all_data = pd.DataFrame(df)
    all_data.to_excel(writer, 'PLS SUMMARY')
    writer.save()
