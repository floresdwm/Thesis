import pandas as pd


def check_matrix_rows(labels, x, y):
    try:
        labels_rows = pd.DataFrame(labels).shape[0]
        x_rows = pd.DataFrame(x).shape[0]
        y_rows = pd.DataFrame(y).shape[0]

        if x_rows == y_rows == labels_rows:
            return True
        else:
            return False
    except:
        print("An exception occurred: matrix size (n rows) doesn't match.")
        return False


def check_matrix_empty(labels, x, y):
    try:
        labels_rows = pd.DataFrame(labels).shape[0]
        x_rows = pd.DataFrame(x).shape[0]
        y_rows = pd.DataFrame(y).shape[0]

        if x_rows != 0 and y_rows != 0 and labels_rows != 0:
            return False
        else:
            return True
    except:
        print("An exception occurred: some imported matrix could be empty.")
        return True