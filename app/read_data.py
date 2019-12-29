import pandas as pd


def read_data(filename='norm.xlsx'):
    xl_file = pd.ExcelFile(filename)
    # print(xl_file.sheet_names[0])
    dfs = xl_file.parse(xl_file.sheet_names[0])
    dfd = dfs.as_matrix()
    t = dfs.T.columns.values.tolist()
    return t, dfd
