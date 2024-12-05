import pandas as pd
def load_data(dataset1, dataset2, dataset3):
    data1 = pd.read_csv(dataset1, parse_dates=True)

    data2 = pd.read_csv(dataset2, parse_dates=True)

    data3 = pd.read_csv(dataset3, parse_dates=True)
    return data1, data2, data3