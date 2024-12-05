import pandas as pd
def preprocess(data1: pd.DataFrame, data2: pd.DataFrame, data3: pd.DataFrame) -> pd.DataFrame:

    data = pd.concat([data1, data2, data3])
    data['date_time'] = pd.to_datetime(data['date']).values.astype("float64")

    return data
