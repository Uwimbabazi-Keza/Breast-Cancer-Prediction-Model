import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data(data_path):
    df = pd.read_csv(data_path)
    df = df.dropna(axis=1)
    
    lb = LabelEncoder()
    df['diagnosis'] = lb.fit_transform(df['diagnosis'])
    
    x = df.drop(labels=["diagnosis", "id"], axis=1)
    y = df["diagnosis"].values

    s = MinMaxScaler()
    s.fit(x)
    x = s.transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    
    return x_train, x_test, y_train, y_test
