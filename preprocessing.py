from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_data(df):
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

if __name__ == "__main__":
    file_path = r'C:\Users\jude4\OneDrive\Desktop\DI DIPLOMA PROJECT\Hackathon 2\bank-full.csv'
    df = pd.read_csv(file_path, delimiter=';')
    df = preprocess_data(df)
    df.to_csv('preprocessed_data.csv', index=False)
