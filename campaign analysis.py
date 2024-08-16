import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def calculate_metrics(df):
     
    print("Column names:", df.columns)

     
    if 'y_yes' not in df.columns:
        raise ValueError("Column 'y_yes' is missing from the DataFrame")
    
    poutcome_columns = ['poutcome_other', 'poutcome_success', 'poutcome_unknown']
    if not any(col in df.columns for col in poutcome_columns):
        raise ValueError("None of the 'poutcome' related columns are present in the DataFrame")

     
    conversion_rate = df['y_yes'].mean() * 100
    print(f"Overall Conversion Rate: {conversion_rate:.2f}%")

    
    response_rate = df.groupby('campaign')['y_yes'].mean()
    print("\nResponse Rate by Campaign:")
    print(response_rate)

     
    poutcome_effectiveness = df[poutcome_columns].mul(df['y_yes'], axis=0).sum() / df[poutcome_columns].sum()
    print("\nEffectiveness of Previous Campaign Outcomes:")
    print(poutcome_effectiveness)

    return conversion_rate, response_rate, poutcome_effectiveness

if __name__ == "__main__":
    file_path = 'C:/Users/jude4/OneDrive/Desktop/DI DIPLOMA PROJECT/Hackathon 2/preprocessed_data.csv'

    df = load_data(file_path)

    conversion_rate, response_rate, poutcome_effectiveness = calculate_metrics(df)

    response_rate.to_csv('response_rate_by_campaign.csv')
    poutcome_effectiveness.to_csv('poutcome_effectiveness.csv')
    print("Analysis results saved to 'response_rate_by_campaign.csv' and 'poutcome_effectiveness.csv'.")






