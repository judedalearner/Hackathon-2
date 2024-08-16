import pandas as pd
 
df = pd.read_csv(r'C:\Users\jude4\OneDrive\Desktop\DI DIPLOMA PROJECT\Hackathon 2\preprocessed_data.csv')  # Make sure this path is correct
 
df_new = df.drop(columns=['y_yes'])
 
df_new.to_csv('new_data.csv', index=False)
