import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

 
model = joblib.load(r'C:\Users\jude4\OneDrive\Desktop\DI DIPLOMA PROJECT\Hackathon 2\logistic_regression_model.pkl')

 
df = pd.read_csv(r'C:\Users\jude4\OneDrive\Desktop\DI DIPLOMA PROJECT\Hackathon 2\preprocessed_data.csv')

 
X = df.drop(columns=['y_yes'])  # Assuming 'y_yes' is your target column
y = df['y_yes']
 
feature_names = X.columns

 
importance = model.coef_[0]

 
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(importance)})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

 
print(importance_df)

 
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()
