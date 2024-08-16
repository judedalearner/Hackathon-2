import joblib
import pandas as pd
 
model = joblib.load(r'C:\Users\jude4\OneDrive\Desktop\DI DIPLOMA PROJECT\Hackathon 2\logistic_regression_model.pkl')
 
df_new = pd.read_csv(r'C:\Users\jude4\OneDrive\Desktop\DI DIPLOMA PROJECT\Hackathon 2\new_data.csv')  # Ensure 'new_data.csv' is the correct file name
 
X_new = df_new 
 
y_pred = model.predict(X_new)
 
print("Predictions:", y_pred)
 
output = pd.DataFrame({'Prediction': y_pred})
output.to_csv(r'C:\Users\jude4\OneDrive\Desktop\DI DIPLOMA PROJECT\Hackathon 2\predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'")