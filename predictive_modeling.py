import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
 
df = pd.read_csv(r'C:\Users\jude4\OneDrive\Desktop\DI DIPLOMA PROJECT\Hackathon 2\preprocessed_data.csv')
 
X = df.drop(columns=['y_yes']) 
y = df['y_yes'] 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
 
y_pred = model.predict(X_test)
 
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))

import joblib
 
joblib.dump(model, 'logistic_regression_model.pkl')
print("Model saved to 'logistic_regression_model.pkl'")

