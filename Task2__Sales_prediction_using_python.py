import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\dwive\\OneDrive\\Desktop\\Oasis Infobyte Internship\\sales prediction csv\\Advertising.csv')

# Split  data into a training set and a testing set
X = data[["TV","Radio","Newspaper","Sales"]]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# By Linear Regression we would make a graph
model = LinearRegression()
model.fit(X_train, y_train)

# sales predictions
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.show()