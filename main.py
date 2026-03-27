import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# load data
data = pd.read_csv("data.csv")
X = data[['Hours']]
y = data['Marks']

# model
model = LinearRegression()
model.fit(X, y)

# prediction
hours = float(input("Enter study hours: "))
if hours < 0:
    print("Hours can't be negative!")
else:
    predicted_marks = model.predict([[hours]])
    print("Predicted Marks:", round(predicted_marks[0], 2))

    # graph
    plt.scatter(X, y)
    plt.plot(X, model.predict(X), color='red')
    plt.xlabel("Study Hours")
    plt.ylabel("Marks")
    plt.title("Study Hours vs Marks")
    plt.show()