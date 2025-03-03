 ML Decision Tree on High-Paying Jobs

This project applies a **Decision Tree Machine Learning Model** to predict high-paying job salaries based on various features from the provided `2025 Job Salary CSV` file.

 Prerequisites
Ensure you have **Anaconda** installed on your system. If not, download and install it from:
[https://www.anaconda.com/download](https://www.anaconda.com/download)

 Steps to Run the Model on Anaconda

1. Create and Activate a New Virtual Environment (Optional)
```bash
conda create --name ml_decision_tree python=3.9 -y
conda activate ml_decision_tree
```

2. Install Required Libraries
Run the following command in Anaconda Prompt:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

 3. Load the Dataset
Ensure your CSV file (e.g., `high_salary_jobs.csv`) is in the working directory.
```python
import pandas as pd
data = pd.read_csv("high_salary_jobs.csv")
print(data.head())
```

 4. Preprocess the Data
Convert categorical variables into numerical form for training.
```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data['Company'] = encoder.fit_transform(data['Company'])
data['Job Role'] = encoder.fit_transform(data['Job Role'])
data['Required Degree'] = encoder.fit_transform(data['Required Degree'])
data['Salary'] = data['Average Salary (USD)'].str.replace("$", "").str.replace(",", "").astype(float)
data.drop(['Average Salary (USD)'], axis=1, inplace=True)
```

 5. Split Data for Training and Testing
```python
from sklearn.model_selection import train_test_split
X = data.drop(columns=['Salary'])
y = data['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

6. Train the Decision Tree Model
```python
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
```

 7. Evaluate the Model
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
```
 8. Visualize the Decision Tree (Optional)
```python
from sklearn
