# Bank Marketing Decision Tree Classifier

This project uses a Decision Tree Classifier to predict whether a customer will subscribe to a term deposit based on demographic and behavioral data.

## 🔗 Dataset
[UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

## 📦 Libraries Used
- pandas
- scikit-learn
- matplotlib
- seaborn

## 🚀 How to Run
1. Clone this repository.
2. Download the dataset and place `bank-additional-full.csv` inside the `data/` folder.
3. Run `bank_marketing_decision_tree.ipynb` using Jupyter Notebook.

## 📊 Output
Accuracy and classification report for predicting term deposit subscription.

---

code:
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/bank-additional-full.csv", sep=';')

# Convert categorical variables using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Features and target
X = df_encoded.drop('y_yes', axis=1)
y = df_encoded['y_yes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build decision tree classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Plot feature importances
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()

# requirments.txt
pandas
scikit-learn
matplotlib
seaborn
