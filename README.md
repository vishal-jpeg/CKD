# CKD
By following the comments and explanations above, you can understand the purpose and functionality of each part of the code. This should help you present your project more effectively and confidently.

### Import Libraries

```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
```
- **Import libraries**: Import necessary libraries for data manipulation, visualization, and machine learning.

### Load Dataset

```
print("Loading dataset...")
df = pd.read_csv(r'/content/kidney_disease.csv')
columns = pd.read_csv(r'/content/data_description.txt', sep='-')
columns = columns.reset_index()
columns.columns = ['cols', 'abb_col_names']
df.columns = columns['abb_col_names'].values
print("Dataset loaded successfully.\n")
```
- **Load dataset**: Load the dataset and rename columns based on a separate file.

### Convert Data Types

```
print("Converting data types for specific features...")
def convert_dtype(df, feature):
    df[feature] = pd.to_numeric(df[feature], errors='coerce')
features = ['packed cell volume', 'white blood cell count', 'red blood cell count']
for feature in features:
    convert_dtype(df, feature)
print("Data types converted.\n")
```
- **Convert data types**: Convert specific features to numeric type.

### Data Wrangling

```
print("Dropping unnecessary columns...")
df.drop('id', axis=1, inplace=True)
print("Columns dropped.\n")
```
- **Drop columns**: Remove unnecessary columns from the dataset.

### Extract Numerical & Categorical Features

```
print("Extracting numerical and categorical features...")
def extract_cat_num(df):
    cat_col = [col for col in df.columns if df[col].dtype == 'object']
    num_col = [col for col in df.columns if df[col].dtype != 'object']
    return cat_col, num_col
cat_col, num_col = extract_cat_num(df)
print(f"Categorical columns: {cat_col}")
print(f"Numerical columns: {num_col}\n")
```
- **Extract features**: Identify and separate numerical and categorical features.

### Replace Incorrect Values

```
print("Replacing incorrect values in categorical columns...")
df['diabetes mellitus'].replace(to_replace={'\tno':'no', '\tyes':'yes', ' yes':'yes'}, inplace=True)
df['coronary artery disease'] = df['coronary artery disease'].replace(to_replace='\tno', value='no')
df['class'] = df['class'].replace(to_replace='ckd\t', value='ckd')
print("Incorrect values replaced.\n")
```
- **Replace incorrect values**: Correct inconsistent entries in categorical columns.

### Handle Missing Values

```
print("Handling missing values...")
data = df.copy()
for col in num_col:
    random_sample = data[col].dropna().sample(data[col].isnull().sum())
    random_sample.index = data[data[col].isnull()].index
    data.loc[data[col].isnull(), col] = random_sample

def impute_mode(feature):
    mode = data[feature].mode()[0]
    data[feature] = data[feature].fillna(mode)
for col in cat_col:
    impute_mode(col)
print("Missing values handled.\n")
```
- **Handle missing values**: Fill missing values in numerical features with random samples and in categorical features with the mode.

### Encode Categorical Features

```
print("Encoding categorical features...")
le = LabelEncoder()
for col in cat_col:
    data[col] = le.fit_transform(data[col])
print("Categorical features encoded.\n")
```
- **Encode features**: Apply label encoding to categorical features.

### Feature Selection

```
print("Selecting top features using SelectKBest...")
X = data.drop('class', axis=1)
y = data['class']
order_rank_features = SelectKBest(score_func=chi2, k=20)
X_new = order_rank_features.fit_transform(X, y)
selected_columns = order_rank_features.get_support(indices=True)
selected_columns_names = X.columns[selected_columns]  # Save selected columns
print(f"Top selected features: {selected_columns_names}\n")
```
- **Feature selection**: Use `SelectKBest` with the chi-squared score to select the top 20 features.

### Plot Feature Importance

```
print("Plotting feature importance...")
features_rank = pd.DataFrame({
    'Feature': X.columns,
    'Importance': order_rank_features.scores_
})
features_rank = features_rank.sort_values(by='Importance', ascending=False).head(10)
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=features_rank)
plt.title('Top 10 Features by Importance')
plt.show()
```
- **Feature importance plot**: Plot the top 10 features based on their importance scores.

### Train-Test Split

```
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
print("Data split completed.\n")
```
- **Train-test split**: Split the data into training and testing sets.

### Model Training

```
print("Training the model...")
classifier = XGBClassifier(
    colsample_bytree=0.3,
    gamma=0.4,
    learning_rate=0.25,
    max_depth=8,
    n_estimators=100,
    objective='binary:logistic'
)
classifier.fit(X_train, y_train)
print("Model trained successfully.\n")
```
- **Model training**: Train an XGBoost classifier with specified hyperparameters.

### Evaluate Model

```
print("Evaluating the model...")
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model is: {accuracy * 100:.2f}%\n")
```
- **Model evaluation**: Predict on the test set and print the accuracy.

### Confusion Matrix Plot

```
print("Plotting confusion matrix...")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not CKD', 'CKD'], yticklabels=['Not CKD', 'CKD'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```
- **Confusion matrix plot**: Plot the confusion matrix to visualize the model's performance.

### Save Label Encoder Classes

```
le_classes = {}
for col in cat_col:
    le_classes[col] = le.classes_
print("Label encoder classes saved for later use.\n")
```
- **Save label encoder classes**: Save the classes of the label encoder for each categorical feature.

### User Input and Prediction Functions

```
def predict_ckd(input_data):
    input_df = pd.DataFrame([input_data], columns=selected_columns_names)
    for col in cat_col:
        if col in input_df.columns:
            input_df[col] = input_df[col].map(lambda x: list(le_classes[col]).index(x) if x in le_classes[col] else np.nan)
    input_df = input_df[selected_columns_names]
    prediction = classifier.predict(input_df)
    return 'CKD' if prediction[0] == 1 else 'Not CKD'
```
- **Prediction function**: Define a function to predict CKD based on user input.

```
def get_user_input():
    print("Please enter the following details:")
    input_data = {}
    input_data['age'] = float(input("Enter age: "))
    input_data['blood pressure'] = float(input("Enter blood pressure: "))
    input_data['specific gravity'] = float(input("Enter specific gravity: "))
    input_data['albumin'] = float(input("Enter albumin: "))
    input_data['sugar'] = float(input("Enter sugar: "))
    input_data['red blood cells'] = input("Enter red blood cells (normal/abnormal): ")
    input_data['pus cell'] = input("Enter pus cell (normal/abnormal): ")
    input_data['pus cell clumps'] = input("Enter pus cell clumps (notpresent/present): ")
    input_data['bacteria'] = input("Enter bacteria (notpresent/present): ")
    input_data['blood glucose random'] = float(input("Enter blood glucose random: "))
    input_data['blood urea'] = float(input("Enter blood urea: "))
    input_data['serum creatinine'] = float(input("Enter serum creatinine: "))
    input_data['sodium'] = float(input("Enter sodium: "))
    input_data['potassium'] = float(input("Enter potassium: "))
    input_data['haemoglobin'] = float(input("Enter haemoglobin: "))
    input_data['packed cell volume'] = float(input("Enter packed cell volume: "))
    input_data['white blood cell count'] = float(input("Enter white blood cell count: "))
    input_data['red blood cell count'] = float(input("Enter red blood cell count: "))
    input_data['hypertension'] = input("Enter hypertension (yes/no): ")
    input_data['diabetes mellitus'] = input("Enter diabetes mellitus (yes/no): ")
    input_data['coronary artery disease'] = input("Enter coronary artery disease (yes/no): ")
    input_data

['appetite'] = input("Enter appetite (good/poor): ")
    input_data['pedal edema'] = input("Enter pedal edema (yes/no): ")
    input_data['anemia'] = input("Enter anemia (yes/no): ")
    return input_data
```
- **User input function**: Define a function to collect user input for the prediction.

### Predict CKD Based on User Input

```
user_input = get_user_input()
result = predict_ckd(user_input)
print(f"The prediction based on the input is: {result}")
```
- **Prediction**: Collect user input and predict CKD, then print the result.

### Additional Plots

```
print("Generating additional plots for presentation...")
plt.figure(figsize=(20, 15))
for i, feature in enumerate(selected_columns_names[:6]):
    plt.subplot(2, 3, i+1)
    df[feature].hist(bins=30)
    plt.title(feature)
plt.tight_layout()
plt.show()

# Scatter matrix for top 3 features
top_3_features = selected_columns_names[:3]
scatter_matrix = pd.plotting.scatter_matrix(df[top_3_features], figsize=(15, 10), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8)
plt.show()
print("Additional plots generated.\n")
```
- **Additional plots**: Generate histograms for the top 6 features and a scatter matrix for the top 3 features to visualize data distribution and relationships.

