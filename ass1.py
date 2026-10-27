import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz


raw_data = {
    'Student_ID': [1001, 1002, 1003, 1004, 1005, 1001, 1006, 1007, 1008, 1009],
    'County': ['Nakuru', 'Narok', 'Nakuru', 'Narok', 'Kakamega',
               'Nakuru', 'Nairobi', None, 'Narok', 'Nakuru'],
    'Study_Hours_Per_Week': [15, 8, 20, None, 12, 15, 30, 10, 25, 18],
    'Attendance Percent': [85, 60, 92, 55, 78, 85, 95, 65, 88, None],
    'Fee_Balance_Ksh': [0, 15000, 0, 25000, 5000, 0, 0, 18000, 2000, 12000],
    'Previous_Grade': [72, 48, 85, 45, 65, 72, 90, 50, 82, 58],
    'Student_Type': ['Regular', 'Regular', 'Scholarship', 'Regular',
                     'Scholarship', 'Regular', 'Regular', 'Scholarship',
                     'Regular', 'Regular'],
    'Academic_Risk': [0, 1, 0, 1, 0, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(raw_data)


df.rename(columns={'Attendance Percent': 'Attendance_Percent'}, inplace=True)

print("--- Initial DataFrame Head ---")
print(df.head(), "\n")



print("--- a. Data Loading and Inspection ---")
print(f"Dimensions (df.shape): {df.shape}")
print("\nData Types (df.dtypes):")
print(df.dtypes)
print("\nSummary Statistics (df.describe()):")
print(df.describe())
print("\n" + "="*50 + "\n")



print("--- b. Data Cleaning ---")


print(f"Duplicate rows count: {df.duplicated().sum()}")

df = df.drop_duplicates().reset_index(drop=True)
print(f"Dimensions after dropping duplicates: {df.shape}")


print("\nMissing values per column before cleaning:")
print(df.isnull().sum())



df['County'] = df['County'].fillna(df['County'].mode()[0])
df['Study_Hours_Per_Week'] = df['Study_Hours_Per_Week'].fillna(df['Study_Hours_Per_Week'].median())
df['Attendance_Percent'] = df['Attendance_Percent'].fillna(df['Attendance_Percent'].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\n" + "="*50 + "\n")



print("--- c. Exploratory Data Analysis (EDA) ---")

sns.set_theme(style="whitegrid")


eda_summary = df.groupby('Academic_Risk')[['Study_Hours_Per_Week', 'Attendance_Percent', 'Fee_Balance_Ksh', 'Previous_Grade']].mean()
print("Mean metrics by Academic Risk status:")
print(eda_summary)


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Academic_Risk', hue='Student_Type')
plt.title('Academic Risk by Student Type')
plt.savefig('academic_risk_by_type.png')
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Academic_Risk', y='Previous_Grade')
plt.title('Previous Grade vs Academic Risk')
plt.savefig('previous_grade_vs_risk.png')
plt.close()
print("\nEDA visualizations saved successfully.")
print("\n" + "="*50 + "\n")



print("--- d. Feature Engineering ---")


def categorize_attendance(att):
    if att < 70:
        return 'Low'
    elif att <= 85:
        return 'Medium'
    else:
        return 'High'

df['Attendance_Category'] = df['Attendance_Percent'].apply(categorize_attendance)


df['Has_Fee_Balance'] = (df['Fee_Balance_Ksh'] > 0).astype(int)

print(df[['Student_ID', 'Attendance_Percent', 'Attendance_Category', 'Fee_Balance_Ksh', 'Has_Fee_Balance']].head())
print("\n" + "="*50 + "\n")



print("--- c & e. Encoding and Feature Scaling ---")


numerical_features = ['Study_Hours_Per_Week', 'Attendance_Percent', 'Fee_Balance_Ksh', 'Previous_Grade']
categorical_features = ['County', 'Student_Type', 'Attendance_Category']

X = df.drop(columns=['Student_ID', 'Academic_Risk'])
y = df['Academic_Risk']


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ],
    remainder='passthrough' 
)

X_processed = preprocessor.fit_transform(X)


cat_encoder = preprocessor.named_transformers_['cat']
encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_features))
all_columns = numerical_features + encoded_cat_features + ['Has_Fee_Balance']

X_processed_df = pd.DataFrame(X_processed, columns=all_columns)
print("Processed and Scaled Data Preview:")
print(X_processed_df.head(3))
print("\n" + "="*50 + "\n")



print("--- f. Correlation Heatmap ---")


plt.figure(figsize=(8, 6))
numeric_df = df[numerical_features + ['Academic_Risk', 'Has_Fee_Balance']]
corr_matrix = numeric_df.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

print("Correlation with Academic_Risk:")
print(corr_matrix['Academic_Risk'].sort_values(ascending=False))
print("\n" + "="*50 + "\n")



print("--- g. Decision Tree and Graphviz ---")


clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_processed, y)


dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=all_columns,
    class_names=['Not At Risk (0)', 'At Risk (1)'],
    filled=True,
    rounded=True,
    special_characters=True
)


graph = graphviz.Source(dot_data)
graph.render("student_academic_risk_tree", format="png", cleanup=True)
print("Decision tree trained and exported to 'student_academic_risk_tree.png'.")


importances = pd.Series(clf.feature_importances_, index=all_columns).sort_values(ascending=False)
print("\nFeature Importances from Decision Tree:")
print(importances)
print(f"\nRoot Decision Feature: {importances.index[0]} (Importance: {importances.iloc[0]:.4f})")
