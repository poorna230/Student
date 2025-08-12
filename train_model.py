import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os

# ===== Load dataset =====
df = pd.read_csv('student.csv')

# ===== Preprocessing =====
# Remove duplicates
df = df.drop_duplicates()

# Handle missing values (if any)
df = df.dropna()

# Map grades to numeric order
grade_mapping = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1}
df['OverallGrade'] = df['OverallGrade'].map(grade_mapping)

# Encode Obedient: Y=1, N=0
df['Obedient'] = df['Obedient'].map({'Y': 1, 'N': 0})

# Encode Recommend (target): Yes=1, No=0
df['Recommend'] = df['Recommend'].map({'Yes': 1, 'No': 0})

# Create Average Score feature
df['AverageScore'] = (df['ResearchScore'] + df['ProjectScore']) / 2

# ===== Feature selection =====
feature_names = ['OverallGrade', 'Obedient', 'ResearchScore', 'ProjectScore', 'AverageScore']
X = df[feature_names]
y = df['Recommend']

# Scale numeric features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ===== Train-test split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== Train model =====
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ===== Predictions =====
y_pred = model.predict(X_test)

# ===== Metrics =====
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
conf_matrix = confusion_matrix(y_test, y_pred)

print("📊 Model Performance Metrics")
print(f"Accuracy:  {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall:    {recall:.2f}%")
print(f"F1 Score:  {f1:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ===== Save model =====
os.makedirs('Model', exist_ok=True)
os.makedirs('Scaler', exist_ok=True)

joblib.dump(model, 'student_grant_model.pkl')
joblib.dump(scaler, 'student_scaler.pkl')
joblib.dump(feature_names, 'feature_columns.pkl')

print("\n✅ Model, scaler, and feature columns saved successfully.")
