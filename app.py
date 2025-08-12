# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and features
model = joblib.load('student_grant_model.pkl')
scaler = joblib.load('student_scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Title
st.title("🎓 Student Grant Recommendation App")
st.write("Fill in the student details below:")

# Inputs
name = st.text_input("Student Name", "John Doe")
overall_grade = st.selectbox("Overall Grade", ['A', 'B', 'C', 'D', 'E', 'F'])
obedient = st.selectbox("Obedient?", ['Y', 'N'])
research_score = st.slider("Research Score", 0, 100, 50)
project_score = st.slider("Project Score", 0, 100, 50)

if st.button("Predict Grant Recommendation"):
    # Map grade to numeric order
    grade_mapping = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'F': 1}
    overall_grade_num = grade_mapping[overall_grade]

    # Map obedience to binary
    obedient_num = 1 if obedient == 'Y' else 0

    # Calculate average score
    average_score = (research_score + project_score) / 2

    # Create DataFrame in the same format as training
    input_df = pd.DataFrame([{
        'OverallGrade': overall_grade_num,
        'Obedient': obedient_num,
        'ResearchScore': research_score,
        'ProjectScore': project_score,
        'AverageScore': average_score
    }])

    # Scale numeric features (entire input_df since all are numeric now)
    input_df = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_df)[0]  # 1 or 0

    # Display result
    if prediction == 1:
        st.success(f"🎯 Student '{name}' is **Recommended** for the grant.")
    else:
        st.error(f"❌ Student '{name}' is **Not Recommended** for the grant.")
