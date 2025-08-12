# 🎓 Student Grant Recommendation App

This is a **Streamlit-based machine learning application** that predicts whether a student should be recommended for a grant based on their academic and behavioral performance.  
The prediction is powered by a **Random Forest Classifier** trained on a dataset containing student grades, obedience, research scores, and project scores.

---

## 📌 Features
- **Grant Recommendation Prediction** based on student details.
- **Prediction Confidence Score** showing how certain the model is about its prediction.
- **Feature Importance Visualization** to understand which features influence the decision.
- **Interactive Inputs** for quick predictions.
- **Model Trained on CSV Dataset** (`student.csv`).

---

## 🛠 Technologies Used
- **Python 3.9+**
- **Streamlit** for the user interface.
- **Pandas** and **NumPy** for data processing.
- **scikit-learn** for machine learning.
- **Joblib** for saving/loading the model and preprocessing objects.

---

## 📂 Project Structure
📦 Student-Grant-App
┣ 📜 app.py # Streamlit application
┣ 📜 train_model.py # Script to train and save the model
┣ 📜 student.csv # Dataset
┣ 📂 Model/
┃ ┣ 📜 student_grant_model.pkl
┃ ┣ 📜 feature_columns.pkl
┣ 📂 Scaler/
┃ ┣ 📜 student_scaler.pkl
┣ 📜 README.md
┣ 📜 requirements.txt



---

## ⚙️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/student-grant-app.git
cd student-grant-app

Install dependencies
pip install -r requirements.txt

Train the model (if not already trained)
python train_model.py

Run the Streamlit app
streamlit run app.py

🧠 How the Model Works
Data Preprocessing
Standardizes numeric features (ResearchScore, ProjectScore) using StandardScaler.
One-hot encodes categorical features (OverallGrade, Obedient).

Model Training
Uses RandomForestClassifier with 100 trees for better accuracy and interpretability.

Prediction
Model predicts "Yes" or "No" for grant recommendation.

Also provides the probability (confidence) for "Yes".
📊 Example Prediction
Input:

Name: John Doe
Grade: A
Obedient: Yes
Research Score: 85
Project Score: 90

Output:
Recommended for Grant ✅


