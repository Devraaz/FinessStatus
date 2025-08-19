# 🚴 Fitness Status Prediction App

This is a **Streamlit web application** that predicts your **fitness status** based on your daily habits, health indicators, and lifestyle.  
It uses **multiple machine learning models** (Logistic Regression, Support Vector Classifier, and Random Forest) and combines their predictions using a **majority voting system**.

---

## 📌 Features

- Input your health and lifestyle details:

  - Age
  - Gender
  - Height (cm)
  - Weight (kg)
  - Heart Rate
  - Blood Pressure
  - Sleep Hours
  - Nutrition Quality
  - Activity Index (0–5)
  - Smoking Habit

- Encodes categorical inputs (`gender`, `smokes`) automatically.
- Runs predictions using:
  - Logistic Regression
  - Support Vector Machine (SVC)
  - Random Forest
- Displays:
  - Individual predictions from each model
  - Final result based on **majority vote**

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git https://github.com/Devraaz/Fitness-Status.git
cd fitness-prediction-app
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

```

Then install requirements:

```bash
pip install -r requirements.txt
```

### 3. Required Files

- fitness_dataset.csv → The dataset used for encoding categorical features.

- le.pkl → Trained LabelEncoder pickle file.

- lr.pkl → Logistic Regression model pickle.

- svc.pkl → Support Vector Classifier pickle.

- rf.pkl → Random Forest model pickle.

Make sure these files are in the root directory.

## ▶️ Running the App

```bash
streamlit run app.py
```

## 📊 Example Output

- Logistic Regression prediction: Fit

- SVM prediction: Unfit

- Random Forest prediction: Fit

✅ Final Decision: Fit (Majority Vote)

## 🧑‍💻 Tech Stack

- Python

- Streamlit

- Pandas, NumPy

- Scikit-learn (ML models)

## 🔗 Live Demo

👉 [Click here to try it out on Streamlit](https://devraaz-fitness-status.streamlit.app/)

---

## 💻 GitHub Repo

🔗 [View Full Source Code on GitHub](https://github.com/Devraaz/Fitness-Status)

## 🙋‍♂️ Author

Made with ❤️ by [Devraj Dora](https://github.com/devraaz)
