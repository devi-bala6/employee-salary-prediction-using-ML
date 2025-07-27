# Employee Salary Prediction using Machine Learning



This project is a web application that predicts whether an individual's annual income is likely to be more or less than $50,000. It uses a machine learning model trained on the Adult Census Income dataset from the UCI Machine Learning Repository.



![Salary Prediction App Screenshot](placeholder_for_your_screenshot.png)

*(You should add a screenshot of your running Streamlit app here)*



## 📋 Table of Contents

* [Features](#✨-features)

* [Dataset](#📊-dataset)

* [Technologies & Libraries](#🛠️-technologies--libraries)

* [How to Run](#🚀-how-to-run)

* [File Structure](#📁-file-structure)



## ✨ Features

* **Data Cleaning & Preprocessing:** The dataset is cleaned by handling missing values and removing irrelevant categories.

* **Model Training:** A **LightGBM Classifier** is trained on the data to achieve high accuracy and fast performance.

* **Interactive Web App:** A user-friendly interface built with **Streamlit** allows for real-time predictions based on user input.

* **Data Visualization:** The app includes an optional section with graphs and charts to explore the underlying dataset.



## 📊 Dataset

This project uses the **Adult Census Income** dataset from the UCI Machine Learning Repository. It contains 15 features, including age, workclass, education, and other demographic information.



* **Source:** [UCI Machine Learning Repository: Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)



## 🛠️ Technologies & Libraries

The project is built using the following technologies and libraries:

* **Python**

* **Jupyter Notebook** (for model development)

* **Streamlit** (for the web application)

* **Pandas** (for data manipulation)

* **Scikit-learn** (for data preprocessing and evaluation)

* **LightGBM** (for the prediction model)

* **Seaborn & Matplotlib** (for data visualization)

* **Joblib** (for saving the model)

🚀 How to Run
To run this project on your local machine, follow these steps:

1. Clone the Repository
Open your terminal or command prompt and clone this repository to your local machine:
git clone https://github.com/YourUsername/your-repo-name.git
cd your-repo-name
2. Install Dependencies
It's recommended to create a Python virtual environment first. Once your environment is activated, install the required libraries:
pip install streamlit pandas scikit-learn lightgbm seaborn matplotlib joblib
3. Run the Streamlit App
Make sure you are in the project's root directory where the app.py file is located. Run the following command in your terminal:
streamlit run app.py
Your web browser will automatically open with the running application.

📁 File Structure
├── employee_salary_model.pkl    # The saved, trained machine learning model
├── model_columns.pkl            # The list of columns the model was trained on
├── app.py                       # The Python script for the Streamlit web application
├── adult 3.csv                  # The dataset file
├── employee salary prediction using ML.ipynb # The Jupyter Notebook for model development
└── README.md                    # This file






