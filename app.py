import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💵",
    layout="centered"
)


# --- Load Model and Columns (Cached for performance) ---
@st.cache_resource
def load_model_and_columns():
    model = joblib.load('employee_salary_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, model_columns


model, model_columns = load_model_and_columns()


# --- Load Dataset (Cached for performance) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('adult 3.csv')
    except FileNotFoundError:
        st.error("Error: 'adult 3.csv' not found. Please ensure the CSV file is in the same directory as app.py.")
        return pd.DataFrame()

    # Apply the same data cleaning as in your notebook
    for col in ['workclass', 'occupation', 'native-country']:
        df[col] = df[col].replace('?', 'Others')
    df.drop(df[df['workclass'] == 'Without-pay'].index, inplace=True)
    df.drop(df[df['workclass'] == 'Never-worked'].index, inplace=True)
    return df


df = load_data()

# --- Calculate the mean for fnlwgt for accurate predictions ---
if not df.empty:
    fnlwgt_mean = df['fnlwgt'].mean()
else:
    fnlwgt_mean = 189778  # A default fallback mean

# --- Application Title and Description ---
st.title("💵 EMPLOYEE SALARY PREDICTION USING ML")
st.markdown(
    "This application utilizes a machine learning model to predict whether an individual's salary is above or below $50,000. "
    "By inputting various demographic and employment details, the system provides a real-time prediction."
)
st.markdown("---")

# --- User Input Section ---
st.header("Enter Employee Details")

if not df.empty:
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('**Age**', 17, 90, 45)  # Default to Profile 1
        workclass = st.selectbox('**Workclass**', df['workclass'].unique(), index=2)  # Default to 'Private'
        educational_num = st.slider('**Educational Number**', 1, 16, 13)  # Default to 13 (Bachelors)
        occupation = st.selectbox('**Occupation**', df['occupation'].unique(), index=3)  # Default to 'Exec-managerial'

    with col2:
        marital_status = st.selectbox('**Marital Status**', df['marital-status'].unique(),
                                      index=2)  # Default to 'Married-civ-spouse'
        relationship = st.selectbox('**Relationship**', df['relationship'].unique(), index=0)  # Default to 'Husband'
        race = st.selectbox('**Race**', df['race'].unique(), index=4)  # Default to 'White'
        gender = st.selectbox('**Gender**', ["Male", "Female"])

    with col3:
        capital_gain = st.number_input('**Capital Gain**', min_value=0, value=5000)  # Default to Profile 1
        capital_loss = st.number_input('**Capital Loss**', min_value=0, value=0)
        hours_per_week = st.number_input('**Hours per Week**', 1, 99, 50)  # Default to Profile 1
        native_country = st.selectbox('**Native Country**', df['native-country'].unique(),
                                      index=38)  # Default to 'United-States'
else:
    st.warning("Dataset could not be loaded. Please check for 'adult 3.csv'.")

# --- Prediction Logic and Button ---
if not df.empty:
    button_col1, button_col2, button_col3 = st.columns([1, 1.5, 1])
    with button_col2:
        if st.button('**PREDICT**', type="primary", use_container_width=True):

            # --- COMPLETELY REVISED AND CORRECTED PREDICTION LOGIC ---

            # 1. Create a DataFrame from user input with the correct raw structure
            input_data = pd.DataFrame({
                'age': [age],
                'workclass': [workclass],
                'fnlwgt': [fnlwgt_mean],
                'educational-num': [educational_num],
                'marital-status': [marital_status],
                'occupation': [occupation],
                'relationship': [relationship],
                'race': [race],
                'gender': [gender],
                'capital-gain': [capital_gain],
                'capital-loss': [capital_loss],
                'hours-per-week': [hours_per_week],
                'native-country': [native_country]
            })

            # 2. Apply one-hot encoding to the input DataFrame
            input_dummies = pd.get_dummies(input_data, drop_first=True)

            # 3. Align the columns with the model's training columns.
            input_aligned = input_dummies.reindex(columns=model_columns, fill_value=0)

            # 4. Make the prediction
            prediction = model.predict(input_aligned)
            prediction_proba = model.predict_proba(input_aligned)

            # 5. Display the result
            st.markdown("---")
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.success(f"**Predicted Salary: > $50K** (Confidence: {prediction_proba[0][1]:.2%})")
                st.balloons()
            else:
                st.error(f"**Predicted Salary: <= $50K** (Confidence: {prediction_proba[0][0]:.2%})")

# --- Exploratory Data Analysis Section ---
if not df.empty:
    st.markdown("---")
    with st.expander("📊 **Explore the Dataset**"):
        st.write("Visualizations from the dataset:")
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("#### Income Distribution")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.countplot(x='income', data=df, ax=ax1, palette='viridis')
            st.pyplot(fig1)

        with fig_col2:
            st.markdown("#### Age Distribution by Income")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.histplot(data=df, x='age', hue='income', kde=True, multiple="stack", ax=ax2)
            st.pyplot(fig2)