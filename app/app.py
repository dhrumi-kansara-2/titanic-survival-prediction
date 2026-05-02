import streamlit as st
import pickle
import numpy as np

# Load model
with open("model/titanic_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Titanic Survival Predictor")

st.title("Titanic Survival Predictor")
st.write("Enter passenger details to predict survival.")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex_input = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouse Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 30.0)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Convert inputs
sex = 0 if sex_input == "male" else 1
embarked_q = 1 if embarked == "Q" else 0
embarked_s = 1 if embarked == "S" else 0

family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# IMPORTANT: order must match your training X columns
input_data = np.array([[
    pclass,
    sex,
    age,
    sibsp,
    parch,
    fare,
    embarked_q,
    embarked_s,
    family_size,
    is_alone
]])

if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    not_survived_prob = probability[0]
    survived_prob = probability[1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Passenger likely survived.")
    else:
        st.error("Passenger likely did not survive.")

    st.write(f"Survived probability: **{survived_prob:.2%}**")
    st.write(f"Not survived probability: **{not_survived_prob:.2%}**")

    st.caption("Prediction based on a trained Random Forest model.")