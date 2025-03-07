import streamlit as st
import pickle
import re
import string

# Load the model and vectorizer
with open("toxic_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Strip spaces
    return text

# Streamlit app
st.title("Toxic Text Classification")
st.write("Enter a text to check if it's toxic or safe.")

# Input text box
user_input = st.text_area("Enter your text here:")

if st.button("Predict"):
    if user_input:
        # Preprocess and vectorize the input text
        processed_text = preprocess_text(user_input)
        text_vectorized = vectorizer.transform([processed_text])

        # Make a prediction
        prediction = model.predict(text_vectorized)
        result = "Toxic" if prediction[0] == 1 else "Safe"

        # Display the result
        st.write(f"Prediction: **{result}**")
    else:
        st.write("Please enter some text.")
