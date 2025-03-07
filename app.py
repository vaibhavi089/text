import streamlit as st  # Correct import
import joblib
import re
import string

# Load the model and vectorizer
try:
    model = joblib.load("toxic_classifier.joblib")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

try:
    vectorizer = joblib.load("vectorizer.joblib")
    st.success("Vectorizer loaded successfully!")
except Exception as e:
    st.error(f"Error loading vectorizer: {e}")
    st.stop()

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
        try:
            # Preprocess and vectorize the input text
            processed_text = preprocess_text(user_input)
            text_vectorized = vectorizer.transform([processed_text])

            # Make a prediction
            prediction = model.predict(text_vectorized)
            result = "Toxic" if prediction[0] == 1 else "Safe"

            # Display the result
            st.write(f"Prediction: **{result}**")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.write("Please enter some text.")
