from fastapi import FastAPI
from pydantic import BaseModel
import pickle  # Use pickle for .pkl files
import re
import string

# Initialize FastAPI app
app = FastAPI()

# Load the model and vectorizer using pickle
try:
    with open("toxic_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

try:
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("Vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading vectorizer: {e}")
    exit()

# Define request body schema
class TextInput(BaseModel):
    text: str

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Strip spaces
    return text

# Prediction endpoint
@app.post("/predict/")
async def predict(input_text: TextInput):
    try:
        # Preprocess and vectorize the input text
        processed_text = preprocess_text(input_text.text)
        text_vectorized = vectorizer.transform([processed_text])

        # Make a prediction
        prediction = model.predict(text_vectorized)
        result = "Toxic" if prediction[0] == 1 else "Safe"

        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Toxic Text Classification API is running!"}
