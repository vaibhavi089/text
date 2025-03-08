from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import string

# Initialize FastAPI app
app = FastAPI()

# Load the model and vectorizer
try:
    model = joblib.load("toxic_classifier.joblib")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

try:
    vectorizer = joblib.load("vectorizer.joblib")
    print("✅ Vectorizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading vectorizer: {e}")
    exit(1)

# Input data model
class TextInput(BaseModel):
    text: str

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Strip spaces
    return text

# API Endpoint
@app.post("/predict/")
async def predict(data: TextInput):
    try:
        processed_text = preprocess_text(data.text)
        text_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(text_vectorized)
        result = "Toxic" if prediction[0] == 1 else "Safe"
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}
