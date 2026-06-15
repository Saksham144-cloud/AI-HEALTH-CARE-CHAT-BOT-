import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

st.title("🧠 Calmbot 2.0 – AI Mental Health Assistant")

# Device
device = torch.device("cpu")

# Load model
model = DistilBertForSequenceClassification.from_pretrained("calmbot_model")
model.to(device)
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("calmbot_model")

# Emotion labels
emotion_labels = [
    "admiration","amusement","anger","annoyance","approval","caring",
    "confusion","curiosity","desire","disappointment","disapproval",
    "disgust","embarrassment","excitement","fear","gratitude","grief",
    "joy","love","nervousness","optimism","pride","realization",
    "relief","remorse","sadness","surprise","neutral"
]

user_input = st.text_area("How are you feeling today?")

if st.button("Analyze"):
    if user_input:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_class = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
        emotion = emotion_labels[predicted_class]

        st.subheader(f"Detected Emotion: {emotion}")

        # Smart responses
        if emotion in ["sadness", "grief", "remorse"]:
            st.write("I'm really sorry you're feeling this way. Do you want to talk about it? ❤️")
        elif emotion in ["anger", "annoyance"]:
            st.write("It sounds frustrating. Want to share what’s bothering you?")
        elif emotion in ["joy", "love", "gratitude"]:
            st.write("That’s amazing to hear! 😊 Keep enjoying the moment!")
        elif emotion in ["fear", "nervousness"]:
            st.write("It’s okay to feel this way. You’re not alone.")
        else:
            st.write("I’m here for you. Tell me more.")

    else:
        st.warning("Please enter some text.")
        
