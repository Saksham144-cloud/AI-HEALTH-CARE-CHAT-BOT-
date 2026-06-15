# AI-HEALTH-CARE-CHAT-BOT-
Developed an AI-powered Healthcare Chatbot designed to provide users with instant health guidance, symptom assessment, and medical information through natural language conversations the chatbot uses  Artificial Intelligence and Natural Language Processing (NLP) .
## Features
- Emotion detection using Transformer model (DistilBERT)
- Trained on GoEmotions dataset
- Interactive chatbot UI using Streamlit

## Tech Stack
- Python
- PyTorch
- Transformers
- Streamlit

## How to Run

1. Install dependencies:
pip install torch transformers streamlit datasets scikit-learn

2. Train the model:
python train.py

3. Run the chatbot:
streamlit run app.py

## Project Structure
calmbot/
│── train.py
│── app.py
│── calmbot_model/
│── README.md
