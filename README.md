# GUVI Multilingual GPT Chatbot

This project is a multilingual chatbot built with Streamlit and Hugging Face Transformers. It can understand user queries in various languages, process them, and respond in the user's original language.

## 🚀 About This Project

[cite_start]The primary goal of this chatbot is to assist learners on platforms like GUVI by providing real-time, multilingual support[cite: 7]. [cite_start]It helps break down language barriers for non-English-speaking users[cite: 19].

## ⚙️ How It Works

The chatbot follows a three-step process to answer a user's query:
1.  [cite_start]**Translate to English**: The user's input, if in a non-English language, is first translated into English using the `facebook/mbart-large-50-many-to-many-mnli` model[cite: 6].
2.  [cite_start]**Generate Response**: The translated English query is passed to a pre-trained GPT-2 model to generate a relevant response[cite: 5].
3.  [cite_start]**Translate Back**: The English response from the model is then translated back into the user's original language before being displayed[cite: 6].

## 🛠️ Tech Stack
- Python
- Streamlit (for the web interface)
- Hugging Face Transformers (for AI models)
- PyTorch

## 📋 How to Run This Project

1.  **Clone the Repository**
    ```bash
    git clone <your-github-repo-url>
    cd <your-repo-name>
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit App**
    ```bash
    streamlit run app.py
    ```
The application will then be running and accessible in your web browser.

## 🔗 Deployment

This application is deployed on Hugging Face Spaces and can be accessed here:

[**<< PASTE YOUR HUGGING FACE SPACES LINK HERE >>**]
