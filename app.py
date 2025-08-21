import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import openai  # if you use OpenAI API

# ---- Load models once ----
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision="main")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, revision="main")
    return tokenizer, model

tokenizer_en_indic, model_en_indic = load_model("ai4bharat/indictrans2-en-indic-1B")
tokenizer_indic_en, model_indic_en = load_model("ai4bharat/indictrans2-indic-en-1B")

# ---- Translation functions ----
def translate(text, src, tgt):
    """Translate text between EN and Indic languages"""
    if src == "en" and tgt != "en":
        tokenizer, model = tokenizer_en_indic, model_en_indic
        formatted = f">>{tgt}<< {text}"
    elif src != "en" and tgt == "en":
        tokenizer, model = tokenizer_indic_en, model_indic_en
        formatted = f">>en<< {text}"
    elif src != "en" and tgt != "en":
        # pivot through English
        mid = translate(text, src, "en")
        return translate(mid, "en", tgt)
    else:
        return text

    inputs = tokenizer(formatted, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---- Chatbot brain (English) ----
def ask_ai(prompt):
    # Example with OpenAI (replace with Hugging Face if needed)
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": prompt}]
    )
    return resp["choices"][0]["message"]["content"]

# ---- Streamlit UI ----
st.set_page_config(page_title="🌐 Multilingual Chatbot", layout="centered")
st.title("🌐 Multilingual Chatbot")

user_text = st.text_area("✍️ Ask me anything:")

if st.button("Send"):
    if user_text.strip():
        with st.spinner("Thinking..."):
            try:
                # Detect language
                detected_lang = detect(user_text)

                # Translate user text -> English
                if detected_lang != "en":
                    query_en = translate(user_text, detected_lang, "en")
                else:
                    query_en = user_text

                # Get AI response in English
                answer_en = ask_ai(query_en)

                # Translate back to user’s lang
                if detected_lang != "en":
                    answer_user = translate(answer_en, "en", detected_lang)
                else:
                    answer_user = answer_en

                st.success(answer_user)

                st.info(f"Detected Source: {detected_lang.upper()}")

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("⚠️ Please enter some text.")
