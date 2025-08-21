import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect
import torch
import openai

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="Multilingual Chatbot", layout="centered")
st.title("🌐 Multilingual Chatbot")

# ----------------------------
# Supported Languages
# ----------------------------
SUPPORTED_LANGS = ["as","bn","gu","hi","kn","ml","mr","or","pa","ta","te","en"]

def normalize_lang(code: str) -> str:
    """Ensure detected language is mapped to IndicTrans2 supported codes."""
    code = code.lower()
    if code in SUPPORTED_LANGS:
        return code
    mapping = {
        "en": "en","hi": "hi","ta": "ta","te": "te","bn": "bn","ml": "ml",
        "kn": "kn","gu": "gu","mr": "mr","pa": "pa","or": "or","as": "as",
    }
    return mapping.get(code, "en")  # fallback to English

# ----------------------------
# Load Translation Models
# ----------------------------
@st.cache_resource
def load_models():
    en_indic = "ai4bharat/indictrans2-en-indic-1B"
    indic_en = "ai4bharat/indictrans2-indic-en-1B"

    tok_en_indic = AutoTokenizer.from_pretrained(en_indic, trust_remote_code=True)
    model_en_indic = AutoModelForSeq2SeqLM.from_pretrained(en_indic, trust_remote_code=True)

    tok_indic_en = AutoTokenizer.from_pretrained(indic_en, trust_remote_code=True)
    model_indic_en = AutoModelForSeq2SeqLM.from_pretrained(indic_en, trust_remote_code=True)

    return tok_en_indic, model_en_indic, tok_indic_en, model_indic_en

tok_en_indic, model_en_indic, tok_indic_en, model_indic_en = load_models()

# ----------------------------
# Translation Function
# ----------------------------
def translate(text, src, tgt):
    if not text.strip():
        return text

    if src == "en" and tgt != "en":
        tokenizer, model = tok_en_indic, model_en_indic
        formatted = f">>{tgt}<< {text}"
    elif src != "en" and tgt == "en":
        tokenizer, model = tok_indic_en, model_indic_en
        formatted = text
    elif src != "en" and tgt != "en":
        mid = translate(text, src, "en")
        return translate(mid, "en", tgt)
    else:
        return text

    inputs = tokenizer(formatted, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ----------------------------
# Chatbot Backends
# ----------------------------
@st.cache_resource
def load_hf_pipeline():
    return pipeline("text-generation", model="tiiuae/falcon-7b-instruct", trust_remote_code=True, device_map="auto")

def chatbot_reply(text, backend, api_key=None):
    if backend == "HuggingFace":
        generator = load_hf_pipeline()
        out = generator(text, max_length=200, do_sample=True, temperature=0.7)
        return out[0]["generated_text"]

    elif backend == "OpenAI":
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": text}]
        )
        return resp["choices"][0]["message"]["content"]

    else:
        return "❌ Invalid backend selected."

# ----------------------------
# UI
# ----------------------------
backend = st.sidebar.radio("Choose Backend:", ["HuggingFace", "OpenAI"])

api_key = None
if backend == "OpenAI":
    api_key = st.sidebar.text_input("🔑 Enter OpenAI API Key", type="password")

user_text = st.text_area("✍️ Ask me anything:")

if user_text:
    detected = detect(user_text)
    src_lang = normalize_lang(detected)
    st.write(f"Detected Source: **{src_lang.upper()}**")

    tgt_lang = st.selectbox("Choose Target Language", SUPPORTED_LANGS, index=0)

    try:
        # Step 1: Translate user input to English
        if src_lang != "en":
            text_en = translate(user_text, src_lang, "en")
        else:
            text_en = user_text

        # Step 2: Get AI reply in English
        reply_en = chatbot_reply(text_en, backend, api_key)

        # Step 3: Translate back to source language
        if src_lang != "en":
            reply_final = translate(reply_en, "en", src_lang)
        else:
            reply_final = reply_en

        st.success(f"🤖 {reply_final}")

    except Exception as e:
        st.error(f"Error: {e}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("""
    <div style="text-align:center;margin-top:20px;font-size:14px;color:#888;">
        © 2025 Aswinprasath V | Supported by GUVI
    </div>
""", unsafe_allow_html=True)
