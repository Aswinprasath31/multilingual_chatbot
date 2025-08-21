import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Multilingual Chatbot", layout="centered")
st.title("🌐 Multilingual Chatbot")

# ✅ MBART language code mapping
LANG_CODE_MAP = {
    "en": "en_XX",
    "hi": "hi_IN",
    "ta": "ta_IN",
    "te": "te_IN",
    "fr": "fr_XX",
    "de": "de_DE",
    "es": "es_XX",
    "zh-cn": "zh_CN",
    "zh-tw": "zh_TW",
    "ar": "ar_AR",
}

def detect_language(text):
    try:
        lang = detect(text)
        return LANG_CODE_MAP.get(lang, "en_XX")  # fallback to English
    except Exception:
        return "en_XX"

# -----------------------
# Load Hugging Face Models
# -----------------------
@st.cache_resource
def load_translator():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("translation", model=model, tokenizer=tokenizer)

translator = load_translator()

def translate(text, src, tgt):
    try:
        result = translator(text, src_lang=src, tgt_lang=tgt)
        return result[0]["translation_text"]
    except Exception:
        return text  # fallback

@st.cache_resource
def load_chatbot():
    try:
        return pipeline("text-generation", model="tiiuae/falcon-7b-instruct", tokenizer="tiiuae/falcon-7b-instruct")
    except Exception:
        return pipeline("text2text-generation", model="google/flan-t5-large")

chatbot = load_chatbot()

def hf_answer(prompt):
    try:
        if "text-generation" in str(chatbot.task):
            resp = chatbot(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
            return resp[0]["generated_text"]
        else:
            resp = chatbot(prompt, max_new_tokens=200)
            return resp[0]["generated_text"]
    except Exception as e:
        return f"[HF Error: {e}]"

# -----------------------
# Chat UI
# -----------------------
user_text = st.text_area("✍️ Ask me anything:")

if user_text:
    src_lang = detect_language(user_text)
    st.write(f"Detected Source: **{src_lang}**")

    # Translate to English
    english_text = translate(user_text, src_lang, "en_XX")

    # Get AI answer in English
    answer_en = hf_answer(english_text)

    # Translate back to user’s language
    final_answer = translate(answer_en, "en_XX", src_lang)

    # ✅ Safe mode: if translation didn’t work, show English
    if final_answer.strip() == answer_en.strip():
        st.success(answer_en)
    else:
        st.success(final_answer)

st.markdown("© 2025 Aswinprasath V | Supported by GUVI")
