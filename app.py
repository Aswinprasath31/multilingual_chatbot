import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect

# -------------------------------
# Load translation models
# -------------------------------
@st.cache_resource
def load_translation_models():
    model_en_indic = "ai4bharat/indictrans2-en-indic-1B"
    model_indic_en = "ai4bharat/indictrans2-indic-en-1B"

    tokenizer_en_indic = AutoTokenizer.from_pretrained(model_en_indic, trust_remote_code=True, revision="main")
    model_en_indic = AutoModelForSeq2SeqLM.from_pretrained(model_en_indic, trust_remote_code=True, revision="main")

    tokenizer_indic_en = AutoTokenizer.from_pretrained(model_indic_en, trust_remote_code=True, revision="main")
    model_indic_en = AutoModelForSeq2SeqLM.from_pretrained(model_indic_en, trust_remote_code=True, revision="main")

    return tokenizer_en_indic, model_en_indic, tokenizer_indic_en, model_indic_en

tokenizer_en_indic, model_en_indic, tokenizer_indic_en, model_indic_en = load_translation_models()

# -------------------------------
# Translation helper
# -------------------------------
def translate(text, src, tgt):
    if not text.strip():
        return text

    if src == "en" and tgt != "en":
        tokenizer, model = tokenizer_en_indic, model_en_indic
        formatted = f">>{tgt}<< {text}"
    elif src != "en" and tgt == "en":
        tokenizer, model = tokenizer_indic_en, model_indic_en
        formatted = text  # ✅ no >>en<< tag
    elif src != "en" and tgt != "en":
        mid = translate(text, src, "en")
        return translate(mid, "en", tgt)
    else:
        return text

    inputs = tokenizer(formatted, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------------
# Load LLM (Hugging Face)
# -------------------------------
@st.cache_resource
def load_llm():
    model_name = "google/gemma-2b-it"  # small instruction-tuned model
    generator = pipeline("text-generation", model=model_name, device=0 if st.secrets.get("use_gpu", False) else -1)
    return generator

llm = load_llm()

def chatbot_response(query_en):
    """Get response from LLM in English"""
    response = llm(query_en, max_new_tokens=200, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Multilingual Chatbot", layout="centered")
st.title("🌐 Multilingual Chatbot")

user_text = st.text_area("✍️ Ask me anything:", height=150)

if user_text.strip():
    try:
        # Detect user language
        detected_lang = detect(user_text)
        st.write(f"Detected Source: **{detected_lang.upper()}**")

        # Translate to English
        if detected_lang != "en":
            text_en = translate(user_text, detected_lang, "en")
        else:
            text_en = user_text

        # Generate response in English (LLM)
        bot_reply_en = chatbot_response(text_en)

        # Translate response back to user language
        if detected_lang != "en":
            bot_reply = translate(bot_reply_en, "en", detected_lang)
        else:
            bot_reply = bot_reply_en

        st.success(bot_reply)

    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("""
    <div style='text-align: center; margin-top: 20px; font-size: 14px; color: #888;'>
        &copy; 2025 Aswinprasath V | Supported by GUVI
    </div>
""", unsafe_allow_html=True)
