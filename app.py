import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect

# ---------------------------
# Load translation models
# ---------------------------
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision="main")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, revision="main")
    return tokenizer, model

tokenizer_en_indic, model_en_indic = load_model("ai4bharat/indictrans2-en-indic-1B")
tokenizer_indic_en, model_indic_en = load_model("ai4bharat/indictrans2-indic-en-1B")

# ---------------------------
# Translation helper
# ---------------------------
def translate(text, src, tgt):
    """Translate text between EN and Indic languages with pivot if needed"""
    if not text.strip():
        return text

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
        return text  # same language

    inputs = tokenizer(formatted, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------
# Load free chatbot model (English brain)
# ---------------------------
@st.cache_resource
def load_chat_model():
    return pipeline("text-generation", model="facebook/blenderbot-400M-distill")

chatbot = load_chat_model()

def ask_ai(prompt):
    """Ask the English chatbot"""
    response = chatbot(prompt, max_length=200, do_sample=True, temperature=0.7)
    return response[0]["generated_text"]

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="🌐 Multilingual Chatbot", layout="centered")
st.title("🌐 Multilingual Chatbot")

user_text = st.text_area("✍️ Ask me anything:", height=150)

if st.button("Send"):
    if user_text.strip():
        with st.spinner("Thinking..."):
            try:
                # 1. Detect language
                detected_lang = detect(user_text)

                # 2. Translate → English
                if detected_lang != "en":
                    query_en = translate(user_text, detected_lang, "en")
                else:
                    query_en = user_text

                # 3. Get AI response (in English)
                answer_en = ask_ai(query_en)

                # 4. Translate back to user language
                if detected_lang != "en":
                    answer_user = translate(answer_en, "en", detected_lang)
                else:
                    answer_user = answer_en

                # Show answer
                st.success(answer_user)
                st.info(f"Detected Source Language: {detected_lang.upper()}")

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("⚠️ Please enter some text.")

# Footer
st.markdown("""
<div style="text-align:center; margin-top:20px; font-size:14px; color:#888;">
    &copy; 2025 Aswinprasath V | Supported by GUVI
</div>
""", unsafe_allow_html=True)
