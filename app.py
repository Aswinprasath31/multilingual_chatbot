import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langdetect import detect

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="Multilanguage Chatbot", layout="centered")
st.title("🌐 Multilanguage Chatbot")

# ---------------------------
# Language info (codes → flags + names)
# ---------------------------
LANG_INFO = {
    "en": {"flag": "🇬🇧", "name": "English"},
    "hi": {"flag": "🇮🇳", "name": "Hindi"},
    "ta": {"flag": "🇮🇳", "name": "Tamil"},
    "te": {"flag": "🇮🇳", "name": "Telugu"},
    "bn": {"flag": "🇧🇩", "name": "Bengali"},
    "ml": {"flag": "🇮🇳", "name": "Malayalam"},
    "kn": {"flag": "🇮🇳", "name": "Kannada"},
    "gu": {"flag": "🇮🇳", "name": "Gujarati"},
    "mr": {"flag": "🇮🇳", "name": "Marathi"},
    "pa": {"flag": "🇮🇳", "name": "Punjabi"},
    "or": {"flag": "🇮🇳", "name": "Odia"},
    "unknown": {"flag": "❓", "name": "Unknown"},
    "not detected": {"flag": "⚪", "name": "Not Detected"}
}

# ---------------------------
# Load IndicTrans2 model
# ---------------------------
@st.cache_resource
def load_model():
    """Load the IndicTrans2 translation model."""
    model_name = "ai4bharat/indictrans2-en-indic-1B"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision="main")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, revision="main")
        return pipeline("translation", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

translator = load_model()

# ---------------------------
# Translation function
# ---------------------------
def translate_text(text, tgt_lang: str):
    """
    Translate text using detected source language and target language.
    """
    if not text.strip():
        return None, "⚠️ Please enter some text to translate."

    src_lang = st.session_state.get("detected_lang", None)

    if not src_lang or src_lang in ["unknown", "not detected"]:
        return None, "⚠️ Could not detect source language."

    if src_lang == tgt_lang:
        return None, "⚠️ Source and target languages cannot be the same."

    try:
        # Add target language token
        if tgt_lang == "en":
            formatted_text = f">>en<< {text}"
        else:
            formatted_text = f">>{tgt_lang}<< {text}"

        output = translator(formatted_text)
        return output[0]["translation_text"], None
    except Exception as e:
        return None, f"Unexpected error: {e}"

# ---------------------------
# Sidebar controls
# ---------------------------
languages = ["en", "hi", "ta", "te", "bn", "ml", "kn", "gu", "mr", "pa", "or"]

# Build options with flag + code + name
lang_options = [f"{LANG_INFO[code]['flag']} {code.upper()} ({LANG_INFO[code]['name']})" for code in languages]

# Dropdown for target language
selected_label = st.sidebar.selectbox("Target Language", lang_options, index=1)

# Extract code back from label
tgt_lang = selected_label.split(" ")[1].lower()

# Show detected source language in sidebar
if "detected_lang" not in st.session_state:
    st.session_state.detected_lang = "not detected"

detected_lang = st.session_state.detected_lang
flag = LANG_INFO.get(detected_lang, {"flag": "❓", "name": "Unknown"})["flag"]
name = LANG_INFO.get(detected_lang, {"flag": "❓", "name": "Unknown"})["name"]

st.sidebar.markdown(f"**Detected Source:** {flag} {detected_lang.upper()} ({name})")

# ---------------------------
# Main input
# ---------------------------
user_text = st.text_area("✍️ Enter text to translate:", height=150)

# Auto-detect language as user types
if user_text.strip():
    try:
        detected_lang = detect(user_text)
        st.session_state.detected_lang = detected_lang
    except:
        st.session_state.detected_lang = "unknown"
else:
    st.session_state.detected_lang = "not detected"

# ---------------------------
# Translation button + layout
# ---------------------------
if st.button("Translate"):
    with st.spinner("Translating..."):
        translation, error = translate_text(user_text, tgt_lang)

        if error:
            st.error(error)
        else:
            # Two-column layout like Google Translate
            col1, col2 = st.columns(2)

            with col1:
                src_flag = LANG_INFO[st.session_state.detected_lang]["flag"]
                src_name = LANG_INFO[st.session_state.detected_lang]["name"]
                st.markdown(f"**Source ({src_flag} {st.session_state.detected_lang.upper()} – {src_name})**")
                st.info(user_text)

            with col2:
                tgt_flag = LANG_INFO[tgt_lang]["flag"]
                tgt_name = LANG_INFO[tgt_lang]["name"]
                st.markdown(f"**Target ({tgt_flag} {tgt_lang.upper()} – {tgt_name})**")
                st.success(translation)

# ---------------------------
# Clear button
# ---------------------------
if st.button("Clear"):
    st.session_state.detected_lang = "not detected"
    st.experimental_rerun()

# ---------------------------
# Footer
# ---------------------------
st.markdown("""
    <div style="text-align: center; margin-top: 20px; font-size: 14px; color: #888;">
        &copy; 2025 Aswinprasath V | Supported by GUVI
    </div>
""", unsafe_allow_html=True)
