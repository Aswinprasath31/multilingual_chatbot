import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# App title
st.set_page_config(page_title="Multilanguage Chatbot", layout="centered")
st.title("🌐 Multilanguage Chatbot ")

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

# Custom CSS for styling
st.markdown("""
    <style>
    .stButton {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        transition: background-color 0.3s;
    }
    .stButton:hover {
        background-color: #45a049;
    }
    .stTextArea {
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 10px;
    }
    .stSelectbox {
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 10px;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 14px;
        color: #888;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar: language options
st.sidebar.header("⚙️ Settings")
src_lang = st.sidebar.selectbox("Source Language", ["en"])
tgt_lang = st.sidebar.selectbox("Target Language", ["hi", "ta", "te", "bn", "ml", "kn", "gu", "mr", "pa", "or"])

# Input box
user_text = st.text_area("✍️ Enter text to translate:", height=150)

if st.button("Translate"):
    if user_text.strip():
        with st.spinner("Translating..."):
            # Prepare the input for the model
            try:
                # Directly pass the user input to the translator
                output = translator(user_text, src_lang=src_lang, tgt_lang=tgt_lang)
                st.success("Translation: " + output[0]["translation_text"])
            except AssertionError as e:
                st.error(f"Error during translation: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("⚠️ Please enter some text to translate.")

# Clear button
if st.button("Clear"):
    st.experimental_rerun()

# Footer with copyright notice
st.markdown("""
    <div class="footer">
        &copy; 2025 Aswinprasath V | Supported by GUVI
    </div>
""", unsafe_allow_html=True)
