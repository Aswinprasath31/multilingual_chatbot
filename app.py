import streamlit as st

# Try safe imports
missing_libs = []
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
except ImportError:
    missing_libs.append("transformers")

try:
    from langdetect import detect
except ImportError:
    missing_libs.append("langdetect")

openai_available = True
try:
    import openai
except ImportError:
    openai_available = False

# Streamlit UI
st.set_page_config(page_title="Multilingual Chatbot", layout="centered")
st.title("🌐 Multilingual Chatbot")

if missing_libs:
    st.error(f"⚠️ Missing dependencies: {', '.join(missing_libs)}. Please install them.")
    st.stop()

backend = st.sidebar.radio("Choose Backend:", ["HuggingFace", "OpenAI"])
if backend == "OpenAI" and not openai_available:
    st.warning("⚠️ OpenAI package not installed. Falling back to HuggingFace.")
    backend = "HuggingFace"

# -----------------------
# Helpers
# -----------------------
def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "en"  # fallback

@st.cache_resource
def load_hf_translator():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("translation", model=model, tokenizer=tokenizer)

translator = load_hf_translator()

def translate(text, src, tgt):
    try:
        result = translator(text, src_lang=src, tgt_lang=tgt)
        return result[0]["translation_text"]
    except Exception:
        return text

def openai_answer(prompt):
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful multilingual assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=300
    )
    return resp["choices"][0]["message"]["content"]

# -----------------------
# Chat UI
# -----------------------
user_text = st.text_area("✍️ Ask me anything:")

if user_text:
    # Step 1: Detect user language
    src_lang = detect_language(user_text)
    st.write(f"Detected Source: **{src_lang.upper()}**")

    # Step 2: Translate to English (pivot language)
    english_text = translate(user_text, src_lang, "en_XX")

    # Step 3: Get answer in English
    if backend == "OpenAI" and openai_available:
        answer_en = openai_answer(english_text)
    else:
        # Dummy fallback: Echo the text
        answer_en = f"(HF backend) Answer to: {english_text}"

    # Step 4: Translate answer back to user language
    final_answer = translate(answer_en, "en_XX", src_lang)

    # Show result
    st.success(final_answer)

st.markdown("© 2025 Aswinprasath V | Supported by GUVI")
