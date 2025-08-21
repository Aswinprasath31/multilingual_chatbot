import streamlit as st

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

st.set_page_config(page_title="Multilingual Chatbot", layout="centered")
st.title("🌐 Multilingual Chatbot")

if missing_libs:
    st.error(f"⚠️ Missing dependencies: {', '.join(missing_libs)}. Please install them.")
    st.stop()

backend = st.sidebar.radio("Choose Backend:", ["HuggingFace", "OpenAI"])
if backend == "OpenAI" and not openai_available:
    st.warning("⚠️ OpenAI package not installed. Falling back to HuggingFace.")
    backend = "HuggingFace"

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
        # ✅ Safe mode: if translation fails, just return original
        return text

@st.cache_resource
def load_hf_chatbot():
    try:
        return pipeline("text-generation", model="tiiuae/falcon-7b-instruct", tokenizer="tiiuae/falcon-7b-instruct")
    except Exception:
        return pipeline("text2text-generation", model="google/flan-t5-large")

hf_chatbot = load_hf_chatbot()

def hf_answer(prompt):
    try:
        if "text-generation" in str(hf_chatbot.task):
            resp = hf_chatbot(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
            return resp[0]["generated_text"]
        else:
            resp = hf_chatbot(prompt, max_new_tokens=200)
            return resp[0]["generated_text"]
    except Exception as e:
        return f"[HF Error: {e}]"

def openai_answer(prompt):
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful multilingual assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[OpenAI Error: {e}]"

# -----------------------
# Chat UI
# -----------------------
user_text = st.text_area("✍️ Ask me anything:")

if user_text:
    src_lang = detect_language(user_text)
    st.write(f"Detected Source: **{src_lang}**")

    english_text = translate(user_text, src_lang, "en_XX")

    if backend == "OpenAI" and openai_available:
        answer_en = openai_answer(english_text)
    else:
        answer_en = hf_answer(english_text)

    final_answer = translate(answer_en, "en_XX", src_lang)

    # ✅ Safe mode: if translation didn’t work, show English answer
    if final_answer.strip() == answer_en.strip():
        st.success(answer_en)
    else:
        st.success(final_answer)

st.markdown("© 2025 Aswinprasath V | Supported by GUVI")
