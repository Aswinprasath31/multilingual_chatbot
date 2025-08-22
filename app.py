# 1. IMPORT NECESSARY LIBRARIES
import streamlit as st
from transformers import pipeline

# 2. SET UP THE USER INTERFACE (UI)
st.set_page_config(layout="wide") # Use the full screen width
st.title("GUVI Multilingual AI Chatbot 🤖")
st.write("This chatbot can answer your questions in different languages. Select a language and ask anything!")

# 3. LOAD THE AI MODELS (using a cache to speed up the app)
# This is a special Streamlit command that caches the models so they don't reload every time you do something.
@st.cache_resource
def load_models():
    """Loads and returns the translation and text generation pipelines."""
    # The translator can translate between 50 different languages.
    translator = pipeline("translation", model="facebook/mbart-large-50-many-to-many-mnli")
    # The generator is a standard pre-trained model for creating text.
    generator = pipeline("text-generation", model="gpt2")
    return translator, generator

# Load the models and show a status message.
with st.spinner("Loading AI models... This might take a moment."):
    translator, generator = load_models()
st.success("AI Models Loaded Successfully!")
st.markdown("---") # Adds a horizontal line

# 4. CREATE THE INTERACTIVE PART OF THE APP
# Define the languages we support. The codes (e.g., "es_XX") are required by the translation model.
LANG_CODE_MAP = {
    "Spanish": "es_XX",
    "French": "fr_XX",
    "Hindi": "hi_IN",
    "German": "de_DE",
    "Tamil": "ta_IN" # Example of adding a regional language
}

# Create a dropdown menu for the user to select their language.
selected_lang_name = st.selectbox("Select your language:", list(LANG_CODE_MAP.keys()))
lang_code = LANG_CODE_MAP[selected_lang_name]

# Create a text input box for the user's question.
user_input = st.text_input(f"Ask your question in {selected_lang_name}...", placeholder="e.g., What is Data Science?")

# 5. DEFINE THE CORE LOGIC
if user_input:
    # Show the user's original question.
    st.subheader("Your Query:")
    st.write(user_input)
    
    with st.spinner(f"Translating from {selected_lang_name} to English..."):
        # Step A: Translate the user's input from their language to English.
        translated_to_en = translator(user_input, src_lang=lang_code, tgt_lang="en_XX")
        english_query = translated_to_en[0]['translation_text']
    
    st.info(f"**Translated to English:** {english_query}")

    with st.spinner("Thinking..."):
        # Step B: Send the English text to the GPT-2 model to get an answer.
        # We add a prompt to guide the model's response.
        prompt = f"Answer the following question about online courses and career development: {english_query}"
        gpt_response = generator(prompt, max_length=150, num_return_sequences=1)
        english_response = gpt_response[0]['generated_text']

    st.info(f"**Bot's Response (in English):** {english_response}")
    
    with st.spinner(f"Translating from English to {selected_lang_name}..."):
        # Step C: Translate the English answer back to the user's original language.
        translated_to_orig = translator(english_response, src_lang="en_XX", tgt_lang=lang_code)
        final_response = translated_to_orig[0]['translation_text']
        
    # Step D: Display the final answer in the user's language.
    st.subheader("Final Answer:")
    st.success(final_response)
