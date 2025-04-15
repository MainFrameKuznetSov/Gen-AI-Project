import streamlit as st
from groq import Groq
import re
import os


# Set page config at the very beginning
st.set_page_config(
    page_title="✨ Text Summarization Tool",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1E88E5;
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 2rem !important;
    }
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
    }
    .stTextArea textarea:focus {
        border-color: #1E88E5 !important;
        box-shadow: 0 0 0 2px rgba(30,136,229,0.2) !important;
    }
    .stButton>button {
        border-radius: 20px !important;
        padding: 0.5rem 2rem !important;
        background-color: #1E88E5 !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #1565C0 !important;
        transform: translateY(-2px) !important;
    }
    .sidebar-content {
        padding: 1.5rem !important;
    }
    .summary-box {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin-top: 1rem;
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def simple_sentence_tokenize(text):
    """Simple sentence tokenizer that splits on periods, exclamation marks, and question marks"""
    text = re.sub(r'\s+', ' ', text)
    sentences = re.split('[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def simple_word_tokenize(text):
    """Simple word tokenizer that splits on whitespace and removes punctuation"""
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.lower().split()

class TextSummarizer:
    def __init__(self, groq_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)
        # Common English stop words
        self.stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                          "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                          'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                          'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                          'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom'}

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the input text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.]', '', text)
        return text.strip()

    def extractive_summarize(self, text: str, num_sentences: int = 3) -> str:
        """Generate extractive summary using frequency-based approach."""
        try:
            if not text or len(text.split()) < 10:
                return "Text is too short to summarize."

            sentences = simple_sentence_tokenize(text)
            if len(sentences) <= num_sentences:
                return text

            words = simple_word_tokenize(text)
            words = [word for word in words if word not in self.stop_words and word.strip()]
            
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

            sentence_scores = {}
            for sentence in sentences:
                score = 0
                sentence_words = simple_word_tokenize(sentence)
                for word in sentence_words:
                    if word in word_freq:
                        score += word_freq[word]
                if sentence.strip():
                    sentence_scores[sentence] = score

            summary_sentences = sorted(sentence_scores.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:num_sentences]
            
            ordered_summary = [sentence for sentence, score in 
                            sorted(summary_sentences, 
                                  key=lambda x: sentences.index(x[0]))]
            
            return ' '.join(ordered_summary)
        except Exception as e:
            return f"Error in extractive summarization: {str(e)}"

    def abstractive_summarize(self, text: str) -> str:
        """Generate abstractive summary using Groq LLM."""
        try:
            if not text or len(text.split()) < 10:
                return "Text is too short to summarize."

            prompt = f"""Please provide a concise summary of the following text. 
            Focus on the main points and key insights while maintaining clarity and coherence:
            
            {text}
            
            Summary:"""

            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="gemma2-9b-it",
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error in abstractive summarization: {str(e)}"

def main():
    # Title with emoji
    st.markdown("<h1 style='text-align: center;'>📝 Text Summarization Tool ✨</h1>", unsafe_allow_html=True)
    
    # Add a description
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    Transform long texts into concise summaries using AI-powered extractive and abstractive summarization techniques.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with improved styling
    with st.sidebar:
        #st.markdown("### 🔑 API Configuration")
        groq_api_key='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'
        
        st.markdown("### ⚙️ Summarization Settings")
        summary_type = st.radio(
            "Choose summarization type:",
            ["Extractive", "Abstractive", "Both"],
            help="Select the type of summarization you want to perform"
        )
        
        if summary_type in ["Extractive", "Both"]:
            num_sentences = st.slider(
                "Number of sentences:",
                min_value=1,
                max_value=10,
                value=3,
                help="Select the number of sentences for extractive summary"
            )
            
        # Add some information about the models
        st.markdown("""
        ---
        ### ℹ️ About the Models
        
        **Extractive Summarization:**
        - Selects key sentences from the original text
        - Maintains original wording
        - Faster processing
        
        **Abstractive Summarization:**
        - Generates new text using AI
        - More natural summaries
        - Powered by Gemma2-9b-it
        """)

    # Main content area with custom styling
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    
    text_input = st.text_area(
        "Enter your text to summarize:",
        height=200,
        placeholder="Paste your text here (minimum 10 words recommended)...",
        help="Enter the text you want to summarize. For best results, use clear, well-structured text."
    )

    # Center the button
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        generate_button = st.button("✨ Generate Summary")

    if generate_button:
        if not groq_api_key:
            st.error("🔑 Please enter your Groq API key in the sidebar.")
            return
        
        if not text_input:
            st.error("📝 Please enter some text to summarize.")
            return

        try:
            summarizer = TextSummarizer(groq_api_key)
            preprocessed_text = summarizer.preprocess_text(text_input)

            with st.spinner("🤖 Generating your summary..."):
                if summary_type == "Extractive":
                    st.markdown("### 📋 Extractive Summary")
                    with st.container():
                        st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
                        extractive_summary = summarizer.extractive_summarize(
                            preprocessed_text,
                            num_sentences
                        )
                        st.write(extractive_summary)
                        st.markdown("</div>", unsafe_allow_html=True)

                elif summary_type == "Abstractive":
                    st.markdown("### 🤖 Abstractive Summary")
                    with st.container():
                        st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
                        abstractive_summary = summarizer.abstractive_summarize(preprocessed_text)
                        st.write(abstractive_summary)
                        st.markdown("</div>", unsafe_allow_html=True)

                else:  # Both
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### 📋 Extractive Summary")
                        with st.container():
                            st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
                            extractive_summary = summarizer.extractive_summarize(
                                preprocessed_text,
                                num_sentences
                            )
                            st.write(extractive_summary)
                            st.markdown("</div>", unsafe_allow_html=True)

                    with col2:
                        st.markdown("### 🤖 Abstractive Summary")
                        with st.container():
                            st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
                            abstractive_summary = summarizer.abstractive_summarize(preprocessed_text)
                            st.write(abstractive_summary)
                            st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

    # Add a footer
    st.markdown("""
    ---
    <div style='text-align: center; color: #666; padding: 1rem;'>
    Built with ❤️ by Team Quadratic     Querysolvers
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()