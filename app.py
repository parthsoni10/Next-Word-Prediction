import streamlit as st
import time
import pandas as pd
import numpy as np
from predictor import NextWordPredictor

# Page configuration
st.set_page_config(
    page_title="Next Word Predictor - LSTM",    
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and dark glassmorphism styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Styling */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    /* Header Gradient Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px 35px;
        margin-bottom: 25px;
        backdrop-filter: blur(12px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 20px;
    }
    
    /* Badge Pills */
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 10px;
        margin-bottom: 8px;
    }
    .badge-purple {
        background: rgba(168, 85, 247, 0.2);
        color: #d8b4fe;
        border: 1px solid rgba(168, 85, 247, 0.4);
    }
    .badge-blue {
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .badge-green {
        background: rgba(34, 197, 94, 0.2);
        color: #86efac;
        border: 1px solid rgba(34, 197, 94, 0.4);
    }

    /* Prediction Card */
    .pred-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: all 0.2s ease-in-out;
        backdrop-filter: blur(8px);
    }
    .pred-card:hover {
        transform: translateY(-2px);
        border-color: rgba(129, 140, 248, 0.5);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
    }
    
    .word-rank {
        font-size: 0.9rem;
        font-weight: 700;
        color: #818cf8;
    }
    
    .word-text {
        font-size: 1.35rem;
        font-weight: 700;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .prob-percent {
        font-size: 1.1rem;
        font-weight: 700;
        color: #c084fc;
    }

    /* Custom Input Box styling */
    .stTextInput > div > div > input {
        font-size: 1.2rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        background-color: rgba(15, 23, 42, 0.8) !important;
        border: 2px solid rgba(129, 140, 248, 0.3) !important;
        border-radius: 14px !important;
        color: #ffffff !important;
        padding: 14px 18px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.25) !important;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 22px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }
    
    /* Quick Pill Button styling */
    .pill-btn {
        background: rgba(51, 65, 85, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        color: #e2e8f0;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .pill-btn:hover {
        background: #818cf8;
        color: #ffffff;
    }

    /* Code & Output Box */
    .generated-box {
        background: rgba(15, 23, 42, 0.9);
        border: 1px dashed rgba(192, 132, 252, 0.4);
        border-radius: 16px;
        padding: 22px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.2rem;
        color: #f1f5f9;
        line-height: 1.7;
    }
    
    .gen-word {
        color: #38bdf8;
        font-weight: 700;
        background: rgba(56, 189, 248, 0.15);
        padding: 2px 8px;
        border-radius: 6px;
    }
    
    /* Stat Cards */
    .stat-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .stat-num {
        font-size: 2.2rem;
        font-weight: 800;
        color: #818cf8;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Predictor with caching to load only once
@st.cache_resource
def load_predictor():
    return NextWordPredictor()

try:
    predictor = load_predictor()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Failed to load model: {e}")

# App Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🔮 Next Word Predictor & Text Generator</div>
    <div class="hero-subtitle">Interactive Deep Learning Web App powered by LSTM (Long Short-Term Memory) Neural Network</div>
    <div>
        <span class="badge badge-green">● Model Active</span>
        <span class="badge badge-purple">📚 Vocab: 8,979 Words</span>
        <span class="badge badge-blue">⚙️ Seq Length: 745</span>
        <span class="badge badge-purple">⚡ Speed: &lt; 5ms</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Options
with st.sidebar:
    st.image("https://img.icons8.com/isometric-line/100/brain.png", width=70)
    st.title("⚙️ Model Settings")
    
    top_k = st.slider("Top Predictions (Top-K)", min_value=3, max_value=20, value=5, step=1)
    
    st.markdown("---")
    st.subheader("🛠️ Model Specifications")
    st.markdown("""
    - **Architecture**: LSTM Sequential
    - **Embedding Dim**: 50
    - **LSTM Units**: 128
    - **Dense Vocab**: 10,000
    - **Parameters**: ~1.88 Million
    - **Backend**: Direct HDF5 / NumPy
    """)
    
    st.markdown("---")
    st.subheader("📁 Project Files")
    st.markdown("""
    - ✅ `lstm_model.h5` (22.6 MB)
    - ✅ `tokenizer.pkl` (359 KB)
    - ✅ `max_len.pkl` (15 Bytes)
    """)
    
    st.markdown("---")
    st.caption("Built with Streamlit & Deep Learning")

if model_loaded:
    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Live Predictor & Composer", "⚡ Auto Text Generator", "📊 Architecture & Vocab"])

    # --------------------------------------------------------------------
    # TAB 1: LIVE PREDICTOR & COMPOSER
    # --------------------------------------------------------------------
    with tab1:
        st.subheader("Type a sentence to get real-time next-word predictions")
        
        # Session state for prompt
        if "user_prompt" not in st.session_state:
            st.session_state["user_prompt"] = "how are"

        # Preset Quick Prompt Buttons
        st.markdown("**Quick Prompts:**")
        cols_preset = st.columns(6)
        presets = ["how are", "what is the", "once upon a", "thank you for", "i want to", "the secret of"]
        
        for idx, preset in enumerate(presets):
            if cols_preset[idx].button(preset, key=f"btn_preset_{idx}"):
                st.session_state["user_prompt"] = preset

        # Text input field
        user_input = st.text_input(
            "Input Text Prompt:",
            value=st.session_state["user_prompt"],
            key="input_text_box",
            placeholder="Type your sentence here..."
        )
        
        # Keep session state updated
        st.session_state["user_prompt"] = user_input

        if user_input.strip():
            start_time = time.time()
            predictions = predictor.predict_next_words(user_input, top_k=top_k)
            latency = (time.time() - start_time) * 1000

            col_left, col_right = st.columns([1.2, 1])

            with col_left:
                st.markdown(f"### 🏆 Top {top_k} Next Word Predictions")
                st.caption(f"Inference latency: {latency:.2f} ms")

                if predictions:
                    # Render sleek prediction cards
                    for rank, pred in enumerate(predictions, start=1):
                        word = pred['word']
                        prob = pred['probability']
                        percent = prob * 100
                        
                        card_html = f"""
                        <div class="pred-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                                <div>
                                    <span class="word-rank">#{rank}</span>
                                    <span class="word-text" style="margin-left: 10px;">{word}</span>
                                </div>
                                <div class="prob-percent">{percent:.1f}%</div>
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
                        st.progress(float(prob))

            with col_right:
                st.markdown("### 🖱️ Smart Sentence Composer")
                st.write("Click any predicted word below to automatically append it to your sentence:")
                
                # Interactive word pills that append word to prompt
                st.markdown("<br>", unsafe_allow_html=True)
                pill_cols = st.columns(2)
                for i, pred in enumerate(predictions):
                    w = pred['word']
                    p_pct = pred['probability'] * 100
                    col_idx = i % 2
                    if pill_cols[col_idx].button(f"➕ {w} ({p_pct:.1f}%)", key=f"append_word_{i}"):
                        st.session_state["user_prompt"] = (user_input.strip() + " " + w).strip()
                        st.rerun()

                st.markdown("---")
                if st.button("🧹 Clear Input", use_container_width=True):
                    st.session_state["user_prompt"] = ""
                    st.rerun()

        else:
            st.info("💡 Type any phrase or click a quick prompt above to see next-word predictions!")

    # --------------------------------------------------------------------
    # TAB 2: AUTO TEXT GENERATOR
    # --------------------------------------------------------------------
    with tab2:
        st.subheader("Generate multi-word text completions using the LSTM model")
        
        gen_col1, gen_col2 = st.columns([1, 1])
        
        with gen_col1:
            seed_text = st.text_input("Seed Prompt:", value="once upon a", key="gen_seed_input")
            num_gen_words = st.slider("Number of Words to Generate:", min_value=1, max_value=30, value=10, step=1)
            temperature = st.slider("Creativity (Temperature):", min_value=0.1, max_value=2.0, value=0.8, step=0.1, 
                                    help="Lower temperature = more deterministic, Higher temperature = more creative/random.")
            
            generate_btn = st.button("✨ Generate Completion", use_container_width=True)

        with gen_col2:
            st.markdown("### 📝 Generated Text Output")
            if generate_btn and seed_text.strip():
                with st.spinner("LSTM model generating sequence..."):
                    full_text, new_words = predictor.generate_text(seed_text, num_words=num_gen_words, temperature=temperature)
                
                # Formatted output display
                highlighted_new = " ".join([f'<span class="gen-word">{w}</span>' for w in new_words])
                out_html = f"""
                <div class="generated-box">
                    <span>{seed_text}</span> {highlighted_new}
                </div>
                """
                st.markdown(out_html, unsafe_allow_html=True)
                st.success(f"Successfully generated {len(new_words)} words!")
            else:
                st.markdown("""
                <div class="generated-box" style="color: #64748b; font-style: italic;">
                    Click "Generate Completion" to see the LSTM model complete the text...
                </div>
                """, unsafe_allow_html=True)

    # --------------------------------------------------------------------
    # TAB 3: ARCHITECTURE & VOCAB
    # --------------------------------------------------------------------
    with tab3:
        st.subheader("Neural Network Architecture & Model Stats")
        
        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-num">8,979</div>
                <div class="stat-label">Tokenizer Vocabulary Size</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-num">745</div>
                <div class="stat-label">Max Sequence Length</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-num">128</div>
                <div class="stat-label">LSTM Hidden Units</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-num">1.88M</div>
                <div class="stat-label">Total Parameters</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Architecture Breakdown
        st.markdown("### 🏗️ Layer-by-Layer Architecture")
        arch_data = [
            {"Layer": "Input Layer", "Type": "Input", "Output Shape": "(Batch, 745)", "Parameters": "0"},
            {"Layer": "Embedding Layer", "Type": "Embedding", "Output Shape": "(Batch, 745, 50)", "Parameters": "500,000 (10k × 50)"},
            {"Layer": "LSTM Layer", "Type": "Recurrent (LSTM)", "Output Shape": "(Batch, 128)", "Parameters": "91,648"},
            {"Layer": "Dense Layer", "Type": "Dense (Softmax)", "Output Shape": "(Batch, 10000)", "Parameters": "1,290,000 (128 × 10k + 10k)"},
        ]
        st.table(pd.DataFrame(arch_data))

        # Vocabulary Explorer
        st.markdown("### 🔍 Vocabulary Explorer")
        search_query = st.text_input("Search word in vocabulary:", value="music", key="vocab_search_key")
        if search_query.strip():
            sq = search_query.strip().lower()
            token_id = predictor.word_index.get(sq, None)
            if token_id:
                st.success(f"Word **'{sq}'** found! Token ID: `{token_id}`")
            else:
                st.warning(f"Word **'{sq}'** not in vocabulary (treated as `<OOV>` token_id = 1).")
                
        # Sample Top 20 Vocabulary Words
        with st.expander("📖 View Top 50 Most Frequent Words in Vocabulary"):
            top_50_items = list(predictor.word_index.items())[:50]
            df_vocab = pd.DataFrame(top_50_items, columns=["Word", "Token ID"])
            st.dataframe(df_vocab, width='stretch')
