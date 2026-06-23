import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configurations
st.set_page_config(
    page_title="Spam Detection System Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling CSS
st.markdown("""
<style>
    /* Main body styling */
    .reportview-container {
        background: #F4F6F7;
    }
    /* Main Header card styling */
    .main-header {
        background: linear-gradient(135deg, #2C3E50 0%, #16A085 100%);
        padding: 30px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        margin: 0;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 10px 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    /* Outcome Cards */
    .result-card-spam {
        background-color: #FADBD8;
        border-left: 6px solid #C0392B;
        color: #78281F;
        padding: 20px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.3rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-top: 15px;
    }
    .result-card-ham {
        background-color: #D4EFDF;
        border-left: 6px solid #27AE60;
        color: #145A32;
        padding: 20px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.3rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-top: 15px;
    }
    /* Info Card container */
    .metric-box {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-top: 4px solid #16A085;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2C3E50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7F8C8D;
        text-transform: uppercase;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Path Resolution
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_PATH = os.path.join(MODELS_DIR, "results.json")
REPORT_PATH = os.path.join(ROOT_DIR, "report.pdf")

# Sidebar Configuration
st.sidebar.markdown("### 🛡️ System Control Center")
st.sidebar.info("This system is running a Production Scikit-Learn Pipeline combining a TF-IDF text vectorizer with a machine learning classifier.")

# Try to load model evaluation results
results = None
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)
else:
    st.sidebar.warning("⚠️ Training metadata not found. Please run model training to enable metrics & visualizations.")

# Load predict_spam function safely
try:
    from predict import predict_spam_with_confidence
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.sidebar.error(f"⚠️ Model Pipeline not found: {e}")

# Sidebar PDF Download
if os.path.exists(REPORT_PATH):
    with open(REPORT_PATH, "rb") as f:
        pdf_bytes = f.read()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Project Report")
    st.sidebar.download_button(
        label="Download Project Report (PDF)",
        data=pdf_bytes,
        file_name="Spam_Detection_Project_Report.pdf",
        mime="application/pdf"
    )
else:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Project Report")
    st.sidebar.warning("Generate report first to enable PDF download.")

# Main Application Banner
st.markdown("""
<div class="main-header">
    <h1>SMS Spam Detection System</h1>
    <p>Enterprise NLP Text Classifier and Analytics Dashboard</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["🚀 Real-time Classifier", "📊 Model Performance & Insights"])

# ----------------------------------------------------
# TAB 1: REAL-TIME SPAM CLASSIFIER
# ----------------------------------------------------
with tab1:
    st.markdown("### Test the Classifier")
    st.write("Enter a SMS text message below to analyze whether it is classified as **Ham (Legitimate)** or **Spam**.")

    # Main text input
    user_input = st.text_area("Message Input:", placeholder="Type or paste your SMS message here...", height=150)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        classify_btn = st.button("Analyze Message", use_container_width=True)
        
    if classify_btn or user_input:
        if not user_input.strip():
            st.warning("Please enter a valid message first.")
        elif not model_loaded:
            st.error("Model pipeline is not trained or loaded. Please run model training first.")
        else:
            # Run prediction
            label, confidence = predict_spam_with_confidence(user_input)
            
            # Show output styling based on label
            if label == "Spam":
                st.markdown(f"""
                <div class="result-card-spam">
                    🚨 Warning: This message is classified as SPAM!
                </div>
                """, unsafe_allow_html=True)
                
                # Visual gauge
                st.write("")
                st.markdown(f"**Model Confidence:** {confidence * 100:.2f}%")
                st.progress(float(confidence))
                
            else:
                st.markdown(f"""
                <div class="result-card-ham">
                    ✅ Safe: This message is classified as HAM (Legitimate).
                </div>
                """, unsafe_allow_html=True)
                
                # Visual gauge
                st.write("")
                st.markdown(f"**Model Confidence:** {confidence * 100:.2f}%")
                st.progress(float(confidence))
                
            # Quick Text Metrics
            with st.expander("🔍 Input Text Details"):
                st.write(f"**Character Count:** {len(user_input)}")
                st.write(f"**Word Count:** {len(user_input.split())}")
                
                # Check for preprocessing result
                from preprocess import clean_text
                cleaned = clean_text(user_input)
                st.write(f"**Standardized Text passed to model:** *\"{cleaned}\"*")

# ----------------------------------------------------
# TAB 2: MODEL PERFORMANCE & INSIGHTS
# ----------------------------------------------------
with tab2:
    if results is None:
        st.info("📊 Run model training (`python src/train.py`) to generate analytics charts.")
    else:
        # Display Key Performance Box metrics
        best_model_name = results['best_model']
        best_metrics = results['nb_metrics'] if best_model_name == "Multinomial Naive Bayes" else results['lr_metrics']
        
        st.markdown(f"### Champion Model Performance Metrics: `{best_model_name}`")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{best_metrics['accuracy']:.2%}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{best_metrics['precision_spam']:.2%}</div>
                <div class="metric-label">Precision (Spam)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{best_metrics['recall_spam']:.2%}</div>
                <div class="metric-label">Recall (Spam)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{best_metrics['f1_spam']:.2%}</div>
                <div class="metric-label">F1-Score (Spam)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Grid of charts
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("#### 📊 Dataset Class Distribution")
            # Draw label distribution plot
            spam_dist = results['spam_distribution']
            ham_count = spam_dist.get('0', spam_dist.get(0, 4825))
            spam_count = spam_dist.get('1', spam_dist.get(1, 747))
            
            fig_dist, ax_dist = plt.subplots(figsize=(6, 4))
            ax_dist.pie(
                [ham_count, spam_count], 
                labels=["Ham (Legitimate)", "Spam"], 
                autopct='%1.1f%%', 
                colors=['#27AE60', '#C0392B'],
                startangle=90, 
                explode=(0, 0.1),
                textprops={'fontsize': 10, 'weight': 'bold'}
            )
            ax_dist.axis('equal')
            plt.tight_layout()
            st.pyplot(fig_dist)
            st.write(f"The dataset is imbalanced: **{ham_count}** Ham messages vs **{spam_count}** Spam messages.")
            
        with col_c2:
            st.markdown("#### 🎯 Confusion Matrix Heatmap")
            # Load and display pre-plotted confusion matrix image
            cm_img_path = os.path.join(MODELS_DIR, "best_confusion_matrix.png")
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, caption="Champion Model Confusion Matrix on Test Set", use_container_width=True)
            else:
                st.info("Confusion matrix image not found.")

        st.markdown("---")
        st.markdown("### 🔑 TF-IDF Feature Importance: Top Vocabulary Keywords")
        st.write("These keywords represent terms with the highest relative significance in distinguishing classes, computed from the model's coefficients or log probability differentials.")
        
        col_k1, col_k2 = st.columns(2)
        
        # Plot top Spam keywords
        with col_k1:
            st.markdown("#### 🔴 Top Spam Keywords")
            spam_words = results['spam_keywords']
            if spam_words:
                words, scores = zip(*spam_words)
                df_spam = pd.DataFrame({'Keyword': words, 'Weight': scores}).sort_values('Weight', ascending=True)
                
                fig_s, ax_s = plt.subplots(figsize=(6, 4.5))
                sns.barplot(x='Weight', y='Keyword', data=df_spam, ax=ax_s, color='#C0392B')
                ax_s.set_title("Spam Class Indicators", fontweight='bold')
                ax_s.set_xlabel("Classifier Coefficient / Relative Log-Odds")
                plt.tight_layout()
                st.pyplot(fig_s)
            else:
                st.info("No Spam keywords available.")

        # Plot top Ham keywords
        with col_k2:
            st.markdown("#### 🟢 Top Ham Keywords")
            ham_words = results['ham_keywords']
            if ham_words:
                words, scores = zip(*ham_words)
                # Keep original order, but since they are negative coefficients, sorting by absolute strength
                # is equivalent to plotting negative values. Let's make them positive magnitudes for easier viewing
                df_ham = pd.DataFrame({'Keyword': words, 'Importance': [abs(s) for s in scores]}).sort_values('Importance', ascending=True)
                
                fig_h, ax_h = plt.subplots(figsize=(6, 4.5))
                sns.barplot(x='Importance', y='Keyword', data=df_ham, ax=ax_h, color='#27AE60')
                ax_h.set_title("Ham Class Indicators (Magnitude)", fontweight='bold')
                ax_h.set_xlabel("Classifier Coefficient / Relative Log-Odds (Absolute)")
                plt.tight_layout()
                st.pyplot(fig_h)
            else:
                st.info("No Ham keywords available.")

        st.markdown("---")
        # Comparative Performance Table
        st.markdown("### 📈 Model Comparison Details")
        nb_m = results['nb_metrics']
        lr_m = results['lr_metrics']
        
        compare_df = pd.DataFrame({
            "Metric (Spam Class Focus)": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Multinomial Naive Bayes": [f"{nb_m['accuracy']:.4%}", f"{nb_m['precision_spam']:.4%}", f"{nb_m['recall_spam']:.4%}", f"{nb_m['f1_spam']:.4%}"],
            "Logistic Regression": [f"{lr_m['accuracy']:.4%}", f"{lr_m['precision_spam']:.4%}", f"{lr_m['recall_spam']:.4%}", f"{lr_m['f1_spam']:.4%}"]
        })
        st.table(compare_df)
