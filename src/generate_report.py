import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

# ReportLab imports for generating structured, styled PDF documents
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

# Paths setup
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_PATH = os.path.join(MODELS_DIR, "results.json")
REPORT_PATH = os.path.join(ROOT_DIR, "report.pdf")

class NumberedCanvas(canvas.Canvas):
    """
    Custom canvas to implement a two-pass page numbering.
    This dynamically calculates and adds footer text and "Page X of Y" on every page.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_elements(num_pages)
            super().showPage()
        super().save()

    def draw_page_elements(self, page_count):
        # We skip the cover page (Page 1) for running header/footer
        if self._pageNumber == 1:
            return
            
        self.saveState()
        
        # Running Header
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#2C3E50"))
        self.drawString(54, 750, "SPAM DETECTION SYSTEM - ML PROJECT REPORT")
        
        # Header line
        self.setStrokeColor(colors.HexColor("#BDC3C7"))
        self.setLineWidth(0.5)
        self.line(54, 742, 558, 742)
        
        # Running Footer
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#7F8C8D"))
        self.drawString(54, 36, "Confidential - Academic & Professional Portfolio")
        
        # Footer line
        self.line(54, 48, 558, 48)
        
        # Page Number right-aligned
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_str)
        
        self.restoreState()


def generate_plots_if_needed(spam_distribution):
    """
    Generates data visualization files if they don't exist.
    """
    # 1. Label distribution chart
    label_dist_path = os.path.join(MODELS_DIR, "label_distribution.png")
    if not os.path.exists(label_dist_path):
        # Translate keys
        ham_count = spam_distribution.get("0", spam_distribution.get(0, 4825))
        spam_count = spam_distribution.get("1", spam_distribution.get(1, 747))
        
        df_plot = pd.DataFrame({
            'Category': ['Ham (Not Spam)', 'Spam'],
            'Count': [ham_count, spam_count]
        })
        
        plt.figure(figsize=(5, 3.5))
        # Warm, harmonic styling
        sns.barplot(x='Category', y='Count', data=df_plot, palette=['#1B5E20', '#B71C1C'])
        
        # Annotate counts on bars
        for idx, row in df_plot.iterrows():
            plt.text(idx, row['Count'] + 100, f"{row['Count']} ({row['Count']/(ham_count+spam_count)*100:.1f}%)", 
                     ha='center', va='bottom', fontweight='bold', fontsize=9)
                     
        plt.title("Class Distribution of SMS Messages", fontsize=11, fontweight='bold', pad=10)
        plt.ylabel("Number of Messages", fontsize=9)
        plt.xlabel("")
        plt.ylim(0, max(ham_count, spam_count) + 800)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(label_dist_path, dpi=300)
        plt.close()
        print(f"Generated distribution plot at {label_dist_path}")
        
    return label_dist_path

def compile_pdf():
    # Load model results metadata
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Cannot find results.json at {RESULTS_PATH}. Please run training first!")
        
    with open(RESULTS_PATH, "r") as f:
        res = json.load(f)
        
    best_model = res['best_model']
    nb_metrics = res['nb_metrics']
    lr_metrics = res['lr_metrics']
    spam_dist = res['spam_distribution']
    ham_keywords = res['ham_keywords']
    spam_keywords = res['spam_keywords']
    
    # Pre-generate plots
    dist_img_path = generate_plots_if_needed(spam_dist)
    best_cm_img_path = os.path.join(MODELS_DIR, "best_confusion_matrix.png")
    
    # Verify confusion matrix plot exists
    if not os.path.exists(best_cm_img_path):
        best_cm_img_path = None
        print("Warning: best_confusion_matrix.png not found, omitting from PDF.")

    # Initialize PDF document
    # Top and bottom margin accommodate header and footer (margin = 54 pt = 0.75 in)
    doc = SimpleDocTemplate(
        REPORT_PATH,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    # Establish styles
    styles = getSampleStyleSheet()
    
    # Create customized styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=32,
        leading=38,
        textColor=colors.HexColor('#2C3E50'),
        alignment=1, # Center
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=16,
        leading=22,
        textColor=colors.HexColor('#7F8C8D'),
        alignment=1, # Center
        spaceAfter=40
    )
    
    meta_style = ParagraphStyle(
        'CoverMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=16,
        textColor=colors.HexColor('#34495E'),
        alignment=1 # Center
    )
    
    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#2C3E50'),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#16A085'),
        spaceBefore=12,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14.5,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=10
    )
    
    equation_style = ParagraphStyle(
        'EquationText',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#2980B9'),
        backColor=colors.HexColor('#ECF0F1'),
        borderPadding=6,
        alignment=1, # Center
        spaceBefore=8,
        spaceAfter=8
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        textColor=colors.white,
        alignment=1
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        alignment=1
    )

    table_cell_bold_style = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        alignment=1
    )

    story = []
    
    # ----------------------------------------------------
    # PAGE 1: COVER PAGE
    # ----------------------------------------------------
    story.append(Spacer(1, 1.5 * inch))
    # Elegant Top Decorative Bar
    d_bar = Table([['']], colWidths=[504], rowHeights=[6])
    d_bar.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#16A085')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(d_bar)
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("SMS Spam Detection System", title_style))
    story.append(Paragraph("A Comparative Machine Learning Study & Production Deployment", subtitle_style))
    
    story.append(Spacer(1, 1 * inch))
    
    # Metainfo Box
    date_str = datetime.now().strftime("%B %d, %Y")
    meta_html = f"""
    <b>Prepared By:</b> Machine Learning Intern<br/>
    <b>Project Category:</b> Natural Language Processing (NLP)<br/>
    <b>Deployment Stack:</b> Scikit-Learn, Streamlit, Joblib<br/>
    <b>Date:</b> {date_str}<br/>
    <b>Status:</b> Production Ready
    """
    story.append(Paragraph(meta_html, meta_style))
    story.append(Spacer(1, 1.5 * inch))
    
    # Elegant Bottom Accent
    story.append(d_bar)
    story.append(PageBreak())
    
    # ----------------------------------------------------
    # PAGE 2: OBJECTIVE & PREPROCESSING
    # ----------------------------------------------------
    story.append(Paragraph("1. Executive Summary & Objective", h1_style))
    obj_text = """
    SMS communication remains a prominent channel for fast notifications, transactional verification codes, and personal updates. However, it is also highly susceptible to unsolicited promotions, scams, and malicious phishing attempts, collectively categorized as <b>Spam</b>. 
    This project builds a robust, data-driven machine learning system to classify incoming text messages as either <b>Ham (Legitimate)</b> or <b>Spam</b> in real-time. By utilizing term-weighting features and comparing probabilistic and linear classifiers, the final system selects the best performing model based on the F1-score metric and deploys it via an interactive Streamlit web application.
    """
    story.append(Paragraph(obj_text, body_style))
    
    story.append(Paragraph("2. Dataset Analysis & Preprocessing", h1_style))
    prep_text = """
    The model was trained and evaluated on the public <b>SMS Spam Collection Dataset</b>, which contains 5,572 raw English SMS messages annotated with ground truth labels (<i>ham</i> or <i>spam</i>).
    <br/><br/>
    To construct clean, numeric representations of language, text messages are subjected to a multi-stage preprocessing pipeline:
    """
    story.append(Paragraph(prep_text, body_style))
    
    # Bullet points
    bullets = [
        "<b>Lowercasing:</b> All text is normalized to lowercase to prevent terms like 'Win' and 'win' from being indexed as separate features.",
        "<b>Punctuation & Character Filtering:</b> Special symbols, punctuation, and digits are removed using regular expressions, focusing purely on alpha characters.",
        "<b>Whitespace Standardization:</b> Multiple consecutive spaces and leading/trailing spaces are stripped to ensure clean word delimiters.",
        "<b>Label Mapping:</b> Categorical targets are encoded as binary numeric labels, where <b>Ham = 0</b> and <b>Spam = 1</b>."
    ]
    for b in bullets:
        story.append(Paragraph(f"• {b}", ParagraphStyle('Bullet', parent=body_style, leftIndent=15, spaceAfter=5)))
        
    story.append(Spacer(1, 10))
    
    # Embed Label Distribution chart side-by-side or centered
    if os.path.exists(dist_img_path):
        story.append(Paragraph("<b>Figure 1:</b> Dataset class distribution showing a significant class imbalance typical of real-world spam detection scenarios.", ParagraphStyle('FigCap', parent=body_style, fontSize=8.5, textColor=colors.HexColor('#7F8C8D'), alignment=1)))
        story.append(Spacer(1, 5))
        story.append(Image(dist_img_path, width=4.0*inch, height=2.8*inch))
        
    story.append(PageBreak())
    
    # ----------------------------------------------------
    # PAGE 3: METHODOLOGY
    # ----------------------------------------------------
    story.append(Paragraph("3. Feature Engineering & Vectorization", h1_style))
    vector_text = """
    Machine learning models require numerical feature vectors. We utilize the <b>TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer</b>. 
    TF-IDF measures how important a word is to a document relative to the entire corpus. The mathematical representation is:
    """
    story.append(Paragraph(vector_text, body_style))
    
    story.append(Paragraph("TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)<br/>"
                           "Where: TF(t, d) = f_t,d / sum(f_t',d)<br/>"
                           "IDF(t, D) = log( (1 + |D|) / (1 + |{d ∈ D : t ∈ d}|) ) + 1", equation_style))
    
    vector_text_2 = """
    This formulation scales down words that appear frequently across all documents (such as 'the', 'is', 'to') and highlights words that carry high informative value in specific contexts. We enforce English stop-word removal and exclude words appearing in only a single document (min_df=2) to reduce noise.
    """
    story.append(Paragraph(vector_text_2, body_style))
    
    story.append(Paragraph("4. Predictive Models", h1_style))
    
    story.append(Paragraph("Multinomial Naive Bayes", h2_style))
    nb_explain = """
    Naive Bayes is a probabilistic classifier based on Bayes' Theorem, with the "naive" assumption of conditional independence between features given the label. The class probability is calculated as:
    """
    story.append(Paragraph(nb_explain, body_style))
    story.append(Paragraph("P(C_k | x) = ( P(C_k) × ∏_i P(x_i | C_k) ) / P(x)", equation_style))
    story.append(Paragraph("For text classification, <b>Multinomial Naive Bayes</b> models word frequencies. Due to its computational efficiency, it performs exceptionally well on high-dimensional text vectors and acts as an excellent baseline.", body_style))
    
    story.append(Paragraph("Logistic Regression", h2_style))
    lr_explain = """
    Logistic Regression is a linear classification model that estimates the probability of the binary target using a sigmoid function:
    """
    story.append(Paragraph(lr_explain, body_style))
    story.append(Paragraph("P(Y=1 | X) = σ(W^T X + b) = 1 / (1 + e^{-(W^T X + b)})", equation_style))
    story.append(Paragraph("Weights (coefficients) are learned by minimizing the binary cross-entropy loss. Logistic Regression models the continuous dependencies of features, identifying structural word boundaries and allowing us to inspect feature coefficients directly to understand key vocabulary predictors.", body_style))
    
    story.append(PageBreak())
    
    # ----------------------------------------------------
    # PAGE 4: RESULTS AND MODEL COMPARISON
    # ----------------------------------------------------
    story.append(Paragraph("5. Model Performance & Selection", h1_style))
    results_intro = """
    Both pipelines were evaluated on a stratified holdout test set (20% of the corpus). Precision, Recall, and F1-Scores are focused primarily on the <b>Spam class (positive class = 1)</b> to ensure the model detects malicious messages reliably while limiting false positives (legitimate messages flagged as spam).
    """
    story.append(Paragraph(results_intro, body_style))
    
    # Comparison Table
    table_data = [
        [
            Paragraph("<b>Model Evaluation Metric (Spam Class Focus)</b>", table_header_style), 
            Paragraph("<b>Multinomial Naive Bayes</b>", table_header_style), 
            Paragraph("<b>Logistic Regression</b>", table_header_style)
        ],
        [
            Paragraph("Accuracy (Overall)", table_cell_bold_style),
            Paragraph(f"{nb_metrics['accuracy']:.4%}", table_cell_style),
            Paragraph(f"{lr_metrics['accuracy']:.4%}", table_cell_style)
        ],
        [
            Paragraph("Precision (Spam Class)", table_cell_bold_style),
            Paragraph(f"{nb_metrics['precision_spam']:.4%}", table_cell_style),
            Paragraph(f"{lr_metrics['precision_spam']:.4%}", table_cell_style)
        ],
        [
            Paragraph("Recall (Spam Class)", table_cell_bold_style),
            Paragraph(f"{nb_metrics['recall_spam']:.4%}", table_cell_style),
            Paragraph(f"{lr_metrics['recall_spam']:.4%}", table_cell_style)
        ],
        [
            Paragraph("F1-Score (Spam Class)", table_header_style),
            Paragraph(f"<b>{nb_metrics['f1_spam']:.4%}</b>" if best_model == "Multinomial Naive Bayes" else f"{nb_metrics['f1_spam']:.4%}", table_cell_style),
            Paragraph(f"<b>{lr_metrics['f1_spam']:.4%}</b>" if best_model == "Logistic Regression" else f"{lr_metrics['f1_spam']:.4%}", table_cell_style)
        ]
    ]
    
    comparison_table = Table(table_data, colWidths=[200, 150, 150])
    comparison_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2C3E50')),
        ('BACKGROUND', (0,4), (-1,4), colors.HexColor('#EAFAF1')), # Highlight F1 row
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#BDC3C7')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    
    story.append(comparison_table)
    story.append(Spacer(1, 15))
    
    decision_text = f"""
    Based on the evaluation criteria, the system selected the <b>{best_model}</b> model, which achieved a Spam class F1-Score of <b>{res['nb_metrics']['f1_spam'] if best_model == 'Multinomial Naive Bayes' else res['lr_metrics']['f1_spam']:.4f}</b> on the holdout test dataset. 
    While both models exhibit high performance, the chosen model provides an optimal balance between precision and recall, ensuring highly reliable filtration.
    """
    story.append(Paragraph(decision_text, body_style))
    
    # Embed Confusion Matrix
    if best_cm_img_path and os.path.exists(best_cm_img_path):
        story.append(Paragraph(f"<b>Figure 2:</b> Confusion Matrix Heatmap for the selected champion model ({best_model}) evaluated on the test partition.", ParagraphStyle('FigCap2', parent=body_style, fontSize=8.5, textColor=colors.HexColor('#7F8C8D'), alignment=1)))
        story.append(Spacer(1, 5))
        story.append(Image(best_cm_img_path, width=3.8*inch, height=3.1*inch))
        
    story.append(PageBreak())
    
    # ----------------------------------------------------
    # PAGE 5: FEATURE COEFFICIENTS & CONCLUSION
    # ----------------------------------------------------
    story.append(Paragraph("6. Feature Insights: Top Diagnostic Keywords", h1_style))
    features_intro = f"""
    By analyzing the parameter weights of the trained champion model, we extracted the most predictive terms for both categories. These terms present strong indicators that trigger classification decisions:
    """
    story.append(Paragraph(features_intro, body_style))
    
    # Create Table of Top Keywords
    kw_data = [[
        Paragraph("<b>Spam Predictor Words</b>", table_header_style), 
        Paragraph("<b>Weight / Score</b>", table_header_style),
        Paragraph("<b>Ham Predictor Words</b>", table_header_style), 
        Paragraph("<b>Weight / Score</b>", table_header_style)
    ]]
    
    for i in range(min(len(spam_keywords), len(ham_keywords))):
        s_w, s_s = spam_keywords[i]
        h_w, h_s = ham_keywords[i]
        kw_data.append([
            Paragraph(s_w, table_cell_bold_style),
            Paragraph(f"{s_s:.4f}", table_cell_style),
            Paragraph(h_w, table_cell_bold_style),
            Paragraph(f"{h_s:.4f}", table_cell_style),
        ])
        
    kw_table = Table(kw_data, colWidths=[130, 120, 130, 124])
    kw_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#C0392B')), # Red header for Spam
        ('BACKGROUND', (2,0), (3,0), colors.HexColor('#27AE60')), # Green header for Ham
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#BDC3C7')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    
    story.append(kw_table)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("7. Streamlit Web Deployment & Architecture", h1_style))
    app_text = """
    To make this machine learning pipeline accessible, a production-grade <b>Streamlit Web Application</b> was built. The backend loads the serialized pipeline (<code>spam_pipeline.pkl</code>) containing the TF-IDF Vectorizer and the trained classifier. 
    When a user inputs a text message:
    <br/><br/>
    1. The raw text is passed through the same vectorizer vocabulary.<br/>
    2. The model outputs predicted class probabilities.<br/>
    3. The UI highlights the result with color-coded layouts (Red alert for Spam, Green alert for Ham) alongside an interactive confidence score indicator and diagnostic data graphs.
    """
    story.append(Paragraph(app_text, body_style))
    
    story.append(Paragraph("8. Conclusions & Next Steps", h1_style))
    conclusion_text = """
    This project demonstrates an end-to-end NLP workflow, from raw message collection to feature engineering, classifier comparison, validation, and web deployment. 
    The selected pipeline is highly effective, maintaining robust generalization boundaries. Future iterations could incorporate word embeddings (e.g. Word2Vec, GloVe) or lightweight transformer models (e.g., DistilBERT) to analyze contextual structures beyond standalone keywords, further reducing edge-case false positives.
    """
    story.append(Paragraph(conclusion_text, body_style))
    
    # Compile the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated project report at {REPORT_PATH}")

if __name__ == "__main__":
    compile_pdf()
