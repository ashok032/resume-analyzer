import os
import streamlit as st

# ========== TORCH BOOTSTRAP ==========
try:
    import torch
except ImportError:
    with st.spinner("Installing PyTorch (CPU only)... This will take ~1-2 minutes ⏳"):
        exit_code = os.system(
            "pip install torch==2.2.1+cpu torchvision==0.17.1+cpu torchaudio==2.2.1+cpu "
            "--extra-index-url https://download.pytorch.org/whl/cpu"
        )
        if exit_code != 0:
            st.error("❌ Failed to install PyTorch. Please check logs.")
        else:
            import torch

# After torch is ready
st.success(f"✅ PyTorch installed successfully! Using torch {torch.__version__}")

# ========== APP START ==========
import pandas as pd
import numpy as np
import pdfplumber
import docx
import spacy
import nltk
from sentence_transformers import SentenceTransformer, util

# Load spacy model
nlp = spacy.load("en_core_web_sm")

st.title("📄 AI-Powered Resume Analyzer")

uploaded_file = st.file_uploader("Upload your resume (PDF/DOCX)", type=["pdf", "docx"])

if uploaded_file:
    file_text = ""

    if uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                file_text += page.extract_text() + "\n"

    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            file_text += para.text + "\n"

    st.subheader("📑 Extracted Text")
    st.write(file_text[:1000] + "..." if len(file_text) > 1000 else file_text)

    # Example NLP
    doc = nlp(file_text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    st.subheader("👤 Extracted Names")
    if names:
        st.write(names[:3])
    else:
        st.write("No names detected.")

    # Example embeddings check
    st.subheader("🤖 Embeddings Test (SentenceTransformer)")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(file_text[:512])
    st.write("✅ Generated embeddings with shape:", emb.shape)
