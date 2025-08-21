import streamlit as st
import pandas as pd
import hashlib
import os
import pdfplumber
import docx
import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

# Auto-download nltk data
nltk.download('stopwords')
nltk.download('punkt')

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# --------------------
# Helper functions
# --------------------
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + ' '
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ' '.join([para.text for para in doc.paragraphs])
    return text

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    # fallback
    lines = text.split('\n')
    for line in lines[:10]:
        if line.strip():
            return line.strip().split()[0] + " " + (line.strip().split()[1] if len(line.strip().split()) > 1 else "")
    return "Unknown"

def extract_skills(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [w for w in words if w.isalpha() and w not in stop_words]
    return list(set(words))

# --------------------
# Streamlit App
# --------------------
st.set_page_config(page_title="Resume Analyzer", layout="wide")
st.title("📝 AI-Powered Resume Analyzer")

uploaded_file = st.file_uploader("Upload your resume (PDF/DOCX)", type=["pdf", "docx"])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_docx(uploaded_file)

    st.subheader("Extracted Text")
    st.write(text[:500] + "...")  # show first 500 chars

    name = extract_name(text)
    st.subheader("Candidate Name")
    st.write(name)

    skills = extract_skills(text)
    st.subheader("Extracted Skills")
    st.write(skills)

    # Sample Job Description
    job_desc = "Python, Machine Learning, Data Analysis, SQL, Pandas"
    st.subheader("Job Description")
    st.write(job_desc)

    # Skill Matching
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    resume_embedding = model.encode(' '.join(skills), convert_to_tensor=True)
    job_embedding = model.encode(job_desc, convert_to_tensor=True)
    score = util.cos_sim(resume_embedding, job_embedding).item() * 100

    st.subheader("Match Score")
    st.write(f"{score:.2f}%")

    missing_skills = [skill for skill in job_desc.lower().split(",") if skill.strip().lower() not in skills]
    st.subheader("Missing Skills")
    st.write(missing_skills)
