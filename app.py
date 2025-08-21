import streamlit as st
import pandas as pd
import hashlib
import pdfplumber
import docx
import re
import spacy
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

# Download NLTK resources
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------
# Helper functions
# ----------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_contact_info(text):
    email = re.search(r"[\w\.-]+@[\w\.-]+", text)
    phone = re.search(r"\+?\d[\d -]{8,}\d", text)
    name = "Unknown"
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break
    return name, email.group() if email else None, phone.group() if phone else None

def extract_skills(text):
    tokens = [w.lower() for w in re.findall(r"\w+", text) if w.lower() not in stop_words]
    return set(tokens)

def match_skills(resume_skills, job_skills):
    resume_embeddings = model.encode(list(resume_skills), convert_to_tensor=True)
    job_embeddings = model.encode(list(job_skills), convert_to_tensor=True)
    cos_sim = util.cos_sim(resume_embeddings, job_embeddings)

    matched = []
    for i, rs in enumerate(resume_skills):
        for j, js in enumerate(job_skills):
            if cos_sim[i][j] > 0.7:  # semantic similarity threshold
                matched.append(js)

    matched = set(matched)
    missing = job_skills - matched
    score = round((len(matched) / len(job_skills)) * 100, 2) if job_skills else 0
    return score, matched, missing

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("📄 AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload your resume (PDF/DOCX)", type=["pdf", "docx"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_docx(uploaded_file)

    name, email, phone = extract_contact_info(text)

    st.subheader("👤 Candidate Info")
    st.write(f"**Name:** {name}")
    st.write(f"**Email:** {email}")
    st.write(f"**Phone:** {phone}")

    # Example job description
    job_description = """
    We are hiring a Data Scientist skilled in Python, Machine Learning, NLP, Deep Learning, and SQL.
    """
    st.subheader("💼 Job Description")
    st.write(job_description)

    resume_skills = extract_skills(text)
    job_skills = extract_skills(job_description)

    score, matched, missing = match_skills(resume_skills, job_skills)

    st.subheader("📊 Match Results")
    st.write(f"**Match Score:** {score}%")
    st.write(f"✅ Matched Skills: {', '.join(matched) if matched else 'None'}")
    st.write(f"❌ Missing Skills: {', '.join(missing) if missing else 'None'}")
