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
from supabase import create_client, Client
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import json

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
model = SentenceTransformer('all-MiniLM-L6-v2')

# Supabase client
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------- HELPERS --------------------

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_name(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    ignore = ["contact", "email", "phone", "address", "linkedin", "resume"]
    for line in lines[:10]:
        if any(k in line.lower() for k in ignore):
            continue
        if line.replace(" ", "").isalpha() and line.isupper() and 2 <= len(line.split()) <= 4:
            return line.title()
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.title()
    return "Not found"

def extract_email_from_text(text):
    m = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return m.group(0) if m else "Not found"

def extract_phone_from_text(text):
    m = re.search(r'(\+?\d{1,3}[\s\-]?)?(\(?\d{2,5}\)?[\s\-]?)?\d{3,5}[\s\-]?\d{3,5}', text)
    return m.group(0) if m else "Not found"

def extract_keywords(text):
    words = word_tokenize(text.lower())
    keywords = [w for w in words if w.isalpha() and w not in stop_words]
    return list(set(keywords))

def match_resume_to_job(resume_keywords, jd_skills):
    embeddings_resume = model.encode(resume_keywords, convert_to_tensor=True)
    embeddings_jd = model.encode(jd_skills, convert_to_tensor=True)
    similarity = util.cos_sim(embeddings_resume, embeddings_jd).cpu().numpy()
    matched = []
    for i, skill in enumerate(jd_skills):
        if any(similarity[:,i] > 0.7):
            matched.append(skill)
    missing = [s for s in jd_skills if s not in matched]
    score = round(len(matched)/len(jd_skills)*100,2) if jd_skills else 0
    return matched, missing, score

# -------------------- AUTH --------------------

def register_user(email, password, role):
    hashed = hash_password(password)
    res = supabase.table("users").insert({"email": email, "password_hash": hashed, "role": role}).execute()
    return res

def login_user(email, password):
    hashed = hash_password(password)
    user = supabase.table("users").select("*").eq("email", email).eq("password_hash", hashed).execute()
    if user.data:
        return user.data[0]
    return None

# -------------------- SIDEBAR --------------------
def logout_sidebar():
    with st.sidebar:
        if "username" in st.session_state:
            st.markdown(f"👤 Logged in as `{st.session_state['username']}`")
            if st.button("🚪 Logout"):
                st.session_state.clear()
                st.rerun()

# -------------------- UI --------------------
def login_register_ui():
    st.title("🔐 AI Resume Analyzer")
    tabs = st.tabs(["🔑 Login", "🆕 Register"])
    with tabs[0]:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login_user(email, password)
            if user:
                st.session_state["username"] = user["email"]
                st.session_state["role"] = user["role"]
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    with tabs[1]:
        email = st.text_input("Email", key="reg_email")
        password = st.text_input("Password", type="password", key="reg_pass")
        role = st.selectbox("Select Role", ["User","HR"], key="reg_role")
        if st.button("Register"):
            register_user(email, password, role)
            st.success("Registered successfully! Please login.")

# -------------------- USER VIEW --------------------
def user_view():
    st.header("👤 Candidate Dashboard")
    # Fetch jobs
    jobs = supabase.table("jobs").select("*").execute().data
    if not jobs:
        st.info("No job postings available.")
        return
    job_options = [f"{job['title']} — {job['description']}" for job in jobs]
    selected_job = st.selectbox("Select Job", job_options)
    job = jobs[job_options.index(selected_job)]
    
    uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf","docx"])
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else extract_text_from_docx(uploaded_file)
        name = extract_name(text)
        email_in_resume = extract_email_from_text(text)
        phone = extract_phone_from_text(text)
        resume_keywords = extract_keywords(text)
        matched, missing, score = match_resume_to_job(resume_keywords, job["skills"])
        
        # Show summary
        st.markdown(f"**Name:** {name}")
        st.markdown(f"**Email:** {email_in_resume}")
        st.markdown(f"**Phone:** {phone}")
        st.markdown(f"**Matched Skills:** {', '.join(matched) if matched else 'None'}")
        st.markdown(f"**Missing Skills:** {', '.join(missing) if missing else 'None'}")
        st.markdown(f"**Match Score:** {score}%")
        
        # Submit application to Supabase
        if st.button("Submit Application"):
            supabase.table("applications").insert({
                "user_id": None,  # can link user id if needed
                "resume_url": None,  # not storing resume
                "match_score": score,
                "phase": "Round 1 (Pending)",
                "submission_date": datetime.utcnow(),
                "job_id": job["uuid"]
            }).execute()
            st.success("Application submitted!")

# -------------------- HR VIEW --------------------
def hr_view():
    st.header("🏢 HR Dashboard")
    apps = supabase.table("applications").select("*").execute().data
    if not apps:
        st.info("No applications yet.")
        return
    for app in apps:
        st.markdown(f"**Job ID:** {app['job_id']} | **Score:** {app['match_score']} | **Phase:** {app['phase']}")

# -------------------- MAIN --------------------
if "username" not in st.session_state:
    login_register_ui()
else:
    logout_sidebar()
    if st.session_state["role"]=="User":
        user_view()
    elif st.session_state["role"]=="HR":
        hr_view()
    else:
        st.error("Unknown role")
