import streamlit as st
import pandas as pd
import hashlib
import pdfplumber
import docx
import re
import spacy
import nltk
import smtplib
from email.mime.text import MIMEText
from sentence_transformers import SentenceTransformer, util
from supabase import create_client, Client

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load sentence transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect Supabase
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Utility functions
# -----------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def extract_text_from_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text

def extract_text_from_docx(file) -> str:
    doc = docx.Document(file)
    return " ".join([p.text for p in doc.paragraphs])

def extract_contact_info(text: str):
    doc = nlp(text)
    name = None
    email = None
    phone = None
    
    # Name
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    # Email
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if email_match:
        email = email_match.group(0)

    # Phone
    phone_match = re.search(r"\+?\d[\d -]{8,}\d", text)
    if phone_match:
        phone = phone_match.group(0)

    return name, email, phone

def semantic_match(resume_skills, jd_skills):
    if not resume_skills or not jd_skills:
        return [], jd_skills, 0
    
    resume_emb = model.encode(resume_skills, convert_to_tensor=True)
    jd_emb = model.encode(jd_skills, convert_to_tensor=True)

    sim_matrix = util.cos_sim(resume_emb, jd_emb)
    matched = []
    for i, jd_skill in enumerate(jd_skills):
        best_match_idx = sim_matrix[:, i].argmax().item()
        if sim_matrix[best_match_idx, i] > 0.6:  # similarity threshold
            matched.append(jd_skill)
    missing = list(set(jd_skills) - set(matched))
    score = int((len(matched) / len(jd_skills)) * 100)
    return matched, missing, score

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

st.title("📄 AI Resume Analyzer")

tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])

with tab1:
    st.subheader("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    role = st.selectbox("Role", ["User", "HR"])
    if st.button("Login"):
        hashed_pw = hash_password(password)
        res = supabase.table("users").select("*").eq("email", email).eq("password", hashed_pw).eq("role", role).execute()
        if res.data:
            st.session_state["logged_in"] = True
            st.session_state["role"] = role
            st.session_state["email"] = email
            st.rerun()
        else:
            st.error("Invalid credentials")

with tab2:
    st.subheader("Register")
    reg_email = st.text_input("Register Email")
    reg_password = st.text_input("Register Password", type="password")
    reg_role = st.selectbox("Register as", ["User", "HR"])
    if st.button("Register"):
        hashed_pw = hash_password(reg_password)
        # Prevent duplicate emails
        res = supabase.table("users").select("*").eq("email", reg_email).execute()
        if res.data:
            st.error("Email already registered")
        else:
            supabase.table("users").insert({"email": reg_email, "password": hashed_pw, "role": reg_role}).execute()
            st.success("Registered successfully! Please login.")

# -----------------------------
# Post-login Dashboard
# -----------------------------
if "logged_in" in st.session_state and st.session_state["logged_in"]:
    st.sidebar.success(f"👤 Logged in as {st.session_state['email']} ({st.session_state['role']})")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    if st.session_state["role"] == "User":
        st.header("📤 Upload Resume")
        uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            else:
                text = extract_text_from_docx(uploaded_file)

            name, email, phone = extract_contact_info(text)
            st.write(f"**Extracted Name:** {name}")
            st.write(f"**Extracted Email:** {email}")
            st.write(f"**Extracted Phone:** {phone}")

            # Example JD skills
            jd_skills = ["Python", "Machine Learning", "SQL", "Communication"]
            resume_skills = [token.text for token in nlp(text) if token.is_alpha]

            matched, missing, score = semantic_match(resume_skills, jd_skills)

            st.metric("Match Score", f"{score}%")
            st.write("✅ Matched Skills:", matched)
            st.write("❌ Missing Skills:", missing)

    elif st.session_state["role"] == "HR":
        st.header("👔 HR Dashboard")
        st.info("Here HR can review candidates, schedule interviews, and update status.")
