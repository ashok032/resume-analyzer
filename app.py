import streamlit as st
import pandas as pd
import hashlib
import pdfplumber
import docx
import re
import spacy
from datetime import datetime
from supabase import create_client
from sentence_transformers import SentenceTransformer, util
import nltk

# Download NLTK stopwords if not already
nltk.download("punkt")
nltk.download("stopwords")

# ========================
# 1. SUPABASE CONNECTION
# ========================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ========================
# 2. UTILITIES
# ========================
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def extract_text_from_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file) -> str:
    doc = docx.Document(file)
    return " ".join([para.text for para in doc.paragraphs])

def extract_text(file) -> str:
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       "application/msword"]:
        return extract_text_from_docx(file)
    else:
        return ""

def extract_name(text: str) -> str:
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    # fallback: first line heuristic
    first_line = text.strip().split("\n")[0]
    return first_line.strip()

def extract_email(text: str) -> str:
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else None

def extract_phone(text: str) -> str:
    match = re.search(r"\+?\d[\d\s-]{8,}\d", text)
    return match.group(0) if match else None

def extract_skills(text: str):
    words = nltk.word_tokenize(text.lower())
    return list(set(words))

def match_resume_to_job(resume_skills, jd_skills):
    if not resume_skills or not jd_skills:
        return [], jd_skills, 0.0
    resume_embeddings = embedder.encode(resume_skills, convert_to_tensor=True)
    jd_embeddings = embedder.encode(jd_skills, convert_to_tensor=True)

    cosine_scores = util.cos_sim(resume_embeddings, jd_embeddings)
    matched = []
    for i, jd_skill in enumerate(jd_skills):
        if max(cosine_scores[:, i]).item() > 0.6:
            matched.append(jd_skill)

    missing = list(set(jd_skills) - set(matched))
    score = (len(matched) / len(jd_skills)) * 100 if jd_skills else 0
    return matched, missing, round(score, 2)

# ========================
# 3. AUTHENTICATION
# ========================
def register_user(email, password, role):
    hashed = hash_password(password)
    response = supabase.table("users").insert({
        "email": email,
        "password_hash": hashed,
        "role": role
    }).execute()
    return response

def login_user(email, password):
    hashed = hash_password(password)
    response = supabase.table("users").select("*").eq("email", email).eq("password_hash", hashed).execute()
    if response.data:
        return response.data[0]
    return None

# ========================
# 4. STREAMLIT APP
# ========================
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

if "user" not in st.session_state:
    st.session_state.user = None

# Sidebar
with st.sidebar:
    st.title("🔐 Resume Analyzer")
    if st.session_state.user:
        st.write(f"👤 {st.session_state.user['email']}")
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()

# Login / Register
if not st.session_state.user:
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login_user(email, password)
            if user:
                st.session_state.user = user
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        email = st.text_input("Email (register)")
        password = st.text_input("Password (register)", type="password")
        role = st.selectbox("Role", ["User", "HR"])
        if st.button("Register"):
            try:
                register_user(email, password, role)
                st.success("Registered! Please login.")
            except Exception as e:
                st.error("Registration failed. Maybe email already exists.")

# ========================
# 5. DASHBOARD
# ========================
else:
    role = st.session_state.user["role"]
    st.header(f"Welcome, {role}")

    # HR View
    if role == "HR":
        st.subheader("📂 Manage Applications")

        job_roles = supabase.table("job_descriptions").select("role").execute().data
        job_options = [j["role"] for j in job_roles]
        selected_job = st.selectbox("Select Job Role", job_options)

        resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
        if resume_file and selected_job:
            text = extract_text(resume_file)
            name = extract_name(text)
            email = extract_email(text)
            phone = extract_phone(text)
            resume_keywords = extract_skills(text)

            jd = supabase.table("job_descriptions").select("*").eq("role", selected_job).execute().data[0]
            jd_skills = jd["skills"]

            matched, missing, score = match_resume_to_job(resume_keywords, jd_skills)

            st.write(f"**Candidate:** {name}")
            st.write(f"📧 {email}")
            st.write(f"📱 {phone}")
            st.write(f"✅ Matched Skills: {matched}")
            st.write(f"❌ Missing Skills: {missing}")
            st.metric("Match Score", f"{score}%")

            if st.button("Submit Application"):
                supabase.table("applications").insert({
                    "user_id": st.session_state.user["id"],
                    "job_role": selected_job,
                    "resume_url": resume_file.name,
                    "match_score": score,
                    "phase": "Round 1" if score > 70 else "Rejected"
                }).execute()
                st.success("Application submitted & phase set.")

    # User View
    elif role == "User":
        st.subheader("📄 My Applications")
        apps = supabase.table("applications").select("*").eq("user_id", st.session_state.user["id"]).execute().data
        if apps:
            df = pd.DataFrame(apps)
            st.dataframe(df[["job_role", "match_score", "phase", "submission_date"]])
        else:
            st.info("No applications submitted yet.")
