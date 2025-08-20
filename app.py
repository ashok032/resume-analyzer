import streamlit as st
import pandas as pd
import hashlib
import pdfplumber
import docx
import re
import spacy
from supabase import create_client, Client
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import nltk

# Initialize NLP & model
nlp = spacy.load("en_core_web_sm")
nltk.download("punkt", quiet=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- Supabase Connection ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Helpers ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_contact_info(text):
    name, email, phone = None, None, None
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not name:
            name = ent.text
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    phone_match = re.search(r"\+?\d[\d \-\(\)]{8,}\d", text)
    if email_match: email = email_match.group(0)
    if phone_match: phone = phone_match.group(0)
    return name, email, phone

def get_job_descriptions():
    res = supabase.table("job_descriptions").select("*").execute()
    return {row["role"]: row for row in res.data} if res.data else {}

def match_resume_to_job(resume_text, jd_skills):
    resume_sentences = nltk.sent_tokenize(resume_text)
    jd_embeddings = model.encode(jd_skills, convert_to_tensor=True)
    resume_embeddings = model.encode(resume_sentences, convert_to_tensor=True)

    scores = []
    for jd_emb in jd_embeddings:
        sim = util.cos_sim(jd_emb, resume_embeddings).max().item()
        scores.append(sim)
    avg_score = round((sum(scores) / len(scores)) * 100, 2) if scores else 0
    matched = [s for s in jd_skills if s.lower() in resume_text.lower()]
    missing = [s for s in jd_skills if s.lower() not in resume_text.lower()]
    return matched, missing, avg_score

# --- UI ---
st.title("📄 AI Resume Analyzer")

# Tabs for Login / Register
tab1, tab2 = st.tabs(["Login", "Register"])

with tab1:
    st.subheader("Login")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login"):
        res = supabase.table("users").select("*").eq("email", email).execute()
        if res.data and res.data[0]["password"] == hash_password(password):
            st.session_state["user"] = res.data[0]
            st.success(f"Welcome {email}!")
            st.rerun()
        else:
            st.error("Invalid credentials")

with tab2:
    st.subheader("Register")
    new_email = st.text_input("Email", key="reg_email")
    new_pass = st.text_input("Password", type="password", key="reg_pass")
    role = st.selectbox("Role", ["User", "HR"])
    if st.button("Register"):
        res = supabase.table("users").select("*").eq("email", new_email).execute()
        if res.data:
            st.error("Email already registered")
        else:
            supabase.table("users").insert({
                "email": new_email,
                "password": hash_password(new_pass),
                "role": role
            }).execute()
            st.success("Account created! Please login.")

# --- Main after login ---
if "user" in st.session_state:
    user = st.session_state["user"]
    st.sidebar.write(f"👤 Logged in as {user['email']} ({user['role']})")
    if st.sidebar.button("Logout"):
        st.session_state.pop("user")
        st.rerun()

    if user["role"] == "User":
        st.header("Upload Resume")
        uploaded = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
        job_roles = get_job_descriptions()
        role = st.selectbox("Select Job Role", list(job_roles.keys()))

        if uploaded and st.button("Submit Resume"):
            text = extract_text_from_pdf(uploaded) if uploaded.type == "application/pdf" else extract_text_from_docx(uploaded)
            name, resume_email, phone = extract_contact_info(text)
            jd_skills = job_roles[role]["skills"]
            matched, missing, score = match_resume_to_job(text, jd_skills)

            supabase.table("applications").insert({
                "user_email": user["email"],
                "candidate_name": name,
                "candidate_email": resume_email,
                "phone": phone,
                "job_role": role,
                "score": score,
                "phase": "Submitted",
                "submitted_at": datetime.utcnow().isoformat()
            }).execute()

            st.success("✅ Resume submitted!")
            st.write(f"Match Score: {score}%")
            st.write("Matched Skills:", matched)
            st.write("Missing Skills:", missing)

    elif user["role"] == "HR":
        st.header("Applications")
        res = supabase.table("applications").select("*").execute()
        df = pd.DataFrame(res.data) if res.data else pd.DataFrame()
        st.dataframe(df)

    else:
        st.error("Unknown role")
