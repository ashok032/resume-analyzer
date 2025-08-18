import streamlit as st
import pandas as pd
import hashlib
import pdfplumber
import docx
import re
import spacy
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
import os

# Load SpaCy
nlp = spacy.load("en_core_web_sm")

# Load semantic model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Supabase connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Utility functions
# -----------------------------

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(email, password, role):
    hashed_pw = hash_password(password)
    existing = supabase.table("users").select("*").eq("email", email).execute()
    if existing.data:
        return False, "Email already exists."
    supabase.table("users").insert({
        "email": email,
        "password_hash": hashed_pw,
        "role": role
    }).execute()
    return True, "Registration successful."

def login_user(email, password):
    hashed_pw = hash_password(password)
    res = supabase.table("users").select("*").eq("email", email).eq("password_hash", hashed_pw).execute()
    if res.data:
        return True, res.data[0]
    return False, None

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_name_email_phone(text):
    name = None
    email = None
    phone = None

    # Email
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if email_match:
        email = email_match.group(0)

    # Phone
    phone_match = re.search(r"\+?\d[\d -]{8,}\d", text)
    if phone_match:
        phone = phone_match.group(0)

    # Name (NER)
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    return name, email, phone

def extract_keywords(text):
    doc = nlp(text.lower())
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return list(set(keywords))

def match_resume_to_job(resume_keywords, jd_skills):
    if not resume_keywords or not jd_skills:
        return [], jd_skills, 0.0

    resume_emb = model.encode(resume_keywords, convert_to_tensor=True)
    jd_emb = model.encode(jd_skills, convert_to_tensor=True)

    sim = util.cos_sim(resume_emb, jd_emb)
    matched = []
    for i, skill in enumerate(jd_skills):
        if max(sim[:, i]).item() > 0.6:
            matched.append(skill)

    missing = [s for s in jd_skills if s not in matched]
    score = (len(matched) / len(jd_skills)) * 100 if jd_skills else 0

    return matched, missing, round(score, 2)

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Resume Analyzer", layout="wide")

if "user" not in st.session_state:
    st.session_state.user = None

st.title("📄 AI Resume Analyzer")

if not st.session_state.user:
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            success, user = login_user(email, password)
            if success:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid credentials.")

    with tab_register:
        new_email = st.text_input("New Email")
        new_password = st.text_input("New Password", type="password")
        role = st.selectbox("Role", ["User", "HR"])
        if st.button("Register"):
            success, msg = register_user(new_email, new_password, role)
            if success:
                st.success(msg)
            else:
                st.error(msg)

else:
    user = st.session_state.user
    st.sidebar.write(f"👤 Logged in as {user['email']} ({user['role']})")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

    # -----------------------------
    # User View
    # -----------------------------
    if user["role"] == "User":
        st.header("Candidate Dashboard")
        uploaded = st.file_uploader("Upload Resume", type=["pdf", "docx"])

        job_roles = supabase.table("job_descriptions").select("*").execute().data
        role_names = [j["role"] for j in job_roles]
        selected_role = st.selectbox("Select Job Role", role_names)

        if uploaded and selected_role:
            if uploaded.name.endswith(".pdf"):
                text = extract_text_from_pdf(uploaded)
            else:
                text = extract_text_from_docx(uploaded)

            name, email, phone = extract_name_email_phone(text)
            jd = [j for j in job_roles if j["role"] == selected_role][0]
            jd_skills = jd["skills"]

            resume_keywords = extract_keywords(text)
            matched, missing, score = match_resume_to_job(resume_keywords, jd_skills)

            st.subheader("📊 Results")
            st.write(f"**Name:** {name}")
            st.write(f"**Email:** {email}")
            st.write(f"**Phone:** {phone}")
            st.write(f"**Match Score:** {score}%")
            st.write(f"✅ Matched Skills: {', '.join(matched)}")
            st.write(f"❌ Missing Skills: {', '.join(missing)}")

            # Save application in DB
            supabase.table("applications").insert({
                "user_id": user["id"],
                "job_role": selected_role,
                "resume_url": uploaded.name,
                "match_score": score,
                "phase": "Round 1" if score > 70 else "Rejected"
            }).execute()

    # -----------------------------
    # HR View
    # -----------------------------
    elif user["role"] == "HR":
        st.header("HR Dashboard")
        apps = supabase.table("applications").select("*, users(email)").execute().data
        df = pd.DataFrame(apps)
        if not df.empty:
            st.dataframe(df)
        else:
            st.info("No applications yet.")
