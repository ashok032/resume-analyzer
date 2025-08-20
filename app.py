import streamlit as st
import pandas as pd
import hashlib
import pdfplumber
import docx
import re
import spacy
import nltk
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
from datetime import datetime
import smtplib
from email.mime.text import MIMEText

# =============================
# INIT
# =============================
nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Supabase
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Email
EMAIL = st.secrets["EMAIL"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]


# =============================
# HELPERS
# =============================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def send_email(to_email, subject, body):
    msg = MIMEText(body)
    msg["From"] = EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL, EMAIL_PASSWORD)
        server.sendmail(EMAIL, to_email, msg.as_string())


def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            return " ".join(page.extract_text() or "" for page in pdf.pages)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return " ".join([para.text for para in doc.paragraphs])
    return ""


def extract_name_email_phone(text):
    doc = nlp(text)
    name = ""
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break
    email = re.search(r"[\w\.-]+@[\w\.-]+", text)
    phone = re.search(r"\+?\d[\d -]{8,}\d", text)
    return name, email.group() if email else "", phone.group() if phone else ""


def extract_skills(text):
    tokens = nltk.word_tokenize(text.lower())
    return list(set(tokens))


def compute_match_score(resume_skills, job_skills):
    if not resume_skills or not job_skills:
        return 0
    resume_emb = model.encode(resume_skills, convert_to_tensor=True)
    job_emb = model.encode(job_skills, convert_to_tensor=True)
    sim = util.cos_sim(resume_emb, job_emb)
    return float(sim.mean().item() * 100)


def get_user(email):
    res = supabase.table("users").select("*").eq("email", email).execute()
    return res.data[0] if res.data else None


def insert_user(email, password, role):
    supabase.table("users").insert({
        "email": email,
        "password_hash": hash_password(password),
        "role": role
    }).execute()


# =============================
# STREAMLIT APP
# =============================
st.set_page_config(page_title="Resume Analyzer", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None

st.sidebar.title("Navigation")

if not st.session_state.logged_in:
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = get_user(email)
            if user and user["password_hash"] == hash_password(password):
                st.session_state.logged_in = True
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab_register:
        new_email = st.text_input("New Email")
        new_password = st.text_input("New Password", type="password")
        role = st.selectbox("Role", ["User", "HR"])
        if st.button("Register"):
            if get_user(new_email):
                st.error("Email already exists")
            else:
                insert_user(new_email, new_password, role)
                st.success("Registration successful. Please login.")
else:
    user = st.session_state.user
    st.sidebar.write(f"👤 Logged in as {user['email']} ({user['role']})")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.rerun()

    if user["role"] == "User":
        st.header("Upload Resume")
        job_roles = supabase.table("job_descriptions").select("role").execute().data
        job_role = st.selectbox("Select Job Role", [j["role"] for j in job_roles])

        uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
        if uploaded_file and st.button("Submit Resume"):
            text = extract_text_from_file(uploaded_file)
            name, email, phone = extract_name_email_phone(text)
            resume_skills = extract_skills(text)

            job_data = supabase.table("job_descriptions").select("*").eq("role", job_role).execute().data[0]
            match_score = compute_match_score(resume_skills, job_data["skills"])

            phase = "Round 1 Scheduled" if match_score > 70 else "Rejected"
            supabase.table("applications").insert({
                "user_id": user["id"],
                "job_role": job_role,
                "resume_url": uploaded_file.name,
                "match_score": match_score,
                "phase": phase,
                "submission_date": datetime.utcnow().isoformat()
            }).execute()

            st.success(f"Submitted! Match Score: {match_score:.2f}% | Phase: {phase}")

    elif user["role"] == "HR":
        st.header("HR Dashboard")
        applications = supabase.table("applications").select("*, users(email)").execute().data
        if applications:
            df = pd.DataFrame(applications)
            st.dataframe(df)

            app_id = st.selectbox("Select Application ID", [a["id"] for a in applications])
            new_phase = st.selectbox("Update Phase", ["Round 1", "Round 2", "Final", "Rejected"])
            if st.button("Update Phase"):
                supabase.table("applications").update({"phase": new_phase}).eq("id", app_id).execute()
                st.success("Phase updated")

                # send email
                app = [a for a in applications if a["id"] == app_id][0]
                user_email = app["users"]["email"]
                send_email(user_email, "Application Update", f"Your application phase is now: {new_phase}")
