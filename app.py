import streamlit as st
import pandas as pd
import hashlib
import os
import pdfplumber
import docx
import re
import spacy
import nltk
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from supabase import create_client, Client
from dotenv import load_dotenv
from job_descriptions import job_descriptions

# Load environment variables
load_dotenv()

# Supabase connection
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Load NLP models
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Helpers ----------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def extract_text_from_file(uploaded_file):
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

def extract_contact_info(text):
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    phone_match = re.search(r'\+?\d[\d -]{8,}\d', text)
    return (
        email_match.group(0) if email_match else "",
        phone_match.group(0) if phone_match else "",
    )

def extract_name(text):
    doc = nlp(text[:500])  # only first few lines
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    # fallback heuristic: first line with 2 words
    for line in text.split("\n")[:5]:
        if len(line.split()) >= 2:
            return line.strip()
    return "Unknown"

def extract_skills(text):
    words = nltk.word_tokenize(text.lower())
    return list(set(words))

def match_resume_to_job(resume_skills, jd_skills):
    if not resume_skills or not jd_skills:
        return [], jd_skills, 0
    resume_emb = embedder.encode(resume_skills, convert_to_tensor=True)
    jd_emb = embedder.encode(jd_skills, convert_to_tensor=True)
    cos_sim = util.cos_sim(resume_emb, jd_emb)
    matched, missing = [], []
    for i, skill in enumerate(jd_skills):
        if cos_sim[:, i].max().item() > 0.6:
            matched.append(skill)
        else:
            missing.append(skill)
    score = int(len(matched) / len(jd_skills) * 100)
    return matched, missing, score

def send_email(to_email, subject, body):
    sender = os.getenv("EMAIL")
    password = os.getenv("EMAIL_PASSWORD")
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, to_email, msg.as_string())
        server.quit()
    except Exception as e:
        st.error(f"❌ Email send failed: {e}")

# ---------- Auth ----------
def register_user(email, password, role):
    hashed_pw = hash_password(password)
    existing = supabase.table("users").select("*").eq("email", email).execute()
    if existing.data:
        return False, "Email already registered!"
    supabase.table("users").insert({
        "email": email,
        "password": hashed_pw,
        "role": role
    }).execute()
    return True, "Registration successful!"

def login_user(email, password):
    hashed_pw = hash_password(password)
    res = supabase.table("users").select("*").eq("email", email).eq("password", hashed_pw).execute()
    return res.data[0] if res.data else None

# ---------- UI ----------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("📄 AI Resume Analyzer")

if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login_user(email, password)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid credentials")
    with tab2:
        email = st.text_input("New Email")
        password = st.text_input("New Password", type="password")
        role = st.selectbox("Role", ["User", "HR"])
        if st.button("Register"):
            ok, msg = register_user(email, password, role)
            st.success(msg) if ok else st.error(msg)

else:
    user = st.session_state.user
    with st.sidebar:
        st.write(f"👤 Logged in as: {user['email']} ({user['role']})")
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()

    if user["role"] == "User":
        st.subheader("📂 Upload Resume & Apply")
        uploaded = st.file_uploader("Upload Resume", type=["pdf", "docx"])
        job_role = st.selectbox("Select Job Role", list(job_descriptions.keys()))
        if uploaded and job_role and st.button("Submit"):
            text = extract_text_from_file(uploaded)
            name = extract_name(text)
            email, phone = extract_contact_info(text)
            resume_skills = extract_skills(text)
            jd_skills = job_descriptions[job_role]["skills"]
            matched, missing, score = match_resume_to_job(resume_skills, jd_skills)

            # save to Supabase
            supabase.table("applications").insert({
                "user_email": user["email"],
                "name": name,
                "resume_email": email,
                "phone": phone,
                "job_role": job_role,
                "score": score,
                "phase": "Round 1" if score > 70 else "Rejected",
                "timestamp": str(datetime.now())
            }).execute()

            if score > 70:
                send_email(email, f"Interview Scheduled - {job_role}",
                           f"Dear {name},\n\nCongratulations! Your resume passed screening. Round 1 interview scheduled.\n\n- HR Team")
                st.success(f"✅ Resume submitted. Match Score: {score}% — Round 1 Scheduled!")
            else:
                send_email(email, f"Application Update - {job_role}",
                           f"Dear {name},\n\nThank you for applying. Unfortunately, your resume did not pass the screening.\n\n- HR Team")
                st.warning(f"❌ Resume submitted. Match Score: {score}% — Not Selected.")

        # Show past applications
        st.subheader("📊 Application Status")
        apps = supabase.table("applications").select("*").eq("user_email", user["email"]).execute()
        if apps.data:
            st.dataframe(pd.DataFrame(apps.data))

    elif user["role"] == "HR":
        st.subheader("👔 HR Dashboard")
        apps = supabase.table("applications").select("*").execute()
        df = pd.DataFrame(apps.data)
        if not df.empty:
            st.dataframe(df)
            app_id = st.selectbox("Select Application ID", df["id"] if "id" in df else [])
            decision = st.radio("Mark Result", ["Pass", "Fail"])
            if st.button("Update Result"):
                app_row = [a for a in apps.data if a["id"] == app_id][0]
                email = app_row["resume_email"]
                name = app_row["name"]
                job_role = app_row["job_role"]
                if decision == "Pass":
                    if app_row["phase"] == "Round 1":
                        new_phase = "Round 2"
                        subject = f"Round 2 Interview Scheduled - {job_role}"
                        body = f"Dear {name},\n\nCongratulations! You passed Round 1. Round 2 interview scheduled.\n\n- HR Team"
                    elif app_row["phase"] == "Round 2":
                        new_phase = "Final Round"
                        subject = f"Final Interview Scheduled - {job_role}"
                        body = f"Dear {name},\n\nGreat work! You passed Round 2. Final round interview scheduled.\n\n- HR Team"
                    else:
                        new_phase = "Selected"
                        subject = f"Job Offer - {job_role}"
                        body = f"Dear {name},\n\nCongratulations! You passed all rounds. Our team will reach out with the offer.\n\n- HR Team"
                else:
                    new_phase = "Rejected"
                    subject = f"Application Update - {job_role}"
                    body = f"Dear {name},\n\nThank you for interviewing. Unfortunately, you have not been selected.\n\n- HR Team"

                supabase.table("applications").update({"phase": new_phase}).eq("id", app_id).execute()
                send_email(email, subject, body)
                st.success(f"✅ Application updated: {new_phase}")
                st.rerun()
