import streamlit as st
import pandas as pd
import hashlib
import os
import pdfplumber
import docx
import re
import spacy
import en_core_web_sm   # ✅ Fixed for Streamlit Cloud
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from job_descriptions import job_descriptions
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords

# ===================== SETUP =====================
nltk.download("punkt")
nltk.download("stopwords")
nlp = en_core_web_sm.load()
model = SentenceTransformer("all-MiniLM-L6-v2")
USERS_FILE = "users.csv"
APPLICATIONS_FILE = "applications.csv"

# ===================== HELPERS =====================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        return pd.DataFrame(columns=["email", "password", "role"])
    return pd.read_csv(USERS_FILE)

def save_users(df):
    df.to_csv(USERS_FILE, index=False)

def load_applications():
    if not os.path.exists(APPLICATIONS_FILE):
        return pd.DataFrame(columns=["email", "role", "resume_text", "job_role", "score", "matched_skills", "missing_skills", "phase", "status", "candidate_name"])
    return pd.read_csv(APPLICATIONS_FILE)

def save_applications(df):
    df.to_csv(APPLICATIONS_FILE, index=False)

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_contact_info(text):
    # Email
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    email = email_match.group(0) if email_match else None

    # Phone
    phone_match = re.search(r"(\+?\d{1,3}[-.\s]?)?\d{10}", text)
    phone = phone_match.group(0) if phone_match else None

    # Name via spaCy NER
    doc = nlp(text)
    name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    return name, email, phone

def extract_keywords(text):
    words = nltk.word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    keywords = [w for w in words if w.isalpha() and w not in stop_words]
    return list(set(keywords))

def match_resume_to_job(resume_keywords, jd_skills):
    resume_embeddings = model.encode(resume_keywords, convert_to_tensor=True)
    jd_embeddings = model.encode(jd_skills, convert_to_tensor=True)
    cosine_sim = util.cos_sim(resume_embeddings, jd_embeddings)

    matched = []
    missing = []
    for i, skill in enumerate(jd_skills):
        if cosine_sim[:, i].max().item() > 0.6:
            matched.append(skill)
        else:
            missing.append(skill)

    score = (len(matched) / len(jd_skills)) * 100 if jd_skills else 0
    return matched, missing, round(score, 2)

def send_email(to_email, subject, body):
    try:
        from_email = st.secrets["EMAIL_USER"]
        app_password = st.secrets["EMAIL_PASS"]

        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(from_email, app_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"❌ Email error: {e}")
        return False

# ===================== UI =====================
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.email = None

st.sidebar.title("🔐 Login / Register")

if not st.session_state.logged_in:
    tab1, tab2 = st.sidebar.tabs(["Login", "Register"])

    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            df = load_users()
            hashed = hash_password(password)
            user = df[(df["email"] == email) & (df["password"] == hashed)]
            if not user.empty:
                st.session_state.logged_in = True
                st.session_state.email = email
                st.session_state.role = user.iloc[0]["role"]
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        email = st.text_input("New Email")
        password = st.text_input("New Password", type="password")
        role = st.selectbox("Role", ["User", "HR"])
        if st.button("Register"):
            df = load_users()
            if email in df["email"].values:
                st.error("Email already exists")
            else:
                new_user = pd.DataFrame([[email, hash_password(password), role]], columns=["email", "password", "role"])
                df = pd.concat([df, new_user], ignore_index=True)
                save_users(df)
                st.success("Registration successful. Please login.")

else:
    st.sidebar.write(f"👤 Logged in as: {st.session_state.email}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # ===================== USER VIEW =====================
    if st.session_state.role == "User":
        st.title("📄 Resume Analyzer - User View")

        df_app = load_applications()
        user_apps = df_app[df_app["email"] == st.session_state.email]

        st.subheader("Your Applications")
        if not user_apps.empty:
            st.dataframe(user_apps[["job_role", "score", "phase", "status"]])
        else:
            st.info("No applications submitted yet.")

        uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
        job_role = st.selectbox("Select Job Role", list(job_descriptions.keys()))

        if uploaded_file and st.button("Submit Resume"):
            file_text = extract_text_from_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else extract_text_from_docx(uploaded_file)
            name, email, phone = extract_contact_info(file_text)
            resume_keywords = extract_keywords(file_text)
            jd_skills = job_descriptions[job_role]["skills"]
            matched, missing, score = match_resume_to_job(resume_keywords, jd_skills)

            phase = "Round 1" if score > 70 else "Rejected"
            status = "Pending" if score > 70 else "Rejected"

            new_app = pd.DataFrame([[
                st.session_state.email, "User", file_text, job_role, score,
                ",".join(matched), ",".join(missing), phase, status, name
            ]], columns=["email", "role", "resume_text", "job_role", "score", "matched_skills", "missing_skills", "phase", "status", "candidate_name"])

            df_app = pd.concat([df_app, new_app], ignore_index=True)
            save_applications(df_app)

            st.success(f"Resume submitted! Match Score: {score}%")
            if score > 70:
                send_email(email, f"Interview Scheduled - {job_role}", f"Dear {name},\n\nYou are shortlisted for {job_role}. Round 1 scheduled.\n\nBest Regards,\nHR Team")
            else:
                send_email(email, f"Application Update - {job_role}", f"Dear {name},\n\nWe regret to inform you that you were not shortlisted for {job_role}.\n\nBest Regards,\nHR Team")

    # ===================== HR VIEW =====================
    elif st.session_state.role == "HR":
        st.title("👔 HR Dashboard")

        df_app = load_applications()
        st.subheader("Candidate Applications")
        st.dataframe(df_app[["candidate_name", "email", "job_role", "score", "phase", "status"]])

        candidate_email = st.selectbox("Select Candidate", df_app["email"].unique() if not df_app.empty else [])
        if candidate_email:
            cand = df_app[df_app["email"] == candidate_email].iloc[0]
            st.write(f"**Candidate:** {cand['candidate_name']}")
            st.write(f"**Job Role:** {cand['job_role']}")
            st.write(f"**Score:** {cand['score']}%")
            st.write(f"**Phase:** {cand['phase']}")
            st.write(f"**Status:** {cand['status']}")

            result = st.radio("Mark Result", ["Pass", "Fail"])
            if st.button("Submit Result"):
                if result == "Pass":
                    if cand["phase"] == "Round 1":
                        df_app.loc[df_app["email"] == candidate_email, "phase"] = "Round 2"
                        send_email(cand["email"], "Round 2 Interview", f"Dear {cand['candidate_name']},\n\nYou have passed Round 1. Round 2 scheduled.\n\nBest Regards,\nHR Team")
                    elif cand["phase"] == "Round 2":
                        df_app.loc[df_app["email"] == candidate_email, "phase"] = "Final Round"
                        send_email(cand["email"], "Final Interview", f"Dear {cand['candidate_name']},\n\nYou have passed Round 2. Final round scheduled.\n\nBest Regards,\nHR Team")
                    elif cand["phase"] == "Final Round":
                        df_app.loc[df_app["email"] == candidate_email, "phase"] = "Selected"
                        send_email(cand["email"], "Congratulations!", f"Dear {cand['candidate_name']},\n\nCongratulations! You are selected for {cand['job_role']}.\n\nBest Regards,\nHR Team")
                else:
                    df_app.loc[df_app["email"] == candidate_email, "phase"] = "Rejected"
                    df_app.loc[df_app["email"] == candidate_email, "status"] = "Rejected"
                    send_email(cand["email"], "Application Update", f"Dear {cand['candidate_name']},\n\nWe regret to inform you that you have been rejected.\n\nBest Regards,\nHR Team")

                save_applications(df_app)
                st.success("Result updated successfully!")

    else:
        st.error("Unknown role")
