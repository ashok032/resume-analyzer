import streamlit as st
import pandas as pd
import hashlib
import os
import pdfplumber
import docx
import re
import spacy
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from job_descriptions import job_descriptions
from match import extract_keywords, match_resume_to_job

# ✅ Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# Helpers
# -----------------------------
USERS_FILE = "users.csv"
APPLICATIONS_FILE = "applications.csv"

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
        return pd.DataFrame(columns=["email", "name", "job_role", "match_score", "matched_skills",
                                     "missing_skills", "phase", "status"])
    return pd.read_csv(APPLICATIONS_FILE)

def save_applications(df):
    df.to_csv(APPLICATIONS_FILE, index=False)

def send_email(to_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg["From"] = st.secrets["EMAIL_USER"]
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(st.secrets["EMAIL_USER"], st.secrets["EMAIL_PASS"])
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join([page.extract_text() or "" for page in pdf.pages])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return " ".join([para.text for para in doc.paragraphs])

def parse_resume(file):
    text = ""
    if file.type == "application/pdf":
        text = extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(file)

    doc = nlp(text)

    # Extract name using NER fallback
    name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break
    if not name:
        lines = text.split("\n")
        for line in lines[:5]:
            if line.strip().isupper():
                name = line.strip()
                break
    if not name:
        name = "Unknown"

    # Extract email
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    email = email_match.group(0) if email_match else ""

    return name, email, text

def update_phase(application, result):
    if application["phase"] == "Applied":
        if application["match_score"] > 70:
            application["phase"] = "Round 1"
            send_email(application["email"], "Round 1 Interview Scheduled",
                       "Congratulations! Your Round 1 interview has been scheduled.")
        else:
            application["status"] = "Rejected"
            send_email(application["email"], "Application Update",
                       "We regret to inform you that your application was not shortlisted.")
    elif application["phase"] == "Round 1":
        if result == "Pass":
            application["phase"] = "Round 2"
            send_email(application["email"], "Round 2 Interview Scheduled",
                       "Good news! You have cleared Round 1. Round 2 interview scheduled.")
        else:
            application["status"] = "Rejected"
            send_email(application["email"], "Application Update",
                       "Thank you for attending. Unfortunately, you did not clear Round 1.")
    elif application["phase"] == "Round 2":
        if result == "Pass":
            application["phase"] = "Final"
            send_email(application["email"], "Final Interview Scheduled",
                       "Congratulations! You cleared Round 2. Final interview scheduled.")
        else:
            application["status"] = "Rejected"
            send_email(application["email"], "Application Update",
                       "Unfortunately, you did not clear Round 2.")
    elif application["phase"] == "Final":
        if result == "Pass":
            application["status"] = "Selected"
            send_email(application["email"], "Offer Letter",
                       "Congratulations! You have been selected. Offer details will follow.")
        else:
            application["status"] = "Rejected"
            send_email(application["email"], "Application Update",
                       "Unfortunately, you were not selected in the final round.")
    return application


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.email = None

st.sidebar.title("Navigation")

if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])

    with tab1:
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            users = load_users()
            user = users[users["email"] == email]
            if not user.empty and user.iloc[0]["password"] == hash_password(password):
                st.session_state.logged_in = True
                st.session_state.role = user.iloc[0]["role"]
                st.session_state.email = email
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        st.subheader("Register")
        email = st.text_input("New Email")
        password = st.text_input("New Password", type="password")
        role = st.selectbox("Role", ["User", "HR"])
        if st.button("Register"):
            users = load_users()
            if email in users["email"].values:
                st.error("Email already registered")
            else:
                new_user = pd.DataFrame([[email, hash_password(password), role]],
                                        columns=["email", "password", "role"])
                users = pd.concat([users, new_user], ignore_index=True)
                save_users(users)
                st.success("Registered successfully! Please login.")
else:
    st.sidebar.write(f"👤 Logged in as: {st.session_state.email}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.session_state.email = None
        st.rerun()

    role = st.session_state.role
    email = st.session_state.email

    if role == "User":
        st.header("📄 Resume Analyzer - User View")

        apps = load_applications()
        user_apps = apps[apps["email"] == email]

        if not user_apps.empty:
            st.subheader("Your Applications")
            st.dataframe(user_apps[["job_role", "match_score", "phase", "status"]])
        else:
            st.info("You have not applied to any jobs yet.")

        st.subheader("Upload Resume")
        file = st.file_uploader("Upload Resume", type=["pdf", "docx"])
        job_role = st.selectbox("Select Job Role", list(job_descriptions.keys()))
        if st.button("Submit Application"):
            if file and job_role:
                name, resume_email, text = parse_resume(file)
                resume_keywords = extract_keywords(text)
                jd_skills = job_descriptions[job_role]["skills"]
                matched, missing, score = match_resume_to_job(resume_keywords, jd_skills)

                apps = load_applications()
                new_app = {
                    "email": email,
                    "name": name,
                    "job_role": job_role,
                    "match_score": score,
                    "matched_skills": ",".join(matched),
                    "missing_skills": ",".join(missing),
                    "phase": "Applied",
                    "status": "Pending"
                }
                apps = pd.concat([apps, pd.DataFrame([new_app])], ignore_index=True)
                save_applications(apps)
                st.success("Application submitted!")
                st.rerun()

    elif role == "HR":
        st.header("👔 Resume Analyzer - HR View")

        apps = load_applications()
        if not apps.empty:
            for idx, app in apps.iterrows():
                with st.expander(f"{app['name']} - {app['job_role']} ({app['phase']})"):
                    st.write(f"**Email:** {app['email']}")
                    st.write(f"**Match Score:** {app['match_score']}")
                    st.write(f"**Matched Skills:** {app['matched_skills']}")
                    st.write(f"**Missing Skills:** {app['missing_skills']}")
                    st.write(f"**Phase:** {app['phase']}")
                    st.write(f"**Status:** {app['status']}")

                    if app["status"] == "Pending":
                        if app["phase"] == "Applied":
                            if st.button("Process Application", key=f"process_{idx}"):
                                apps.loc[idx] = update_phase(app, None)
                                save_applications(apps)
                                st.rerun()
                        elif app["phase"] in ["Round 1", "Round 2", "Final"]:
                            result = st.radio("Result", ["Pass", "Fail"], key=f"result_{idx}")
                            if st.button("Submit Result", key=f"submit_{idx}"):
                                apps.loc[idx] = update_phase(app, result)
                                save_applications(apps)
                                st.rerun()
        else:
            st.info("No applications submitted yet.")
