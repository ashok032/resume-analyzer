import streamlit as st
import pandas as pd
import hashlib
import pdfplumber
import docx
import re
import spacy
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
from job_descriptions import job_descriptions
from match_skills import extract_keywords, match_resume_to_job
from supabase import create_client, Client

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Load email config from Streamlit secrets
SMTP_EMAIL = st.secrets["SMTP_EMAIL"]
SMTP_PASSWORD = st.secrets["SMTP_PASSWORD"]

# Supabase setup
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

nlp = spacy.load("en_core_web_sm")

# -------------------- Helpers --------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_user(username):
    res = supabase.table("users").select("*").eq("email", username).execute()
    if res.data:
        return res.data[0]
    return None

def add_user(username, password_hash, role):
    supabase.table("users").insert({"email": username, "password_hash": password_hash, "role": role}).execute()

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

def send_email_with_ics(to_email, subject, body, meeting_start=None, meeting_end=None):
    if not to_email or to_email == "Not found":
        st.error("❌ Email is blank, cannot send invite.")
        return False

    msg = MIMEMultipart("mixed")
    msg["From"] = SMTP_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    if meeting_start and meeting_end:
        ics_content = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//ResumeAnalyzer//EN
BEGIN:VEVENT
UID:{datetime.now().strftime('%Y%m%dT%H%M%S')}
DTSTAMP:{datetime.now().strftime('%Y%m%dT%H%M%SZ')}
DTSTART:{meeting_start.strftime('%Y%m%dT%H%M%SZ')}
DTEND:{meeting_end.strftime('%Y%m%dT%H%M%SZ')}
SUMMARY:{subject}
DESCRIPTION:{body}
END:VEVENT
END:VCALENDAR
"""
        part = MIMEBase("text", "calendar", method="REQUEST", name="invite.ics")
        part.set_payload(ics_content)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment; filename=invite.ics")
        msg.attach(part)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)
        st.success(f"📨 Email sent to {to_email}")
        return True
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")
        return False

def logout_sidebar():
    with st.sidebar:
        st.markdown(f"👤 Logged in as `{st.session_state['username']}`")
        if st.button("🚪 Logout", key="logout_btn"):
            st.session_state.clear()
            st.rerun()

# -------------------- Login/Register UI --------------------
def login_register_ui():
    st.title("🔐 AI Resume Analyzer")
    tabs = st.tabs(["🔑 Login", "🆕 Register"])
    with tabs[0]:
        username = st.text_input("👤 Email")
        password = st.text_input("🔑 Password", type="password")
        if st.button("Login"):
            user = get_user(username)
            if user and user["password_hash"] == hash_password(password):
                st.session_state["username"] = username
                st.session_state["role"] = user["role"]
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials.")
    with tabs[1]:
        new_username = st.text_input("👤 Email", key="reg_user")
        new_password = st.text_input("🔑 Password", type="password", key="reg_pass")
        role = st.selectbox("Select Role", ["User", "HR"], key="reg_role")
        if st.button("Register"):
            if get_user(new_username):
                st.warning("Email already registered.")
            else:
                add_user(new_username, hash_password(new_password), role)
                st.success("Registered successfully! Please login.")

# -------------------- User View --------------------
def user_view():
    st.header("👤 Candidate Dashboard")
    # Fetch applications
    apps_res = supabase.table("applications").select("*").eq("user_id", get_user(st.session_state["username"])["id"]).execute()
    my_apps = apps_res.data

    st.subheader("📜 My Applications")
    if my_apps:
        st.dataframe(pd.DataFrame(my_apps))
    else:
        st.info("No applications yet.")

    job_role = st.selectbox("Select a Job Role", options=list(job_descriptions.keys()))
    company = st.selectbox("Select Company", options=list(job_descriptions[job_role].keys()))

    uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else extract_text_from_docx(uploaded_file)
        name = extract_name(text)
        email_in_resume = extract_email_from_text(text)
        phone = extract_phone_from_text(text)
        candidate_email = email_in_resume if email_in_resume != "Not found" else ""

        resume_keywords = extract_keywords(text)
        jd_skills = job_descriptions[job_role][company]["skills"]
        matched, missing, score = match_resume_to_job(resume_keywords, jd_skills)

        if score >= 70 and candidate_email:
            phase = "Round 1 (Interview Pending Scheduling)"
            status = "In Progress"
        else:
            phase = "Not Selected"
            status = "Rejected"

        # Insert into Supabase
        user_id = get_user(st.session_state["username"])["id"]
        supabase.table("applications").insert({
            "user_id": user_id,
            "resume_url": uploaded_file.name,
            "match_score": score,
            "phase": phase,
            "submission_date": datetime.now().isoformat(),
            "job_id": list(job_descriptions.keys()).index(job_role)  # simple mapping
        }).execute()

        st.success(f"Application submitted for **{company} — {job_role}** (Score: {score}%)")
        st.markdown(f"**Name:** {name}")
        st.markdown(f"**Email:** {candidate_email if candidate_email else 'Blank'}")
        st.markdown(f"**Phone:** {phone}")
        st.markdown(f"**Matched Skills:** {', '.join(matched) if matched else 'None'}")
        st.markdown(f"**Missing Skills:** {', '.join(missing) if missing else 'None'}")
        st.markdown(f"**Phase:** {phase}")

# -------------------- HR View --------------------
def hr_view():
    st.header("🏢 HR Dashboard")
    apps_res = supabase.table("applications").select("*").execute()
    df = pd.DataFrame(apps_res.data)

    if df.empty:
        st.warning("No candidates yet.")
        return

    st.subheader("All Candidates Overview")
    st.dataframe(df)

# -------------------- Main --------------------
if "username" not in st.session_state:
    login_register_ui()
else:
    logout_sidebar()
    if st.session_state["role"] == "User":
        user_view()
    elif st.session_state["role"] == "HR":
        hr_view()
    else:
        st.error("Unknown role")
