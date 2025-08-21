import streamlit as st
import pandas as pd
import hashlib
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
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# -------------------- SPAcy --------------------
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# -------------------- NLTK --------------------
try:
    stopwords.words("english")
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# -------------------- SUPABASE --------------------
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------- EMAIL --------------------
SMTP_EMAIL = st.secrets.get("SMTP_EMAIL")
SMTP_PASSWORD = st.secrets.get("SMTP_PASSWORD")

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
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    return [t for t in tokens if t.isalpha() and t not in stop_words]

def match_resume_to_job(resume_keywords, job_skills):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    matched = []
    missing = []
    for skill in job_skills:
        sim_scores = [util.cos_sim(model.encode(skill), model.encode(rk)).item() for rk in resume_keywords]
        if sim_scores and max(sim_scores) > 0.7:
            matched.append(skill)
        else:
            missing.append(skill)
    score = round(len(matched)/len(job_skills)*100,2) if job_skills else 0
    return matched, missing, score

def send_email_with_ics(to_email, subject, body, meeting_start=None, meeting_end=None):
    if not to_email:
        st.error("❌ Email blank, cannot send invite.")
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

# -------------------- AUTH --------------------
def login_register_ui():
    st.title("🔐 AI Resume Analyzer")
    tabs = st.tabs(["🔑 Login", "🆕 Register"])
    with tabs[0]:
        email = st.text_input("👤 Email")
        password = st.text_input("🔑 Password", type="password")
        if st.button("Login"):
            res = supabase.table("users").select("*").eq("email", email).execute()
            user = res.data[0] if res.data else None
            if user and user["password_hash"] == hash_password(password):
                st.session_state["email"] = email
                st.session_state["role"] = user["role"]
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials.")
    with tabs[1]:
        email = st.text_input("👤 Email", key="reg_email")
        password = st.text_input("🔑 Password", type="password", key="reg_pass")
        role = st.selectbox("Select Role", ["User", "HR"], key="reg_role")
        if st.button("Register"):
            res = supabase.table("users").select("*").eq("email", email).execute()
            if res.data:
                st.warning("Email already registered.")
            else:
                supabase.table("users").insert({
                    "email": email,
                    "password_hash": hash_password(password),
                    "role": role
                }).execute()
                st.success("Registered successfully! Please login.")

# -------------------- SIDEBAR LOGOUT --------------------
def logout_sidebar():
    with st.sidebar:
        st.markdown(f"👤 Logged in as `{st.session_state['email']}`")
        if st.button("🚪 Logout"):
            st.session_state.clear()
            st.rerun()

# -------------------- USER VIEW --------------------
def user_view():
    st.header("👤 Candidate Dashboard")
    jobs_res = supabase.table("jobs").select("*").execute()
    jobs = jobs_res.data

    if not jobs:
        st.info("No job postings yet.")
        return

    job_titles = list({job["title"] for job in jobs})
    selected_title = st.selectbox("Select Job Role", job_titles)
    filtered_jobs = [j for j in jobs if j["title"] == selected_title]
    companies = [j["description"] for j in filtered_jobs]
    selected_company_desc = st.selectbox("Select Company", companies)
    selected_job = next(j for j in filtered_jobs if j["description"]==selected_company_desc)

    uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else extract_text_from_docx(uploaded_file)
        name = extract_name(text)
        candidate_email = extract_email_from_text(text)
        phone = extract_phone_from_text(text)
        resume_keywords = extract_keywords(text)
        matched, missing, score = match_resume_to_job(resume_keywords, selected_job["skills"])

        phase = "Round 1 (Interview Pending Scheduling)" if score >= 70 else "Not Selected"
        status = "In Progress" if score >= 70 else "Rejected"

        supabase.table("applications").insert({
            "user_id": st.session_state["email"],
            "resume_url": None,
            "match_score": score,
            "phase": phase,
            "submission_date": datetime.now(),
            "job_id": selected_job["uuid"]
        }).execute()

        st.success(f"Application submitted for **{selected_company_desc} — {selected_title}** (Score: {score}%)")
        st.markdown(f"**Name:** {name}")
        st.markdown(f"**Email:** {candidate_email if candidate_email else 'Blank'}")
        st.markdown(f"**Phone:** {phone}")
        st.markdown(f"**Matched Skills:** {', '.join(matched) if matched else 'None'}")
        st.markdown(f"**Missing Skills:** {', '.join(missing) if missing else 'None'}")
        st.markdown(f"**Phase:** {phase}")

# -------------------- HR VIEW --------------------
def hr_view():
    st.header("🏢 HR Dashboard")
    apps_res = supabase.table("applications").select("*").execute()
    applications = apps_res.data

    if not applications:
        st.info("No candidate applications yet.")
        return

    for app in applications:
        st.markdown(f"**Candidate ID:** {app['user_id']} | **Job ID:** {app['job_id']} | **Score:** {app['match_score']}% | **Phase:** {app['phase']}")

# -------------------- MAIN --------------------
if "email" not in st.session_state:
    login_register_ui()
else:
    logout_sidebar()
    if st.session_state["role"] == "User":
        user_view()
    elif st.session_state["role"] == "HR":
        hr_view()
    else:
        st.error("Unknown role")
