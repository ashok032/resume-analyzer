import streamlit as st
import pandas as pd
import hashlib
import os
import pdfplumber
import docx
import re
import spacy
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta

# --- This is your original file, make sure it is in the same folder ---
from match_skills import extract_keywords, match_resume_to_job

# --- Import the new database models ---
from database import SessionLocal, User, Application, JobDescription, init_db
import sqlalchemy

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# --- Securely load secrets from environment variables (set in Streamlit Cloud) ---
SMTP_EMAIL = os.environ.get("SMTP_EMAIL")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")

# --- Load the spaCy model ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("spaCy model not found. This is an issue with the deployment environment.")
    st.stop()

# --- Initialize the database connection and create tables if they don't exist ---
try:
    init_db()
except Exception as e:
    st.error(f"Error connecting to the database. Please check your DATABASE_URL secret. Details: {e}")
    st.stop()

# -------------------- Database Helper Functions --------------------
def get_user_by_email(email):
    db = SessionLocal()
    try:
        return db.query(User).filter(User.email == email).first()
    finally:
        db.close()

def add_new_user(email, password_hash, role):
    db = SessionLocal()
    try:
        new_user = User(email=email, password_hash=password_hash, role=role)
        db.add(new_user)
        db.commit()
    finally:
        db.close()

def load_applications():
    db = SessionLocal()
    try:
        return pd.read_sql(db.query(Application).statement, db.bind)
    finally:
        db.close()

def add_application(data_dict):
    db = SessionLocal()
    try:
        new_application = Application(**data_dict)
        db.add(new_application)
        db.commit()
    finally:
        db.close()

def update_application(record_id, updates):
    db = SessionLocal()
    try:
        db.query(Application).filter(Application.id == record_id).update(updates)
        db.commit()
    finally:
        db.close()
        
@st.cache_data(ttl=300) # Cache for 5 minutes
def load_job_descriptions():
    db = SessionLocal()
    try:
        jobs = db.query(JobDescription).all()
        job_dict = {}
        for job in jobs:
            if job.role not in job_dict:
                job_dict[job.role] = {"id": job.id, "skills": job.skills, "description": job.description}
        return job_dict
    finally:
        db.close()

def add_job_description(role, description, skills):
    db = SessionLocal()
    try:
        # Skills should be a list of strings
        skills_list = [s.strip() for s in skills.split(',')]
        new_job = JobDescription(role=role, description=description, skills=skills_list)
        db.add(new_job)
        db.commit()
        st.cache_data.clear() # Clear cache after adding a new job
    finally:
        db.close()

# -------------------- Core Helper Functions --------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.title()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        first_line = lines[0]
        if len(first_line.split()) < 4: return first_line.title()
    return "Not Found"

def extract_email_from_text(text):
    m = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return m.group(0) if m else "Not Found"

def send_email_with_ics(to_email, subject, body, meeting_start=None, meeting_end=None):
    if not all([SMTP_EMAIL, SMTP_PASSWORD]):
        st.error("Email credentials are not configured. Cannot send email.")
        return False
    if not to_email or to_email == "Not Found":
        st.error("❌ Candidate email not found, cannot send invite.")
        return False

    msg = MIMEMultipart("mixed")
    msg["From"] = SMTP_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    if meeting_start and meeting_end:
        ics_content = f"""BEGIN:VCALENDAR...""" # Your ICS content here
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

# -------------------- UI Sections --------------------
def logout_sidebar():
    with st.sidebar:
        st.markdown(f"👤 Logged in as `{st.session_state['email']}`")
        if st.button("🚪 Logout", key="logout_btn"):
            st.session_state.clear()
            st.rerun()

def login_register_ui():
    st.title("🔐 AI Resume Analyzer")
    tabs = st.tabs(["🔑 Login", "🆕 Register"])
    with tabs[0]:
        email = st.text_input("👤 Email")
        password = st.text_input("🔑 Password", type="password")
        if st.button("Login"):
            hashed = hash_password(password)
            user = get_user_by_email(email)
            if user and user.password_hash == hashed:
                st.session_state["email"] = user.email
                st.session_state["role"] = user.role
                st.session_state["user_id"] = user.id
                st.rerun()
            else:
                st.error("Invalid credentials.")
    with tabs[1]:
        new_email = st.text_input("👤 Email", key="reg_email")
        new_password = st.text_input("🔑 Password", type="password", key="reg_pass")
        role = st.selectbox("Select Role", ["User", "HR"], key="reg_role")
        if st.button("Register"):
            if get_user_by_email(new_email):
                st.warning("Email already registered.")
            else:
                add_new_user(new_email, hash_password(new_password), role)
                st.success("Registered successfully! Please login.")

def user_view():
    st.header("👤 Candidate Dashboard")
    df_all = load_applications()
    my_apps = df_all[df_all["user_id"] == st.session_state["user_id"]]
    st.subheader("📜 My Applications")
    if not my_apps.empty:
        display_cols = ['submission_date', 'job_role', 'match_score', 'phase', 'status']
        st.dataframe(my_apps[display_cols])
    else:
        st.info("You have not submitted any applications yet.")

    job_descriptions = load_job_descriptions()
    if not job_descriptions:
        st.warning("No job roles are available. Please check back later.")
        return

    job_role = st.selectbox("Select a Job Role to Apply For", options=list(job_descriptions.keys()))
    
    with st.expander("View Job Description"):
        st.write(job_descriptions[job_role]['description'])
        st.write("**Skills Required:**", ", ".join(job_descriptions[job_role]['skills']))

    uploaded_file = st.file_uploader("Upload Your Resume (PDF or DOCX)", type=["pdf", "docx"])
    
    if uploaded_file:
        if st.button("Analyze and Submit Application"):
            text = extract_text_from_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else extract_text_from_docx(uploaded_file)
            candidate_name = extract_name(text)
            candidate_email = extract_email_from_text(text)
            
            jd_skills = job_descriptions[job_role]["skills"]
            resume_keywords = extract_keywords(text)
            matched, missing, score = match_resume_to_job(resume_keywords, jd_skills)

            phase = "Rejected"
            status = "Rejected"
            if score >= 70:
                phase = "Screening"
                status = "In Progress"

            new_application_data = {
                "user_id": st.session_state["user_id"],
                "job_role": job_role,
                "match_score": score,
                "phase": phase,
                "status": status,
                "candidate_name": candidate_name,
                "candidate_email": candidate_email,
            }
            add_application(new_application_data)

            st.success(f"Application submitted for **{job_role}** (Score: {score}%)")
            st.rerun()

def hr_view():
    st.header("🏢 HR Dashboard")
    
    hr_tabs = st.tabs(["Manage Job Descriptions", "View Applications"])

    with hr_tabs[0]:
        st.subheader("Add or Update Job Descriptions")
        with st.form("job_form", clear_on_submit=True):
            role = st.text_input("Job Role Title")
            description = st.text_area("Job Description")
            skills = st.text_input("Required Skills (comma-separated)")
            submitted = st.form_submit_button("Add Job")
            if submitted:
                if role and description and skills:
                    add_job_description(role, description, skills)
                    st.success(f"Job '{role}' added successfully!")
                else:
                    st.error("Please fill out all fields.")
        
        st.subheader("Current Job Listings")
        jobs_df = pd.DataFrame.from_dict(load_job_descriptions(), orient='index')
        if not jobs_df.empty:
            st.dataframe(jobs_df)
        else:
            st.info("No jobs found in the database.")

    with hr_tabs[1]:
        st.subheader("All Candidate Applications")
        df = load_applications()
        if df.empty:
            st.warning("No candidates have applied yet.")
            return
        
        for index, row in df.iterrows():
            db_id = row['id']
            st.markdown("---")
            cols = st.columns([3, 2, 1, 2, 2, 2])
            cols[0].write(f"**{row['candidate_name']}** ({row['candidate_email']})")
            cols[1].write(f"**Role:** {row['job_role']}")
            cols[2].write(f"**Score:** {row['match_score']}%")
            cols[3].write(f"**Phase:** {row['phase']}")
            cols[4].write(f"**Status:** {row['status']}")
            
            with cols[5]:
                if row["status"] == "In Progress":
                    if st.button("✅ Pass to Next Round", key=f"pass_{db_id}"):
                        # Define your hiring stages
                        phases = ["Screening", "Round 1", "Round 2", "Final", "Offer"]
                        current_phase_index = phases.index(row['phase']) if row['phase'] in phases else -1
                        if current_phase_index < len(phases) - 1:
                            next_phase = phases[current_phase_index + 1]
                            update_application(db_id, {"phase": next_phase})
                            if next_phase == "Offer":
                                update_application(db_id, {"status": "Selected"})
                        else:
                            update_application(db_id, {"status": "Selected"})
                        st.rerun()

                    if st.button("❌ Reject", key=f"fail_{db_id}"):
                        update_application(db_id, {"status": "Rejected", "phase": "Rejected"})
                        st.rerun()
                
                elif row["status"] == "Selected":
                    st.success("🎉 Selected")
                
                elif row["status"] == "Rejected":
                    st.error("❌ Rejected")

# -------------------- Main Application Logic --------------------
if "email" not in st.session_state:
    login_register_ui()
else:
    logout_sidebar()
    if st.session_state["role"] == "User":
        user_view()
    elif st.session_state["role"] == "HR":
        hr_view()
    else:
        st.error("Unknown role. Please contact support.")
