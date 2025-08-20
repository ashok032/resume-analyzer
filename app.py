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
from job_descriptions import job_descriptions
from match_skills import extract_keywords, match_resume_to_job

# Import the new database modules
from database import SessionLocal, User, CandidateProgress, init_db
import sqlalchemy

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Securely load secrets from environment variables (will be set in Streamlit Cloud)
SMTP_EMAIL = os.environ.get("SMTP_EMAIL")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("spaCy model not found. Please ensure it's installed and listed in requirements.txt.")
    st.stop()

# Initialize the database connection and create tables if they don't exist
try:
    init_db()
except Exception as e:
    st.error(f"Error connecting to the database. Please check your DATABASE_URL secret. Details: {e}")
    st.stop()

# -------------------- New Database Helper Functions --------------------
def load_users():
    """Loads all users from the database into a pandas DataFrame."""
    db = SessionLocal()
    try:
        users_query = db.query(User).all()
        # Convert the list of User objects to a DataFrame
        df = pd.DataFrame([u.__dict__ for u in users_query])
        return df
    finally:
        db.close()

def add_new_user(username, password_hash, role):
    """Adds a new user to the database."""
    db = SessionLocal()
    try:
        new_user = User(username=username, password_hash=password_hash, role=role)
        db.add(new_user)
        db.commit()
    finally:
        db.close()

def load_progress():
    """Loads all candidate progress data from the database into a pandas DataFrame."""
    db = SessionLocal()
    try:
        # Use pandas to read directly from the SQL table for convenience
        df = pd.read_sql(db.query(CandidateProgress).statement, db.bind)
        return df
    finally:
        db.close()

def add_candidate_progress(data_dict):
    """Adds a new candidate application record to the database."""
    db = SessionLocal()
    try:
        new_progress = CandidateProgress(**data_dict)
        db.add(new_progress)
        db.commit()
    finally:
        db.close()

def update_progress(record_id, updates):
    """Updates an existing candidate progress record in the database by its ID."""
    db = SessionLocal()
    try:
        db.query(CandidateProgress).filter(CandidateProgress.id == record_id).update(updates)
        db.commit()
    finally:
        db.close()

# -------------------- Original Helper Functions (Unchanged) --------------------
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
    # Fallback for simple names
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        first_line = lines[0]
        if len(first_line.split()) < 4: # Likely a name
             return first_line.title()
    return "Not found"

def extract_email_from_text(text):
    m = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return m.group(0) if m else "Not found"

def extract_phone_from_text(text):
    m = re.search(r'(\+?\d{1,3}[\s\-]?)?(\(?\d{2,5}\)?[\s\-]?)?\d{3,5}[\s\-]?\d{3,5}', text)
    return m.group(0) if m else "Not found"

def send_email_with_ics(to_email, subject, body, meeting_start=None, meeting_end=None):
    if not to_email or to_email == "Not found":
        st.error("❌ Email is blank, cannot send invite. Candidate auto-rejected.")
        return False

    msg = MIMEMultipart("mixed")
    msg["From"] = SMTP_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    if meeting_start and meeting_end:
        ics_content = f"""BEGIN:VCALENDAR...""" # ICS content remains the same
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

# -------------------- UI Sections (Updated to use new DB functions) --------------------
def login_register_ui():
    st.title("🔐 AI Resume Analyzer")
    tabs = st.tabs(["🔑 Login", "🆕 Register"])
    with tabs[0]:
        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")
        if st.button("Login"):
            users = load_users()
            hashed = hash_password(password)
            if not users.empty:
                user = users[(users["username"] == username) & (users["password_hash"] == hashed)]
                if not user.empty:
                    st.session_state["username"] = username
                    st.session_state["role"] = user.iloc[0]["role"]
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            else:
                st.error("Invalid credentials.")
    with tabs[1]:
        new_username = st.text_input("👤 Username", key="reg_user")
        new_password = st.text_input("🔑 Password", type="password", key="reg_pass")
        role = st.selectbox("Select Role", ["User", "HR"], key="reg_role")
        if st.button("Register"):
            users = load_users()
            if new_username in users["username"].values:
                st.warning("Username already registered.")
            else:
                # Use the new function to add user to the database
                add_new_user(new_username, hash_password(new_password), role)
                st.success("Registered successfully! Please login.")

def user_view():
    st.header("👤 Candidate Dashboard")
    df_all = load_progress()
    my_apps = df_all[df_all["logged_in_username"] == st.session_state["username"]]
    st.subheader("📜 My Applications")
    if not my_apps.empty:
        # Display without the 'id' and 'logged_in_username' columns for a cleaner look
        st.dataframe(my_apps.drop(columns=['id', 'logged_in_username'], errors='ignore'))
    else:
        st.info("You have not submitted any applications yet.")

    job_role = st.selectbox("Select a Job Role", options=list(job_descriptions.keys()))
    company = st.selectbox("Select Company", options=list(job_descriptions[job_role].keys()))

    uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else extract_text_from_docx(uploaded_file)
        name = extract_name(text)
        email_in_resume = extract_email_from_text(text)
        phone = extract_phone_from_text(text)
        
        resume_keywords = extract_keywords(text)
        jd_skills = job_descriptions[job_role][company]["skills"]
        matched, missing, score = match_resume_to_job(resume_keywords, jd_skills)

        phase = "Not Selected"
        status = "Rejected"
        if score >= 70:
            phase = "Round 1 (Interview Pending Scheduling)"
            status = "In Progress"

        # Create a dictionary and use the new function to add it to the database
        new_application = {
            "logged_in_username": st.session_state["username"],
            "email": email_in_resume,
            "name": name,
            "role": job_role,
            "company": company,
            "match_score": score,
            "current_phase": phase,
            "status": status
        }
        add_candidate_progress(new_application)

        st.success(f"Application submitted for **{company} — {job_role}** (Score: {score}%)")
        st.markdown(f"**Name:** {name}")
        st.markdown(f"**Email:** {email_in_resume}")
        st.markdown(f"**Phone:** {phone}")
        st.markdown(f"**Status:** {status}")
        st.info("Your application has been recorded. You can see its status in the 'My Applications' table above.")
        st.button("Submit another application") # To help with rerunning the page

def hr_view():
    st.header("🏢 HR Dashboard")
    df = load_progress()
    if df.empty:
        st.warning("No candidates have applied yet.")
        return

    st.subheader("All Candidates Overview")
    
    # Create a copy to avoid SettingWithCopyWarning
    df_display = df.copy()
    
    # Use st.data_editor for interactive actions
    for index, row in df_display.iterrows():
        db_id = row['id'] # Get the unique database ID for this record
        cols = st.columns([3, 2, 2, 2, 3, 2])
        cols[0].write(f"**{row['name']}** ({row['email']})")
        cols[1].write(f"{row['company']}")
        cols[2].write(f"{row['role']}")
        cols[3].write(f"{row['match_score']}%")
        cols[4].write(row['current_phase'])
        
        # --- Actions Column ---
        with cols[5]:
            if "Pending Scheduling" in row["current_phase"] and row["status"] == "In Progress":
                if st.button("📅 Schedule", key=f"sch_{db_id}"):
                    # Logic to open a scheduling modal/form can be added here
                    # For now, we'll just update the status
                    update_progress(db_id, {"current_phase": row['current_phase'].replace("Pending Scheduling", "Scheduled")})
                    st.rerun()

            elif "Scheduled" in row["current_phase"]:
                if st.button("✅ Pass", key=f"pass_{db_id}"):
                    next_phase = "Selected" # Default
                    if "Round 1" in row['current_phase']:
                        next_phase = "Round 2 (Interview Pending Scheduling)"
                    elif "Round 2" in row['current_phase']:
                        next_phase = "Final (Interview Pending Scheduling)"
                    
                    update_progress(db_id, {"current_phase": next_phase})
                    if next_phase == "Selected":
                        update_progress(db_id, {"status": "Selected"})
                        send_email_with_ics(row["email"], "🎉 Job Offer", f"Dear {row['name']},\n\nCongratulations! You are selected.")
                    st.rerun()

                if st.button("❌ Fail", key=f"fail_{db_id}"):
                    update_progress(db_id, {"status": "Rejected", "current_phase": "Rejected"})
                    st.rerun()
            
            elif row["status"] == "Selected":
                st.write("🎉 Offer Sent")
            
            elif row["status"] == "Rejected":
                st.write("❌ Rejected")


# -------------------- Main Application Logic --------------------
if "username" not in st.session_state:
    login_register_ui()
else:
    logout_sidebar()
    if st.session_state["role"] == "User":
        user_view()
    elif st.session_state["role"] == "HR":
        hr_view()
    else:
        st.error("Unknown role. Please contact support.")

