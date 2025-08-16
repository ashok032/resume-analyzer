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
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
from job_descriptions import job_descriptions
from match_skills import extract_keywords, match_resume_to_job

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

with open("config.json", "r") as cfg:
    email_config = json.load(cfg)

SMTP_EMAIL = email_config["EMAIL"]
SMTP_PASSWORD = email_config["PASSWORD"]

nlp = spacy.load("en_core_web_sm")

USERS_FILE = "users.csv"
PROGRESS_FILE = "candidate_progress.csv"

# -------------------- Helpers --------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    return pd.read_csv(USERS_FILE) if os.path.exists(USERS_FILE) else pd.DataFrame(columns=["username", "password_hash", "role"])

def load_progress():
    cols = ["logged_in_username", "email", "name", "role", "company", "match_score", "current_phase", "status"]
    if os.path.exists(PROGRESS_FILE):
        df = pd.read_csv(PROGRESS_FILE)
        for c in cols:
            if c not in df.columns:
                df[c] = ""
        df = df.fillna("")
        return df[cols]
    return pd.DataFrame(columns=cols)

def save_progress(df):
    df.to_csv(PROGRESS_FILE, index=False)

# Resume parsing
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

# Email sending with ICS
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

# Sidebar logout
def logout_sidebar():
    with st.sidebar:
        st.markdown(f"👤 Logged in as `{st.session_state['username']}`")
        if st.button("🚪 Logout", key="logout_btn"):
            st.session_state.clear()
            st.rerun()

# Login/Register
def login_register_ui():
    st.title("🔐 AI Resume Analyzer")
    tabs = st.tabs(["🔑 Login", "🆕 Register"])
    with tabs[0]:
        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")
        if st.button("Login"):
            users = load_users()
            hashed = hash_password(password)
            user = users[(users["username"] == username) & (users["password_hash"] == hashed)]
            if not user.empty:
                st.session_state["username"] = username
                st.session_state["role"] = user.iloc[0]["role"]
                st.success("Login successful!")
                st.rerun()
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
                new_user = pd.DataFrame([[new_username, hash_password(new_password), role]], columns=["username", "password_hash", "role"])
                updated = pd.concat([users, new_user], ignore_index=True)
                updated.to_csv(USERS_FILE, index=False)
                st.success("Registered successfully! Please login.")

# -------------------- User View --------------------
def user_view():
    st.header("👤 Candidate Dashboard")
    df_all = load_progress()
    my_apps = df_all[df_all["logged_in_username"] == st.session_state["username"]]
    st.subheader("📜 My Applications")
    if not my_apps.empty:
        st.dataframe(my_apps)
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

        if set(jd_skills) <= set(resume_keywords):
            matched_display = "All Matched"
            missing_display = "None"
        elif not set(matched):
            matched_display = "None"
            missing_display = ", ".join(missing)
        else:
            matched_display = ", ".join(matched)
            missing_display = ", ".join(missing) if missing else "None"

        if score >= 70 and candidate_email:
            phase = "Round 1 (Interview Pending Scheduling)"
            status = "In Progress"
        else:
            phase = "Not Selected"
            status = "Rejected"

        df = load_progress()
        df.loc[len(df)] = [st.session_state["username"], candidate_email, name, job_role, company, score, phase, status]
        save_progress(df)

        st.success(f"Application submitted for **{company} — {job_role}** (Score: {score}%)")
        st.markdown(f"**Name:** {name}")
        st.markdown(f"**Email:** {candidate_email if candidate_email else 'Blank'}")
        st.markdown(f"**Phone:** {phone}")
        st.markdown(f"**Matched Skills:** {matched_display}")
        st.markdown(f"**Missing Skills:** {missing_display}")
        st.markdown(f"**Phase:** {phase}")

# -------------------- HR View --------------------
def hr_view():
    st.header("🏢 HR Dashboard")
    df = load_progress()
    if df.empty:
        st.warning("No candidates yet.")
        return

    toggle_all = st.toggle("📋 Show All Candidates")

    if toggle_all:
        st.subheader("All Candidates Overview")
        for idx, row in df.iterrows():
            cols = st.columns([3,2,2,2,2,2])
            cols[0].write(f"**{row['name']}** ({row['email']})")
            cols[1].write(f"{row['company']}")
            cols[2].write(f"{row['role']}")
            cols[3].write(f"{row['match_score']}%")
            cols[4].write(row['current_phase'])
            cols[5].write(row['status'])

            if "Pending Scheduling" in row["current_phase"] and row["status"]=="In Progress":
                if cols[5].button("📅 Schedule", key=f"sch_{idx}"):
                    st.session_state["selected_idx"] = idx
                    st.session_state["mode"] = "schedule"
                    st.rerun()
            elif "Scheduled" in row["current_phase"]:
                if cols[5].button("✅ Pass", key=f"pass_{idx}"):
                    df.loc[idx,"current_phase"] = "Round 2 (Interview Pending Scheduling)" if row["current_phase"].startswith("Round 1") else "Final (Interview Pending Scheduling)"
                    save_progress(df)
                    st.rerun()
                if cols[5].button("❌ Fail", key=f"fail_{idx}"):
                    df.loc[idx,["status","current_phase"]] = ["Rejected","Rejected"]
                    save_progress(df)
                    st.rerun()
            elif row["current_phase"]=="Selected":
                cols[5].write("🎉 Offer Sent")

    else:
        company_sel = st.selectbox("Select Company", sorted(df["company"].unique()))
        filtered_company = df[(df["company"] == company_sel) & (df["status"]=="In Progress") & (df["match_score"]>=70)]

        if filtered_company.empty:
            st.info("No eligible candidates to process.")
            return

        role_sel = st.selectbox("Select Role", sorted(filtered_company["role"].unique()))
        filtered_role = filtered_company[filtered_company["role"] == role_sel]

        selected_email = st.selectbox("Select Candidate", filtered_role["email"].unique())
        candidate_row = filtered_role[filtered_role["email"] == selected_email]
        candidate = candidate_row.iloc[0]

        st.markdown(f"**Name:** {candidate['name']}")
        st.markdown(f"**Role:** {candidate['role']}")
        st.markdown(f"**Company:** {candidate['company']}")
        st.markdown(f"**Score:** {candidate['match_score']}%")
        st.markdown(f"**Phase:** {candidate['current_phase']}")
        st.markdown(f"**Status:** {candidate['status']}")

        if "Pending Scheduling" in candidate["current_phase"]:
            meeting_date = st.date_input("Meeting Date")
            meeting_time = st.time_input("Meeting Time")
            duration_mins = st.number_input("Duration (minutes)", min_value=15, max_value=240, step=15)
            meet_link = st.text_input("Google Meet Link")
            if st.button("📅 Send Interview Invite"):
                start_dt = datetime.combine(meeting_date, meeting_time)
                end_dt = start_dt + timedelta(minutes=duration_mins)
                sent = send_email_with_ics(candidate["email"], f"Interview for {candidate['role']} - {candidate['company']}",
                                f"Dear {candidate['name']},\n\nYour interview is scheduled.\n\nMeet link: {meet_link}", start_dt, end_dt)
                if sent:
                    df.loc[df["email"] == selected_email, "current_phase"] = candidate["current_phase"].replace("Pending Scheduling", "Scheduled")
                    save_progress(df)
                    st.success("Interview scheduled successfully!")

        elif "Scheduled" in candidate["current_phase"]:
            result = st.radio("Result of Current Round", ["Pass", "Fail"])
            if st.button("Submit Result"):
                if result=="Fail":
                    df.loc[df["email"] == selected_email, ["status","current_phase"]] = ["Rejected","Rejected"]
                else:
                    if candidate["current_phase"].startswith("Round 1"):
                        df.loc[df["email"] == selected_email,"current_phase"] = "Round 2 (Interview Pending Scheduling)"
                    elif candidate["current_phase"].startswith("Round 2"):
                        df.loc[df["email"] == selected_email,"current_phase"] = "Final (Interview Pending Scheduling)"
                    elif candidate["current_phase"].startswith("Final"):
                        df.loc[df["email"] == selected_email,["status","current_phase"]] = ["Selected","Selected"]
                        send_email_with_ics(candidate["email"], "🎉 Job Offer",
                                            f"Dear {candidate['name']},\n\nCongratulations! You are selected for {candidate['role']} at {candidate['company']}.\n\nWelcome aboard!")
                save_progress(df)
                st.success("Result updated successfully!")

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
