# app.py

import os
import re
import hashlib
from datetime import datetime

import streamlit as st
import pandas as pd

# Resume parsing deps
import pdfplumber
import docx

# NLP
import spacy

# Email
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Supabase
from supabase import create_client, Client

# Skills matching (your file)
from match_skills import extract_keywords, match_resume_to_job


# ==============================
# CONFIG & CLIENTS
# ==============================

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Secrets (Streamlit)
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    EMAIL_USER = st.secrets["EMAIL_USER"]
    EMAIL_PASS = st.secrets["EMAIL_PASS"]
except Exception as e:
    st.error("Missing Streamlit secrets. Please set SUPABASE_URL, SUPABASE_KEY, EMAIL_USER, EMAIL_PASS in .streamlit/secrets.toml")
    st.stop()

# Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# spaCy model (graceful fallback if not installed)
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        # lightweight fallback: blank English pipeline
        nlp_blank = spacy.blank("en")
        return nlp_blank

nlp = load_spacy()


# ==============================
# UTILS
# ==============================

def sha256_hash(password: str) -> str:
    """Hash password with SHA-256 to match your existing DB schema."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def send_email(to_email: str, subject: str, body: str) -> None:
    """Send email via Gmail (App Password recommended)."""
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_USER
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        st.warning(f"Email sending failed: {e}")

def extract_text_from_file(uploaded_file) -> str:
    """Read PDF/DOCX to text. No file is stored anywhere."""
    text = ""
    if uploaded_file is None:
        return text

    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
    elif name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        st.error("Unsupported file type. Please upload PDF or DOCX.")
    return text

def extract_contact_info(text: str):
    """Extract name (best-effort), email, phone from resume text."""
    # Email
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    email = email_match.group(0) if email_match else None

    # Phone
    phone_match = re.search(r"\+?\d[\d\s\-()]{8,}\d", text)
    phone = phone_match.group(0) if phone_match else None

    # Name (NER if model has it; else heuristic = first non-empty line)
    name = None
    if "ner" in nlp.pipe_names:  # only if the small model loaded
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                break
    if not name:
        for line in text.splitlines():
            line = line.strip()
            if line and len(line.split()) <= 5:  # short-ish line
                name = line
                break

    return name, email, phone


# ==============================
# SUPABASE DATA ACCESS
# ==============================

# Users
def register_user(email: str, password: str, role: str):
    return supabase.table("users").insert({
        "email": email,
        "password_hash": sha256_hash(password),
        "role": role
    }).execute()

def login_user(email: str, password: str):
    hashed = sha256_hash(password)
    res = supabase.table("users").select("*").eq("email", email).eq("password_hash", hashed).execute()
    return res.data[0] if res.data else None

# Jobs (merged table: id uuid, title, description, skills[])
def get_jobs():
    res = supabase.table("jobs").select("*").order("created_at", desc=True).execute()
    return res.data or []

def add_job(title: str, description: str, skills_list: list[str]):
    return supabase.table("jobs").insert({
        "title": title,
        "description": description,
        "skills": skills_list
    }).execute()

def update_job(job_id: str, title: str, description: str, skills_list: list[str]):
    return supabase.table("jobs").update({
        "title": title,
        "description": description,
        "skills": skills_list
    }).eq("id", job_id).execute()

def delete_job(job_id: str):
    return supabase.table("jobs").delete().eq("id", job_id).execute()

# Applications (assumes schema with job_id uuid; we also try fallback to job_role if needed)
def save_application(user_id: int, job_id: str, match_score: float, phase: str):
    """We DO NOT store the resume file/URL. Only metadata."""
    # Preferred: applications with job_id
    try:
        return supabase.table("applications").insert({
            "user_id": user_id,
            "job_id": job_id,
            "resume_url": "",           # explicitly empty to ensure no storage
            "match_score": match_score,
            "phase": phase,
            "submission_date": datetime.utcnow().isoformat()
        }).execute()
    except Exception:
        # Fallback for old schema with job_role TEXT (if migration not yet applied)
        # Fetch job title for compatibility
        job = supabase.table("jobs").select("title").eq("id", job_id).single().execute().data
        job_role = job["title"] if job else ""
        return supabase.table("applications").insert({
            "user_id": user_id,
            "job_role": job_role,
            "resume_url": "",
            "match_score": match_score,
            "phase": phase,
            "submission_date": datetime.utcnow().isoformat()
        }).execute()

def get_all_applications():
    res = supabase.table("applications").select("*").order("submission_date", desc=True).execute()
    return res.data or []

def get_user_applications(user_id: int):
    res = supabase.table("applications").select("*").eq("user_id", user_id).order("submission_date", desc=True).execute()
    return res.data or []

def update_application_phase(app_id: int, new_phase: str):
    return supabase.table("applications").update({"phase": new_phase}).eq("id", app_id).execute()


# ==============================
# PHASE WORKFLOW
# ==============================

PHASES = [
    "Not Selected",
    "Round 1 Scheduled",
    "Round 2 Scheduled",
    "Final Round Scheduled",
    "Offer Extended",
    "Rejected"
]

def advance_phase(current_phase: str) -> str:
    order = ["Round 1 Scheduled", "Round 2 Scheduled", "Final Round Scheduled", "Offer Extended"]
    if current_phase not in order:
        return current_phase
    idx = order.index(current_phase)
    if idx < len(order) - 1:
        return order[idx + 1]
    return "Offer Extended"

def notify_phase_update(candidate_email: str, candidate_name: str | None, new_phase: str):
    sal = candidate_name or "Candidate"
    if new_phase == "Rejected" or new_phase == "Not Selected":
        subject = "Application Update"
        body = f"Dear {sal},\n\nThank you for applying. At this time, we are unable to proceed.\n\nRegards"
    elif new_phase == "Round 1 Scheduled":
        subject = "Interview Scheduled - Round 1"
        body = f"Dear {sal},\n\nCongratulations! Your Round 1 interview has been scheduled.\n\nRegards"
    elif new_phase == "Round 2 Scheduled":
        subject = "Interview Scheduled - Round 2"
        body = f"Dear {sal},\n\nGreat news! You have progressed to Round 2.\n\nRegards"
    elif new_phase == "Final Round Scheduled":
        subject = "Interview Scheduled - Final Round"
        body = f"Dear {sal},\n\nYou have progressed to the Final Round.\n\nRegards"
    elif new_phase == "Offer Extended":
        subject = "Offer Extended"
        body = f"Dear {sal},\n\nCongratulations! We are pleased to extend an offer.\n\nRegards"
    else:
        subject = "Application Update"
        body = f"Dear {sal},\n\nYour application status is now: {new_phase}.\n\nRegards"

    send_email(candidate_email, subject, body)


# ==============================
# SESSION
# ==============================

if "user" not in st.session_state:
    st.session_state.user = None


# ==============================
# AUTH VIEWS
# ==============================

def show_auth():
    tab_login, tab_register = st.tabs(["🔑 Login", "🆕 Register"])

    with tab_login:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", type="primary"):
            user = login_user(email, password)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab_register:
        reg_email = st.text_input("Email", key="reg_email")
        reg_pass = st.text_input("Password", type="password", key="reg_pass")
        role = st.selectbox("Role", ["User", "HR"], key="reg_role")
        if st.button("Create account"):
            if not reg_email or not reg_pass:
                st.warning("Please enter email and password.")
            else:
                # unique email constraint handled by DB; show friendly message if duplicate
                res = register_user(reg_email, reg_pass, role)
                if getattr(res, "error", None):
                    st.error(f"Registration failed: {res.error}")
                else:
                    st.success("Registered successfully. Please login.")


# ==============================
# USER PORTAL
# ==============================

def show_user_portal():
    st.header("📄 Candidate Portal")

    # Jobs
    jobs = get_jobs()
    if not jobs:
        st.info("No jobs available yet. Please check back later.")
        return

    titles = [j["title"] for j in jobs]
    selected_title = st.selectbox("Select a role", options=titles)
    job = next((j for j in jobs if j["title"] == selected_title), None)

    if job:
        with st.expander("Job Details", expanded=True):
            st.markdown(f"**Title:** {job['title']}")
            st.markdown(f"**Description:** {job['description']}")
            st.markdown("**Required Skills:** " + ", ".join(job.get("skills", [])))

    uploaded = st.file_uploader("Upload your resume (PDF/DOCX)", type=["pdf", "docx"])

    if uploaded and job:
        # Parse in memory (NOT storing)
        text = extract_text_from_file(uploaded)

        # Extract contact
        name, resume_email, phone = extract_contact_info(text)

        # Extract keywords -> match to job skills
        resume_skills = extract_keywords(text)
        matched, missing, score = match_resume_to_job(resume_skills, job.get("skills", []))

        st.subheader("Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Match Score", f"{score}%")
        with col2:
            st.metric("Matched Skills", str(len(matched)))
        with col3:
            st.metric("Missing Skills", str(len(missing)))

        st.write("**Matched:**", ", ".join(matched) if matched else "—")
        st.write("**Missing:**", ", ".join(missing) if missing else "—")
        st.write("**Extracted Name:**", name or "—")
        st.write("**Extracted Email:**", resume_email or "—")
        st.write("**Phone:**", phone or "—")

        # Submit application
        if st.button("Submit Application", type="primary"):
            phase = "Round 1 Scheduled" if score > 70 else "Not Selected"
            # Save minimal metadata; NO resume persisted
            res = save_application(
                user_id=st.session_state.user["id"],
                job_id=job["id"],
                match_score=score,
                phase=phase
            )
            if getattr(res, "error", None):
                st.error(f"Failed to save application: {res.error}")
            else:
                # notify user via registered account email (not resume email to keep consistent)
                notify_phase_update(
                    candidate_email=st.session_state.user["email"],
                    candidate_name=name,
                    new_phase=phase
                )
                st.success("Application submitted.")
                st.rerun()

    # List user's applications
    st.subheader("Your Applications")
    my_apps = get_user_applications(st.session_state.user["id"])
    if my_apps:
        df = pd.DataFrame(my_apps)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("You haven't submitted any applications yet.")


# ==============================
# HR PORTAL
# ==============================

def show_hr_portal():
    st.header("👔 HR Portal")

    # ---- Manage Jobs
    st.subheader("Manage Jobs")
    with st.expander("➕ Add Job", expanded=False):
        title = st.text_input("Job Title")
        desc = st.text_area("Job Description")
        skills_csv = st.text_input("Skills (comma separated)", placeholder="Python, SQL, ML, NLP")
        if st.button("Add Job"):
            skills = [s.strip() for s in skills_csv.split(",") if s.strip()]
            if not title or not desc or not skills:
                st.warning("Please provide title, description, and at least one skill.")
            else:
                res = add_job(title, desc, skills)
                if getattr(res, "error", None):
                    st.error(f"Failed to add job: {res.error}")
                else:
                    st.success("Job added.")
                    st.rerun()

    jobs = get_jobs()
    if jobs:
        with st.expander("✏️ Edit / Delete Jobs", expanded=False):
            job_map = {f"{j['title']}": j for j in jobs}
            pick = st.selectbox("Select Job", list(job_map.keys()))
            jsel = job_map[pick]
            etitle = st.text_input("Title", value=jsel["title"])
            edesc = st.text_area("Description", value=jsel["description"])
            eskills = st.text_input("Skills (comma separated)", value=", ".join(jsel.get("skills", [])))
            colA, colB = st.columns(2)
            with colA:
                if st.button("Update Job"):
                    skills_list = [s.strip() for s in eskills.split(",") if s.strip()]
                    res = update_job(jsel["id"], etitle, edesc, skills_list)
                    if getattr(res, "error", None):
                        st.error(f"Update failed: {res.error}")
                    else:
                        st.success("Job updated.")
                        st.rerun()
            with colB:
                if st.button("Delete Job"):
                    res = delete_job(jsel["id"])
                    if getattr(res, "error", None):
                        st.error(f"Delete failed: {res.error}")
                    else:
                        st.success("Job deleted.")
                        st.rerun()
    else:
        st.info("No jobs found. Add one above.")

    # ---- Applications
    st.subheader("Applications")
    apps = get_all_applications()
    if not apps:
        st.info("No applications yet.")
        return

    # Try to enrich with job titles if job_id present
    jobs_by_id = {j["id"]: j for j in jobs}
    for a in apps:
        if "job_id" in a and a["job_id"]:
            a["job_title"] = jobs_by_id.get(a["job_id"], {}).get("title", "")
        elif "job_role" in a:
            a["job_title"] = a.get("job_role", "")
        else:
            a["job_title"] = ""

    df = pd.DataFrame(apps)
    show_cols = [c for c in ["id", "user_id", "job_id", "job_title", "match_score", "phase", "submission_date"] if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

    st.markdown("### Update Application Phase")
    app_ids = [a["id"] for a in apps]
    chosen_id = st.selectbox("Select Application ID", app_ids)
    chosen = next(a for a in apps if a["id"] == chosen_id)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Current Phase:** {chosen['phase']}")
        st.write(f"**User ID:** {chosen['user_id']}")
        st.write(f"**Job:** {chosen.get('job_title','')}")

    with col2:
        if st.button("Mark PASS ➜ Next Phase", key=f"pass_{chosen_id}"):
            if chosen["phase"] in ["Not Selected", "Rejected", "Offer Extended"]:
                st.warning("Cannot advance this application.")
            else:
                new_phase = advance_phase(chosen["phase"])
                update_application_phase(chosen_id, new_phase)
                # Email to account email on file
                user_row = supabase.table("users").select("email").eq("id", chosen["user_id"]).single().execute().data
                candidate_email = user_row["email"] if user_row else None
                if candidate_email:
                    notify_phase_update(candidate_email, None, new_phase)
                st.success(f"Phase updated to {new_phase}")
                st.rerun()

    with col3:
        if st.button("Mark FAIL ➜ Rejected", key=f"fail_{chosen_id}"):
            update_application_phase(chosen_id, "Rejected")
            # Email to account email on file
            user_row = supabase.table("users").select("email").eq("id", chosen["user_id"]).single().execute().data
            candidate_email = user_row["email"] if user_row else None
            if candidate_email:
                notify_phase_update(candidate_email, None, "Rejected")
            st.success("Application marked as Rejected.")
            st.rerun()


# ==============================
# MAIN
# ==============================

if st.session_state.user is None:
    show_auth()
else:
    st.sidebar.markdown(f"**👤 {st.session_state.user['email']}**  \nRole: `{st.session_state.user['role']}`")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

    role = st.session_state.user["role"]
    if role == "User":
        show_user_portal()
    elif role == "HR":
        show_hr_portal()
    else:
        st.warning("Unknown role. Please contact admin.")
