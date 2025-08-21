# app.py
import streamlit as st
import hashlib
import pdfplumber
import docx
import re
import os
import json
from datetime import datetime, timedelta
from difflib import SequenceMatcher

# Supabase client
from supabase import create_client, Client

# spaCy (NER for name extraction)
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Download model in runtime if missing
        from spacy.cli import download as spacy_download
        spacy_download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
except Exception as e:
    nlp = None  # fallback, name extraction will be heuristic

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI Resume Analyzer (Supabase-backed, no resume storage)", layout="wide")

# Load secrets (Streamlit Cloud secrets)
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SMTP_EMAIL = st.secrets.get("email", {}).get("EMAIL")
SMTP_PASSWORD = st.secrets.get("email", {}).get("PASSWORD")

# -------------------- Helpers --------------------
def hash_password(password: str) -> str:
    """SHA-256 password hashing (stored in DB)."""
    return hashlib.sha256(password.encode()).hexdigest()

# Resume text extraction (in-memory only)
def extract_text_from_pdf(file) -> str:
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)

def extract_text_from_docx(file) -> str:
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

# Name extraction: heuristic + spaCy fallback
def extract_name(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    ignore = ["contact", "email", "phone", "address", "linkedin", "resume", "objective"]
    for line in lines[:12]:
        low = line.lower()
        if any(k in low for k in ignore):
            continue
        # ALL CAPS likely name
        if line.replace(" ", "").isalpha() and line.isupper() and 2 <= len(line.split()) <= 4:
            return line.title()
        # Title case short line
        if line.istitle() and 2 <= len(line.split()) <= 4:
            return line
    # spaCy NER fallback
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text.title()
    return "Not found"

def extract_email_from_text(text: str) -> str:
    m = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return m.group(0) if m else "Not found"

def extract_phone_from_text(text: str) -> str:
    m = re.search(r'(\+?\d{1,3}[\s\-\(\)]*)?(\d{2,4}[\s\-\(\)]*)?\d{3,4}[\s\-\(\)]*\d{3,4}', text)
    return m.group(0) if m else "Not found"

# Normalize tokens
def normalize_tokens(text: str):
    tokens = re.findall(r'[a-zA-Z0-9+#\-\._]+', text.lower())
    stop = set(["and","or","the","a","an","in","on","for","with","to","of","by","from"])
    filtered = [t for t in tokens if t and t not in stop and len(t) > 1]
    return set(filtered)

def fuzzy_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def compute_skill_match_score(resume_text: str, job_skills) -> (list, list, float):
    resume_tokens = normalize_tokens(resume_text)
    matched = []
    missing = []
    partial_matches = []

    for skill in job_skills:
        if not isinstance(skill, str):
            skill = str(skill)
        skill_norm = skill.lower().strip()
        skill_tokens = normalize_tokens(skill_norm)
        # Exact token overlap
        if skill_tokens and skill_tokens & resume_tokens:
            matched.append(skill)
            continue
        # fuzzy checks
        fscore_text = fuzzy_similarity(skill_norm, resume_text)
        fscore_tokens = max((fuzzy_similarity(skill_norm, tok) for tok in resume_tokens), default=0)
        fscore = max(fscore_text, fscore_tokens)
        if fscore >= 0.7:
            matched.append(skill)
            partial_matches.append((skill, fscore))
        elif fscore >= 0.45:
            missing.append(skill)  # keep in missing but record partial
            partial_matches.append((skill, fscore))
        else:
            missing.append(skill)

    total_skills = max(len(job_skills), 1)
    exact_count = len(matched)
    partial_effective = sum(score for (_, score) in partial_matches if score < 0.7 and score >= 0.45)
    matched_effective = exact_count + partial_effective
    raw_ratio = matched_effective / total_skills
    score_percent = round(raw_ratio * 100, 2)
    return matched, missing, score_percent

# Email sending (optional)
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib

def send_email_with_ics(to_email, subject, body, meeting_start=None, meeting_end=None):
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        return False, "SMTP not configured in secrets"
    if not to_email or to_email == "Not found":
        return False, "Recipient email missing"
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
DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}
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
        return True, "Email sent"
    except Exception as e:
        return False, str(e)

# -------------------- Supabase helpers --------------------
def get_user_by_email(email: str):
    res = supabase.table("users").select("*").eq("email", email).limit(1).execute()
    return res.data[0] if res.data else None

def register_user_supabase(email: str, password: str, role: str):
    existing = get_user_by_email(email)
    if existing:
        return False, "Email already exists"
    hashed = hash_password(password)
    payload = {"email": email, "password_hash": hashed, "role": role}
    res = supabase.table("users").insert(payload).execute()
    if res.status_code in (200,201) or res.data:
        return True, "Registered"
    return False, f"Failed to insert user: {res.error}"

def authenticate_supabase(email: str, password: str):
    user = get_user_by_email(email)
    if not user:
        return None
    if user.get("password_hash") == hash_password(password):
        return user  # full user object including id and role
    return None

def save_application_supabase(user_id: int, job_id, match_score: float, phase: str):
    # Per your earlier schema: applications columns include resume_url (text) which we will keep NULL
    payload = {
        "user_id": user_id,
        "resume_url": None,
        "match_score": float(match_score),
        "phase": phase,
        "submission_date": datetime.utcnow().isoformat(),
        "job_id": job_id
    }
    res = supabase.table("applications").insert(payload).execute()
    return res

def update_application_phase(application_id, new_phase, status=None):
    payload = {"phase": new_phase}
    if status is not None:
        payload["status"] = status
    res = supabase.table("applications").update(payload).eq("id", application_id).execute()
    return res

def fetch_jobs():
    res = supabase.table("jobs").select("*").execute()
    return res.data or []

def fetch_applications_for_company(job_id=None):
    # return applications, optionally filtered by job_id
    q = supabase.table("applications").select("*")
    if job_id:
        q = q.eq("job_id", job_id)
    res = q.execute()
    return res.data or []

# -------------------- UI --------------------
if "user" not in st.session_state:
    st.session_state.user = None  # will hold authenticated user object or None

st.title("📄 AI Resume Analyzer (Supabase-backed — resumes NOT stored)")

# Authentication UI
with st.sidebar:
    st.header("Authentication (Supabase)")
    if not st.session_state.user:
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login", key="login_btn"):
                u = authenticate_supabase(email, password)
                if u:
                    st.session_state.user = u
                    st.success("Logged in")
                    st.experimental_rerun()
                else:
                    st.error("Invalid credentials")
        with tab2:
            reg_email = st.text_input("Email (register)", key="reg_email")
            reg_pass = st.text_input("Password (register)", type="password", key="reg_pass")
            reg_role = st.selectbox("Role", ["User", "HR"], key="reg_role")
            if st.button("Register", key="reg_btn"):
                ok, msg = register_user_supabase(reg_email, reg_pass, reg_role)
                if ok:
                    st.success("Registered successfully. Please login.")
                else:
                    st.error(msg)
    else:
        st.markdown(f"**Logged in:** {st.session_state.user.get('email')}  \n**Role:** {st.session_state.user.get('role')}")
        if st.button("Logout"):
            st.session_state.user = None
            st.experimental_rerun()

# Not logged in -> stop after asking to login
if not st.session_state.user:
    st.info("Sign in (or register) via the sidebar to proceed.")
    st.stop()

# Read jobs for selection
jobs = fetch_jobs()
if not jobs:
    st.warning("No jobs found in Supabase `jobs` table.")
    st.stop()

# Normalize job display
def job_display_title(j):
    for k in ("title", "role", "name"):
        if k in j and j[k]:
            return j[k]
    return f"Job {j.get('id')}"

job_map = {job_display_title(j): j for j in jobs}

role = st.session_state.user.get("role", "User")

if role == "User":
    st.header("👤 Candidate Dashboard")
    st.markdown("Upload your resume — it will be analyzed in-memory only. An application record will be created but the resume file will NOT be stored.")

    selected_job_display = st.selectbox("Select Job", ["-- select --"] + list(job_map.keys()))
    uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX) — not stored", type=["pdf", "docx"])

    if uploaded_file and selected_job_display != "-- select --":
        # Extract text in-memory
        try:
            if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = extract_text_from_docx(uploaded_file)
        except Exception as e:
            st.error(f"Failed to parse resume: {e}")
            st.stop()

        name = extract_name(resume_text)
        email_in_resume = extract_email_from_text(resume_text)
        phone = extract_phone_from_text(resume_text)

        # prepare job skills from job record
        job = job_map[selected_job_display]
        job_skills = []
        if job.get("skills"):
            if isinstance(job["skills"], list):
                job_skills = [s for s in job["skills"] if isinstance(s, str)]
            else:
                job_skills = [s.strip() for s in str(job["skills"]).split(",") if s.strip()]
        elif job.get("description"):
            # fallback: extract some tokens from description (not ideal)
            job_skills = list(normalize_tokens(str(job.get("description", ""))))[:10]
        else:
            job_skills = []

        matched, missing, score = compute_skill_match_score(resume_text, job_skills)

        # Determine phase/status per your logic
        if score > 70:
            phase = "Round 1 (Interview Pending Scheduling)"
            status = "In Progress"
        else:
            phase = "Not Selected"
            status = "Rejected"

        # Insert application record into Supabase (resume_url NULL)
        user_id = st.session_state.user.get("id")
        job_id = job.get("id")
        res = save_application_supabase(user_id, job_id, score, phase)
        if res.error:
            st.error(f"Failed to save application: {res.error}")
        else:
            st.success(f"Application submitted (Score: {score}%). Phase: {phase}")
            st.markdown(f"**Name (extracted):** {name}")
            st.markdown(f"**Email (in resume):** {email_in_resume}")
            st.markdown(f"**Phone:** {phone}")
            st.markdown(f"**Matched skills:** {matched if matched else 'None'}")
            st.markdown(f"**Missing skills:** {missing if missing else 'None'}")

elif role == "HR":
    st.header("🏢 HR Dashboard")
    st.markdown("View applications and manage rounds. Resumes are not stored by the system — HR can upload a resume to re-analyze in-memory.")

    # Option to filter by job
    job_filter = st.selectbox("Filter by Job", ["All"] + list(job_map.keys()))
    selected_job_id = None
    if job_filter != "All":
        selected_job_id = job_map[job_filter].get("id")

    # Fetch applications from Supabase (optionally filter)
    apps = fetch_applications_for_company(job_id=selected_job_id)
    if not apps:
        st.info("No applications found for selected job/filter.")
    else:
        st.subheader("Applications")
        # Show each application with actions
        for app in sorted(apps, key=lambda x: x.get("submission_date") or "", reverse=True):
            # fetch job title
            j = next((jj for jj in jobs if jj.get("id") == app.get("job_id")), None)
            jtitle = job_display_title(j) if j else f"Job {app.get('job_id')}"
            cols = st.columns([3,2,1,2,2])
            cols[0].markdown(f"**Application #{app.get('id')}** — **{jtitle}**")
            cols[1].markdown(f"Score: **{app.get('match_score')}%**")
            cols[2].markdown(f"Phase: **{app.get('phase')}**")
            cols[3].markdown(f"Status: **{app.get('status') or 'N/A'}**")
            # Buttons: schedule (if pending), pass/fail if scheduled
            if "Pending Scheduling" in (app.get("phase") or "") and (app.get("status") in (None, "", "In Progress")):
                if cols[4].button("📅 Schedule", key=f"sch_{app.get('id')}"):
                    # Save selected application id in session to open scheduling UI below
                    st.session_state["to_schedule_id"] = app.get("id")
                    st.session_state["to_schedule_job_id"] = app.get("job_id")
                    st.experimental_rerun()
            elif "Scheduled" in (app.get("phase") or ""):
                if cols[4].button("✅ Pass", key=f"pass_{app.get('id')}"):
                    # Advance phase
                    cur_phase = app.get("phase","")
                    if cur_phase.startswith("Round 1"):
                        update_application_phase(app.get("id"), "Round 2 (Interview Pending Scheduling)")
                    elif cur_phase.startswith("Round 2"):
                        update_application_phase(app.get("id"), "Final (Interview Pending Scheduling)")
                    elif cur_phase.startswith("Final"):
                        update_application_phase(app.get("id"), "Selected", status="Selected")
                    st.experimental_rerun()
                if cols[4].button("❌ Fail", key=f"fail_{app.get('id')}"):
                    update_application_phase(app.get("id"), "Rejected", status="Rejected")
                    st.experimental_rerun()

    # Scheduling UI (when HR clicked schedule button)
    if st.session_state.get("to_schedule_id"):
        st.markdown("---")
        st.subheader("Schedule Interview (session action + update application record)")
        app_id = st.session_state["to_schedule_id"]
        selected_date = st.date_input("Meeting date", key="sch_date")
        selected_time = st.time_input("Meeting time", key="sch_time")
        duration = st.number_input("Duration (minutes)", min_value=15, max_value=240, step=15, value=30, key="sch_dur")
        meet_link = st.text_input("Meeting link (optional)", key="sch_link")
        if st.button("Send Invite & Mark Scheduled"):
            # find application row to get candidate email (we didn't store resume_url but user might have an email in another column)
            # Here we attempt to fetch candidate email from applications table — if you stored it, use it. Otherwise, cannot send.
            app_row = supabase.table("applications").select("*").eq("id", app_id).limit(1).execute().data
            if not app_row:
                st.error("Application row not found.")
            else:
                app_row = app_row[0]
                candidate_email = app_row.get("email")  # only if you store email column; else None
                start_dt = datetime.combine(selected_date, selected_time)
                end_dt = start_dt + timedelta(minutes=duration)
                # update application phase to Scheduled
                update_application_phase(app_id, app_row.get("phase", "").replace("Pending Scheduling", "Scheduled"))
                # optional: send email if configured and email present
                if SMTP_EMAIL and SMTP_PASSWORD and candidate_email:
                    subj = f"Interview Invitation — {job_display_title(next((jj for jj in jobs if jj.get('id')==app_row.get('job_id')),{}))}"
                    body = f"Dear Candidate,\n\nYour interview is scheduled at {start_dt}.\nLink: {meet_link}\n\nRegards."
                    sent, msg = send_email_with_ics(candidate_email, subj, body, start_dt, end_dt)
                    if sent:
                        st.success("Invite sent and application marked Scheduled.")
                    else:
                        st.warning(f"Application marked Scheduled. Email failed: {msg}")
                else:
                    st.success("Application marked Scheduled. (No email sent — SMTP missing or candidate email not present.)")
                # cleanup session schedule id
                del st.session_state["to_schedule_id"]
                st.experimental_rerun()

# Footer
st.markdown("---")
st.caption("This app reads jobs and uses Supabase for users/applications. Uploaded resumes are parsed in-memory only and are NOT stored anywhere (resume_url inserted NULL).")
