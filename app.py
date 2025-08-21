# app.py
import streamlit as st
import hashlib
import pdfplumber
import docx
import re
import spacy
import nltk
from datetime import datetime, timedelta
from supabase import create_client, Client
from difflib import SequenceMatcher
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib

# -------------------- Minimal NLTK setup --------------------
# downloads are idempotent; Streamlit Cloud will reuse cached files when available.
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))

# -------------------- CONFIG --------------------
st.set_page_config(page_title="AI Resume Analyzer (In-memory)", layout="wide")

# Supabase connection (read-only for jobs)
SUPABASE_URL = st.secrets["supabase"]["url"]
SUPABASE_KEY = st.secrets["supabase"]["key"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Optional email config (for scheduling invites) - set in Streamlit secrets if you want
SMTP_EMAIL = st.secrets.get("email", {}).get("EMAIL")
SMTP_PASSWORD = st.secrets.get("email", {}).get("PASSWORD")

# Load spaCy (name/entity extraction)
nlp = spacy.load("en_core_web_sm")

# -------------------- Helpers --------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def extract_text_from_pdf(file) -> str:
    """Extract text from an uploaded PDF file-like object."""
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages)

def extract_text_from_docx(file) -> str:
    """Extract text from uploaded DOCX file-like object."""
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_name(text: str) -> str:
    """Heuristic + spaCy NER fallback to extract a candidate name from resume text."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    ignore = ["contact", "email", "phone", "address", "linkedin", "resume", "curriculum", "objective"]
    # Heuristic: first lines likely contain name (ALL CAPS or Title Case)
    for line in lines[:12]:
        lower = line.lower()
        if any(k in lower for k in ignore):
            continue
        # All-caps name
        if line.replace(" ", "").isalpha() and line.isupper() and 2 <= len(line.split()) <= 4:
            return line.title()
        # Title-case short line
        if line.istitle() and 2 <= len(line.split()) <= 4:
            return line
    # spaCy named entity detection fallback
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.title()
    return "Not found"

def extract_email_from_text(text: str) -> str:
    m = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return m.group(0) if m else "Not found"

def extract_phone_from_text(text: str) -> str:
    # Accepts international-like formats and many variants
    m = re.search(r'(\+?\d{1,3}[\s\-\(\)]*)?(\d{2,4}[\s\-\(\)]*)?\d{3,4}[\s\-\(\)]*\d{3,4}', text)
    return m.group(0) if m else "Not found"

def normalize_tokens(text: str):
    """Lowercase, tokenize, remove stopwords & non-alpha tokens, return set."""
    tokens = word_tokenize(text.lower())
    words = [re.sub(r'[^a-z0-9#+\-\.]', "", t) for t in tokens]
    words = [w for w in words if w and (w.isalnum() or any(ch in w for ch in ["+", "#", "-", "."])) and w not in STOPWORDS]
    return set(words)

def fuzzy_similarity(a: str, b: str) -> float:
    """Return a fuzzy similarity ratio between 0 and 1 for two strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def compute_skill_match_score(resume_text: str, job_skills) -> (list, list, float):
    """
    Compute matched skills, missing skills, and a composite score (0-100).
    - job_skills: list[str] expected from Supabase (array text)
    """
    resume_tokens = normalize_tokens(resume_text)
    matched = []
    missing = []
    partial_matches = []

    for skill in job_skills:
        # Skill normalized tokens
        skill_norm = skill.lower().strip()
        skill_tokens = normalize_tokens(skill_norm)
        # Exact token overlap (strong signal)
        if skill_tokens and skill_tokens & resume_tokens:
            matched.append(skill)
            continue
        # fuzzy check against resume text and tokens
        # 1) fuzzy against entire resume text
        fscore_text = fuzzy_similarity(skill_norm, resume_text)
        # 2) fuzzy vs tokens joined
        fscore_tokens = max((fuzzy_similarity(skill_norm, tok) for tok in resume_tokens), default=0)
        fscore = max(fscore_text, fscore_tokens)
        if fscore >= 0.7:
            matched.append(skill)
            partial_matches.append((skill, fscore))
        elif fscore >= 0.45:
            # treat as partial — gives half credit
            missing.append(skill)
            partial_matches.append((skill, fscore))
        else:
            missing.append(skill)

    # Score calculation:
    # base = exact matches count, partials give half credit weighted by fscore
    exact_count = len(matched)
    partial_credit = sum((score for (_, score) in partial_matches if score < 1.0 and score >= 0.45), 0.0)
    # To avoid double counting, cap computations:
    total_skills = max(len(job_skills), 1)
    # combine: exact (1.0), partial credit in [0.45..0.69] scaled
    # For simplicity: treat matched as full, each partial contributes (fscore)
    # But matched list already included strong fuzzy >=0.7; partial_credit contains only those <1.0 and >=0.45
    # We'll compute matched_count_effective = exact_count + sum(partial_score_scaled)
    partial_effective = sum(score for (_, score) in partial_matches if score < 0.7 and score >= 0.45)
    matched_effective = exact_count + partial_effective
    raw_ratio = matched_effective / total_skills
    score_percent = round(raw_ratio * 100, 2)
    return matched, missing, score_percent

def send_email_with_ics(to_email, subject, body, meeting_start=None, meeting_end=None):
    """Optional: send email with .ics invite (requires SMTP_EMAIL & SMTP_PASSWORD in secrets)."""
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        st.error("Email not configured in secrets (EMAIL/PASSWORD).")
        return False
    if not to_email or to_email == "Not found":
        st.error("Recipient email missing.")
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
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# -------------------- UI / Flow --------------------
# Ensure session-state keys
if "candidates_session" not in st.session_state:
    # store last-analyzed candidates for the session (ephemeral only)
    st.session_state["candidates_session"] = []  # list of dicts: {name,email,phone,job,title,score,matched,missing,datetime}

# Top-level title
st.title("📄 AI Resume Analyzer — In-Memory (No Resume Storage)")

# Authentication (session-only)
def login_register_ui():
    st.sidebar.header("🔐 Login / Register (session-only)")
    tabs = st.sidebar.tabs(["🔑 Login", "🆕 Register"])

    with tabs[0]:
        login_email = st.text_input("Email", key="login_email")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", key="login_btn"):
            if login_email and login_pass:
                # session-only "authentication"
                st.session_state["username"] = login_email
                # Allow selecting role at login for demo or read role from email domain heuristics (default User)
                st.session_state["role"] = st.radio("Select role for this session", ["User", "HR"], index=0, key="role_radio_login")
                st.success("Logged in (session-only). Use the sidebar to logout.")
                st.experimental_rerun()
            else:
                st.error("Enter email and password to login.")

    with tabs[1]:
        reg_email = st.text_input("Email (register)", key="reg_email")
        reg_pass = st.text_input("Password (register)", type="password", key="reg_pass")
        reg_role = st.selectbox("Role (session-only)", ["User", "HR"], key="reg_role")
        if st.button("Register", key="reg_btn"):
            if reg_email and reg_pass:
                # no storing — just prefill session and instruct user to login
                st.success("Registered (session-only). Now switch to Login and sign in.")
            else:
                st.warning("Please provide both email and password.")

if "username" not in st.session_state:
    login_register_ui()
    st.info("This app reads job postings from Supabase but **does not store resumes**. Register/login is session-only (no persistent storage).")
    st.stop()

# Sidebar: logged-in info and logout
with st.sidebar:
    st.markdown(f"👤 Logged in as **{st.session_state['username']}**")
    st.markdown(f"🔖 Role: **{st.session_state.get('role','User')}**")
    if st.button("🚪 Logout (clear session)"):
        # clear all session keys (ephemeral)
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

# Read jobs (only read)
jobs_res = supabase.table("jobs").select("*").execute()
jobs_data = jobs_res.data or []
if not jobs_data:
    st.warning("No jobs found in Supabase `jobs` table. Please add job rows and reload.")
    st.stop()

# Present role-specific views
role = st.session_state.get("role", "User")

# Common: prepare job selection mapping. Accept different job schema column names (title/text/description/skills)
job_title_key = None
for k in ("title", "role", "name"):
    if any(k in j and j.get(k) for j in jobs_data):
        job_title_key = k
        break
if not job_title_key:
    # fallback: use id or index
    job_title_key = "title"  # may be empty strings but at least key exists

job_map = {}
for j in jobs_data:
    display = j.get(job_title_key) or f"Job {j.get('id') or '[no title]'}"
    job_map[display] = j

if role == "User":
    st.header("👤 Candidate Dashboard")
    st.markdown("Upload your resume. It will be analyzed in-memory only and not stored anywhere.")
    selected_job_display = st.selectbox("Choose job to check against", ["-- select --"] + list(job_map.keys()))
    uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX) — **will NOT be stored**", type=["pdf", "docx"])

    if uploaded_file and selected_job_display != "-- select --":
        # Extract resume text
        try:
            if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = extract_text_from_docx(uploaded_file)
        except Exception as e:
            st.error(f"Failed to extract resume text: {e}")
            st.stop()

        # Extract personal info
        name = extract_name(resume_text)
        email = extract_email_from_text(resume_text)
        phone = extract_phone_from_text(resume_text)

        # Pull job skills from Supabase record; handle multiple storage forms
        job = job_map[selected_job_display]
        # Try possible keys used earlier: skills (array), text fields that may contain skills divided by commas, description, etc.
        job_skills = []
        if job.get("skills"):
            # If Supabase returns Postgres array, it should be a list
            if isinstance(job["skills"], list):
                job_skills = [s for s in job["skills"] if isinstance(s, str)]
            else:
                # comma-separated string
                job_skills = [s.strip() for s in str(job["skills"]).split(",") if s.strip()]
        elif job.get("text"):
            # some tables store skill list in text JSON or plain text
            # attempt comma split from text
            job_skills = [s.strip() for s in str(job.get("text")).split(",") if s.strip()]
        elif job.get("description"):
            # fallback: try to extract common skill-like tokens from description (simple heuristic)
            desc_tokens = normalize_tokens(str(job.get("description", "")))
            job_skills = list(desc_tokens)[:10]  # short sample
        else:
            job_skills = []

        # compute skill match
        matched, missing, score = compute_skill_match_score(resume_text, job_skills)

        # Prepare ephemeral candidate record and store in session list (not persisted)
        candidate_record = {
            "datetime": datetime.now().isoformat(),
            "name": name,
            "email": email,
            "phone": phone,
            "job": selected_job_display,
            "job_id": job.get("id"),
            "score": score,
            "matched": matched,
            "missing": missing,
            "raw_text_sample": (resume_text[:1000] + "...") if len(resume_text) > 1000 else resume_text,
        }
        st.session_state["candidates_session"].insert(0, candidate_record)  # newest first

        # Display results
        st.subheader("👤 Extracted Candidate Info")
        st.write(f"**Name:** {name}")
        st.write(f"**Email:** {email}")
        st.write(f"**Phone:** {phone}")

        st.subheader("💼 Job & Matching")
        st.write(f"**Job:** {selected_job_display}")
        st.write(f"**Skill set (from job row):** {job_skills if job_skills else 'No skills present in job record'}")
        st.write(f"**Matched skills:** {matched if matched else 'None'}")
        st.write(f"**Missing skills:** {missing if missing else 'None'}")
        st.metric(label="Match Score (%)", value=score, help="Computed using token overlap + fuzzy matching heuristics")
        st.success("Resume analyzed in-memory — nothing was stored or uploaded to external storage.")

        # Optional: allow user to request interview scheduling email (only if SMTP configured)
        if SMTP_EMAIL and SMTP_PASSWORD and email != "Not found":
            st.markdown("---")
            st.info("Optional: Send an interview invite email (uses SMTP credentials from secrets).")
            meet = st.checkbox("Send interview invite email now", key="send_invite_user")
            if meet:
                meeting_date = st.date_input("Meeting date", key="user_meet_date")
                meeting_time = st.time_input("Meeting time", key="user_meet_time")
                duration = st.number_input("Duration (minutes)", min_value=15, max_value=240, step=15, value=30, key="user_duration")
                link = st.text_input("Meeting link (optional)", key="user_meet_link")
                if st.button("Send Invite (ephemeral)", key="user_send_invite"):
                    start_dt = datetime.combine(meeting_date, meeting_time)
                    end_dt = start_dt + timedelta(minutes=duration)
                    subj = f"Interview: {selected_job_display}"
                    body = f"Dear {name if name!='Not found' else 'Candidate'},\n\nYour interview is scheduled at {start_dt}.\n\nLink: {link}\n\nBest regards."
                    ok = send_email_with_ics(email, subj, body, start_dt, end_dt)
                    if ok:
                        st.success("Invite sent (email).")
        else:
            st.info("Email invite disabled (missing SMTP credentials in Streamlit secrets).")

    # Show session-only recent analyses
    if st.session_state["candidates_session"]:
        st.markdown("---")
        st.subheader("🔁 Recent Analyses (session-only)")
        for idx, c in enumerate(st.session_state["candidates_session"][:10]):
            with st.expander(f"{c['name']} — {c['job']} — {c['score']}% — {c['datetime']}", expanded=(idx==0)):
                st.write(f"**Email:** {c['email']}  |  **Phone:** {c['phone']}")
                st.write(f"**Matched:** {c['matched']}")
                st.write(f"**Missing:** {c['missing']}")
                st.write(f"Sample text (truncated):")
                st.code(c["raw_text_sample"][:1000])

elif role == "HR":
    st.header("🏢 HR Dashboard (session-only, no resume storage)")
    st.markdown("You can load candidate resumes for analysis (in-memory) and manage scheduling or pass/fail — nothing will be persisted.")

    # HR can optionally upload a resume to analyze on the spot (same as User flow)
    selected_job_display = st.selectbox("Choose job to view candidates / analyze resume", ["-- select --"] + list(job_map.keys()))
    uploaded_file = st.file_uploader("Upload candidate resume (PDF or DOCX) — will NOT be stored", type=["pdf", "docx"], key="hr_upload")

    if uploaded_file and selected_job_display != "-- select --":
        try:
            if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = extract_text_from_docx(uploaded_file)
        except Exception as e:
            st.error(f"Failed to extract resume text: {e}")
            st.stop()

        name = extract_name(resume_text)
        email = extract_email_from_text(resume_text)
        phone = extract_phone_from_text(resume_text)

        job = job_map[selected_job_display]
        # determine skills same as above
        job_skills = []
        if job.get("skills"):
            if isinstance(job["skills"], list):
                job_skills = [s for s in job["skills"] if isinstance(s, str)]
            else:
                job_skills = [s.strip() for s in str(job["skills"]).split(",") if s.strip()]
        elif job.get("text"):
            job_skills = [s.strip() for s in str(job.get("text")).split(",") if s.strip()]
        elif job.get("description"):
            job_skills = list(normalize_tokens(str(job.get("description", ""))))[:10]
        else:
            job_skills = []

        matched, missing, score = compute_skill_match_score(resume_text, job_skills)

        # transient record for HR session view
        candidate_record = {
            "datetime": datetime.now().isoformat(),
            "name": name,
            "email": email,
            "phone": phone,
            "job": selected_job_display,
            "job_id": job.get("id"),
            "score": score,
            "matched": matched,
            "missing": missing,
            "raw_text_sample": (resume_text[:1000] + "...") if len(resume_text) > 1000 else resume_text,
            "hr_status": "In Review",  # default ephemeral status
        }
        st.session_state["candidates_session"].insert(0, candidate_record)

        # Display
        st.subheader("Candidate Snapshot")
        st.write(f"**Name:** {name}")
        st.write(f"**Email:** {email}")
        st.write(f"**Phone:** {phone}")
        st.write(f"**Matched:** {matched}")
        st.write(f"**Missing:** {missing}")
        st.metric(label="Match Score (%)", value=score)

        # HR actions (ephemeral)
        st.markdown("---")
        st.subheader("HR Actions (session-only)")
        if score >= 70:
            st.success("Candidate is eligible for interview based on score threshold (>=70%).")
        else:
            st.info("Candidate below score threshold (>=70%).")

        # Schedule interview (ephemeral + optional email)
        meeting_date = st.date_input("Meeting Date (optional)", key=f"hr_date_{len(st.session_state['candidates_session'])}")
        meeting_time = st.time_input("Meeting Time (optional)", key=f"hr_time_{len(st.session_state['candidates_session'])}")
        duration = st.number_input("Duration (mins)", min_value=15, max_value=240, step=15, value=30, key=f"hr_dur_{len(st.session_state['candidates_session'])}")
        meet_link = st.text_input("Meeting link (optional)", key=f"hr_link_{len(st.session_state['candidates_session'])}")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Mark Pass (ephemeral)", key=f"hr_pass_{len(st.session_state['candidates_session'])}"):
                st.session_state["candidates_session"][0]["hr_status"] = "Pass"
                st.success("Marked Pass (session-only).")
        with col2:
            if st.button("Mark Fail (ephemeral)", key=f"hr_fail_{len(st.session_state['candidates_session'])}"):
                st.session_state["candidates_session"][0]["hr_status"] = "Fail"
                st.warning("Marked Fail (session-only).")
        with col3:
            if st.button("Schedule & Send Invite (ephemeral)", key=f"hr_sched_{len(st.session_state['candidates_session'])}"):
                start_dt = datetime.combine(meeting_date, meeting_time)
                end_dt = start_dt + timedelta(minutes=duration)
                # attempt to send email if configured and candidate email present
                if SMTP_EMAIL and SMTP_PASSWORD and email != "Not found":
                    subj = f"Interview Invitation — {selected_job_display}"
                    body = f"Dear {name},\n\nYou are invited for an interview for {selected_job_display}.\n\nWhen: {start_dt}\nLink: {meet_link}\n\nRegards."
                    ok = send_email_with_ics(email, subj, body, start_dt, end_dt)
                    if ok:
                        st.success("Invite sent (ephemeral). Candidate remains session-only.")
                    else:
                        st.error("Failed to send invite.")
                else:
                    st.info("SMTP not configured or candidate email missing – scheduling recorded only in session (ephemeral).")
                    st.session_state["candidates_session"][0]["hr_status"] = "Scheduled (session-only)"

    # HR: show session-only candidate list and allow ephemeral actions
    st.markdown("---")
    st.subheader("Session Candidates (ephemeral)")
    if not st.session_state["candidates_session"]:
        st.info("No candidates analyzed in this session yet.")
    else:
        for i, c in enumerate(st.session_state["candidates_session"]):
            with st.expander(f"{c['name']} — {c['job']} — {c['score']}% — status: {c.get('hr_status','In Review')}", expanded=(i==0)):
                st.write(f"**Email:** {c['email']}  |  **Phone:** {c['phone']}")
                st.write(f"**Matched:** {c['matched']}")
                st.write(f"**Missing:** {c['missing']}")
                st.write(f"**Session status:** {c.get('hr_status','In Review')}")
                # ephemeral actions
                if st.button("Set status: Offer (session-only)", key=f"offer_{i}"):
                    st.session_state["candidates_session"][i]["hr_status"] = "Offer"
                    st.success("Status set to Offer (session-only).")
                if st.button("Set status: Reject (session-only)", key=f"reject_{i}"):
                    st.session_state["candidates_session"][i]["hr_status"] = "Rejected"
                    st.warning("Status set to Rejected (session-only).")

else:
    st.error("Unknown role. Please logout and login again selecting User or HR.")

# Footer reminder
st.markdown("---")
st.caption("Note: This app analyzes resumes in-memory only. Resumes are never stored on disk or in Supabase. Session data (recent analyses) lives only in the current session and is not persisted.")
