import streamlit as st
import pandas as pd
import re
import pdfplumber
import docx
import spacy
import hashlib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util
from supabase import create_client, Client

# =========================
# CONFIG & CLIENTS
# =========================
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Read secrets (set these in .streamlit/secrets.toml on Streamlit Cloud)
SUPABASE_URL: str = st.secrets["SUPABASE_URL"]
SUPABASE_KEY: str = st.secrets["SUPABASE_KEY"]
EMAIL_USER: str = st.secrets["EMAIL_USER"]          # your Gmail
EMAIL_PASS: str = st.secrets["EMAIL_PASS"]          # your Gmail App Password

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# NLP models
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    st.error(
        "spaCy model 'en_core_web_sm' not found. "
        "Add the wheel URL to requirements.txt and redeploy:\n\n"
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
    )
    st.stop()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# HELPERS
# =========================
def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def send_email_with_optional_ics(to_email: str, subject: str, body: str,
                                 meeting_start_utc: datetime | None = None,
                                 meeting_end_utc: datetime | None = None) -> bool:
    """Send email using Gmail SMTP; attach .ics if start/end provided (UTC datetimes)."""
    if not to_email:
        st.error("❌ Missing recipient email.")
        return False

    msg = MIMEMultipart("mixed")
    msg["From"] = EMAIL_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    if meeting_start_utc and meeting_end_utc:
        ics = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//AI Resume Analyzer//EN
CALSCALE:GREGORIAN
METHOD:REQUEST
BEGIN:VEVENT
UID:{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}@resume-analyzer
DTSTAMP:{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}
DTSTART:{meeting_start_utc.strftime('%Y%m%dT%H%M%SZ')}
DTEND:{meeting_end_utc.strftime('%Y%m%dT%H%M%SZ')}
SUMMARY:{subject}
DESCRIPTION:{body}
END:VEVENT
END:VCALENDAR
"""
        part = MIMEBase("text", "calendar", method="REQUEST", name="invite.ics")
        part.set_payload(ics)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment; filename=invite.ics")
        msg.attach(part)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"❌ Email failed: {e}")
        return False

def extract_text_from_pdf(file) -> str:
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text

def extract_text_from_docx(file) -> str:
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text(uploaded_file) -> str:
    if uploaded_file.name.lower().endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.lower().endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload PDF or DOCX.")
        return ""

def extract_contact_info(text: str):
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    phone_match = re.search(r"(\+?\d{1,3}[\s\-]?)?(\(?\d{2,5}\)?[\s\-]?)?\d{3,5}[\s\-]?\d{3,5}", text)

    # Try good-enough person detection from top lines; fallback to NER PERSON
    name = None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    ignore = {"contact", "email", "phone", "address", "linkedin", "resume", "curriculum", "vitae"}
    for line in lines[:8]:
        lower = line.lower()
        if any(k in lower for k in ignore):
            continue
        if line.replace(" ", "").isalpha() and (line.isupper() or 2 <= len(line.split()) <= 4):
            name = line.title()
            break
    if not name:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and 2 <= len(ent.text.split()) <= 5:
                name = ent.text.title()
                break

    return (
        name or "Not found",
        email_match.group(0) if email_match else "Not found",
        phone_match.group(0) if phone_match else "Not found",
    )

def extract_resume_tokens(text: str) -> list[str]:
    doc = nlp(text.lower())
    return [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]

def semantic_match(resume_tokens: list[str], jd_skills: list[str]):
    """Return (matched, missing, score%)."""
    if not jd_skills:
        return [], [], 0
    if not resume_tokens:
        return [], jd_skills, 0

    # Encode tokens and skills
    res_emb = embedder.encode(resume_tokens, convert_to_tensor=True)
    jd_emb = embedder.encode(jd_skills, convert_to_tensor=True)
    sim = util.cos_sim(res_emb, jd_emb)

    matched = []
    for j, skill in enumerate(jd_skills):
        # any resume token >= threshold to count as matched
        if float(sim[:, j].max()) >= 0.60:
            matched.append(skill)

    matched = sorted(set(matched), key=lambda x: jd_skills.index(x))
    missing = [s for s in jd_skills if s not in matched]
    score = int(round((len(matched) / len(jd_skills)) * 100)) if jd_skills else 0
    return matched, missing, score

# =========================
# SUPABASE DATA ACCESS
# =========================
# USERS
def supa_login(email: str, password: str):
    resp = supabase.table("users").select("*").eq("email", email).execute()
    if resp.data:
        row = resp.data[0]
        return row if row.get("password") == hash_password(password) else None
    return None

def supa_register(email: str, password: str, role: str) -> tuple[bool, str]:
    exists = supabase.table("users").select("email").eq("email", email).execute()
    if exists.data:
        return False, "Email already registered."
    supabase.table("users").insert({
        "email": email,
        "password": hash_password(password),
        "role": role
    }).execute()
    return True, "Registration successful."

# JOB DESCRIPTIONS
def jd_list() -> list[dict]:
    resp = supabase.table("job_descriptions").select("*").order("role").execute()
    return resp.data or []

def jd_upsert(role: str, description: str, skills_csv: str, jd_id: str | None = None):
    skills = [s.strip() for s in skills_csv.split(",") if s.strip()]
    payload = {"role": role, "description": description, "skills": skills}
    if jd_id:
        return supabase.table("job_descriptions").update(payload).eq("id", jd_id).execute()
    return supabase.table("job_descriptions").insert(payload).execute()

def jd_delete(jd_id: str):
    return supabase.table("job_descriptions").delete().eq("id", jd_id).execute()

# APPLICATIONS
def app_insert(user_email: str, candidate_email: str, name: str, phone: str,
               role: str, company: str, score: int, phase: str, status: str):
    supabase.table("applications").insert({
        "user_email": user_email,
        "candidate_email": candidate_email,
        "candidate_name": name,
        "candidate_phone": phone,
        "role": role,
        "company": company,
        "score": score,
        "phase": phase,
        "status": status,
        "created_at": datetime.utcnow().isoformat()
    }).execute()

def app_list(filter_company: str | None = None):
    q = supabase.table("applications").select("*").order("created_at", desc=True)
    if filter_company:
        q = q.eq("company", filter_company)
    return q.execute().data or []

def app_update(email: str, role: str, **fields):
    q = supabase.table("applications").update(fields).eq("candidate_email", email).eq("role", role)
    return q.execute()

# =========================
# AUTH & NAV
# =========================
if "user" not in st.session_state:
    st.session_state.user = None

with st.sidebar:
    st.markdown("### Navigation")
    if st.session_state.user:
        st.write(f"👤 {st.session_state.user['email']} ({st.session_state.user['role']})")
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()

if not st.session_state.user:
    tab_login, tab_register = st.tabs(["🔑 Login", "🆕 Register"])
    with tab_login:
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        if st.button("Login"):
            user = supa_login(email, pw)
            if user:
                st.session_state.user = user
                st.success("Logged in!")
                st.rerun()
            else:
                st.error("Invalid credentials.")
    with tab_register:
        email = st.text_input("New Email", key="reg_email")
        pw = st.text_input("New Password", type="password", key="reg_pw")
        role = st.selectbox("Role", ["User", "HR"])
        if st.button("Register"):
            ok, msg = supa_register(email, pw, role)
            st.success(msg) if ok else st.error(msg)

else:
    role = st.session_state.user["role"]

    # =========================
    # USER VIEW
    # =========================
    if role == "User":
        st.header("👤 Candidate Dashboard")

        # Pull JDs (roles + optional company)
        jds = jd_list()
        if not jds:
            st.info("No job descriptions available yet. Please check back later.")
        else:
            col1, col2 = st.columns(2)
            job_role = col1.selectbox("Select a Job Role", [row["role"] for row in jds])
            # Optional company selector (you can keep it simple with a free text)
            company = col2.text_input("Company (optional)", value="Acme Inc.")

            uploaded = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
            if uploaded:
                full_text = extract_text(uploaded)
                name, cand_email, phone = extract_contact_info(full_text)
                resume_tokens = extract_resume_tokens(full_text)

                # get JD skills for chosen role
                chosen = next((row for row in jds if row["role"] == job_role), None)
                jd_skills = chosen["skills"] if chosen else []

                matched, missing, score = semantic_match(resume_tokens, jd_skills)

                # Phase/status rule
                if score >= 70 and cand_email != "Not found":
                    phase, status = "Round 1 (Interview Pending Scheduling)", "In Progress"
                else:
                    phase, status = "Not Selected", "Rejected"

                # Save application
                app_insert(
                    user_email=st.session_state.user["email"],
                    candidate_email=cand_email if cand_email != "Not found" else "",
                    name=name,
                    phone=phone if phone != "Not found" else "",
                    role=job_role,
                    company=company,
                    score=score,
                    phase=phase,
                    status=status
                )

                st.success(f"Application submitted for **{company} — {job_role}** (Score: {score}%)")
                st.markdown(f"**Name:** {name}")
                st.markdown(f"**Email:** {cand_email}")
                st.markdown(f"**Phone:** {phone}")
                st.markdown(f"**Matched Skills:** {', '.join(matched) if matched else 'None'}")
                st.markdown(f"**Missing Skills:** {', '.join(missing) if missing else 'None'}")
                st.markdown(f"**Phase:** {phase}")
                st.markdown(f"**Status:** {status}")

        # Show my applications
        st.subheader("📜 My Applications")
        my_apps = [r for r in app_list() if r.get("user_email") == st.session_state.user["email"]]
        if my_apps:
            st.dataframe(pd.DataFrame(my_apps))
        else:
            st.info("No applications yet.")

    # =========================
    # HR VIEW
    # =========================
    elif role == "HR":
        st.header("🏢 HR Dashboard")
        tab_manage_jd, tab_candidates = st.tabs(["📋 Manage Job Descriptions", "🧪 Review Candidates"])

        # ---- Manage Job Descriptions ----
        with tab_manage_jd:
            st.subheader("Existing Roles")
            rows = jd_list()
            if rows:
                df = pd.DataFrame([{
                    "id": r["id"],
                    "role": r["role"],
                    "description": r.get("description", ""),
                    "skills (comma-separated)": ", ".join(r.get("skills", []))
                } for r in rows])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No job descriptions yet.")

            st.markdown("---")
            st.subheader("➕ Add / ✏️ Edit Role")
            edit_mode = st.checkbox("Edit existing?")
            if edit_mode and rows:
                to_edit_role = st.selectbox("Select role to edit", [r["role"] for r in rows])
                editing = next(r for r in rows if r["role"] == to_edit_role)
                role_name = st.text_input("Role", value=editing["role"])
                desc = st.text_area("Description", value=editing.get("description", ""))
                skills_csv = st.text_area("Skills (comma-separated)", value=", ".join(editing.get("skills", [])))
                c1, c2 = st.columns(2)
                if c1.button("Save Changes"):
                    jd_upsert(role_name, desc, skills_csv, jd_id=editing["id"])
                    st.success("Updated.")
                    st.rerun()
                if c2.button("Delete Role", type="primary"):
                    jd_delete(editing["id"])
                    st.success("Deleted.")
                    st.rerun()
            else:
                role_name = st.text_input("Role")
                desc = st.text_area("Description")
                skills_csv = st.text_area("Skills (comma-separated)")
                if st.button("Add Role"):
                    if not role_name or not skills_csv.strip():
                        st.error("Role and skills are required.")
                    else:
                        jd_upsert(role_name, desc, skills_csv)
                        st.success("Added.")
                        st.rerun()

        # ---- Review Candidates ----
        with tab_candidates:
            st.subheader("Filter")
            all_apps = app_list()
            companies = sorted({r.get("company", "") for r in all_apps if r.get("company")})
            company = st.selectbox("Company", ["(All)"] + companies)
            view = all_apps if company == "(All)" else app_list(company)

            if not view:
                st.info("No candidates found.")
            else:
                df = pd.DataFrame(view)
                st.dataframe(df, use_container_width=True)

                st.markdown("---")
                st.subheader("Process Candidate")
                # Pick by email+role combo (unique enough for demo)
                options = [f"{r['candidate_email']} | {r['role']}" for r in view if r.get("candidate_email")]
                if options:
                    pick = st.selectbox("Select candidate", options)
                    email_sel, role_sel = [x.strip() for x in pick.split("|")]
                    email_sel = email_sel
                    role_sel = role_sel

                    current = next((r for r in view if r["candidate_email"] == email_sel and r["role"] == role_sel), None)
                    if current:
                        st.write(f"**Name:** {current.get('candidate_name','')}")
                        st.write(f"**Score:** {current.get('score', 0)}%")
                        st.write(f"**Phase:** {current.get('phase','')}")
                        st.write(f"**Status:** {current.get('status','')}")

                        # If pending scheduling, allow scheduler (sends .ics)
                        if "Pending Scheduling" in current.get("phase", "") and current.get("status") == "In Progress":
                            st.markdown("### 📅 Schedule Interview")
                            date = st.date_input("Date")
                            time = st.time_input("Time")
                            duration = st.number_input("Duration (minutes)", 15, 240, 30, 15)
                            meet_link = st.text_input("Meeting Link (Google Meet/Zoom)")

                            if st.button("Send Invite"):
                                start_dt = datetime.combine(date, time)
                                end_dt = start_dt + timedelta(minutes=int(duration))
                                body = (
                                    f"Dear {current.get('candidate_name','Candidate')},\n\n"
                                    f"You are invited for an interview.\n\n"
                                    f"Role: {current.get('role')}\n"
                                    f"Company: {current.get('company')}\n"
                                    f"Link: {meet_link}\n\n"
                                    "Best regards,\nHR Team"
                                )
                                sent = send_email_with_optional_ics(
                                    to_email=email_sel,
                                    subject=f"Interview for {role_sel} - {current.get('company','')}",
                                    body=body,
                                    meeting_start_utc=start_dt,  # assumed UTC; adjust if needed
                                    meeting_end_utc=end_dt
                                )
                                if sent:
                                    app_update(email_sel, role_sel, phase=current["phase"].replace("Pending Scheduling", "Scheduled"))
                                    st.success("Invite sent & phase updated to Scheduled.")
                                    st.rerun()

                        # If scheduled, allow pass/fail to move to next round
                        if "Scheduled" in current.get("phase", "") and current.get("status") == "In Progress":
                            st.markdown("### 🧪 Update Result")
                            decision = st.radio("Result", ["Pass", "Fail"], horizontal=True)
                            if st.button("Submit Result"):
                                if decision == "Fail":
                                    app_update(email_sel, role_sel, status="Rejected", phase="Rejected")
                                    send_email_with_optional_ics(
                                        to_email=email_sel,
                                        subject="Application Update",
                                        body="Thank you for interviewing with us. We won't be moving forward at this time."
                                    )
                                    st.error("Candidate Rejected.")
                                else:
                                    # advance to next phase
                                    phase = current["phase"]
                                    if phase.startswith("Round 1"):
                                        new_phase = "Round 2 (Interview Pending Scheduling)"
                                    elif phase.startswith("Round 2"):
                                        new_phase = "Final (Interview Pending Scheduling)"
                                    elif phase.startswith("Final"):
                                        new_phase = "Selected"
                                        app_update(email_sel, role_sel, status="Selected", phase=new_phase)
                                        send_email_with_optional_ics(
                                            to_email=email_sel,
                                            subject="🎉 Job Offer",
                                            body=f"Congratulations! You are selected for {role_sel}."
                                        )
                                        st.success("Offer sent.")
                                        st.rerun()
                                    else:
                                        new_phase = "Round 1 (Interview Pending Scheduling)"

                                    if new_phase != "Selected":
                                        app_update(email_sel, role_sel, phase=new_phase)
                                        send_email_with_optional_ics(
                                            to_email=email_sel,
                                            subject=f"Next Round: {new_phase.split('(')[0].strip()}",
                                            body=f"Congrats! You advanced to {new_phase}."
                                        )
                                        st.success(f"Moved to {new_phase}.")
                                st.rerun()
                else:
                    st.info("No selectable candidates yet.")
