# app.py
import streamlit as st
import hashlib
from supabase import create_client
from db import *
from resume_parser import extract_text_from_pdf, extract_text_from_docx, extract_contact_info
from match_skills import match_resume_to_job
from datetime import datetime
import json

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# ---------------------------
# Secrets & Supabase client
# ---------------------------
# Put these in Streamlit secrets (Settings -> Secrets)
# SUPABASE_URL, SUPABASE_KEY, EMAIL_USER, EMAIL_PASS (optional)
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# Helpers
# ---------------------------
def sha256_hash(p: str) -> str:
    return hashlib.sha256(p.encode()).hexdigest()

def require_login():
    if "user" not in st.session_state:
        st.session_state["user"] = None

require_login()

# ---------------------------
# Auth UI
# ---------------------------
def login_register_ui():
    st.title("AI Resume Analyzer")
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            user = get_user(supabase, email)
            if user and user.get("password_hash") == sha256_hash(password):
                st.session_state["user"] = user
                st.success("Logged in")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        st.subheader("Register (creates user in Supabase)")
        email = st.text_input("Email (register)", key="reg_email")
        password = st.text_input("Password", type="password", key="reg_pass")
        role = st.selectbox("Role", ["User", "HR"], key="reg_role")
        if st.button("Register"):
            existing = get_user(supabase, email)
            if existing:
                st.error("Email already exists")
            else:
                insert_user(supabase, email, sha256_hash(password), role)
                st.success("Registered — please log in")

# ---------------------------
# User Portal
# ---------------------------
def candidate_portal():
    st.header("Candidate Portal")
    st.sidebar.write(f"Logged in: {st.session_state['user']['email']} (User)")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.experimental_rerun()

    jobs = get_jobs(supabase)
    if not jobs:
        st.info("No jobs available. Ask HR to add jobs.")
        return

    job_map = {job["title"]: job for job in jobs}
    selected_title = st.selectbox("Select Job", options=list(job_map.keys()))
    selected_job = job_map[selected_title]

    with st.expander("Job details", expanded=True):
        st.markdown(f"**{selected_job['title']}**")
        st.write(selected_job.get("description", ""))
        st.write("Required skills:", ", ".join(selected_job.get("skills", [])))

    uploaded = st.file_uploader("Upload resume (PDF/DOCX) — file is NOT stored", type=["pdf", "docx"])
    if uploaded:
        # parse in memory
        if uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
            text = extract_text_from_pdf(uploaded)
        else:
            text = extract_text_from_docx(uploaded)

        name, email_from_resume, phone = extract_contact_info(text)
        score, matched, missing = match_resume_to_job(text, selected_job.get("skills", []))

        st.subheader("Analysis")
        st.metric("Match score", f"{score}%")
        st.write("Matched:", matched or "—")
        st.write("Missing:", missing or "—")
        st.write("Name (extracted):", name or "—")
        st.write("Email (extracted):", email_from_resume or "—")
        st.write("Phone (extracted):", phone or "—")

        if st.button("Submit application"):
            # store metadata in applications table; DO NOT store file content/URL
            user_row = st.session_state["user"]
            payload = {
                "user_id": user_row["id"],
                "candidate_name": name,
                "candidate_email": user_row["email"],  # use registered email as canonical contact
                "candidate_phone": phone,
                "job_id": selected_job["id"],
                "resume_text": None,   # explicit: not storing resume
                "skills": json.dumps(matched),
                "match_score": score,
                "phase": "Round 1 Scheduled" if score > 70 else "Not Selected",
                "status": "Scheduled" if score > 70 else "Rejected",
                "created_at": datetime.utcnow().isoformat()
            }
            insert_application(supabase, payload)
            st.success("Application submitted and saved in Supabase.")
            st.experimental_rerun()

    # show user's applications
    st.subheader("My applications")
    apps = get_applications(supabase)
    my_apps = [a for a in apps if a.get("user_id") == st.session_state["user"]["id"]]
    if not my_apps:
        st.info("No applications yet")
    else:
        for a in my_apps:
            st.write(f"Job: {get_job_by_id(supabase, a.get('job_id'))['title'] if a.get('job_id') else a.get('job_role')}")
            st.write(f"Score: {a.get('match_score')}")
            st.write(f"Phase: {a.get('phase')} — Status: {a.get('status')}")
            st.write("---")

# ---------------------------
# HR Portal
# ---------------------------
def hr_portal():
    st.header("HR Portal")
    st.sidebar.write(f"Logged in: {st.session_state['user']['email']} (HR)")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.experimental_rerun()

    st.subheader("Create / Edit Jobs")
    with st.form("create_job", clear_on_submit=True):
        title = st.text_input("Title")
        description = st.text_area("Description")
        skills_csv = st.text_input("Skills (comma separated)")
        submitted = st.form_submit_button("Create job")
        if submitted:
            skills = [s.strip() for s in skills_csv.split(",") if s.strip()]
            create_job(supabase, title, description, skills)
            st.success("Job created")
            st.experimental_rerun()

    st.subheader("Applications")
    apps = get_applications(supabase)
    if not apps:
        st.info("No applications yet")
        return

    # display and manage
    for app in apps:
        st.markdown(f"### {app.get('candidate_name')} — {app.get('candidate_email')}")
        job = get_job_by_id(supabase, app.get("job_id")) if app.get("job_id") else None
        st.write("Job:", job["title"] if job else app.get("job_role"))
        st.write("Score:", app.get("match_score"))
        st.write("Phase:", app.get("phase"))
        # actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"Mark PASS [{app.get('id')}]", key=f"pass_{app.get('id')}"):
                # advance phase
                current = app.get("phase", "Not Selected")
                next_phase = current
                if current == "Round 1 Scheduled":
                    next_phase = "Round 2 Scheduled"
                elif current == "Round 2 Scheduled":
                    next_phase = "Final Round Scheduled"
                elif current == "Final Round Scheduled":
                    next_phase = "Offer Extended"
                update_application_phase(supabase, app.get("id"), {"phase": next_phase, "status": "Pending" if "Scheduled" in next_phase else "Passed"})
                st.success(f"Advanced to {next_phase}")
                st.experimental_rerun()
        with col2:
            if st.button(f"Mark FAIL [{app.get('id')}]", key=f"fail_{app.get('id')}"):
                update_application_phase(supabase, app.get("id"), {"phase": "Rejected", "status": "Failed"})
                st.error("Marked Rejected")
                st.experimental_rerun()

# ---------------------------
# Main
# ---------------------------
def main():
    if st.session_state["user"] is None:
        login_register_ui()
    else:
        role = st.session_state["user"]["role"]
        if role == "User":
            candidate_portal()
        elif role == "HR":
            hr_portal()
        else:
            st.error("Unknown role")

if __name__ == "__main__":
    main()
