# app.py

import streamlit as st
import pandas as pd
import hashlib
import pdfplumber
import docx
import re
import spacy
import smtplib
import io
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
from supabase import create_client, Client
from match_skills import extract_keywords, match_resume_to_job # Assuming this file exists

# -------------------- CONFIG AND INITIALIZATION --------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# Load NLP model
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")
nlp = load_spacy_model()

# Initialize Supabase connection
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error("Could not connect to Supabase. Please check your `secrets.toml` file.")
    st.stop()


# Email credentials from secrets
SMTP_EMAIL = st.secrets.get("SMTP_EMAIL", "")
SMTP_PASSWORD = st.secrets.get("SMTP_PASSWORD", "")

# -------------------- DATABASE HELPERS --------------------

@st.cache_data(ttl=600) # Cache for 10 minutes
def fetch_job_descriptions():
    """Fetches all job roles and skills from Supabase."""
    try:
        response = supabase.table('job_descriptions').select('role, skills').execute()
        if response.data:
            return {item['role']: item['skills'] for item in response.data}
        return {}
    except Exception as e:
        st.error(f"Failed to fetch job descriptions: {e}")
        return {}

# -------------------- GENERAL HELPERS --------------------

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Resume parsing functions (unchanged)
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
    # Fallback regex if spaCy fails
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if lines:
        # A simple heuristic: the first line with 2-3 words, likely a name
        first_line_words = lines[0].split()
        if 2 <= len(first_line_words) <= 3:
            return lines[0].title()
    return "Not Found"

def extract_email_from_text(text):
    m = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return m.group(0) if m else "Not Found"

# Email sending with ICS (unchanged logic, uses st.secrets)
def send_email_with_ics(to_email, subject, body, meeting_start=None, meeting_end=None):
    if not all([SMTP_EMAIL, SMTP_PASSWORD]):
        st.error("Email credentials are not configured in secrets.toml. Cannot send email.")
        return False
    if not to_email or to_email == "Not Found":
        st.error("❌ Candidate email is blank, cannot send invite.")
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

# -------------------- UI COMPONENTS --------------------

def logout_sidebar():
    with st.sidebar:
        st.markdown(f"👤 Logged in as `{st.session_state['email']}`")
        if st.button("🚪 Logout", key="logout_btn"):
            st.session_state.clear()
            st.rerun()

def login_register_ui():
    st.title("🔐 AI Resume Analyzer")
    tabs = st.tabs(["🔑 Login", "🆕 Register"])

    with tabs[0]: # Login
        email = st.text_input("👤 Email")
        password = st.text_input("🔑 Password", type="password")
        if st.button("Login"):
            hashed_pw = hash_password(password)
            try:
                user = supabase.table('users').select('*').eq('email', email).eq('password_hash', hashed_pw).execute()
                if user.data:
                    st.session_state["email"] = user.data[0]['email']
                    st.session_state["role"] = user.data[0]['role']
                    st.session_state["user_id"] = user.data[0]['id']
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            except Exception as e:
                st.error(f"Login failed: {e}")


    with tabs[1]: # Register
        new_email = st.text_input("👤 Email", key="reg_user")
        new_password = st.text_input("🔑 Password", type="password", key="reg_pass")
        role = st.selectbox("Select Role", ["User", "HR"], key="reg_role")
        if st.button("Register"):
            if not all([new_email, new_password, role]):
                st.warning("Please fill all fields.")
                return

            try:
                existing_user = supabase.table('users').select('id').eq('email', new_email).execute()
                if existing_user.data:
                    st.warning("Email already registered.")
                else:
                    new_user = {
                        "email": new_email,
                        "password_hash": hash_password(new_password),
                        "role": role
                    }
                    supabase.table('users').insert(new_user).execute()
                    st.success("Registered successfully! Please login.")
            except Exception as e:
                st.error(f"Registration failed: {e}")


# -------------------- USER VIEW --------------------
def user_view():
    st.header("👤 Candidate Dashboard")

    # Display My Applications
    st.subheader("📜 My Applications")
    try:
        my_apps_response = supabase.table('applications').select('*').eq('user_id', st.session_state['user_id']).order('submission_date', desc=True).execute()
        if my_apps_response.data:
            df_apps = pd.DataFrame(my_apps_response.data)
            df_display = df_apps[['job_role', 'match_score', 'phase', 'submission_date']]
            st.dataframe(df_display, use_container_width=True)
        else:
            st.info("You have not submitted any applications yet.")
    except Exception as e:
        st.error(f"Could not load your applications: {e}")


    # Submit New Application
    st.subheader("🚀 Submit New Application")
    job_descriptions_dict = fetch_job_descriptions()
    if not job_descriptions_dict:
        st.warning("No job descriptions found in the database. Cannot submit applications.")
        return

    job_role = st.selectbox("Select a Job Role", options=list(job_descriptions_dict.keys()))
    uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])

    if uploaded_file and job_role:
        if st.button("Analyze and Submit"):
            with st.spinner("Analyzing resume..."):
                file_bytes = io.BytesIO(uploaded_file.getvalue())
                text = extract_text_from_pdf(file_bytes) if uploaded_file.name.endswith(".pdf") else extract_text_from_docx(file_bytes)

                resume_keywords = extract_keywords(text)
                jd_skills = job_descriptions_dict[job_role]
                matched, missing, score = match_resume_to_job(resume_keywords, jd_skills)

                st.markdown(f"**Match Score:** `{score:.2f}%` for the role of **{job_role}**")
                st.info(f"**Matched Skills:** {', '.join(matched) if matched else 'None'}")
                st.warning(f"**Missing Skills:** {', '.join(missing) if missing else 'None'}")

                phase = "Round 1 Interview" if score >= 70 else "Rejected"

                # Upload resume to Supabase Storage
                try:
                    file_ext = uploaded_file.name.split('.')[-1]
                    file_path_in_bucket = f"{st.session_state['user_id']}_{int(datetime.now().timestamp())}.{file_ext}"
                    supabase.storage.from_('resumes').upload(file_path_in_bucket, uploaded_file.getvalue())
                    resume_url = supabase.storage.from_('resumes').get_public_url(file_path_in_bucket)
                except Exception as e:
                    st.error(f"Failed to upload resume file: {e}")
                    return

                # Insert application into database
                try:
                    application_data = {
                        "user_id": st.session_state['user_id'],
                        "job_role": job_role,
                        "resume_url": resume_url,
                        "match_score": score,
                        "phase": phase
                    }
                    supabase.table('applications').insert(application_data).execute()
                    st.success("Application submitted successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Failed to save application to database: {e}")


# -------------------- HR VIEW --------------------
def hr_view():
    st.header("🏢 HR Dashboard")
    try:
        # Fetch applications and join with users table to get email
        response = supabase.table('applications').select('*, users(email)').order('submission_date', desc=True).execute()
        if not response.data:
            st.warning("No candidate applications found.")
            return
        
        candidates_df = pd.DataFrame(response.data)
        # Extract email from the nested user data
        candidates_df['email'] = candidates_df['users'].apply(lambda x: x['email'] if isinstance(x, dict) else 'N/A')
        candidates_df = candidates_df.drop(columns=['users'])

    except Exception as e:
        st.error(f"Failed to load candidates: {e}")
        return

    # UI for HR dashboard
    st.subheader("All Candidate Applications")
    
    # Simple filters
    job_roles = ["All"] + sorted(list(candidates_df['job_role'].unique()))
    selected_role = st.selectbox("Filter by Role", job_roles)
    
    phases = ["All"] + sorted(list(candidates_df['phase'].unique()))
    selected_phase = st.selectbox("Filter by Phase", phases)

    filtered_df = candidates_df.copy()
    if selected_role != "All":
        filtered_df = filtered_df[filtered_df['job_role'] == selected_role]
    if selected_phase != "All":
        filtered_df = filtered_df[filtered_df['phase'] == selected_phase]

    if filtered_df.empty:
        st.info("No candidates match the current filters.")
        return

    # Display candidates
    for index, row in filtered_df.iterrows():
        app_id = row['id']
        with st.expander(f"**{row['email']}** | Role: {row['job_role']} | Score: {row['match_score']:.2f}% | Phase: {row['phase']}"):
            st.markdown(f"**Resume:** [View Resume]({row['resume_url']})")
            st.markdown(f"**Submitted on:** {pd.to_datetime(row['submission_date']).strftime('%Y-%m-%d %H:%M')}")
            
            # Action buttons
            cols = st.columns(3)
            if cols[0].button("✅ Advance to Next Round", key=f"adv_{app_id}"):
                # Logic to determine next phase
                next_phase = "Offer Extended" # Default final phase
                if row['phase'] == 'Round 1 Interview':
                    next_phase = 'Round 2 Interview'
                elif row['phase'] == 'Round 2 Interview':
                    next_phase = 'Final Interview'
                
                try:
                    supabase.table('applications').update({'phase': next_phase}).eq('id', app_id).execute()
                    st.success(f"Advanced {row['email']} to {next_phase}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Update failed: {e}")


            if cols[1].button("❌ Reject", key=f"rej_{app_id}"):
                try:
                    supabase.table('applications').update({'phase': 'Rejected'}).eq('id', app_id).execute()
                    st.warning(f"Rejected {row['email']}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Update failed: {e}")

            # Simplified scheduling - can be expanded
            with cols[2].form(key=f"schedule_{app_id}"):
                st.write("Schedule Interview:")
                meeting_link = st.text_input("Meeting Link", key=f"link_{app_id}")
                submitted = st.form_submit_button("Send Invite")
                if submitted and meeting_link:
                    subject = f"Interview for {row['job_role']}"
                    body = f"Dear Candidate,\n\nYour interview for the {row['job_role']} role has been scheduled.\n\nPlease join using this link: {meeting_link}"
                    send_email_with_ics(row['email'], subject, body)


# -------------------- MAIN APP LOGIC --------------------
def main():
    if "email" not in st.session_state:
        login_register_ui()
    else:
        logout_sidebar()
        if st.session_state.get("role") == "User":
            user_view()
        elif st.session_state.get("role") == "HR":
            hr_view()
        else:
            st.error("Unknown role. Please log out and try again.")
            if st.button("Force Logout"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()