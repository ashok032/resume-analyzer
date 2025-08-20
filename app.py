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
from datetime import datetime
from supabase import create_client, Client
from match_skills import extract_keywords, match_resume_to_job # Your custom matching logic file

# -------------------- CONFIG AND INITIALIZATION --------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide", initial_sidebar_state="auto")

# Load NLP model using Streamlit's caching for efficiency
@st.cache_resource
def load_spacy_model():
    """Load the spaCy model and cache it to avoid reloading on every run."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model not found. Please ensure 'en_core_web_sm' is listed in your requirements.txt.")
        st.stop()

nlp = load_spacy_model()

# Initialize Supabase connection using secrets
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except KeyError:
    st.error("🚨 **Error**: Supabase credentials not found. Please add `SUPABASE_URL` and `SUPABASE_KEY` to your Streamlit secrets.")
    st.stop()

# Email credentials from secrets for sending notifications
SMTP_EMAIL = st.secrets.get("SMTP_EMAIL", "")
SMTP_PASSWORD = st.secrets.get("SMTP_PASSWORD", "")

# -------------------- DATABASE HELPER FUNCTIONS --------------------

@st.cache_data(ttl=600) # Cache job descriptions for 10 minutes
def fetch_job_descriptions():
    """Fetches all job roles and their associated skills from the Supabase 'job_descriptions' table."""
    try:
        response = supabase.table('job_descriptions').select('role, skills').execute()
        if response.data:
            # Create a dictionary mapping each role to its list of skills
            return {item['role']: item['skills'] for item in response.data}
        return {}
    except Exception as e:
        st.error(f"DB Error: Failed to fetch job descriptions. Details: {e}")
        return {}

# -------------------- GENERAL HELPER FUNCTIONS --------------------

def hash_password(password: str) -> str:
    """Hashes a password using SHA256 for secure storage."""
    return hashlib.sha256(password.encode()).hexdigest()

def extract_text(file_bytes: io.BytesIO, file_name: str) -> str:
    """Extracts text from PDF or DOCX files based on file extension."""
    if file_name.endswith(".pdf"):
        with pdfplumber.open(file_bytes) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif file_name.endswith(".docx"):
        doc = docx.Document(file_bytes)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""

def extract_name(text: str) -> str:
    """Extracts a person's name from text using spaCy's NER, with a fallback."""
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.title()
    # Fallback if NER fails
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines and len(lines[0].split()) in [2, 3]:
        return lines[0].title()
    return "Name Not Found"
    
def extract_email_from_text(text: str) -> str:
    """Extracts an email address from text using regex."""
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else "Email Not Found"

def send_email(to_email: str, subject: str, body: str):
    """Sends an email notification to a candidate."""
    if not all([SMTP_EMAIL, SMTP_PASSWORD]):
        st.error("Email credentials are not configured in secrets. Cannot send email.")
        return False
    if not to_email or to_email == "Email Not Found":
        st.error("❌ Candidate email is blank, cannot send notification.")
        return False

    msg = MIMEMultipart()
    msg["From"] = SMTP_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

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
    """Displays a logout button in the sidebar."""
    with st.sidebar:
        st.markdown(f"👤 Logged in as `{st.session_state['email']}`")
        if st.button("🚪 Logout", key="logout_btn", use_container_width=True):
            st.session_state.clear()
            st.rerun()

def login_register_ui():
    """Handles the user login and registration UI."""
    st.title("🔐 AI Resume Analyzer")
    st.write("Please log in or register to continue.")
    
    tabs = st.tabs(["🔑 Login", "🆕 Register"])

    with tabs[0]: # Login
        with st.form("login_form"):
            email = st.text_input("👤 Email")
            password = st.text_input("🔑 Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if not email or not password:
                    st.warning("Please enter both email and password.")
                    return
                
                hashed_pw = hash_password(password)
                try:
                    user_response = supabase.table('users').select('*').eq('email', email).eq('password_hash', hashed_pw).execute()
                    if user_response.data:
                        user = user_response.data[0]
                        st.session_state["email"] = user['email']
                        st.session_state["role"] = user['role']
                        st.session_state["user_id"] = user['id']
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Please try again.")
                except Exception as e:
                    st.error(f"Login failed: {e}")

    with tabs[1]: # Register
        with st.form("register_form"):
            new_email = st.text_input("👤 Email", key="reg_email")
            new_password = st.text_input("🔑 Password", type="password", key="reg_pass")
            role = st.selectbox("Select Role", ["User", "HR"], key="reg_role")
            submitted = st.form_submit_button("Register")
            if submitted:
                if not all([new_email, new_password, role]):
                    st.warning("Please fill all fields.")
                    return

                try:
                    existing_user = supabase.table('users').select('id').eq('email', new_email).execute()
                    if existing_user.data:
                        st.warning("Email already registered. Please login.")
                    else:
                        new_user = {"email": new_email, "password_hash": hash_password(new_password), "role": role}
                        supabase.table('users').insert(new_user).execute()
                        st.success("Registered successfully! Please proceed to the Login tab.")
                except Exception as e:
                    st.error(f"Registration failed: {e}")

# -------------------- USER (CANDIDATE) VIEW --------------------
def user_view():
    st.header("👤 Candidate Dashboard")

    # Display My Applications
    st.subheader("📜 My Applications")
    try:
        apps_response = supabase.table('applications').select('*').eq('user_id', st.session_state['user_id']).order('submission_date', desc=True).execute()
        if apps_response.data:
            df_apps = pd.DataFrame(apps_response.data)
            # Ensure required columns exist, even if resume_url was removed from DB
            if 'resume_url' in df_apps.columns:
                 df_apps = df_apps.drop(columns=['resume_url'])
            df_display = df_apps[['job_role', 'match_score', 'phase', 'submission_date']]
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.info("You have not submitted any applications yet.")
    except Exception as e:
        st.error(f"Could not load your applications: {e}")

    # Submit New Application
    st.subheader("🚀 Submit New Application")
    job_descriptions_dict = fetch_job_descriptions()
    if not job_descriptions_dict:
        st.warning("No job roles are currently available. Please check back later.")
        return

    job_role = st.selectbox("Select a Job Role", options=list(job_descriptions_dict.keys()))
    uploaded_file = st.file_uploader("Upload Your Resume (PDF or DOCX)", type=["pdf", "docx"])

    if st.button("Analyze and Submit Application", disabled=(not uploaded_file or not job_role)):
        with st.spinner("Analyzing resume and submitting application..."):
            file_bytes = io.BytesIO(uploaded_file.getvalue())
            text = extract_text(file_bytes, uploaded_file.name)
            
            resume_keywords = extract_keywords(text)
            jd_skills = job_descriptions_dict[job_role]
            matched, missing, score = match_resume_to_job(resume_keywords, jd_skills)

            st.markdown(f"**Analysis Complete for '{job_role}' role:**")
            st.metric(label="Match Score", value=f"{score:.2f}%")
            st.info(f"**✅ Matched Skills:** {', '.join(matched) if matched else 'None'}")
            st.warning(f"**❌ Missing Skills:** {', '.join(missing) if missing else 'None'}")

            phase = "Screening" if score >= 60 else "Rejected"

            # Insert application into database (WITHOUT resume URL)
            try:
                application_data = {
                    "user_id": st.session_state['user_id'],
                    "job_role": job_role,
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
        candidates_df['email'] = candidates_df['users'].apply(lambda x: x['email'] if isinstance(x, dict) else 'N/A')
        candidates_df = candidates_df.drop(columns=['users'])
    except Exception as e:
        st.error(f"Failed to load candidates: {e}")
        return

    # UI for HR dashboard
    st.subheader("All Candidate Applications")
    
    # Filters
    col1, col2 = st.columns(2)
    job_roles = ["All"] + sorted(list(candidates_df['job_role'].unique()))
    selected_role = col1.selectbox("Filter by Role", job_roles)
    
    phases = ["All"] + sorted(list(candidates_df['phase'].unique()))
    selected_phase = col2.selectbox("Filter by Phase", phases)

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
        expander_title = f"**{row['email']}** | Role: {row['job_role']} | Score: {row['match_score']:.1f}% | Phase: **{row['phase']}**"
        with st.expander(expander_title):
            # REMOVED the resume link
            st.markdown(f"**Submitted on:** {pd.to_datetime(row['submission_date']).strftime('%Y-%m-%d %H:%M')}")
            
            cols = st.columns((1, 1, 2))
            
            # Action buttons
            if cols[0].button("✅ Advance", key=f"adv_{app_id}", help="Advance candidate to the next hiring stage"):
                phase_map = {"Screening": "Round 1 Interview", "Round 1 Interview": "Round 2 Interview", "Round 2 Interview": "Offer"}
                next_phase = phase_map.get(row['phase'], row['phase'])
                if next_phase != row['phase']:
                    try:
                        supabase.table('applications').update({'phase': next_phase}).eq('id', app_id).execute()
                        st.success(f"Advanced {row['email']} to {next_phase}")
                        send_email(row['email'], f"Update on your application for {row['job_role']}", f"Hello,\n\nCongratulations! You have been advanced to the next stage: {next_phase}.\n\nWe will be in touch shortly with more details.\n\nBest regards,\nThe Hiring Team")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Update failed: {e}")

            if cols[1].button("❌ Reject", key=f"rej_{app_id}", help="Reject candidate application"):
                try:
                    supabase.table('applications').update({'phase': 'Rejected'}).eq('id', app_id).execute()
                    st.warning(f"Rejected {row['email']}")
                    send_email(row['email'], f"Update on your application for {row['job_role']}", f"Hello,\n\nThank you for your interest in the {row['job_role']} role. After careful consideration, we have decided not to move forward with your application at this time.\n\nWe wish you the best in your job search.\n\nBest regards,\nThe Hiring Team")
                    st.rerun()
                except Exception as e:
                    st.error(f"Update failed: {e}")

# -------------------- MAIN APP LOGIC --------------------
def main():
    """Main function to run the Streamlit app."""
    if "email" not in st.session_state:
        login_register_ui()
    else:
        logout_sidebar()
        if st.session_state.get("role") == "User":
            user_view()
        elif st.session_state.get("role") == "HR":
            hr_view()
        else:
            st.error("Unknown user role. Please log out and try again.")
            if st.button("Force Logout"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()
