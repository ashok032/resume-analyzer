# match_skills.py

from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')
stop_words = set(stopwords.words('english'))

def extract_keywords(text):
    """Extract cleaned keywords from resume text."""
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return list(set(tokens))

def match_resume_to_job(resume_skills, job_skills, threshold=0.6):
    """Compare resume skills to job skills using semantic similarity."""
    if not resume_skills or not job_skills:
        return [], job_skills, 0

    # Encode in batch (much faster)
    resume_emb = model.encode(resume_skills, convert_to_tensor=True)
    job_emb = model.encode(job_skills, convert_to_tensor=True)

    cosine_sim = util.cos_sim(job_emb, resume_emb)

    matched = []
    missing = []

    for i, job_skill in enumerate(job_skills):
        max_sim = float(cosine_sim[i].max())
        if max_sim >= threshold:
            matched.append(job_skill)
        else:
            missing.append(job_skill)

    score_percent = round(100 * len(matched) / len(job_skills)) if job_skills else 0
    return matched, missing, score_percent
