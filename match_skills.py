
# match_skills.py

from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

model = SentenceTransformer('all-MiniLM-L6-v2')
stop_words = set(stopwords.words('english'))

def extract_keywords(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return list(set(tokens))

def match_resume_to_job(resume_skills, job_skills):
    matched = []
    missing = []

    for job_skill in job_skills:
        job_vec = model.encode(job_skill, convert_to_tensor=True)
        found_match = False
        for resume_skill in resume_skills:
            res_vec = model.encode(resume_skill, convert_to_tensor=True)
            score = util.cos_sim(res_vec, job_vec).item()
            if score >= 0.6:
                matched.append(job_skill)
                found_match = True
                break
        if not found_match:
            missing.append(job_skill)

    score_percent = round(100 * len(matched) / len(job_skills)) if job_skills else 0
    return matched, missing, score_percent
