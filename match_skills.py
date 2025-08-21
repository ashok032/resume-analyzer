import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sentence_transformers import SentenceTransformer, util

# ----------------------------
# Ensure required NLTK data
# ----------------------------
for pkg in ["punkt", "stopwords"]:
    try:
        if pkg == "punkt":
            nltk.data.find("tokenizers/punkt")
        else:
            nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

# ----------------------------
# Initialize model
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Extract keywords from resume
# ----------------------------
def extract_keywords(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # remove special chars/numbers
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return list(set(filtered_tokens))  # unique keywords

# ----------------------------
# Match skills semantically
# ----------------------------
def match_skills(resume_skills, job_skills):
    if not resume_skills or not job_skills:
        return [], job_skills, 0.0

    resume_embeddings = model.encode(resume_skills, convert_to_tensor=True)
    job_embeddings = model.encode(job_skills, convert_to_tensor=True)

    matched = []
    missing = []

    for i, job_skill in enumerate(job_skills):
        sims = util.cos_sim(job_embeddings[i], resume_embeddings)[0]
        best_score = sims.max().item()
        if best_score > 0.6:  # similarity threshold
            matched.append(job_skill)
        else:
            missing.append(job_skill)

    score = (len(matched) / len(job_skills)) * 100 if job_skills else 0
    return matched, missing, round(score, 2)

# ----------------------------
# Wrapper for app.py
# ----------------------------
def match_resume_to_job(resume_text, job_skills):
    resume_keywords = extract_keywords(resume_text)
    return match_skills(resume_keywords, job_skills)
