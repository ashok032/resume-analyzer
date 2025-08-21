# resume_parser.py
import pdfplumber
import docx
import re
import spacy

# load spaCy once
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file) -> str:
    text = ""
    # file can be a BytesIO or UploadedFile
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(file) -> str:
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_contact_info(text: str) -> tuple:
    """
    Returns (name, email, phone)
    Name: first PERSON entity found by spaCy (searches beginning chunk)
    """
    # email
    email = None
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if m:
        email = m.group(0)

    # phone
    phone = None
    m = re.search(r"\+?\d[\d\s\-\(\)]{7,}\d", text)
    if m:
        phone = m.group(0)

    # name using spaCy on the top of the doc
    name = None
    doc = nlp(text[:800])  # search only beginning to reduce noise
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    return name, email, phone
