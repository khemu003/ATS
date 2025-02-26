from dotenv import load_dotenv
import streamlit as st
import os
import io
import base64
from PIL import Image
import pdf2image
import PyPDF2
import google.generativeai as genai
from sklearn.feature_extraction.text import CountVectorizer
import textstat


# Load environment variables
load_dotenv()

# Configure Google Gemini API
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your environment variables.")
    st.stop()

genai.configure(api_key=API_KEY)

# Function to call Google Gemini API
def get_gemini_response(input_text, pdf_content, prompt):
    """Generate a response using Google Gemini API."""
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, pdf_content[0], prompt])
    return response.text

# Function to process PDF and extract first page image
def input_pdf_setup(uploaded_file):
    """Convert first page of uploaded PDF to an image and encode as base64."""
    if uploaded_file:
        uploaded_file.seek(0)  # Reset file pointer
        images = pdf2image.convert_from_bytes(uploaded_file.read()) 
        first_page = images[0]

        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        first_page.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        pdf_parts = [{
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_byte_arr).decode()
        }]
        return pdf_parts
    else:
        raise FileNotFoundError("No File Uploaded")

# Function to extract text from resume
def extract_resume_text(uploaded_file):
    """Extract text from the uploaded PDF resume."""
    if uploaded_file:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    return None

# Function to extract keywords
def extract_keywords(text, num_keywords=15):
    """Extract top keywords from the given text."""
    vectorizer = CountVectorizer(stop_words='english', max_features=num_keywords)
    X = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

# Function to calculate resume match score
def calculate_resume_score(resume_text, job_desc):
    """Calculate match percentage based on keyword overlap."""
    resume_keywords = set(extract_keywords(resume_text, 20))
    job_keywords = set(extract_keywords(job_desc, 20))

    matched_keywords = resume_keywords.intersection(job_keywords)
    score = (len(matched_keywords) / len(job_keywords)) * 100 if job_keywords else 0
    return round(score, 2), matched_keywords

# Function to check resume readability
def readability_score(text):
    """Evaluate readability of resume text."""
    flesch_score = textstat.flesch_reading_ease(text)
    grade_level = textstat.flesch_kincaid_grade(text)
    return flesch_score, grade_level

# Function to check resume format
def check_resume_format(resume_text):
    """Check if key sections exist in resume."""
    sections = ["education", "experience", "skills", "projects", "certifications"]
    missing_sections = [s for s in sections if s not in resume_text.lower()]
    return missing_sections

# Streamlit App
st.set_page_config(page_title="A5 ATS Resume Expert")
st.header("MY A5 PERSONAL ATS")

# Input Fields
input_text = st.text_area("Job Description:", key="input")
uploaded_file = st.file_uploader("Upload your resume (PDF)...", type=['pdf'])

if uploaded_file:
    st.success("PDF Uploaded Successfully.")
    resume_text = extract_resume_text(uploaded_file)  # Extract resume text

# Prompts for Gemini API
input_prompt1 = """
You are an experienced HR with tech expertise in Data Science, Full Stack, Web Development, Big Data Engineering, DevOps, or Data Analysis.
Your task is to review the provided resume against the job description for these roles.
Please evaluate the candidate's profile, highlighting strengths and weaknesses in relation to the specified job role.
"""

input_prompt3 = """
You are a skilled ATS scanner with expertise in Data Science, Full Stack, Web Development, Big Data Engineering, DevOps, and Data Analysis.
Your task is to evaluate the resume against the job description. Provide:
1. The percentage match.
2. Keywords missing.
3. Final evaluation.
"""

input_prompt4 = """
You are an experienced learning coach and technical expert. Create a 6-month personalized study plan for an individual aiming to excel in [Job Role], 
focusing on the skills, topics, and tools specified in the provided job description. Ensure the study plan includes:
- A list of topics and tools for each month.
- Suggested resources (books, online courses, documentation).
- Recommended practical exercises or projects.
- Periodic assessments or milestones.
- Tips for real-world applications.
"""

input_prompt5 = """
You are a career consultant specializing in LinkedIn profile optimization. 
Analyze the provided resume and job description, then suggest improvements for the LinkedIn headline, summary, skills, and endorsements.
"""

# Buttons for different functionalities
submit1 = st.button("Tell Me About the Resume")
submit3 = st.button("Percentage Match")
submit4 = st.button("Personalized Learning Path")
submit5 = st.button("Optimize LinkedIn Profile")

if submit1:
    if uploaded_file:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt1, pdf_content, input_text)
        st.subheader("Resume Review:")
        st.write(response)
    else:
        st.warning("Please upload a resume.")

elif submit3:
    if uploaded_file:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt3, pdf_content, input_text)
        st.subheader("ATS Match Analysis:")
        st.write(response)

        # Extra Analysis
        match_score, matched_keywords = calculate_resume_score(resume_text, input_text)
        st.write(f"**Match Score:** {match_score}%")
        st.write("**Matched Keywords:**", ", ".join(matched_keywords))

        missing_keywords = set(extract_keywords(input_text, 15)) - set(extract_keywords(resume_text, 15))
        st.write("**Missing Keywords in Resume:**", ", ".join(missing_keywords))
    else:
        st.warning("Please upload a resume.")

elif submit4:
    if uploaded_file:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt4, pdf_content, input_text)
        st.subheader("Personalized Learning Path:")
        st.write(response)
    else:
        st.warning("Please upload a resume.")

elif submit5:
    if uploaded_file:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt5, pdf_content, input_text)
        st.subheader("LinkedIn Optimization Suggestions:")
        st.write(response)
    else:
        st.warning("Please upload a resume.")

# Additional Resume Formatting & Readability Analysis
if uploaded_file:
    missing_sections = check_resume_format(resume_text)
    if missing_sections:
        st.warning(f"Your resume is missing these sections: {', '.join(missing_sections)}")
    else:
        st.success("Your resume has all necessary sections.")

    flesch, grade = readability_score(resume_text)
    st.subheader("Resume Readability:")
    st.write(f"**Flesch Reading Ease Score:** {flesch}")
    st.write(f"**Flesch-Kincaid Grade Level:** {grade}")
