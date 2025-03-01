from dotenv import load_dotenv
import streamlit as st
import os
import io
import base64
from PIL import Image
import pdf2image
import google.generativeai as genai
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Load environment variables
load_dotenv()

# Configure Google Gemini API
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your environment variables.")
    st.stop()

genai.configure(api_key=API_KEY)

def get_gemini_response(prompt):
    """Generate a response using Google Gemini API."""
    if not prompt.strip():
        return "Error: Prompt is empty. Please provide a valid prompt."
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([prompt, f"Add unique variations each time this prompt is called: {os.urandom(8).hex()}"])
        if hasattr(response, 'text') and response.text:
            return response.text
        else:
            return "Error: No valid response received from Gemini API."
    except Exception as e:
        st.error(f"API call failed: {str(e)}")
        return f"Error: {str(e)}"

st.set_page_config(page_title="A5 ATS Resume Expert", layout='wide')

# Header with a fresh style
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>MY A5 PERSONAL ATS</h1>
    <hr style='border: 1px solid #4CAF50;'>
""", unsafe_allow_html=True)

# Input section with better layout
col1, col2 = st.columns(2)

with col1:
    input_text = st.text_area("📋 Job Description:", key="input", height=150)

uploaded_file = None
resume_text = ""
with col2:
    uploaded_file = st.file_uploader("📄 Upload your resume (PDF)...", type=['pdf'])
    if uploaded_file:
        st.success("✅ PDF Uploaded Successfully.")
        try:
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                if page and page.extract_text():
                    resume_text += page.extract_text()
        except Exception as e:
            st.error(f"❌ Failed to read PDF: {str(e)}")

# Always visible buttons styled
st.markdown("---")
st.markdown("<h3 style='text-align: center;'>🛠️ Quick Actions</h3>", unsafe_allow_html=True)

action_cols = st.columns(6)
with action_cols[0]:
    if st.button("📖 Tell Me About the Resume"):
        if resume_text:
            response = get_gemini_response(f"Please review the following resume and provide a detailed evaluation: {resume_text}")
            st.write(response)
            st.download_button("💾 Download Resume Evaluation", response, "resume_evaluation.txt")
        else:
            st.warning("⚠️ Please upload a valid resume first.")

with action_cols[1]:
    if st.button("📊 Percentage Match"):
        if resume_text and input_text:
            response = get_gemini_response(f"Evaluate the following resume against this job description and provide a percentage match:\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text}")
            st.write(response)
            st.download_button("💾 Download Percentage Match", response, "percentage_match.txt")
        else:
            st.warning("⚠️ Please upload a resume and provide a job description.")

with action_cols[2]:
    learning_path_duration = st.selectbox("📆 Select Personalized Learning Path Duration:", ["3 Months", "6 Months", "9 Months", "12 Months"])
    if st.button("🎓 Personalized Learning Path"):
        if resume_text and input_text and learning_path_duration:
            response = get_gemini_response(f"Create a detailed and structured personalized learning path for a duration of {learning_path_duration} based on the resume and job description:\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text}")
            if response and "Error" not in response:
                st.write(response)
                pdf_buffer = io.BytesIO()
                doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                styles.add(ParagraphStyle(name='Custom', spaceAfter=12))
                story = [Paragraph(f"Personalized Learning Path ({learning_path_duration})", styles['Title']), Spacer(1, 12)]
                for line in response.split('\n'):
                    story.append(Paragraph(line, styles['Custom']))
                    story.append(Spacer(1, 12))
                doc.build(story)
                st.download_button(f"💾 Download Learning Path PDF", pdf_buffer.getvalue(), f"learning_path_{learning_path_duration.replace(' ', '_').lower()}.pdf", "application/pdf")
            else:
                st.warning("⚠️ No content received for the learning path. Please try again.")
        else:
            st.warning("⚠️ Please upload a resume and provide a job description.")

with action_cols[3]:
    if st.button("📝 Generate Updated Resume"):
        if resume_text:
            response = get_gemini_response(f"Suggest improvements and generate an updated resume for this candidate:\n{resume_text}")
            st.write(response)
            st.download_button("💾 Download Updated Resume", response, "updated_resume.txt")
        else:
            st.warning("⚠️ Please upload a resume first.")

with action_cols[4]:
    if st.button("❓ Generate 30 Interview Questions and Answers"):
        if resume_text:
            response = get_gemini_response("Generate 30 technical interview questions and their detailed answers.")
            st.write(response)
            st.download_button("💾 Download Interview Questions", response, "interview_questions.txt")
        else:
            st.warning("⚠️ Please upload a resume first.")

with action_cols[5]:
    if st.button("🧠 Skill Gap Analysis"):
        if resume_text and input_text:
            response = get_gemini_response(f"Identify skill gaps between the resume and the job description:\n\nJob Description:\n{input_text}\n\nResume:\n{resume_text}")
            st.write(response)
            st.download_button("💾 Download Skill Gap Analysis", response, "skill_gap_analysis.txt")
        else:
            st.warning("⚠️ Please upload a resume and provide a job description.")

st.markdown("<hr style='border: 1px solid #4CAF50;'>", unsafe_allow_html=True)