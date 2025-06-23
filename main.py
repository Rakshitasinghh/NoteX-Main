import streamlit as st
import pdfplumber
import docx
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import re

# Load models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Custom CSS for cleaner look
st.markdown("""
    <style>
        h1, h2, h3 { font-family: 'Poppins', sans-serif; }
        body { background: linear-gradient(135deg, #6e7dff, #ff6ec7); background-size: 400% 400%; animation: gradientBackground 15s ease infinite; margin: 0; height: 100vh; }
        @keyframes gradientBackground { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        .css-1v0mbdj.etr89bj1 { background-color: #f4f9fc; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
        .block-container { padding: 1.5rem 2rem; border-radius: 10px; animation: slideUp 1s ease-out; }
        @keyframes slideUp { 0% { transform: translateY(50px); opacity: 0; } 100% { transform: translateY(0); opacity: 1; } }
        h1:hover, h2:hover, h3:hover { color: #ff6ec7; transform: scale(1.1); transition: all 0.3s ease-in-out; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <h1 style='text-align: center; color:#4682B4;'>NoteX</h1>
    <h4 style='text-align: center; color:#87CEFA;'>Smart summaries, smarter learning.</h4>
    <hr style='border:1px solid #ddd;'/>
""", unsafe_allow_html=True)

# --- Extraction Functions ---
def extract_text_from_pdf(uploaded_file):
    text = ''
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

def get_youtube_transcript(url):
    try:
        video_id = url.split("v=")[-1] if "youtube.com" in url else url.split("/")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([i['text'] for i in transcript])
    except Exception as e:
        return f"Error retrieving transcript: {str(e)}"

def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all('p')
        return "\n".join([p.get_text() for p in paragraphs if p.get_text().strip()])
    except Exception as e:
        return f"Error: {str(e)}"

# --- Summarization ---
def summarize_by_sections(text):
    sections = re.split(r'\n(?=\d+\.\s|Chapter\s+\d+|Section\s+\d+)', text)
    summaries = {}

    for i, sec in enumerate(sections):
        sec = sec.strip()
        word_count = len(sec.split())
        if word_count > 20:
            short_text = sec[:1000]
            max_len = min(150, max(13, int(word_count * 0.5)))
            min_len = max(10, int(max_len * 0.6))
            try:
                adjusted_max = min(max_len, len(short_text.split()))
                adjusted_min = min(min_len, adjusted_max - 1) if adjusted_max > 1 else 1
                summary = summarizer(short_text, max_length=adjusted_max, min_length=adjusted_min, do_sample=False)
                summaries[f"Section {i+1}"] = summary[0]['summary_text']
            except Exception as e:
                summaries[f"Section {i+1}"] = f"Error: {str(e)}"
        else:
            summaries[f"Section {i+1}"] = "Section too short to summarize."
    return summaries

def answer_question(context, question):
    try:
        result = qa_pipeline(question=question, context=context)
        return result['answer']
    except Exception as e:
        return f"Error: {str(e)}"

# Tabs for Inputs 
tab1, tab2, tab3, tab4 = st.tabs(["PDF", "DOCX", "YouTube", "Website"])

text = ""

with tab1:
    uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])
    if uploaded_pdf is not None:
        text = extract_text_from_pdf(uploaded_pdf)
        st.success("✅ PDF content extracted!")

with tab2:
    uploaded_docx = st.file_uploader("Upload Word Document", type=["docx"])
    if uploaded_docx is not None:
        text = extract_text_from_docx(uploaded_docx)
        st.success("✅ DOCX content extracted!")

with tab3:
    youtube_link = st.text_input("Paste the YouTube link:")
    if youtube_link:
        text = get_youtube_transcript(youtube_link)
        st.success("✅ YouTube transcript extracted!")

with tab4:
    website_url = st.text_input("Paste the website article link:")
    if website_url:
        text = extract_text_from_url(website_url)
        st.success("✅ Website article extracted!")

# View Extracted Text 
if text:
    with st.expander("View Extracted Text"):
        st.write(text[:1500] + "..." if len(text) > 1500 else text)

#  Summarization Section 
if st.button("🪄 Summarize"):
    if text:
        with st.spinner("Summarizing..."):
            section_summaries = summarize_by_sections(text)
            st.subheader("Section-wise Summary")
            for title, summary in section_summaries.items():
                st.markdown(f"#### 🔹 {title}")
                st.info(summary)
            st.session_state['summary'] = " ".join(section_summaries.values())
    else:
        st.error("Please upload or paste something first!")

# Q&A Section 
if 'summary' in st.session_state:
    st.subheader("Ask a Doubt")
    question = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        if question:
            answer = answer_question(st.session_state['summary'], question)
            st.success(f"Answer: {answer}")
        else:
            st.warning("Please type your question.")

# --- Footer ---
st.markdown("""
    <hr>
    <p style='text-align: center; font-size: 11px;'>Made by Rakshita| © 2025 EduSummarizer</p>
""", unsafe_allow_html=True)
