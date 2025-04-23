import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

st.set_page_config(page_title="Webpage Q&A Assistant", layout="wide")

if 'urls' not in st.session_state:
    st.session_state.urls = []
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'url_contents' not in st.session_state:
    st.session_state.url_contents = {}

@st.cache_resource
def load_qa_model():
    return pipeline('question-answering', model='deepset/roberta-base-squad2')

qa_pipeline = load_qa_model()

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def fetch_html(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        response = requests.get(url, headers=headers, timeout=10)
        return response.text
    except Exception as e:
        st.error(f"Error fetching {url}: {str(e)}")
        return ""

def extract_main_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup(["script", "style"]):
        script.extract()
    paragraphs = soup.find_all('p')
    text = clean_text(" ".join(p.get_text() for p in paragraphs))
    if not text:
        text = clean_text(soup.get_text())
    return text

def split_text_into_chunks(text, max_words=100):
    words = text.split()
    chunks = [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
    return [chunk for chunk in chunks if len(chunk.strip()) > 0]

def get_top_k_chunks(question, vectorizer, chunks, k=5):
    if not chunks:
        return []
    tfidf_matrix = vectorizer.transform(chunks)
    q_vec = vectorizer.transform([question])
    similarities = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-k:][::-1]
    return [chunks[i] for i in top_indices]

def generate_answer(question, chunks):
    if not chunks:
        return "No content available to answer the question.", 0.0
    context = "\n".join(chunks)
    try:
        result = qa_pipeline(question=question, context=context)
        if result['score'] > 0.1:
            return result['answer'], result['score']
        else:
            return "I don't know based on the provided content.", result['score']
    except Exception as e:
        return f"Error generating answer: {str(e)}", 0.0

st.title("Webpage Q&A Assistant")

st.header("Step 1: Add URLs")
col1, col2 = st.columns([3, 1])

with col1:
    url_input = st.text_input("Enter webpage URL:", placeholder="https://example.com")

with col2:
    if st.button("Add URL"):
        if url_input and url_input not in st.session_state.urls:
            if not url_input.startswith(('http://', 'https://')):
                url_input = 'https://' + url_input
            st.session_state.urls.append(url_input)
            st.success(f"Added: {url_input}")
            
            html = fetch_html(url_input)
            if html:
                text = extract_main_text(html)
                chunks = split_text_into_chunks(text)
                st.session_state.url_contents[url_input] = chunks
                st.session_state.chunks.extend(chunks)
                
                if st.session_state.chunks:
                    st.session_state.vectorizer = TfidfVectorizer().fit(st.session_state.chunks)
        elif url_input in st.session_state.urls:
            st.warning("This URL has already been added.")
        else:
            st.warning("Please enter a valid URL.")

if st.session_state.chunks and st.session_state.vectorizer:
    st.header("Step 2: Ask Questions")
    question = st.text_input("Ask a question about the content:")
    
    if st.button("Get Answer") and question:
        with st.spinner("Finding answer..."):
            relevant_chunks = get_top_k_chunks(
                question, 
                st.session_state.vectorizer, 
                st.session_state.chunks, 
                k=5
            )
            answer, confidence = generate_answer(question, relevant_chunks)
            st.subheader("Answer:")
            st.write(answer)