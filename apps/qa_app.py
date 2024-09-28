import streamlit as st
import os
from pathlib import Path
import pypdf
import nltk
import faiss
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai

# Download the punkt tokenizer for sentence splitting
nltk.download('punkt', quiet=True)

# Configure Gemini API
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))  # Use environment variable

def save_uploaded_file(uploaded_file, save_dir):
    # Create the save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Save the file
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = pypdf.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def chunk_text(text, chunk_size=5):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    
    for sentence in sentences:
        current_chunk.append(sentence)
        if len(current_chunk) >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def create_faiss_index(chunks):
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)
    
    # Convert sparse matrix to dense numpy array
    dense_vectors = chunk_vectors.toarray().astype('float32')
    
    # Create FAISS index
    dimension = dense_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(dense_vectors)
    
    return index, vectorizer

def search_similar_chunks(query, index, vectorizer, chunks, k=3):
    query_vector = vectorizer.transform([query]).toarray().astype('float32')
    _, indices = index.search(query_vector, k)
    return [chunks[i] for i in indices[0]]

def get_gemini_response(question, context):
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = model.generate_content(prompt)
    return response.text

def main():
    st.title("PDF Q&A with Gemini")

    # Check if the API key is set
    if not os.environ.get('GEMINI_API_KEY'):
        st.error("GEMINI_API_KEY is not set in the environment variables. Please set it before running the app.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Display file details
        st.write(f"Filename: {uploaded_file.name}")
        st.write(f"File size: {uploaded_file.size} bytes")

        # Save the file
        save_dir = "uploaded_pdfs"
        saved_path = save_uploaded_file(uploaded_file, save_dir)
        st.success(f"File saved successfully at: {saved_path}")

        # Read PDF content
        pdf_text = read_pdf(saved_path)
        st.write("PDF content extracted successfully.")

        # Chunk the text
        chunks = chunk_text(pdf_text)
        st.write(f"Text chunked into {len(chunks)} segments.")

        # Create FAISS index
        index, vectorizer = create_faiss_index(chunks)
        st.success(f"FAISS index created with {index.ntotal} vectors.")

        # User question input
        user_question = st.text_input("Ask a question about the PDF:")

        if user_question:
            # Search for similar chunks
            relevant_chunks = search_similar_chunks(user_question, index, vectorizer, chunks)
            context = " ".join(relevant_chunks)

            # Get response from Gemini
            answer = get_gemini_response(user_question, context)

            # Display the answer
            st.subheader("Answer:")
            st.write(answer)

            # Optionally, display the relevant chunks used for context
            with st.expander("View relevant context"):
                for i, chunk in enumerate(relevant_chunks, 1):
                    st.write(f"Chunk {i}:")
                    st.write(chunk)
                    st.write("---")

if __name__ == "__main__":
    main()