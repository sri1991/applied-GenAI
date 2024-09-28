# PDF Q&A with Gemini AI

This project showcases an innovative PDF Question-Answering application that leverages Google's Gemini AI and Facebook AI Similarity Search (FAISS). The app allows users to upload PDF documents and ask questions about their content, receiving AI-generated answers based on the document's context.

## Features

- PDF text extraction
- Semantic search using FAISS
- AI-powered question answering with Gemini
- User-friendly interface with Streamlit

## Architecture

![Architecture Diagram](assets\architecture.png)


The application follows this high-level architecture:

1. **PDF Processing**: Extract text from uploaded PDF files using PyPDF.
2. **Text Chunking**: Split the extracted text into manageable chunks using NLTK's sentence tokenizer.
3. **Vector Representation**: Convert text chunks into numerical vectors using Scikit-learn's TfidfVectorizer.
4. **Similarity Search**: Create an efficient index of these vectors using FAISS for rapid similarity searches.
5. **AI-powered Q&A**: Generate human-like responses based on the relevant context using Google's Gemini AI model.
6. **Web Interface**: Provide an intuitive, interactive web application for users to upload PDFs and ask questions using Streamlit.

## How It Works

1. Users upload a PDF file through the Streamlit interface.
2. The app extracts and chunks the text from the PDF.
3. FAISS indexes the text chunks for efficient retrieval.
4. Users input questions about the PDF content.
5. The app finds the most relevant text chunks using FAISS.
6. Gemini AI generates an answer based on the question and relevant context.

## Setup and Installation

1. Clone this repository.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Gemini API key as an environment variable:
   - On Windows:
     ```bash
     set GEMINI_API_KEY=your_api_key_here
     ```
   - On macOS or Linux:
     ```bash
     export GEMINI_API_KEY=your_api_key_here
     ```
   Alternatively, for Streamlit Cloud deployment, add the API key to `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your_api_key_here"
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run apps/qa_app.py
   ```

Make sure to replace `your_api_key_here` with your actual Gemini API key.

## Future Enhancements

- Multi-document support
- Fine-tuning Gemini for specific domains
- Implementing user feedback for continuous improvement

## Contributing

We welcome contributions to this project! Please feel free to submit issues, fork the repository and send pull requests!

## License

[Specify your license here]

## Contact

[Your Name] - [Your Email]

Project Link: [https://github.com/yourusername/pdf-qa-gemini](https://github.com/yourusername/pdf-qa-gemini)
