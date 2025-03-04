import streamlit as st
import PyPDF2
import faiss
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from langchain.llms import Ollama

# Ensure NLTK is ready
nltk.download("punkt")
nltk.data.path.append("C:/Users/admin/AppData/Roaming/nltk_data")
from nltk.tokenize import sent_tokenize

st.title("📚 AI Knowledge Assistant")
st.write("Upload a document and ask questions!")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Mistral model from Ollama
llm = Ollama(model="mistral")

uploaded_file = st.file_uploader("📂 Upload your document", type=["pdf", "txt"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    # Extract text from PDF or TXT
    text = ""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"

    elif uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")

    if not text.strip():
        st.error("⚠ Error extracting text. Try another file.")
    else:
        st.text_area("📜 Extracted Text:", text[:5000], height=300)  # Show first 5000 chars

        # Split text into sentences
        sentences = sent_tokenize(text)

        # Convert text into embeddings
        embeddings = embedding_model.encode(sentences)
        embeddings = np.array(embeddings, dtype="float32")

        # Store embeddings in FAISS
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        st.success("✅ Text converted into embeddings and stored in FAISS!")

        # Ask a question
        query = st.text_input("💬 Ask a question based on the document:")
        if query:
            query_embedding = embedding_model.encode([query])
            _, I = index.search(query_embedding, k=5)  # Retrieve top 5 relevant sentences
            
            # Ensure no out-of-bounds errors
            relevant_sentences = [sentences[i] for i in I[0] if i < len(sentences)]
            context = "\n".join(relevant_sentences) if relevant_sentences else "No relevant context found."

            # Use Mistral 7B to generate an answer
            prompt = f"""
            You are an AI assistant. Use the following document context to answer the question.

            Context:
            {context}

            Question: {query}

            Provide a detailed but concise response.
            """
            response = llm.invoke(prompt)  # ✅ Correct method

            st.write("**📝 Answer:**", response)
