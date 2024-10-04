import io
import streamlit as st
from typing import List
import torch
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import faiss

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('hkunlp/instructor-xl')

@st.cache_resource
def load_llm_model():
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xxl')
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-xxl')
    return tokenizer, model

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        # Streamlit's UploadedFile needs to be read as bytes
        pdf_bytes = pdf.read()
        # Create a BytesIO stream from the read bytes
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        # Extract text from each page in the PDF
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    chunks = []
    start = 0
    end = chunk_size
    while start < len(text):
        chunks.append(text[start:end])
        start = end - chunk_overlap
        end = start + chunk_size
    return chunks

def get_vectorstore(text_chunks: List[str], model: SentenceTransformer):
    embeddings = model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def get_relevant_chunk(query: str, index: faiss.IndexFlatL2, text_chunks: List[str], embeddings: torch.Tensor, model: SentenceTransformer):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=1)
    return text_chunks[I[0][0]], I[0][0]

def generate_response(context: str, query: str, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration) -> str:
    input_text = f"Context: {context}\nQuestion: {query}\nAnswer:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    
    outputs = model.generate(
        input_ids,
        max_length=150,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

class PDFChatBot:
    def __init__(self):
        self.embedding_model = load_embedding_model()
        self.tokenizer, self.generation_model = load_llm_model()
        self.index = None
        self.embeddings = None
        self.text_chunks = None

    def process_pdfs(self, pdf_docs):
        raw_text = get_pdf_text(pdf_docs)
        self.text_chunks = get_text_chunks(raw_text)
        self.index, self.embeddings = get_vectorstore(self.text_chunks, self.embedding_model)

    def ask_question(self, question: str) -> str:
        context, _ = get_relevant_chunk(question, self.index, self.text_chunks, self.embeddings, self.embedding_model)
        response = generate_response(context, question, self.tokenizer, self.generation_model)
        return response
