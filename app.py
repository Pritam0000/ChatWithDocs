from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    model_name = "all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # Use the Hugging Face API key from Streamlit's secrets management
    api_key = st.secrets["HUGGINGFACE_API_KEY"]
    
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large", 
        model_kwargs={"temperature": 0.5, "max_length": 512},
        huggingfacehub_api_token=api_key
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display chat history in reverse order (most recent on top)
    for i, message in enumerate(reversed(st.session_state.chat_history)):
        if i % 2 == 0:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False

    st.header("Chat with docs")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
        if pdf_docs:
            st.success(f"{len(pdf_docs)} document(s) uploaded successfully!")
        
        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF document before processing.")
            else:
                with st.spinner("Processing documents..."):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)
                    
                    st.session_state.processing_complete = True
                st.success("Document processing completed. You can now ask questions!")

    if st.session_state.processing_complete:
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)
    else:
        st.info("Please upload and process at least one PDF document before asking questions.")

if __name__ == '__main__':
    main()
