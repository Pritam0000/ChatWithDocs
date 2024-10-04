import streamlit as st
from pdf_chat import PDFChatBot

st.set_page_config(page_title="Chat with PDF", page_icon="📚")

@st.cache_resource
def get_pdf_chatbot():
    return PDFChatBot()

def main():
    st.title("Chat with PDF 📚")

    pdf_chatbot = get_pdf_chatbot()

    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")

    if uploaded_files:
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                pdf_chatbot.process_pdfs(uploaded_files)
            st.success("PDFs processed successfully!")

        st.subheader("Ask a question about the PDFs")
        user_question = st.text_input("Enter your question:")

        if user_question:
            with st.spinner("Generating answer..."):
                answer = pdf_chatbot.ask_question(user_question)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()
