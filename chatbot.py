import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

# Set your OpenAI API key
OPENAI_API_KEY = ""

# Set up the main header for the app
st.header("Spanish Visas for Canadian Q&A")

# Add some text under the header
st.write("Find detailed information about the different visas available for Canadians looking to move to Spain, including eligibility, requirements, and more.")

# Optional sidebar for uploading PDF documents
# with st.sidebar:
#     st.title("Your Documents")
#     # Upload a PDF file for processing
#     file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")
    
# Upload pdf and extract text 
file_path = 'visas.pdf'
with open(file_path, 'rb') as file:
    pdf_reader = PdfReader(file)
    text = ""
    # Extract text from each page of the PDF
    for page in pdf_reader.pages:
        text += page.extract_text()

# Split the extracted text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",       # Split text at newlines
        chunk_size=1000,       # Size of each text chunk in characters
        chunk_overlap=150,     # Overlap between chunks to maintain context
        length_function=len    # Function to measure chunk length
    )
    chunks = text_splitter.split_text(text)

    # Generate embeddings for the text chunks using OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create a vector store using FAISS to efficiently search the text chunks
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get a user question from the input box
    user_question = st.text_input("Type your question here")

    # Perform similarity search and answer the user's question
    if user_question:
        # Search for the most relevant text chunks based on the user's question
        match = vector_store.similarity_search(user_question)
        
        # Define and configure the language model (LLM) using OpenAI's API
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,       # Lower temperature for more deterministic answers
            max_tokens = 1000,     # Limit the response length to about 750 words
            model_name = "gpt-3.5-turbo"
        )
        
        # Load the question-answering chain and generate a response
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        
        # Display the response to the user
        st.write(response)
