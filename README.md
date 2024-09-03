Methodology: **LLM, OpenAIEmbeddings, ChatOpenAI**

Tools Used: **Python, Streamlit App, OpenAI API** | Category: **LLM, Chatbot** | Year: **2024**

# Spanish Visas for Canadian Q&A Chatbot

### Summarized Approach
This Streamlit app enables users to ask questions about Spanish visas for Canadians by utilizing a pre-uploaded PDF document. The app extracts and processes text from the PDF using PyPDF2, splits the text into manageable chunks with langchain's RecursiveCharacterTextSplitter, and generates text embeddings using OpenAI's embeddings model. A vector store is created with FAISS to perform efficient similarity searches on the text chunks. When a user inputs a question, the app searches for relevant text chunks and uses OpenAI's ChatGPT model to generate a detailed response, which is then displayed to the user.

### Data

The information has been gathered from various sources, including the Spanish Consulate in Canada and immigration consulting blogs. It has been concisely summarized, formatted, and paraphrased for clarity.

### Results
https://chatbot-app-spanish-visas.streamlit.app/


![image](https://github.com/user-attachments/assets/071b7daa-d915-48a7-8854-6541569ec9b5)
