# import dependencies
import chainlit as cl
import google.generativeai as genai 
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# provide API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# load the pdfs and extract the texts
def get_pdf_text(pdf):
    text=""
    # Use relative path
    pdf_reader = PdfReader(pdf)

    # loop through the pages and add their content to text
    for page in pdf_reader.pages:
        text+=page.extract_text()
    return text

# split the extracted text into chunks
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

# convert chunks into vector embeddings
def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    
    # store vector embeddings in the faiss index
    vector_store.save_local("faiss_index")

# initialise the conversational model chain with a prompt template
def get_conversational_chain():
    prompt_template = """
    You are acting like a sales person. Answer the question as detailed as possible from the provided context, 
    make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", 
    don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# handles user interactions
async def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # initialise the database and load the faiss index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # perform similarity search
    docs = new_db.similarity_search(user_question)

    # generate a response with conversational chain
    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    # return the model's response to the user query
    return response["output_text"]

# describes the actions of the application when run
@cl.on_chat_start
async def main():
    files = None
    while files is None:
        # Initiates the user interaction by requesting a PDF file upload.
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin",
            accept=["application/pdf"],
        ).send()
    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`..")
    await msg.send

    # processes the PDF
    raw_text = get_pdf_text()
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    
    # updates the user when processing is complete
    cl.user_session.set("texts", text_chunks)
    msg.content = f"`{file.name}` processed. You can now ask questions"
    await msg.update()

# Responds to user messages (questions) after the initial processing is complete
@cl.on_message
async def process_response(message: cl.Message):
    if message:

        # generates a response through the conversational chain
        response = await user_input(message.content)

        # sends this response to the user.
        await cl.Message(content=response).send()
    
    

