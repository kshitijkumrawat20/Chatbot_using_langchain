from flask import Flask, request, jsonify, render_template
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from typing import Dict
from langchain_community.vectorstores.faiss import FAISS
# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Initialize the chat model
chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))
# if we want to use openai model
# chat = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

# Initialize chat history for chatbot memory
chat_history = ChatMessageHistory()

# Defining the prompt template
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Create the document chain
document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

# Defining the retriever
loader = WebBaseLoader("https://brainlox.com/courses/category/technical")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(data)
vectorstore = FAISS.from_documents(documents=all_splits, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
# vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(k=4)

# Define the query transformation
def parse_retriever_input(params: Dict):
    return params["messages"][-1].content

# Definig prompt template 

query_transform_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else."),
    ]
)

query_transforming_retriever_chain = RunnableBranch(
    (
        lambda x: len(x.get("messages", [])) == 1,
        (lambda x: x["messages"][-1].content) | retriever,
    ),
    query_transform_prompt | chat | StrOutputParser() | retriever,
).with_config(run_name="chat_retriever_chain")

# Creating the  retrieval chain
conversational_retrieval_chain = RunnablePassthrough.assign(
    context=query_transforming_retriever_chain,
).assign(
    answer=document_chain,
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    user_message = request.json.get('message')
    chat_history.add_user_message(user_message)
    
    response = conversational_retrieval_chain.invoke(
        {"messages": chat_history.messages},
    )
    
    chat_history.add_ai_message(response["answer"])
    
    return jsonify({"response": response["answer"]})

if __name__ == '__main__':
    app.run(debug=True)
