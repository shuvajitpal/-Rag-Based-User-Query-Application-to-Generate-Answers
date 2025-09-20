from langchain_community.embeddings import OllamaEmbeddings
import gradio as gr 
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain.output_parsers.rail_parser import GuardrailsOutputParser
from langchain.text_splitter import CharacterTextSplitter

def process_input(question):
    modal_local = ChatOllama(model="llama2")
    
    DATA_PATH="D:\\Shuvajit\\project.txt"
    loader= TextLoader(DATA_PATH, encoding='utf-8')

    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma2",
        embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text')
    )
    retriever = vectorstore.as_retriever()

    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain =(
    {"context": retriever, "question": RunnablePassthrough()}
    |after_rag_prompt   
    |modal_local
    |StrOutputParser())

    return after_rag_chain.invoke(question)

iface = gr.Interface(fn=process_input,
                     inputs=[ gr.Textbox(label="Question")],
                     outputs="text",
                     title="ODISHA LEAVE PROVISIONS",
                     description="Enter a question to query the documents.")
iface.launch()