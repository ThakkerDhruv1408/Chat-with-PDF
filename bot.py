import os, logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings , OllamaLLM
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
# from data import data_dict
import time


start_time = time.time()

llm = OllamaLLM(model='mistral' , temperature=0)
embeddings = OllamaEmbeddings(model='nomic-embed-text')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_NAME = "mistral"
VECTOR_STORE_NAME = "simple-rag"
DATA_FOLDER = "./DATA"

def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


def create_vector_db(chunks):
    """Create a vector database from document chunks."""


    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=VECTOR_STORE_NAME,
    )
    logging.info("Vector database created.")
    return vector_db


def create_retriever(vector_db):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are tasked with generating search queries that will help find relevant information ONLY within the provided document context. Generate two alternative versions of the user's question that:
1. Focus on key terms and concepts that are likely to appear in the document
2. Use different wordings while maintaining the original intent
3. Stay within the scope of the document's topic and domain

Original question: {question}

Generate two alternative questions that would help search within this specific document context.
Return only the two questions, separated by newlines, without any additional text."""
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain"""
    # RAG prompt
    template = '''You are an advanced PDF analysis expert, capable of reading, comprehending, and analyzing PDF documents with precision and accuracy. Your task is to extract relevant information solely from the provided PDF content and answer questions based on context, avoiding assumptions or external knowledge. Follow these guidelines for optimal performance:

                Context-Driven Responses:

                Base your answers strictly on the text, tables, charts, or metadata available in the PDF.
                If information is missing or unclear, respond professionally by stating: ‘The provided PDF does not contain sufficient information to answer this question.’
                Clarity and Conciseness:

                Provide well-structured and concise answers while including necessary details. Avoid overly verbose responses.
                Use bullet points or short paragraphs when summarizing complex information.
                Structured Analysis:

                When summarizing sections, maintain logical flow and highlight key points, such as headings, subtopics, or conclusions.
                If asked for comparisons, ensure key differences and similarities are clearly outlined.
                Professional Tone:

                Maintain a formal and respectful tone. Avoid casual phrases or ambiguous language.
                Your response should demonstrate authority and confidence, reflecting your expertise as a PDF analysis specialist.
                Citation of Context:

                Wherever possible, reference page numbers, headings, or document sections to enhance clarity.
                For tables or charts, briefly describe their key data or findings when relevant to the question.
                Error Handling:

                In case of ambiguous or incomplete queries, ask clarifying questions or provide general context from the PDF.
                Avoid making speculative statements or using external data sources.
                Your goal is to deliver answers that are insightful, reliable, and well-supported by the document, showcasing your analytical expertise and professionalism. Begin by acknowledging the specific PDF-related question, then provide a clear, context-backed response.

                Context: {context}
                Question: {question}
                '''

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created successfully.")
    return chain


def result(path):
    # Load and process the PDF document
    data = ingest_pdf(path)
    if data is None:
        return

    # Split the documents into chunks
    chunks = split_documents(data)

    # Create the vector database
    vector_db = create_vector_db(chunks)

    # Create the retriever
    retriever = create_retriever(vector_db)

    # Create the chain with preserved syntax
    chain = create_chain(retriever, llm)

    return chain
  

def main():
    chain = result("../resume.pdf")
    while True:
        Question = input("Ask Question: \n")
        answer = chain.invoke(input=Question)
        print(answer)

if __name__ == "_main_":
    main()

print("time taken for execution:", time.time() - start_time)