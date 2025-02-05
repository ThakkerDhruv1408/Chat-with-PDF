from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Initialize directories
pdf_path = "" #<Path to your PDF document>
store_directory = "./store"

# Initialize the prompt template
template = """You are an advanced PDF analysis expert, capable of reading, comprehending, and analyzing PDF documents with precision and accuracy. Your task is to extract relevant information solely from the provided PDF content and answer questions based on context, avoiding assumptions or external knowledge. Follow these guidelines for optimal performance:

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
History: {history}

User: {question}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

# Initialize memory
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True,
    input_key="question"
)

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(base_url='http://localhost:11434', model="nomic-embed-text")
vectorstore = Chroma(persist_directory=store_directory, embedding_function=embeddings)

# Initialize LLM
llm = Ollama(
    base_url="http://localhost:11434",
    model="mistral",
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

def process_pdf():
    """Process the PDF and create the vector store"""
    print("Processing PDF document...")
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    all_splits = text_splitter.split_documents(data)

    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=store_directory
    )
    vectorstore.persist()
    return vectorstore

def main():
    # Process PDF and initialize retriever
    vectorstore = process_pdf()
    retriever = vectorstore.as_retriever()

    # Initialize the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": memory,
        }
    )

    print("\nPDF Chatbot initialized. Type 'quit' to exit.")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        if user_input:
            response = qa_chain(user_input)
            print("\nChatbot:", response['result'])

if __name__ == "__main__":
    main()