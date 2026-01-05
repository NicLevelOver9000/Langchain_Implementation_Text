from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

text = """
Java is an object-oriented programming language.
Spring Boot is used to build production-ready Java applications.
JVM allows Java programs to run on multiple platforms.
"""

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
documents = splitter.create_documents([text])

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a junior research assistant.
        Always answer ONLY using the context provided.
        You MUST answer ONLY using the context provided.
        If the answer is not in the context, then reply I don't know.""".strip()
    ),
    (
        "user",
        "Context:\n{context}"
    ),
    (
        "user",
        "Question:\n{question}"
    )
])

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

load_dotenv()

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

rag_chain = (
    RunnableMap({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | llm
)


print(rag_chain.invoke("What is a cat?"))
