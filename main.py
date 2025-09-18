from dotenv import load_dotenv
import os
import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src import DocumentLoader, VectorStoreManager
from src.agent import run_agent

load_dotenv()
llm = ChatOpenAI(model="gpt-5", temperature=1, http_client=httpx.Client(verify=False))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", http_client=httpx.Client(verify=False))
# Temporary SSL bypass; fix certificates with 'conda install ca-certificates certifi' for production.

# Configuration flags (can be set in .env)
use_hierarchical = os.getenv("HIERARCHICAL_CHUNKING", "true").lower() == "true"
rebuild_flag = os.getenv("REBUILD_VDB", "false").lower() == "true"

print("Welcome to the Car Repair RAG Chatbot! Type 'quit' to exit.")
print(f"Config: HIERARCHICAL_CHUNKING={use_hierarchical} | REBUILD_VDB={rebuild_flag}")

manager = VectorStoreManager(embeddings)
if rebuild_flag:
    manager.rebuild()
vectorstore = manager.get_or_create()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

while True:
    user_input = input("\nWhat is your question: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    try:
        result = run_agent(user_input, retriever=retriever, llm=llm)
        print("\n=== ANSWER ===")
        print(result)
    except Exception as e:
        print(f"Error: {e}")