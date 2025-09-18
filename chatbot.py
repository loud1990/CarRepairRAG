from dotenv import load_dotenv
import os
import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src import VectorStoreManager

def main():
    """
    Legacy CLI. Prefer running main.py for full features.
    This script now reads config flags and ensures the vector store is initialized
    with the current settings (including hierarchical chunking).
    """
    load_dotenv()
    use_hierarchical = os.getenv("HIERARCHICAL_CHUNKING", "true").lower() == "true"
    rebuild_flag = os.getenv("REBUILD_VDB", "false").lower() == "true"

    print("Legacy chatbot entrypoint. Use main.py for the interactive experience.")
    print(f"Config: HIERARCHICAL_CHUNKING={use_hierarchical} | REBUILD_VDB={rebuild_flag}")

    # Initialize vector store to honor flags (DocumentLoader reads env internally)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", http_client=httpx.Client(verify=False))
    manager = VectorStoreManager(embeddings)
    if rebuild_flag:
        manager.rebuild()
    else:
        manager.get_or_create()

    print("Vector store is ready. Exiting.")

if __name__ == "__main__":
    main()