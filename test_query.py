import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from json_parser import parse_all_sources
from rag_split import build_and_save_vectorstore
from rag_retrieve import MathRetriever
from config import Config
import os

# Step 1: Parse documents
print("Step 1: Parsing documents...")
docs = parse_all_sources(
    knowledge_graph_path='data/knowledge_graph.json',
    problems_path='data/problems.json',
    course_path='data/course.json'
)
print(f"Total docs: {len(docs)}")

# Step 2: Rebuild vector store
print("\nStep 2: Rebuilding vector store...")
os.makedirs(Config.DB_DIR, exist_ok=True)
# Clear existing database
import shutil
if os.path.exists(Config.DB_DIR):
    for f in os.listdir(Config.DB_DIR):
        p = os.path.join(Config.DB_DIR, f)
        if os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p)

build_and_save_vectorstore(docs)

# Step 3: Test query
print("\nStep 3: Testing query...")
retriever = MathRetriever()
query = "交错级数怎么判断敛散性"
results = retriever.retrieve(query, k=3)

print(f"\nQuery: {query}")
print(f"Found {len(results)} results:")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Source: {doc.metadata.get('source_type', 'unknown')}")
    print(f"Name: {doc.metadata.get('name', doc.metadata.get('title', 'N/A'))}")
    print(f"Content preview: {doc.page_content[:200]}...")
