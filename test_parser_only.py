import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from json_parser import parse_all_sources

# Parse documents
docs = parse_all_sources(
    knowledge_graph_path='data/knowledge_graph.json',
    problems_path='data/problems.json',
    course_path='data/course.json'
)

print(f"Total docs parsed: {len(docs)}\n")

# Search for series-related content
print("=" * 50)
print("Searching for 'series' or 'series' content...")
print("=" * 50)

for doc in docs:
    content = doc.page_content.lower()
    if 'series' in content or 'series' in str(doc.metadata).lower():
        print(f"\n[Source: {doc.metadata.get('source_type', 'unknown')}]")
        print(f"Key: {doc.metadata.get('key', 'N/A')}")
        print(f"Name: {doc.metadata.get('name', doc.metadata.get('title', 'N/A'))}")
        print(f"Preview: {doc.page_content[:300]}...")
        print("-" * 50)
        break
else:
    print("NOT FOUND in parsed documents!")
