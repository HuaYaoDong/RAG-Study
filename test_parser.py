"""测试 JsonRichText 解析器"""
from json_parser import parse_all_sources

print("测试 JsonRichText 解析器...")
print("="*50)

docs = parse_all_sources(
    knowledge_graph_path='data/knowledge_graph.json',
    problems_path='data/problems.json',
    course_path='data/course.json'
)

print(f'\n总计: {len(docs)} 个文档')
print("="*50)

for i, doc in enumerate(docs[:5]):
    print(f'\n--- 文档 {i+1} [{doc.metadata.get("source_type")}] ---')
    print(f'标题: {doc.metadata.get("name") or doc.metadata.get("title", "N/A")}')
    print(f'预览: {doc.page_content[:150]}...')
