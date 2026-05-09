"""
JsonRichText 解析工具
用于解析老师提供的 JsonRichText 格式数据，提取纯文本供 RAG 使用
"""
import json
from typing import List, Dict, Any
from langchain_core.documents import Document


def extract_plain_text_from_inline(inline_node: Dict) -> str:
    """
    从 inline 节点提取纯文本
    按照规范第七章的提取规则
    """
    node_type = inline_node.get("type", "")
    
    if node_type == "text":
        return inline_node.get("text", "")
    elif node_type == "equation":
        # 行内公式：提取 latex 原文
        return f"${inline_node.get('latex', '')}$"
    elif node_type == "link":
        # 链接：提取 label 作为 fallback
        return inline_node.get("label", "")
    else:
        return ""


def extract_plain_text_from_block(block: Dict) -> str:
    """
    从 block 节点提取纯文本
    """
    block_type = block.get("type", "")
    
    # 富文本块：提取 content 中的 inline 文本
    rich_text_types = [
        "paragraph", "heading", "bulleted_list_item", 
        "numbered_list_item", "blockquote", "callout"
    ]
    
    if block_type in rich_text_types:
        content = block.get("content", [])
        texts = [extract_plain_text_from_inline(inline) for inline in content]
        # 段落和标题之间加空格
        if block_type in ["paragraph", "heading"]:
            return " ".join(texts)
        # 列表项之间加换行
        return "\n".join(texts)
    
    # 非富文本块
    if block_type == "equation_block":
        # 独立公式：提取 latex 原文
        return f"\n$${block.get('latex', '')}$$\n"
    elif block_type == "code_block":
        # 代码块：提取 code 字段
        return f"\n{block.get('code', '')}\n"
    elif block_type == "image":
        # 图片：提取 alt 文字
        return block.get("alt", "")
    elif block_type == "divider":
        return "\n---\n"
    else:
        # ref_card 等：跳过
        return ""


def extract_plain_text(blocks: List[Dict]) -> str:
    """
    从 JsonRichText blocks 提取完整纯文本
    按照规范第七章的 extract_plain_text 约定
    """
    texts = []
    for block in blocks:
        text = extract_plain_text_from_block(block)
        if text:
            texts.append(text)
    
    return "\n".join(texts)


def extract_heading_from_blocks(blocks: List[Dict]) -> str:
    """提取第一个标题作为文档标题"""
    for block in blocks:
        if block.get("type") == "heading":
            content = block.get("content", [])
            texts = [extract_plain_text_from_inline(inline) for inline in content]
            return "".join(texts)
    return ""


def parse_knowledge_graph(json_file_path: str) -> List[Document]:
    """
    解析 knowledge_graph.json，提取概念文档
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    docs = []
    concepts = data.get("concepts", [])
    
    for concept in concepts:
        key = concept.get("key", "")
        name = concept.get("name", "")
        tags = concept.get("tags", [])
        weight = concept.get("weight", 0.5)
        description_blocks = concept.get("description", [])
        
        # 提取纯文本
        plain_text = extract_plain_text(description_blocks)
        heading = extract_heading_from_blocks(description_blocks)
        
        metadata = {
            "source_type": "concept",
            "key": key,
            "name": name,
            "tags": ",".join(tags) if tags else "",
            "weight": weight,
            "heading": heading
        }
        
        # 构建完整内容：标题 + 正文
        full_content = f"{name}\n{plain_text}"
        
        doc = Document(
            page_content=full_content,
            metadata=metadata
        )
        docs.append(doc)
    
    print(f"从知识图谱中提取了 {len(docs)} 个概念文档")
    return docs


def parse_problems(json_file_path: str) -> List[Document]:
    """
    解析 problems.json，提取题目文档
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    docs = []
    problems = data.get("problems", [])
    
    for problem in problems:
        key = problem.get("key", "")
        title = problem.get("title", "")
        tags = problem.get("tags", [])
        concepts = problem.get("concepts", [])
        
        # 提取题目内容
        content_blocks = problem.get("content", [])
        content_text = extract_plain_text(content_blocks)
        
        # 提取答案
        answer_blocks = problem.get("answer", [])
        answer_text = extract_plain_text(answer_blocks)
        
        # 提取解答
        solution_blocks = problem.get("solution", [])
        solution_text = extract_plain_text(solution_blocks)
        
        # 构建完整内容
        full_content = f"{title}\n\n题目：{content_text}\n\n答案：{answer_text}\n\n解答：{solution_text}"
        
        metadata = {
            "source_type": "problem",
            "key": key,
            "title": title,
            "tags": ",".join(tags) if tags else "",
            "concepts": ",".join(concepts) if concepts else ""
        }
        
        doc = Document(
            page_content=full_content,
            metadata=metadata
        )
        docs.append(doc)
    
    print(f"从题目库中提取了 {len(docs)} 个题目文档")
    return docs


def parse_course(json_file_path: str) -> List[Document]:
    """
    解析 course.json，提取课程内容文档
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    docs = []
    course = data.get("course", {})
    
    key = course.get("key", "")
    title = course.get("title", "")
    graph_key = course.get("graph_key", "")
    
    # 提取课程简介
    summary_blocks = course.get("summary", [])
    summary_text = extract_plain_text(summary_blocks)
    
    # 提取课程内容
    content_blocks = course.get("content", [])
    content_text = extract_plain_text(content_blocks)
    
    # 构建完整内容
    full_content = f"{title}\n\n课程简介：{summary_text}\n\n课程内容：{content_text}"
    
    metadata = {
        "source_type": "course",
        "key": key,
        "title": title,
        "graph_key": graph_key
    }
    
    doc = Document(
        page_content=full_content,
        metadata=metadata
    )
    docs.append(doc)
    
    print(f"从课程中提取了 {len(docs)} 个文档")
    return docs


def parse_all_sources(
    knowledge_graph_path: str = None,
    problems_path: str = None,
    course_path: str = None
) -> List[Document]:
    """
    解析所有数据源，返回合并的文档列表
    """
    all_docs = []
    
    if knowledge_graph_path:
        try:
            docs = parse_knowledge_graph(knowledge_graph_path)
            all_docs.extend(docs)
        except Exception as e:
            print(f"解析知识图谱失败: {e}")
    
    if problems_path:
        try:
            docs = parse_problems(problems_path)
            all_docs.extend(docs)
        except Exception as e:
            print(f"解析题目库失败: {e}")
    
    if course_path:
        try:
            docs = parse_course(course_path)
            all_docs.extend(docs)
        except Exception as e:
            print(f"解析课程失败: {e}")
    
    print(f"\n共提取了 {len(all_docs)} 个文档")
    return all_docs


if __name__ == "__main__":
    # 测试解析
    docs = parse_all_sources(
        knowledge_graph_path="data/knowledge_graph.json",
        problems_path="data/problems.json",
        course_path="data/course.json"
    )
    
    for i, doc in enumerate(docs[:3]):
        print(f"\n{'='*50}")
        print(f"文档 {i+1} [{doc.metadata.get('source_type')}]")
        print(f"标题: {doc.metadata.get('name') or doc.metadata.get('title', 'N/A')}")
        print(f"内容预览: {doc.page_content[:200]}...")
