import os
import json
import uuid
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pickle

# 设置 HuggingFace 国内镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 引入全局配置
from config import Config


def process_math_markdown(md_file_path: str):
    """读取 Markdown 文件，按标题提取父文档"""
    with open(md_file_path, 'r', encoding='utf-8') as f:
        markdown_document = f.read()

    headers_to_split_on = [
        ("#", "Chapter"),
        ("##", "Section"),
        ("###", "Topic"),
        ("####", "Sub_Topic"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    parent_docs = markdown_splitter.split_text(markdown_document)

    print(f"解析完成！共生成 {len(parent_docs)} 个带有层级标签的父文档。")
    return parent_docs


def process_math_json(json_file_path: str):
    """读取 JSON 文件，构建 Document 列表（兼容旧格式）"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # JSON 应为数组格式，每个元素包含 content 和 metadata
    if not isinstance(data, list):
        raise ValueError("JSON 文件必须是数组格式，每个元素包含 content 和 metadata")
    
    parent_docs = []
    for item in data:
        if 'content' not in item:
            continue
        
        # 提取 metadata（Chapter, Section, Topic, Sub_Topic 等）
        metadata = {k: v for k, v in item.items() if k != 'content'}
        
        doc = Document(
            page_content=item['content'],
            metadata=metadata
        )
        parent_docs.append(doc)
    
    print(f"解析完成！共生成 {len(parent_docs)} 个文档。")
    return parent_docs


def process_teacher_json(
    knowledge_graph_path: str = None,
    problems_path: str = None,
    course_path: str = None
) -> list:
    """
    解析老师提供的 JsonRichText 格式数据
    """
    from json_parser import parse_all_sources
    
    docs = parse_all_sources(
        knowledge_graph_path=knowledge_graph_path,
        problems_path=problems_path,
        course_path=course_path
    )
    
    return docs


def build_and_save_vectorstore(parent_docs: list):
    """构建向量数据库并持久化到本地"""
    print(f"正在加载 Embedding 模型 ({Config.EMBED_MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBED_MODEL_NAME,
        model_kwargs={'device': Config.DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 为每个文档分配唯一ID
    doc_ids = [str(uuid.uuid4()) for _ in parent_docs]

    # 直接存入 Chroma（自带持久化）
    vectorstore = Chroma(
        collection_name="math_parent_child",
        embedding_function=embeddings,
        persist_directory=Config.DB_DIR
    )

    print("正在进行文档切分、向量化并建立索引...")
    texts = [doc.page_content for doc in parent_docs]
    metadatas = [doc.metadata for doc in parent_docs]

    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=doc_ids)
    # 注意：新版 ChromaDB 无需手动调用 persist()，数据会自动持久化
    return vectorstore


if __name__ == "__main__":
    MD_FILE = "data/示例_高等数学.md"

    print(">>> 阶段 1：解析父文档")
    docs = process_math_markdown(MD_FILE)

    print("\n>>> 阶段 2：执行切分并持久化入库")
    build_and_save_vectorstore(docs)
