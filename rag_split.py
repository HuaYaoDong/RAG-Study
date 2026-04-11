import json
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")  # 从 .env 文件获取智谱AI API 密钥
DATABASE_PATH = os.getenv("DATABASE_PATH")  # 从 .env 文件获取数据库路径

def initalize_zhipu(api_key: str):
    """
    设置智谱AI的API密钥，以便后续使用智谱AI的API生成Embedding。
    :param api_key: 智谱AI的API密钥（字符串形式）。
    """

def process_math_markdown(md_file_path: str, output_jsonl_path: str):
    """读取 Markdown 文件，按照层级切分并导出为 JSONL，返回切片列表"""
    with open(md_file_path, 'r', encoding='utf-8') as f:
        markdown_document = f.read()

    headers_to_split_on = [
        ("#", "Chapter"),       
        ("##", "Section"),      
        ("###", "Topic"),       
        ("####", "Sub_Topic"), 
    ]
    
    # 初始化切分器 (伪代码，假设你已经 import 了对应模块)
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(markdown_document)
    
    chunk_size = 512
    chunk_overlap = 100
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，"] 
    )
    
    final_splits = text_splitter.split_documents(md_header_splits)

    processed_data = []
    for i, doc in enumerate(final_splits):
        chunk_data = {
            "chunk_id": f"chunk_{i:04d}",
            "metadata": doc.metadata,          
            "content": doc.page_content,       
            "word_count": len(doc.page_content)
        }
        processed_data.append(chunk_data)

    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"处理完成！共生成 {len(processed_data)} 个带有层级标签的知识块。")
    return final_splits

# ==========================================
# 2：构建并持久化向量数据库
# ==========================================
def build_vector_database(documents: list, persist_dir: str, model_name: str = "BAAI/bge-small-zh-v1.5"):
    """
    将文本片段转化为向量并存入 Chroma 数据库
    """
    print(f"正在加载 Embedding 模型 ({model_name})...")
    model_kwargs = {'device': 'cuda'} 
    encode_kwargs = {'normalize_embeddings': True}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    print("正在将知识转化为向量并存入数据库，请稍候...")
    vectordb = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"向量数据库构建成功，已保存至: {persist_dir}")
    
    return vectordb

# ==========================================
# 3：知识库检索测试
# ==========================================
def test_retrieval(vectordb, query: str, top_k: int = 3):
    """
    输入查询词，从向量库中检索相关片段并打印
    """
    print(f"\n正在检索问题: {query}")
    retrieved_docs = vectordb.similarity_search(query, k=top_k)

    if not retrieved_docs:
        print("未检索到相关内容。")
        return

    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- 召回结果 Top {i+1} ---")
        print(f"📌 来源标签: {doc.metadata}")
        # 只打印前 200 个字符
        print(f"📄 内容片段: {doc.page_content[:200]}...")


# ==========================================
# 主程序入口 
# ==========================================
if __name__ == "__main__":
    # 配置路径参数
    MD_FILE = "第二章第一节.md"
    JSONL_FILE = "math_chunks.jsonl"
    DB_DIR = "./math_vector_db"
    TEST_QUERY = "导数的几何意义是什么？切线方程怎么求？"

    # 1. 预处理数据
    print(">>> 阶段 1：文档切分")
    splits = process_math_markdown(MD_FILE, JSONL_FILE)

    # 2. 构建向量库
    print("\n>>> 阶段 2：向量化与入库")
    vector_store = build_vector_database(documents=splits, persist_dir=DB_DIR)

    # 3. 执行检索测试
    print("\n>>> 阶段 3：检索效果测试")
    test_retrieval(vectordb=vector_store, query=TEST_QUERY)