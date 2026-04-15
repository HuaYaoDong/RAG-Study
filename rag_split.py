import os
import uuid
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.storage import LocalFileStore  # 关键修复：引入本地文件存储
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import EncoderBackedStore
import pickle

# 引入我们刚才写的全局配置
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

def build_and_save_retriever(parent_docs: list):
    print(f"正在加载 Embedding 模型 ({Config.EMBED_MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBED_MODEL_NAME,
        model_kwargs={'device': Config.DEVICE}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Chroma(
        collection_name="math_parent_child",
        embedding_function=embeddings,
        persist_directory=Config.DB_DIR
    )
    
    # ======== 【修改核心部分】 ========
    store_path = os.path.join(Config.DB_DIR, "docstore")
    os.makedirs(store_path, exist_ok=True)
    
    # 1. 建立底层字节存储
    fs = LocalFileStore(store_path)
    
    # 2. 包装一层 Encoder，告诉它如何把 Document 转成字节流 (序列化)
    store = EncoderBackedStore(
        store=fs,
        key_encoder=lambda x: x,
        value_serializer=pickle.dumps,     # 存入时：对象转字节
        value_deserializer=pickle.loads    # 读取时：字节转对象
    )
    # ==================================

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=50,
        separators=["\n\n", "。", "！", "？", "\n", "，", " "] 
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
    )

    print("正在进行子文档切分、向量化并建立父子映射关系...")
    retriever.add_documents(parent_docs, ids=[str(uuid.uuid4()) for _ in parent_docs])
    
    print(f"高级检索器构建成功，向量与文档数据已持久化至: {Config.DB_DIR}")

if __name__ == "__main__":
    # 这里可以后续改成遍历 DATA_DIR 里的所有 md 文件
    MD_FILE = "第二章第一节.md" 
    
    print(">>> 阶段 1：解析父文档")
    docs = process_math_markdown(MD_FILE)

    print("\n>>> 阶段 2：执行切分并持久化入库")
    build_and_save_retriever(docs)