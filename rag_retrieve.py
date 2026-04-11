from zhipuai import ZhipuAI  # 用于调用智谱AI生成Embedding
from dotenv import load_dotenv  # 用于加载 .env 文件中的环境变量
import os
from typing import List, Tuple
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()  

# 获取 API 密钥和数据库路径
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")  
DATABASE_PATH = os.getenv("DATABASE_PATH") 

def initialize_model(api_key: str):
    """
    初始化智谱AI客户端
    """

def generate_embeddings(text: str) -> list[float]:
    """
    调用智谱API生成文本的向量表示
    """
    try:
        client = ZhipuAI(api_key=ZHIPU_API_KEY)
        response = client.embeddings.create(
            model='embedding-3',  # 智谱的中文小模型，适合生成文本向量
            input=text
        )
        return response.data[0].embedding  # 返回生成的向量
    except Exception as e:
        print(f"生成向量时发生错误: {e}")
        return []  # 发生错误时返回空列表
    
def fetch_embeddings_chroma(collection_name: str) -> List[Tuple[str, str, List[float]]]:
    """
    从 Chroma 数据库中提取存储的文本块及其Embedding。
    :param collection_name: Chroma 中的集合名称 (相当于关系型数据库的表名)
    :return: 返回一个包含 (file_name, chunk, embedding) 的列表。
    """
    # 1. 从环境变量获取数据库路径
    db_path = os.getenv("CHROMA_DB_PATH")
    if not db_path:
        raise ValueError("环境变量 CHROMA_DB_PATH 未设置，请检查 .env 文件！")

    # 2. 初始化 Chroma 持久化客户端
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        # 3. 获取对应的集合
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"找不到集合 '{collection_name}': {e}")
        return []

    # 4. 获取集合中的所有数据
    # 注意：Chroma 的 get() 默认不返回 embeddings，必须在 include 中显式声明
    chroma_data = collection.get(
        include=["documents", "metadatas", "embeddings"]
    )
    
    results = []
    
    # 5. 组装结果
    # Chroma 返回的是一个字典，包含 ids, documents, metadatas, embeddings 等列表
    if chroma_data["ids"]:
        for i in range(len(chroma_data["ids"])):
            # 提取元数据中的 file_name (前提是你在存入 Chroma 时，将文件名存入了 metadata)
            metadata = chroma_data["metadatas"][i]
            file_name = metadata.get("file_name", "unknown_file") if metadata else "unknown_file"
            
            chunk = chroma_data["documents"][i]
            embedding = chroma_data["embeddings"][i]
            
            results.append((file_name, chunk, embedding))
            
    return results


def query_similar_text(query: str, db_path: str, top_k: int = 3):
    """
    根据查询文本，直接从 Chroma 数据库中检索最相似的文本块。
    """
    # 1. 统一使用本地 BGE 模型生成问题的 Embedding
    print("正在加载本地 Embedding 模型处理您的问题...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cuda'}, # 你的 4060 显卡在这里发力
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 将用户的问题翻译成 512 维度的向量
    query_embedding = embeddings_model.embed_query(query)

    # 2. 连接 Chroma 数据库
    client = chromadb.PersistentClient(path=db_path)
    
    # 3. 获取集合 (名字必须写死为 "langchain"，因为这是 Langchain 构建时的默认名)
    collection_name = "langchain" 
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"获取集合 '{collection_name}' 失败，请确认数据库是否已正确构建: {e}")
        return []

    # 4. 执行极速检索
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=top_k,                    
        include=["documents", "metadatas", "distances"] 
    )

    similarities = []
    
    # 5. 提取结果
    if results["ids"] and len(results["ids"][0]) > 0:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(documents)):
            # 提取元数据和文本
            metadata = metadatas[i]
            chunk = documents[i]
            
            # 由于使用的是 BAAI/bge-small-zh-v1.5 配合 LangChain 默认设置
            # 这里 Chroma 返回的通常是距离，我们转换一下（具体视模型而定，也可以直接返回距离）
            score = 1.0 - distances[i] if distances[i] <= 1.0 else distances[i]

            similarities.append((metadata, chunk, score))

    return similarities


if __name__ == "__main__":
    # 示例查询
    user_query = "导数的定义"  # 替换为你的查询
    # 删掉 ZHIPU_API_KEY 传参，让它只传 3 个参数
    top_k_results = query_similar_text(user_query, DATABASE_PATH, top_k=3)

    # 打印结果
    print("查询结果：")
    for file_name, chunk, similarity in top_k_results:
        print(f"文件名: {file_name}, 相似度: {similarity:.4f}, 文本块: {chunk}")