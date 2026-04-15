import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.storage import LocalFileStore # 保存到本地
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from langchain_classic.storage import EncoderBackedStore
import pickle

# 引入全局配置
from config import Config

class MathRetriever:
    def __init__(self):
        print(f"正在初始化检索器，加载 Embedding 模型: {Config.EMBED_MODEL_NAME}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBED_MODEL_NAME,
            model_kwargs={'device': Config.DEVICE}, 
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vectorstore = Chroma(
            collection_name="math_parent_child",
            embedding_function=self.embeddings,
            persist_directory=Config.DB_DIR
        )
        
        store_path = os.path.join(Config.DB_DIR, "docstore")
        fs = LocalFileStore(store_path)
        
        # 必须使用与写入时完全相同的反序列化逻辑
        self.store = EncoderBackedStore(
            store=fs,
            key_encoder=lambda x: x,
            value_serializer=pickle.dumps,
            value_deserializer=pickle.loads
        )
        # ==================================
        
        print(f"正在加载 Rerank 模型: {Config.RERANK_MODEL_NAME}")
        self.reranker = CrossEncoder(Config.RERANK_MODEL_NAME, max_length=512, device=Config.DEVICE)
        
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        )

    def retrieve_with_rerank(self, query: str):
        """标准化的两阶段检索接口，供 Agent 随时调用"""
        print(f"\n[第一阶段] 向量召回中: {query}")
        self.retriever.search_kwargs = {"k": Config.RETRIEVER_TOP_K}
        candidate_docs = self.retriever.invoke(query)
        
        if not candidate_docs:
            print("未检索到相关内容。")
            return []

        print(f"召回 {len(candidate_docs)} 个候选父文档，开始交叉重排...")
        
        # 构建输入对并打分
        sentence_pairs = [[query, doc.page_content] for doc in candidate_docs]
        scores = self.reranker.predict(sentence_pairs)
        
        # 排序并提取前 K 个
        doc_score_pairs = list(zip(candidate_docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        top_docs = []
        for i, (doc, score) in enumerate(doc_score_pairs[:Config.RERANK_TOP_K]):
            print(f"--- 精排 Top {i+1} (匹配得分: {score:.4f}) ---")
            top_docs.append(doc)
            
        return top_docs

# 独立测试入口
if __name__ == "__main__":
    TEST_QUERY = "导数的几何意义是什么？切线方程怎么求？"
    
    # 初始化检索引擎
    retriever_engine = MathRetriever()
    
    # 执行检索
    results = retriever_engine.retrieve_with_rerank(TEST_QUERY)
    
    for i, doc in enumerate(results):
        print(f"\n[最终喂给大模型的文档片段 {i+1}]")
        print(f"层级路径: {doc.metadata}")
        print(f"内容摘录: {doc.page_content[:150]}...\n")