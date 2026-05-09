# 引入全局配置
from config import Config

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder


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

        print(f"正在加载 Rerank 模型: {Config.RERANK_MODEL_NAME}")
        self.reranker = CrossEncoder(
            Config.RERANK_MODEL_NAME,
            max_length=512,
            device=Config.DEVICE
        )

    def retrieve_with_rerank(self, query: str):
        """标准化的两阶段检索：向量召回 + 交叉重排"""
        print(f"\n[第一阶段] 向量召回中: {query}")
        candidate_results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=Config.RETRIEVER_TOP_K
        )

        if not candidate_results:
            print("未检索到相关内容。")
            return []

        candidate_docs = [doc for doc, score in candidate_results]

        print(f"召回 {len(candidate_docs)} 个候选文档，开始交叉重排...")

        # 构建输入对并打分
        sentence_pairs = [[query, doc.page_content] for doc in candidate_docs]
        scores = self.reranker.predict(sentence_pairs)

        # 排序并提取前 K 个
        doc_score_pairs = list(zip(candidate_docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        top_docs = []
        for i, (doc, score) in enumerate(doc_score_pairs[:Config.RERANK_TOP_K]):
            print(f"--- 精排 Top {i+1} (匹配得分: {score:.4f}) ---")
            print(f"    层级: {doc.metadata}")
            print(f"    内容: {doc.page_content[:80]}...")
            top_docs.append(doc)

        return top_docs


# 独立测试入口
if __name__ == "__main__":
    TEST_QUERY = "导数的几何意义是什么？切线方程怎么求？"

    print("初始化检索引擎...")
    retriever_engine = MathRetriever()

    print(f"\n执行检索: {TEST_QUERY}")
    results = retriever_engine.retrieve_with_rerank(TEST_QUERY)

    for i, doc in enumerate(results):
        print(f"\n[最终喂给大模型的文档片段 {i+1}]")
        print(f"层级路径: {doc.metadata}")
        print(f"内容摘录: {doc.page_content[:150]}...\n")
