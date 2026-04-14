import os
from dotenv import load_dotenv

# 强制加载项目根目录下的 .env 文件
load_dotenv()

class Config:
    """全局配置中心"""

    # ==========================================
    # 1. 安全与鉴权 (绝不硬编码！)
    # ==========================================
    ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
    if not ZHIPU_API_KEY:
        raise ValueError("🚨 严重错误: 未找到 ZHIPU_API_KEY，请检查 .env 文件！")

    # ==========================================
    # 2. 路径配置
    # ==========================================
    # 向量数据库的持久化目录
    DB_DIR = os.getenv("DB_DIR", "./database")
    # 原始 Markdown 文档存放目录
    DATA_DIR = os.getenv("DATA_DIR", "./data")

    # ==========================================
    # 3. 模型参数配置
    # ==========================================
    # 大语言模型 (生成回答)
    LLM_MODEL_NAME = "glm-4"
    LLM_TEMPERATURE = 0.1  # 数学问题需要严谨，温度设低

    # 向量化模型 (本地运行)
    EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
    
    # 交叉重排模型 (本地运行)
    RERANK_MODEL_NAME = "BAAI/bge-reranker-large"

    # ==========================================
    # 4. 硬件与性能配置
    # ==========================================
    # 既然有 4060 加持，这两个 BGE 模型跑在 GPU 上速度会飞起，直接默认锁死 cuda
    DEVICE = "cuda"
    
    # 检索相关参数
    RETRIEVER_TOP_K = 10     # 第一阶段向量召回的数量
    RERANK_TOP_K = 2         # 第二阶段精排后喂给大模型的数量