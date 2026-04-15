import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 导入其他模块的核心组件
from config import Config
from rag_split import process_math_markdown, build_and_save_retriever
from rag_retrieve import MathRetriever
from rag_agent import router_chain

# 加载环境变量
load_dotenv()

# ==========================================
# 1：知识库初始化 (切片与建库)
# ==========================================
def setup_database(md_file_path: str):
    """
    检查向量数据库是否已存在。如果不存在，则一次性完成 Markdown 解析、切分和持久化建库。
    """
    print("\n>>> 阶段 1：检查知识库状态 <<<")
    # 简单通过判断 docstore 目录是否存在来决定是否需要重新建库
    store_path = os.path.join(Config.DB_DIR, "docstore")
    
    if not os.path.exists(store_path) or not os.listdir(store_path):
        print(f"未检测到已存在的数据库，开始读取 [{md_file_path}] 并构建...")
        if not os.path.exists(md_file_path):
            raise FileNotFoundError(f"找不到指定的 Markdown 文件: {md_file_path}")
            
        docs = process_math_markdown(md_file_path)
        build_and_save_retriever(docs)
        print("知识库构建完毕！\n")
    else:
        print(f"检测到本地数据库已存在 ({Config.DB_DIR})，直接加载。\n")


# ==========================================
# 2：生成模型初始化
# ==========================================
def init_llm(model_name: str = "glm-4", temperature: float = 0.1):
    """
    初始化用于生成回答的大模型。
    """
    zhipu_api_key = os.getenv("ZHIPU_API_KEY")
    if not zhipu_api_key:
        raise ValueError("请在 .env 文件中设置 ZHIPU_API_KEY")

    llm = ChatOpenAI(
        model=model_name,
        api_key=zhipu_api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4/", 
        temperature=temperature, 
        max_tokens=2048
    )
    return llm


# ==========================================
# 3：智能路由与检索分发
# ==========================================
def smart_retrieve(query: str, retriever_engine: MathRetriever) -> list:
    """
    结合大模型路由意图，与本地的检索器进行交互，返回最相关的文档片段。
    """
    print(">>> 阶段 2：意图分析与检索 <<<")
    # 1. 触发 rag_agent 中的路由逻辑
    parsed_request = router_chain.invoke({"query": query})
    print(f"识别意图: [{parsed_request.intent}], 章节限定: [{parsed_request.chapter_filter}], 核心知识点: [{parsed_request.topic_keyword}]")

    vectorstore = retriever_engine.vectorstore
    
    # 2. 根据不同意图执行不同策略的检索
    if parsed_request.intent == "summarize":
        print("走章节总结通道...")
        docs = []
        
        # 尝试 1：严格的元数据过滤匹配
        if parsed_request.chapter_filter:
            try:
                docs = vectorstore.similarity_search(query, k=15, filter={"Chapter": parsed_request.chapter_filter})
            except Exception as e:
                pass # 忽略 Chroma 可能抛出的过滤语法错误
                
        # 尝试 2：降级容错！如果严格过滤找不到，直接放开限制，用原始 query 进行全局语义搜索
        if not docs:
            print(f"精确过滤 [{parsed_request.chapter_filter}] 未命中，降级为全局语义检索...")
            # 去掉 filter，扩大搜索范围
            docs = vectorstore.similarity_search(query, k=15)
            
        return docs

    elif parsed_request.intent == "find_examples":
        print("走例题查找通道...")
        search_target = parsed_request.topic_keyword if parsed_request.topic_keyword else query
        docs = vectorstore.similarity_search(search_target, k=3, filter={"Topic": "例题"})
        # 降级策略：如果没有带有例题标签的内容，走常规检索
        if not docs:
            print("未找到明确的例题标签，自动降级为全文检索。")
            return retriever_engine.retrieve_with_rerank(query)
        return docs

    else:
        print("走默认的高阶重排检索通道...")
        return retriever_engine.retrieve_with_rerank(query)


# ==========================================
# 4：最终生成链
# ==========================================
def generate_math_answer(query: str, retrieved_docs: list, llm) -> str:
    """接收用户问题和检索到的文档，生成最终解答。"""
    context_text = ""
    for i, doc in enumerate(retrieved_docs):
        source_info = doc.metadata if hasattr(doc, 'metadata') else "未知"
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        context_text += f"\n[参考片段 {i+1}] (标签: {source_info})\n{content}\n" + "-" * 30

    system_prompt = """你是一位严谨的高等数学教授。请基于下方提供的【参考资料】来回答用户的提问。

核心约束原则：
1. 严谨性：绝不编造数学定理或公式。如果参考资料中无法得出答案，请回复“根据现有知识库，无法准确解答”。
2. 完整性：包含前提条件、核心公式及推导步骤。
3. 格式规范：所有数学公式必须使用标准 LaTeX 格式（行内 $...$，独立块 $$...$$）。
4. 结构还原：如遇到特殊的占位符或图表标记，请保持逻辑结构不变。

【参考资料】:
{context}
"""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "用户提问：{query}\n请给出你的解答：")
    ])

    chain = prompt_template | llm | StrOutputParser()
    print("大模型推理中...")
    return chain.invoke({"context": context_text, "query": query})


# ==========================================
# 5：主程序入口 (Pipeline 组装)
# ==========================================
def save_to_markdown(query: str, answer: str, filepath: str = "高数RAG问答记录.md"):
    """
    将用户问题和模型解答以追加模式(a)写入 Markdown 文件
    """
    # 使用 "a" 模式（append），保证每次问答都是追加在文件末尾，不会覆盖之前的内容
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"### 提问：{query}\n\n")
        f.write(f"**解答：**\n\n{answer}\n\n")
        f.write("---\n\n")  # 添加一条分割线，保持笔记整洁
    print(f"本次问答已自动保存至 [{filepath}]")


if __name__ == "__main__":
    # 配置你要解析的高数 Markdown 文件路径
    MD_FILE_PATH = "高等数学.md" 
    # 设定输出笔记的文件名
    OUTPUT_MD_FILE = "我的高数学习笔记.md"

    try:
        # 步骤 A：一次性完成切分与数据库构建
        setup_database(MD_FILE_PATH)
        
        # 步骤 B：初始化检索引擎与生成模型
        retriever_engine = MathRetriever()
        my_llm = init_llm()
        
        # 步骤 C：开启多轮交互问答
        print("\n" + "="*50)
        print("高等数学 RAG 助教已就绪！(输入 'quit' 退出)")
        print("="*50)
        
        while True:
            user_query = input("\n请输入你的高数问题: ")
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("再见！祝你高数学习顺利！")
                break
            if not user_query.strip():
                continue

            # 执行意图路由与文档召回
            retrieved_docs = smart_retrieve(user_query, retriever_engine)

            if not retrieved_docs:
                print("未能在知识库中检索到相关解答，请尝试更换关键词。")
                continue

            # 执行生成
            final_answer = generate_math_answer(query=user_query, retrieved_docs=retrieved_docs, llm=my_llm)
            
            print("\n" + "="*20 + " 最终解答 " + "="*20)
            print(final_answer)
            print("="*50)
            
            # 新增：将生成的解答自动保存到 Markdown 文件中
            save_to_markdown(query=user_query, answer=final_answer, filepath=OUTPUT_MD_FILE)
            
    except Exception as e:
        print(f"\n系统运行出错: {e}")