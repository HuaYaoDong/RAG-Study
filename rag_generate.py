import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 必须最先设置！

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 导入其他模块的核心组件
from config import Config
from rag_split import process_math_markdown, process_math_json, process_teacher_json, build_and_save_vectorstore
from rag_retrieve import MathRetriever
from rag_agent import router_chain

# 加载环境变量
load_dotenv()

# ==========================================
# 1：知识库初始化 (切片与建库)
# ==========================================
def setup_database(source_file_path: str = None, use_teacher_data: bool = False):
    """
    检查向量数据库是否已存在。如果不存在，则一次性完成文档解析、切分和持久化建库。
    
    Args:
        source_file_path: 源文件路径（支持 .md 和 .json）
        use_teacher_data: 是否使用老师提供的 JsonRichText 格式数据
    """
    print("\n>>> 阶段 1：检查知识库状态 <<<")
    # 通过判断数据库文件是否存在来决定是否需要重新建库
    db_files = os.listdir(Config.DB_DIR) if os.path.exists(Config.DB_DIR) else []
    
    if not db_files:
        if use_teacher_data:
            # 使用老师提供的 JsonRichText 格式数据
            print("使用老师提供的 JsonRichText 数据源...")
            docs = process_teacher_json(
                knowledge_graph_path="data/knowledge_graph.json",
                problems_path="data/problems.json",
                course_path="data/course.json"
            )
        elif source_file_path:
            # 根据文件扩展名选择解析器
            if not os.path.exists(source_file_path):
                raise FileNotFoundError(f"找不到指定的源文件: {source_file_path}")
            
            file_ext = os.path.splitext(source_file_path)[1].lower()
            if file_ext == '.json':
                print(f"检测到 JSON 文件，开始解析 [{source_file_path}] ...")
                docs = process_math_json(source_file_path)
            elif file_ext == '.md':
                print(f"检测到 Markdown 文件，开始解析 [{source_file_path}] ...")
                docs = process_math_markdown(source_file_path)
            else:
                raise ValueError(f"不支持的文件类型: {file_ext}，仅支持 .md 和 .json")
        else:
            raise ValueError("必须指定 source_file_path 或设置 use_teacher_data=True")
        
        if not docs:
            raise ValueError("未解析到任何文档，请检查数据文件")
        
        build_and_save_vectorstore(docs)
        print("知识库构建完毕！\n")
    else:
        print(f"检测到本地数据库已存在 ({Config.DB_DIR})，直接加载。\n")


# ==========================================
# 2：生成模型初始化
# ==========================================
def init_llm(model_name: str = "qwen-plus", temperature: float = 0.1):
    """
    初始化用于生成回答的大模型（千问）。
    """
    qwen_api_key = os.getenv("QWEN_API_KEY")
    if not qwen_api_key:
        raise ValueError("请在 .env 文件中设置 QWEN_API_KEY")

    llm = ChatOpenAI(
        model=model_name,
        api_key=qwen_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/", 
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

    system_prompt = r"""你是“高数速答助手”。目标是：让学生最快拿到可用结果。

回答规则（必须遵守）：
1. 先给最终答案，再补最少必要说明；禁止寒暄和铺垫。
2. 用户问“定义/公式/定理”时，直接给标准表达式（优先 LaTeX），不展开背景。
3. 用户问“怎么做题”时，只给最短步骤链（最多 3 步）+ 最终结果。
4. 无关内容一律不输出；不重复题目。
5. 若题意不完整，先给最常见默认情形下的答案，并用一句话说明默认条件。
6. 若知识库不足，直接说“当前资料不足以确定答案”，不要编造。
7. 输出尽量短：默认不超过 6 行。

格式规则：
- 数学内容优先用 LaTeX。
- 优先使用“结论：...”开头。
- 除非用户要求，不给例题、不做延伸阅读。

示例风格：
用户：导数的定义
助手：
结论：函数 \(f(x)\) 在 \(x_0\) 处的导数定义为
\[
f'(x_0)=\lim_{h\to 0}\frac{f(x_0+h)-f(x_0)}{h}
\]
等价形式：
\[
f'(x_0)=\lim_{x\to x_0}\frac{f(x)-f(x_0)}{x-x_0}
\]


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
def print_and_save_answer(query: str, answer: str, filepath: str = "高数RAG问答记录.md"):
    """
    同时在命令行输出答案，并以追加模式写入 Markdown 文件。
    """
    # 1) 命令行输出
    print("\n" + "=" * 20 + " 最终解答 " + "=" * 20)
    print(answer)
    print("=" * 50)

    # 2) 追加保存到 Markdown，避免覆盖历史问答
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"### 提问：{query}\n\n")
        f.write(f"**解答：**\n\n{answer}\n\n")
        f.write("---\n\n")

    print(f"本次问答已自动保存至 [{filepath}]")


if __name__ == "__main__":
    # 配置数据源
    # 模式1: 使用老师提供的 JsonRichText 数据（推荐）
    USE_TEACHER_DATA = True
    
    # 模式2: 使用自定义文件（取消 USE_TEACHER_DATA 并设置 SOURCE_FILE_PATH）
    SOURCE_FILE_PATH = "data/示例_高等数学.md"

    # 设定输出笔记的文件名
    OUTPUT_MD_FILE = "我的高数学习笔记.md"

    try:
        # 步骤 A：一次性完成切分与数据库构建
        if USE_TEACHER_DATA:
            setup_database(use_teacher_data=True)
        else:
            setup_database(source_file_path=SOURCE_FILE_PATH)
        
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
            
            # 同时输出到命令行 + 保存到 Markdown
            print_and_save_answer(query=user_query, answer=final_answer, filepath=OUTPUT_MD_FILE)
            
    except Exception as e:
        print(f"\n系统运行出错: {e}")
