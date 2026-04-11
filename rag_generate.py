import os
from zhipuai import ZhipuAI
from rag_retrieve import query_similar_text
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter

load_dotenv()

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
DATABASE_PATH = os.getenv("DATABASE_PATH") 


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    client = ZhipuAI(api_key=ZHIPU_API_KEY)

    try:
        # ================================
        # 结构化拼接 chunks，注入元数据
        # ================================
        context_parts = []
        for i, item in enumerate(retrieved_chunks):

            metadata, chunk_content, score = item
            
            # 提取元数据（例如文件名），如果没有则默认显示编号
            # 此时 metadata 本身就是一个字典，可以直接用 .get()
            source = metadata.get('file_name', f'未命名文档_{i+1}') if metadata else f'未命名文档_{i+1}'
            
            # 使用 Markdown 风格包裹每个 chunk，明确边界
            formatted_chunk = f"### [参考资料 {i+1}] (来源: {source})\n{chunk_content}"
            context_parts.append(formatted_chunk)
            
        context = "\n\n".join(context_parts)
        
        prompt = f"""# 角色设定
你是一位极其严密、专业的高等数学教授。你的任务是基于提供的【参考知识】为用户解答高等数学相关问题。

# 核心纪律（必须严格遵守）
1. **绝对忠于知识库**：你所有的定义、定理、引理和推导过程必须**严格基于【参考知识】**。如果参考知识中不包含解答该问题所需的必要公式或定理，请务必如实回答：“根据知识库提供的参考资料，无法得出完整的解答。” 绝不允许凭空捏造公式或证明！
2. **严谨的 LaTeX 排版**：所有的数学变量、公式和推导必须使用 LaTeX 格式输出。
   - 行内公式使用单美元符号包裹（例如：设函数 $f(x) = x^2$）。
   - 独立居中的公式块使用双美元符号包裹（例如：$$\\int_a^b f(x) dx = F(b) - F(a)$$）。
3. **思维链推导（Chain of Thought）**：对于计算题或证明题，你必须提供按部就班的推导过程（Step-by-Step）。切勿跨越逻辑跳出最终答案，每一步推导必须有文字说明。
4. **精准溯源**：在引用核心公式或定理时，必须在对应步骤后标注其来源（如：[参考资料 1]）。

# 建议的输出结构
- **核心概念/定理**：简述解答该问题涉及的核心数学概念。
- **推导/证明过程**：分步骤展示详细计算或证明逻辑。
- **最终结论**：清晰给出最终的计算结果或证明完成的说明。

=======================
【参考知识】:
{context}

【用户问题】: 
{query}
=======================

请保持严谨的学术态度，开始你的解答："""
        
        response = client.chat.completions.create(
            model='glm-4',  
            messages=[
                {"role": "system", "content": "你是一个基于知识库问答的专家助手。"}, 
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"生成答案时发生错误: {e}")
        return "抱歉，生成答案时发生了错误，请稍后再试。"
    
def retrieve_and_generate(query: str, db_path: str, api_key: str, top_k: int = 3) -> str:
    """
    主流程：结合检索和生成功能，实现RAG（检索增强生成）。
    :param query: 用户的问题。
    :param db_path: 数据库文件路径。
    :param api_key: 智谱AI的API密钥。
    :param top_k: 从数据库中检索的前K个相关文本块。
    :return: 生成的回答。
    """
    # 1. 从数据库中检索与用户问题最相关的文本块
    results = query_similar_text(query=query, db_path=db_path, top_k=top_k)

    # 如果没有找到相关上下文，直接返回提示
    if not results:
        return "抱歉，没有找到相关的上下文。"
    
    # 2. 将检索到的文本块作为上下文，调用生成函数生成答案
    answer = generate_answer(query=query, retrieved_chunks=results)

    return answer  # 返回生成的回答

if __name__ == "__main__":
    # 示例用户查询
    user_query = "导数的定义是什么"  # 用户输入的查询问题

    # 调用主流程函数
    final_answer = retrieve_and_generate(
        query=user_query,  
        db_path=DATABASE_PATH,  
        api_key=ZHIPU_API_KEY,  
        top_k=3  
    )

    # ================================
    # 将结果保存为 Markdown 文件
    # ================================
    output_filename = "解答输出.md"  
    
    # 使用 'w' 模式（写入覆盖）打开文件，编码为 utf-8 防止中文乱码
    with open(output_filename, "w", encoding="utf-8") as f:
        # 先写入一级标题，记录用户的问题
        f.write(f"# 问题：{user_query}\n\n")
        # 写入大模型生成的最终 Markdown 格式回答
        f.write(final_answer)
        
    print(f"\n解答已成功导出至当前目录下的 '{output_filename}' 文件中。")