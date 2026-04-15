from pydantic import BaseModel, Field
from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import os

load_dotenv()  

# 获取 API 密钥
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")  

# 1. 定义期望的数据结构
class RouteQuery(BaseModel):
    intent: str = Field(
        description="用户的真实意图，必须是以下三种之一：'summarize'(章节总结), 'find_examples'(找例题), 'qa'(概念问答或解题指导)"
    )
    chapter_filter: Optional[str] = Field(
        description="如果用户明确提到了某一章，提取章节名称（如'第一章'），否则为 null"
    )
    topic_keyword: Optional[str] = Field(
        description="提取用户查询的核心数学知识点（如'导数证明'、'洛必达法则'），否则为 null"
    )

# 2. 初始化大模型
llm = ChatOpenAI(
    model="glm-4", 
    api_key=ZHIPU_API_KEY, 
    base_url="https://open.bigmodel.cn/api/paas/v4/", 
    temperature=0
)

# 3. 引入 PydanticOutputParser 解析器
# 它会自动生成一段严厉的 prompt 逼迫大模型只输出 JSON
parser = PydanticOutputParser(pydantic_object=RouteQuery)

# 4. 构建强约束的 Prompt
system_prompt = """你是一个高等数学 RAG 系统的查询路由引擎。
你的唯一任务是分析用户的输入，提取出搜索意图和过滤条件。
绝不能回答用户的问题，也绝不能输出任何解释性或客套的文字！

{format_instructions}
"""

route_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{query}")
]).partial(format_instructions=parser.get_format_instructions())

# 5. 重新组装 Chain 
router_chain = route_prompt | llm | parser