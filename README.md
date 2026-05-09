# RAG-Study — 高等数学 RAG 问答系统

> 南京理工大学 RAG 科研训练项目
> 基于 LangChain + 阿里千问 (Qwen) + ChromaDB 构建的本地知识库问答系统

---

## 📁 项目结构

```text
RAG-Study/
├── config.py           # 全局配置（模型、路径、API密钥读取）
├── json_parser.py      # JsonRichText 解析工具（处理老师提供的结构化数据）
├── rag_split.py        # 文档解析与向量库构建（支持 Markdown 与 JSON 解析）
├── rag_retrieve.py     # 两阶段检索（向量召回 + BGE 交叉重排序）
├── rag_generate.py     # 主程序：意图路由 + 检索分发 + 大模型生成
├── rag_agent.py        # 查询意图分析（总结/例题/问答路由）
├── requirements.txt    # 项目依赖包列表
├── .env                # API 密钥配置（不要上传 GitHub！）
└── data/               # 数据存放目录（支持 .md 笔记或结构化 .json 数据）
    ├── knowledge_graph.json
    ├── problems.json
    └── course.json
```

---

### 第一步：准备环境与依赖

推荐使用 Conda 虚拟环境：
```powershell
conda activate rag
pip install -r requirements.txt
```

### 第二步：配置 API Key

在项目根目录创建或编辑 `.env` 文件，填入你的**千问 API Key**：
```env
QWEN_API_KEY="sk-你的真实API_KEY"
```

### 第三步：配置知识库数据源

打开 `rag_generate.py`，滚动到底部主函数区域，根据你的数据来源进行配置：

- **模式 1：使用JsonRichText数据**
  设置 `USE_TEACHER_DATA = True`，系统会自动解析 `data/` 目录下的 `knowledge_graph.json`, `problems.json`, `course.json`。
- **模式 2：使用Markdown笔记**
  设置 `USE_TEACHER_DATA = False`，并修改 `SOURCE_FILE_PATH` 指向你的 `.md` 或 `.json` 文件。

> ⚠️ 如果使用自定义 Markdown 文件，建议使用 `#`、`##`、`###` 等标准标题层级，切块效果最好！

### 第四步：运行主程序

在终端中执行：
```powershell
python rag_generate.py
```

系统会自动：
1. 检查本地知识库状态。若未建库，则读取指定的 Markdown 或 JSON 文件进行解析、切分，并构建 ChromaDB 向量数据库。
2. 加载 BGE Embedding 模型、BGE Reranker 模型和千问大模型。
3. 进入交互问答模式，直接在命令行输入你的高数问题即可！

---

## ⚠️ 注意事项

| 问题 | 解决方法 |
|------|----------|
| 没有 NVIDIA GPU | 默认已在 `config.py` 中将 `DEVICE` 设置为 `"cpu"`。 |
| 首次运行很慢 | 正常现象！首次运行需要下载 BGE 相关的 Embedding 和 Reranker 模型（约需几百 MB），并构建本地向量数据库，请耐心等待。 |
| 问答结果保存在哪 | 默认自动追加保存到当前目录下的 `我的高数学习笔记.md` 文件中。 |

---

## 🧠 系统架构

```text
       JSON 数据 / Markdown 笔记
                   ↓
 [文档解析] json_parser.py / rag_split.py → 按标题或结构提取文本
                   ↓
 [构建题库] HuggingFace BGE Embeddings → ChromaDB 向量数据库
                   ↓
 [意图路由] rag_agent.py (Router) → 识别问题意图（总结/找例题/常规问答）
                   ↓
 [知识检索] rag_retrieve.py → 向量相似度初筛 (Top-K) → BGE Reranker 精排
                   ↓
 [结果生成] rag_generate.py → 组装 Prompt (包含检索内容) → 千问 Qwen-Plus 生成解答
                   ↓
            终端输出 & 保存到本地 Markdown 笔记
```
