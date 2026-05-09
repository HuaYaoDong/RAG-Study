# RAG-Study — 高等数学 RAG 问答系统

> 南京理工大学 RAG 科研训练项目
> 基于 LangChain + 智谱 GLM-4 + ChromaDB 构建的本地知识库问答系统

---

## 📁 项目结构

```
RAG-Study/
├── config.py           # 全局配置（模型、路径、参数）
├── rag_split.py        # 文档解析与向量库构建
├── rag_retrieve.py     # 两阶段检索（向量召回 + 重排序）
├── rag_generate.py     # 主程序：意图路由 + 生成回答
├── rag_agent.py        # 查询意图分析（总结/例题/问答）
├── install.py          # 一键安装依赖脚本
├── .env                # API 密钥配置（不要上传 GitHub！）
└── data/               # 放你的 Markdown 笔记文件
    └── 示例_高等数学.md
```

---

## 🚀 快速开始

### 第一步：安装依赖

```powershell
cd RAG-Study
python install.py
```

### 第二步：配置 API Key

编辑 `.env` 文件，填入你的**智谱 API Key**：

```
ZHIPU_API_KEY=你的密钥（去 https://open.bigmodel.cn 申请）
```

### 第三步：放入你的笔记

把你的 Markdown 格式笔记（如高数、线代等）放入 `data/` 目录。

> ⚠️ Markdown 文件最好用 `#`、`##`、`###` 等标题层级，效果最好！

### 第四步：修改入口文件

打开 `rag_generate.py`，修改第 158 行：

```python
MD_FILE_PATH = "data/你的笔记文件.md"   # 改成你的文件路径
```

### 第五步：运行！

```powershell
python rag_generate.py
```

系统会自动：
1. 读取 Markdown → 切块 → 构建向量数据库（首次运行较慢）
2. 加载检索器和大模型
3. 进入交互问答模式，输入问题即可

---

## ⚠️ 注意事项

| 问题 | 解决方法 |
|------|----------|
| 没有 NVIDIA GPU | 修改 `config.py` 第 42 行：`DEVICE = "cpu"` |
| 首次运行很慢 | 正常！需要下载 Embedding 模型（约 200MB），耐心等待 |
| 问答结果保存在哪 | 自动保存到 `我的高数学习笔记.md` |

---

## 🧠 系统架构

```
你的 Markdown 笔记
       ↓
  rag_split.py    →  按标题切块  →  ChromaDB 向量数据库
       ↓
  rag_agent.py    →  识别问题意图（总结/找例题/问答）
       ↓
  rag_retrieve.py →  向量召回 → BGE Reranker 重排序
       ↓
  rag_generate.py →  GLM-4 生成最终回答
       ↓
  保存到 Markdown 笔记文件
```
