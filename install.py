# RAG-Study 一键安装脚本
# 在 RAG-Study 目录下运行：python install.py

import subprocess
import sys

packages = [
    "python-dotenv",
    "langchain",
    "langchain-openai",
    "langchain-huggingface",
    "langchain-chroma",
    "langchain-text-splitters",
    "sentence-transformers",
    "chromadb",
    "pydantic",
]

print("=" * 50)
print("开始安装 RAG-Study 所需依赖...")
print("=" * 50)

for pkg in packages:
    print(f"\n>>> 安装 {pkg}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "-q"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"    ✓ {pkg} 安装成功")
    else:
        print(f"    ✗ {pkg} 安装失败：{result.stderr.strip()}")

print("\n" + "=" * 50)
print("安装完成！接下来：")
print("1. 编辑 .env 文件，填入你的 ZHIPU_API_KEY")
print("2. 把你的 Markdown 笔记放入 data/ 目录")
print("3. 修改 rag_generate.py 中 MD_FILE_PATH 为你的文件名")
print("4. 运行 python rag_generate.py 开始提问！")
print("=" * 50)
