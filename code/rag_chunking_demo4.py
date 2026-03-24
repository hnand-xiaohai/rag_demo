# Markdown文本示例
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

markdown_text = """
# RAG系统概述

RAG（Retrieval-Augmented Generation）是一种强大的AI框架。

## 核心组件

RAG系统主要包含两个核心组件：

1.  **检索器 (Retriever)**：负责从知识库中快速找到相关文档。
2.  **生成器 (Generator)**：基于检索到的信息生成答案。

### 检索器的技术细节

检索器通常使用向量数据库实现...
"""
markdown_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.MARKDOWN, chunk_size=100, chunk_overlap=0)
chunks = markdown_splitter.split_text(markdown_text)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print(f"(长度: {len(chunk)})\n")



