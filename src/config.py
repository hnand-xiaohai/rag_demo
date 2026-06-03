# config.py
import os

from dotenv import load_dotenv

load_dotenv()

class Settings:
    # DeepSeek API 配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-6d01a34d2c6b4030b0a7598a1d0dd5f4")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # 模型路径配置 (建议本地下载好模型，直接指向本地路径，避免每次联网下载)
    # 示例：EMBED_MODEL_PATH = "C:/models/m3e-base"
    EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", "/Users/xiaohai/IdeaProjects/rag_demo/src/model/m3e-base")
    RERANK_MODEL_PATH = os.getenv("RERANK_MODEL_PATH", "/Users/xiaohai/IdeaProjects/rag_demo/src/model/bge-reranker-base")

    # 数据库路径
    CHROMA_PERSIST_DIR = "./db"
    CHROMA_COLLECTION_NAME = "enterprise_knowledge"

    # 文档路径
    DOCS_DIR = "./data"

settings = Settings()