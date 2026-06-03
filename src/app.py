# app.py
import logging
import os

import chromadb
import gradio as gr
import httpx
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.deepseek import DeepSeek
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.agent_engine import DecomposeRAGAgent
from src.config import settings

# 解决 Windows Gradio 启动时的代理问题
os.environ['NO_PROXY'] = '127.0.0.1,localhost,::1'
os.environ['no_proxy'] = '127.0.0.1,localhost,::1'

logger = logging.getLogger(__file__)


def init_components():
    logger.info("正在初始化模型服务...")

    async_client = httpx.AsyncClient(verify=False, timeout=30.0,trust_env=False)
    http_client = httpx.Client(verify=False,trust_env=False)
    # 配置 LLM (DeepSeek)
    llm = DeepSeek(
        model=settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        api_base=settings.DEEPSEEK_BASE_URL,
        temperature=0.1,
        http_client=http_client,
        async_http_client=async_client
    )

    # 配置embedding
    embedding = HuggingFaceEmbedding(model_name=settings.EMBED_MODEL_PATH, trust_remote_code=True)

    # 全局 Settings 配置
    Settings.llm = llm
    Settings.embed_model = embedding

    # 加载 ChromaDB
    db = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    chroma_collection = db.get_or_create_collection(settings.CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 加载StorageContext(包含之前构建的docstore)
    # 必须加载 docstore 才能使用 BM25
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=settings.CHROMA_PERSIST_DIR
    )

    #     加载索引
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, embed_model=embedding)

    # 配置rerank
    reranker = SentenceTransformerRerank(model=settings.RERANK_MODEL_PATH, top_n=5)

    #    构造混合检索器
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=10)

    bm25_retriever = BM25Retriever.from_defaults(docstore=storage_context.docstore, similarity_top_k=10)

    hybrid_retriever = QueryFusionRetriever(retrievers=[vector_retriever, bm25_retriever], similarity_top_k=10,
                                            mode=FUSION_MODES.RECIPROCAL_RANK, num_queries=1)

    return llm, hybrid_retriever, reranker


llm, retriever, reranker = init_components()


# 2. 定义问答逻辑
async def respond(message, history):

    def extract_text_from_content(content):
        """从 Gradio 的消息格式中提取文本"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Gradio 新版本格式: [{'text': '...', 'type': 'text'}]
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
            return ' '.join(text_parts)
        else:
            return str(content)

    # --- 重建记忆 ---
    chat_history = []

    # 处理历史记录
    if history:
        # 检查历史记录格式
        if isinstance(history[0], dict):
            # 新版本 Gradio 格式: [{"role": "user/assistant", "content": [...]}]
            for msg in history:
                role = msg.get("role", "")
                raw_content = msg.get("content", "")
                # 提取文本内容
                content = extract_text_from_content(raw_content)

                if role in ["user", "assistant"] and content:
                    chat_history.append(ChatMessage(role=role, content=content))

        elif isinstance(history[0], (list, tuple)):
            # 旧版本 Gradio 格式: [(user_msg, assistant_msg), ...]
            for item in history:
                if len(item) >= 2:
                    user_msg = extract_text_from_content(item[0])
                    assistant_msg = extract_text_from_content(item[1])

                    if user_msg:
                        chat_history.append(ChatMessage(role="user", content=user_msg))
                    if assistant_msg:
                        chat_history.append(ChatMessage(role="assistant", content=assistant_msg))

        else:
            # 未知格式，记录警告
            print(f"警告: 未知的聊天历史格式: {type(history[0])}")

    # 初始化agent
    agent = DecomposeRAGAgent(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
        memory=chat_history
    )

    # --- 执行 Agent 工作流 ---
    try:
        full_response = ""
        async for response in agent.astream_chat(message):
            if hasattr(response, 'delta') and response.delta:
                full_response += response.delta
                yield full_response
            elif hasattr(response, 'response') and response.response:
                full_response = response.response
                yield full_response

    except Exception as e:
        yield f"抱歉，系统处理出现异常：{str(e)}"


def launch_app():
    # ========== CSS 样式定义 ==========
    custom_css = """
    /* ========== 蓝白机甲 · 粒子动态背景 ========== */
    .gradio-container {
        background: #0a0f1f;
        font-family: 'Segoe UI', 'Orbitron', 'Poppins', system-ui, sans-serif;
        position: relative;
        overflow: hidden;
        color: #d0e6ff;
    }

    .gradio-container::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: radial-gradient(2px 2px at 20% 40%, #3a86ff, rgba(58,134,255,0.2)),
                          radial-gradient(3px 3px at 60% 80%, #00c8ff, rgba(0,200,255,0.2)),
                          radial-gradient(1px 1px at 85% 15%, #ffffff, rgba(255,255,255,0.3)),
                          radial-gradient(2px 2px at 10% 90%, #4c9aff, rgba(76,154,255,0.2)),
                          radial-gradient(1px 1px at 45% 35%, #00e0ff, rgba(0,224,255,0.3));
        background-repeat: no-repeat;
        background-size: 300px 300px, 400px 400px, 200px 200px, 350px 350px, 250px 250px;
        animation: particleDrift 30s infinite linear;
        pointer-events: none;
        z-index: 0;
    }

    @keyframes particleDrift {
        0% { background-position: 0% 0%, 0% 0%, 0% 0%, 0% 0%, 0% 0%; }
        100% { background-position: 100% 100%, 80% 60%, 120% 80%, 60% 40%, 140% 30%; }
    }

    .gradio-container .main,
    .gradio-container > .gradio-app,
    .gr-box {
        background: rgba(10, 18, 30, 0.75) !important;
        backdrop-filter: blur(8px);
        border: 1px solid #2a6f8f;
        border-radius: 0px !important;
        box-shadow: 0 0 0 1px rgba(0, 200, 255, 0.2), inset 0 0 12px rgba(0, 160, 255, 0.1);
        position: relative;
        z-index: 1;
    }

    /* ========== 聊天区域高度优化 ========== */
    /* 聊天容器整体高度 */
    .gradio-chatbot,
    [data-testid="chatbot"] {
        height: 650px !important;
        min-height: 600px !important;
    }

    /* 消息区域 */
    .gradio-chatbot .messages,
    [data-testid="chatbot"] .messages {
        height: 100% !important;
        max-height: none !important;
    }

    /* 输入框容器高度 */
    .chat-input {
        min-height: 60px !important;
    }

    /* ========== 标题样式 ========== */
    .markdown-text h1,
    .gradio-markdown h1 {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        background: linear-gradient(135deg, #fff, #3a86ff);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        border-left: 5px solid #3a86ff;
        padding-left: 20px;
        margin: 0 0 12px 0;
        text-shadow: 0 0 8px rgba(58,134,255,0.5);
    }

    .markdown-text p {
        color: #b0d4ff;
        font-size: 1rem;
        letter-spacing: 0.5px;
    }

    /* ========== 聊天气泡样式 ========== */
    [data-testid="chatbot"] .user .message,
    [data-testid="chatbot"] .bot .message {
        border-radius: 0px !important;
        padding: 14px 22px !important;
        margin: 10px 0;
        font-size: 0.95rem;
        border: 1px solid #2a6f8f;
        background: rgba(10, 20, 35, 0.9);
        box-shadow: 2px 2px 0 rgba(0,0,0,0.2);
    }

    [data-testid="chatbot"] .user .message {
        background: linear-gradient(135deg, #1e3a6f, #0e2a4a) !important;
        border-left: 4px solid #3a86ff;
        color: #ffffff;
    }

    [data-testid="chatbot"] .bot .message {
        background: rgba(8, 20, 35, 0.9) !important;
        border-left: 1px solid #3a86ff;
        color: #c8e2ff;
    }

    /* ========== 输入框样式 ========== */
    .gradio-textbox,
    textarea,
    .chat-input textarea {
        background: #07121f !important;
        border: 1px solid #2a6f8f !important;
        border-radius: 0px !important;
        color: #e0f0ff !important;
        font-family: 'Segoe UI', monospace;
        font-size: 0.95rem;
        padding: 14px 18px !important;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.5);
    }

    .gradio-textbox:focus,
    textarea:focus {
        border-color: #3a86ff !important;
        box-shadow: 0 0 0 2px rgba(58,134,255,0.3), inset 0 1px 2px #000 !important;
        outline: none;
    }

    /* ========== 按钮样式 ========== */
    button,
    .gradio-button,
    .primary-button,
    .stop-button {
        background: #0e1a2a !important;
        border: 1px solid #3a86ff !important;
        border-radius: 0px !important;
        color: #88ccff !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 10px 26px;
        transition: all 0.2s ease;
        box-shadow: 0 1px 0 rgba(255,255,255,0.05);
    }

    button:hover,
    .gradio-button:hover {
        background: #3a86ff !important;
        color: #0a0f1f !important;
        border-color: #88ccff !important;
        box-shadow: 0 0 8px #3a86ff;
        transform: translateY(-1px);
    }

    /* ========== 示例标签样式 ========== */
    .example-item,
    .examples button {
        background: #0c1725 !important;
        border: 1px solid #2a6f8f;
        border-radius: 0px;
        padding: 8px 16px;
        font-family: monospace;
        font-size: 0.85rem;
        color: #aac9ff;
        transition: 0.1s linear;
    }

    .example-item:hover,
    .examples button:hover {
        background: #3a86ff !important;
        color: #fff !important;
        border-color: #88ccff;
    }

    /* ========== 滚动条样式 ========== */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #0a0f1f;
    }
    ::-webkit-scrollbar-thumb {
        background: #2a6f8f;
        border-radius: 0px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #3a86ff;
    }

    /* ========== 自定义头部样式 ========== */
    .app-header {
        text-align: center;
        padding: 24px 0 16px 0;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(42, 111, 143, 0.3);
    }
    .app-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(to right, #fff, #3a86ff, #00c8ff);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        text-shadow: 0 0 15px rgba(58, 134, 255, 0.4);
        letter-spacing: 1px;
    }
    .app-header p {
        color: #88ccff;
        font-size: 1.05rem;
        margin-top: 10px;
        opacity: 0.85;
    }

    /* ========== 自定义页脚样式 ========== */
    .app-footer {
        text-align: center;
        font-size: 0.8rem;
        border-top: 1px solid #2a6f8f;
        padding: 16px 0 12px 0;
        margin-top: 20px;
        color: #6c8fb0;
    }
    .app-footer a {
        color: #3a86ff;
        text-decoration: none;
        border-bottom: 1px dashed #3a86ff;
    }

    /* ========== 示例区域样式 ========== */
    .examples-container {
        margin-top: 20px;
        padding: 16px;
        background: rgba(10, 18, 30, 0.5);
        border: 1px solid #2a6f8f;
    }
    """

    # ========== 主题配置 ==========
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
    ).set(
        body_background_fill="#0a0f1f",
        button_primary_background_fill="#3a86ff",
    )

    # ========== 构建界面 ==========
    with gr.Blocks(title="威武集团智能客服助手") as demo:
        # 1. 头部区域
        gr.HTML("""
        <div class="app-header">
            <h1>威武集团 · 智能客服助手</h1>
            <p>⚡ 混合检索引擎 + DeepSeek大模型驱动 ⚡</p>
        </div>
        """)

        # 2. 聊天界面
        gr.ChatInterface(
            fn=respond,
            title=None,
            description=None,
            show_progress="full",
            fill_height=True,
            examples=[
                ["介绍一下威武集团？"],
                ["我想买一款适合打游戏的熙哥电脑"],
                ["威武手机和熙哥电脑如何跨屏协同？"],
                ["熙哥电脑的创始人是谁？"]
            ],
            textbox=gr.Textbox(
                placeholder="请输入您关于威武手机或熙哥电脑的问题...",
                show_label=False,
                container=False,
                scale=7,
                submit_btn="发送 ⚡"
            )
        )

        # 3. 页脚区域
        gr.HTML("""
        <div class="app-footer">
            <p>✨ 基于 RAG 技术构建 | 威武集团 & 熙哥电脑 联合出品</p>
            <p>数据安全由 D4 级技术专家团队保驾护航</p>
        </div>
        """)

    # ========== 启动配置 ==========
    demo.queue(default_concurrency_limit=5)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        theme=theme,
        css=custom_css
    )


if __name__ == "__main__":
    launch_app()