
import asyncio
import json
from typing import Optional, List

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import LLM
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore, QueryBundle

DECOMPOSE_PROMPT = """
你是一个专业的查询分析助手。你的任务是将用户的复杂问题拆解为一系列独立的子问题。
如果用户的问题很简单，只需要一个子问题即可。
请直接输出 JSON 格式的列表，不要包含其他解释性文字。

用户问题：{query}

示例输入："产品A多少钱？支持退货吗？"
示例输出：["产品A的价格是多少？", "产品A支持退货吗？"]

请输出子问题列表：
"""


class DecomposeRAGAgent:
    def __init__(
            self,
            llm: LLM,
            retriever: BaseRetriever,
            reranker: SentenceTransformerRerank,
            memory: Optional[List[ChatMessage]] = None,
            top_k_per_subq: int = 5
    ):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.memory = memory or []
        self.top_k_per_subq = top_k_per_subq

    async def _decompose_query(self, query: str) -> List[str]:
        """步骤1: 使用 LLM 分解问题"""
        # 构造包含历史对话的 Prompt，以便处理代词指代
        history_str = "\n".join([f"{m.role}: {m.content}" for m in self.memory[-4:]])  # 取最近2轮
        prompt = DECOMPOSE_PROMPT.format(query=query)
        # 调用 LLM
        response = await self.llm.acomplete(prompt)
        try:
            # 尝试解析 JSON
            sub_questions = json.loads(response.text)
            if isinstance(sub_questions, list):
                return sub_questions
        except json.JSONDecodeError:
            pass

            # 如果解析失败，降级为单问题模式
        return [query]

    async def _retrieve_for_subq(self, sub_query: str) -> List[NodeWithScore]:
        """步骤2: 针对单个子问题进行检索+重排序"""
        # 检索
        nodes = await self.retriever.aretrieve(sub_query)
        # 重排序
        # 注意：LlamaIndex 的 postprocessor 需要在 query_str 存在时效果最好
        processed_nodes = self.reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str=sub_query))
        return processed_nodes[:self.top_k_per_subq]

    async def _parallel_retrieve(self, sub_questions: List[str]) -> List[NodeWithScore]:
        """步骤2 (并发): 对所有子问题并行检索"""
        tasks = [self._retrieve_for_subq(q) for q in sub_questions]
        results = await asyncio.gather(*tasks)
        # 步骤3: 汇总与去重
        all_nodes = []
        seen_ids = set()
        for nodes_list in results:
            for node in nodes_list:
                if node.node.node_id not in seen_ids:
                    all_nodes.append(node)
                    seen_ids.add(node.node.node_id)
        return all_nodes

    async def astream_chat(self, message: str):
        """完整工作流：分解 -> 检索 -> 汇总 -> 生成"""
        # 1. 分解问题
        sub_questions = await self._decompose_query(message)
        print(f"--- [Agent Debug] 分解子问题: {sub_questions}")

        # 2. 并行检索
        gathered_nodes = await self._parallel_retrieve(sub_questions)
        print(f"--- [Agent Debug] 汇总节点数: {len(gathered_nodes)}")

        # 3. 构建上下文
        context_text = "\n\n".join([n.node.get_content() for n in gathered_nodes])

        # 4. 最终生成 Prompt (优化后的 System Prompt)
        SYSTEM_PROMPT = """\
        你是一位专业的企业客服助手。请参考以下信息回答用户的问题。

        要求：
        1. 请直接回答问题，内容要准确、专业。
        2. **禁止使用“根据参考上下文”、“根据提供的信息”等机械性的引用前缀**。
        3. 如果以下信息无法回答用户的问题，请礼貌地说明情况。

        参考信息：
        {context_str}

        用户问题：
        {query_str}
        """
        # 构造完整的 Prompt
        full_prompt = SYSTEM_PROMPT.format(context_str=context_text, query_str=message)
        # 记忆注入 (简单的拼接方式，确保上下文连贯)
        # 在实际生产中，如果上下文过长，需要更智能的截断策略
        messages = self.memory + [ChatMessage(role="user", content=full_prompt)]
        # 流式输出
        response_stream = await self.llm.astream_chat(messages)

        async for response in response_stream:
            # response.delta 是增量文本
            # response.response 是完整响应
            yield response