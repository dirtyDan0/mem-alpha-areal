import asyncio
import json
import os
import re
import sys
from typing import Any

import yaml
import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.workflow_api import RolloutWorkflow
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.experimental.openai import ArealOpenAI
from areal.utils import stats_tracker
from areal.utils import logging

from rank_bm25 import BM25Okapi

try:  # Package-style relative import (works if executed via -m with package context)
    from .utils import (  # type: ignore
        Memory,
        MEMORY_TOOL_IMPL,
        MEMORY_TOOLS,
        TokenCounter,
    )
    from .metrics import evaluate_wrt_source, _extract_answer_from_response  # type: ignore
except ImportError:  # Fallback when executed directly (no package parent known)
    module_dir = os.path.dirname(os.path.abspath(__file__))
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    from utils import (  # type: ignore  # noqa: E402
        Memory,
        MEMORY_TOOL_IMPL,
        MEMORY_TOOLS,
        TokenCounter,
    )
    from metrics import evaluate_wrt_source, _extract_answer_from_response  # type: ignore  # noqa: E402


logger = logging.getLogger("AReaL Mem-Alpha")

DEFAULT_R1_MAX_TOKENS = 2048
DEFAULT_R1_LME_MAX_TOKENS = 50
DEFAULT_R4_MAX_TOKENS = 512


class MemoryAgent:
    def __init__(
        self,
        external_client: ArealOpenAI,
        max_new_tokens: int,
        beta: float,
        gamma: float,
        r1_max_completion_tokens: int,
        r1_lme_max_completion_tokens: int,
        r4_max_completion_tokens: int,
        rollout_stat_scope: str = "rollout",
    ):
        """
        The external client is responsible for:
            1. Answering questions based on memories (which would be used to calculate R1 reward).
            2. Serving as the LLM judge to compute R4 reward.
        """
        self.external_client = external_client
        self.rag_top_k = 20
        self.max_new_tokens = max_new_tokens
        self.beta = beta
        self.gamma = gamma
        self.rollout_stat_scope = rollout_stat_scope
        self.r1_max_completion_tokens = r1_max_completion_tokens
        self.r1_lme_max_completion_tokens = r1_lme_max_completion_tokens
        self.r4_max_completion_tokens = r4_max_completion_tokens
        with open(
            "examples/mem-alpha/prompts_wrt_datasource.yaml", "r", encoding="utf-8"
        ) as handle:
            self.prompts_wrt_datasource = yaml.safe_load(handle) or {}

    async def run_agent(
        self,
        data: dict[str, Any],
        client: ArealOpenAI,
    ) -> dict[str, Any]:
        """
        rt = r1 + r2,t + βr3 + γr4,t
        """
        memory = Memory()
        token_counter = TokenCounter()

        chunks = data["chunks"]
        questions_and_answers = data["questions_and_answers"]
        data_source = data["data_source"]
        pending_r4_tasks = []
        r2_list = []
        completion_ids = []
        turn_records = []
        total_chunk_length = 0

        unified_prompt = self.prompts_wrt_datasource.get("unified_prompt")
        for chunk in chunks:
            chunk_text = (
                unified_prompt.format(context=chunk) if unified_prompt else chunk
            )
            total_chunk_length += token_counter.count_tokens(chunk)
            messages = self._format_prompt_with_memory_and_chunk(memory, chunk_text)

            (
                tool_calls,
                completion_id,
                response_text,
                response_tool_calls,
            ) = await self._get_tool_calls_and_completion_id(messages, client)
            completion_ids.append(completion_id)

            current_r4_tasks = []

            for tool_call in tool_calls:
                current_r4_tasks.append(
                    asyncio.create_task(self._calculate_r4(tool_call))
                )

            tool_call_results = []

            for tool_call in tool_calls:
                tool_call_result = self._execute_memory_tool(memory, tool_call)
                tool_call_results.append(tool_call_result)

            pending_r4_tasks.append(current_r4_tasks)
            turn_records.append(
                {
                    "prompt": messages,
                    "response": response_text,
                    "tool_calls": response_tool_calls,
                    "tool_call_results": tool_call_results,
                }
            )

            # r2: the percentage of successfully executed function calls
            r2 = (
                sum(tool_call_results) / len(tool_call_results)
                if tool_call_results
                else 0.0
            )
            r2_list.append(r2)

        (
            semantic_bm25,
            semantic_items,
            episodic_bm25,
            episodic_items,
        ) = self._build_memory_bm25_index(memory)

        r1_task = asyncio.create_task(
            self._calculate_r1(
                data_source,
                memory,
                questions_and_answers,
                semantic_bm25,
                semantic_items,
                episodic_bm25,
                episodic_items,
            )
        )
        memory_total_length = memory.total_length(token_counter)
        r3 = self._calculate_r3(memory_total_length, total_chunk_length)
        r4_list = await self._collect_r4_list(pending_r4_tasks)
        r1 = await r1_task
        for completion_id, r2, r4 in zip(completion_ids, r2_list, r4_list):
            # log stats
            traj_score = r1 + self.beta * r3
            turn_score = r2 + self.gamma * r4
            reward = traj_score + turn_score

            client.set_reward(completion_id, reward)
            interaction = client.get_interaction(completion_id)
            if interaction is not None:
                interaction.traj_score = float(traj_score)
                interaction.turn_score = float(turn_score)

            rollout_tracker = stats_tracker.get(self.rollout_stat_scope)
            rollout_tracker.scalar(
                r1=r1,
                r2=r2,
                r3=r3,
                r4=r4,
                memory_total_length=memory_total_length,
                reward=reward,
            )
            if self.rollout_stat_scope == "eval-rollout":
                data_source_key = str(data_source).replace("/", "_")
                with rollout_tracker.scope(f"eval/{data_source_key}"):
                    rollout_tracker.scalar(
                        perf=r1,
                        mem=memory_total_length,
                    )

        for turn_record, completion_id, r2, r4 in zip(
            turn_records, completion_ids, r2_list, r4_list
        ):
            stop_reason = None
            interaction = client.get_interaction(completion_id)
            if interaction is not None and interaction.model_response is not None:
                stop_reason = interaction.model_response.stop_reason
            turn_record.update(
                {
                    "r1": float(r1),
                    "r2": float(r2),
                    "r3": float(r3),
                    "r4": float(r4),
                    "stop_reason": stop_reason,
                }
            )

        return {
            "turn_records": turn_records,
            "final_memory_state": memory.export_memory_state(),
            "final_memory_total_length": memory_total_length,
        }

    def _format_prompt_with_memory_and_chunk(self, memory, chunk):
        memory_state = memory.export_memory_state()

        system_prompt = f"""You are a memory assistant. Use tools to update memory based only on the content inside <new_chunk>.

MEMORY TYPES:
- core: brief running summary (user/reading/classification rules)
- semantic: key facts or information
- episodic: specific events with time/context ("At timestamp t, ...")

TOOLS (JSON tool calls):
- new_memory_insert: add semantic/episodic
- memory_update: update core or update semantic/episodic by memory_id
- memory_delete: delete core or delete semantic/episodic by memory_id
Examples:
{{"name": "memory_update", "arguments": {{"memory_type": "core", "new_content": "User prefers concise replies and runs with friends on weekends."}}}}
{{"name": "new_memory_insert", "arguments": {{"memory_type": "semantic", "content": "Project timeline: kickoff in early March; final delivery by May 30."}}}}
{{"name": "memory_delete", "arguments": {{"memory_type": "episodic", "memory_id": "epi_0003"}}}}

CURRENT MEMORY STATE:
{memory_state}

RULES:
- memory_type must be core/semantic/episodic
- update/delete for semantic/episodic must use existing memory_id from CURRENT MEMORY STATE
- Output only tool calls, or 'Done' if no changes are needed; be concise."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk},
        ]

    async def _get_tool_calls_and_completion_id(
        self, messages, client: ArealOpenAI
    ) -> tuple[list[dict] | None, str, str | None, list[dict] | None]:

        def _parse_tool_arguments(raw_arguments: Any) -> dict:
            if isinstance(raw_arguments, dict):
                return raw_arguments
            if isinstance(raw_arguments, str):
                try:
                    return json.loads(raw_arguments)
                except json.JSONDecodeError:
                    try:
                        parsed = yaml.safe_load(raw_arguments)
                    except yaml.YAMLError:
                        return {}
                    return parsed if isinstance(parsed, dict) else {}
            return {}

        try:
            completion = await client.chat.completions.create(
                messages=messages,
                max_completion_tokens=self.max_new_tokens,
                max_total_tokens=16384,
                temperature=1.0,
                tools=MEMORY_TOOLS,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            completion_id = completion.id
            message = completion.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                return [None], completion_id, message.content, None
            response_tool_calls = [
                tool_call.function.to_dict() for tool_call in tool_calls
            ]
            tool_calls = [
                {
                    "arguments": _parse_tool_arguments(tool_call["arguments"]),
                    "name": tool_call["name"],
                }
                for tool_call in response_tool_calls
            ]
            return tool_calls, completion_id, message.content, response_tool_calls
        except RuntimeError as e:
            logger.warning(f"RuntimeError during LLM call_server: {e}")

    def _execute_memory_tool(self, memory, tool_call: str):
        if not tool_call:
            return 0.0
        name, arguments = tool_call.get("name"), tool_call.get("arguments", {})
        if not isinstance(arguments, dict):
            return 0.0
        tool_impl = MEMORY_TOOL_IMPL.get(name, None)
        if not tool_impl:
            logger.warning(f"Tool {name} not found in MEMORY_TOOL_IMPL.")
            return 0.0
        try:
            tool_impl(memory, **arguments)
            return 1.0
        except Exception as e:
            logger.warning(f"Error executing tool {name}: {e}")
            return 0.0

    def _build_memory_bm25_index(self, memory: Memory):
        semantic_items = list(memory.semantic.items())
        episodic_items = list(memory.episodic.items())
        semantic_corpus = [content.split() for _, content in semantic_items] or [[""]]
        episodic_corpus = [content.split() for _, content in episodic_items] or [[""]]
        semantic_bm25 = BM25Okapi(semantic_corpus)
        episodic_bm25 = BM25Okapi(episodic_corpus)
        return semantic_bm25, semantic_items, episodic_bm25, episodic_items

    async def _calculate_r1(
        self,
        data_source,
        memory,
        questions_and_answers,
        semantic_bm25,
        semantic_items,
        episodic_bm25,
        episodic_items,
    ):
        async def _compute_reward_wrt_data_source(
            data_source, prediction, answer, question=None
        ):
            if data_source == "booksum":
                keywords = answer.split(",")
                keywords = [x.strip() for x in keywords]
                hit = 0
                for keyword in keywords:
                    if keyword.lower() in prediction.lower():
                        hit += 1
                return hit / len(keywords)

            elif (
                data_source == "pubmed-rct"
                or "ttl_train" in data_source
                or "icl" in data_source
            ):
                # PUBMED dataset evaluation: MUST be ONLY a single digit
                extracted_answer = _extract_answer_from_response(prediction)

                # Remove quotes and strip whitespace
                extracted_answer = extracted_answer.strip("\"'").strip()

                # STRICT pattern: must be EXACTLY a single digit with nothing else
                single_digit_pattern = r"^\d+$"

                if not re.match(single_digit_pattern, extracted_answer):
                    return 0.0

                gold_num = str(answer).strip("\"'").strip()
                return 1.0 if extracted_answer == gold_num else 0.0
            elif data_source == "lme_train":
                template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
                prompt = template.format(question, answer, prediction)

                response = await self.external_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=self.r1_lme_max_completion_tokens,
                    max_total_tokens=16384,
                    temperature=0.0,
                    store=False,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )

                if (
                    "yes" in response.choices[0].message.content.strip().lower()
                    and "no" not in response.choices[0].message.content.strip().lower()
                ):
                    return 1.0
                else:
                    return 0.0
            elif data_source == "perltqa":
                if ";" in answer:
                    answer = answer.split(";")
                    total_hit = 0
                    for answer in answer:
                        if answer.lower().strip() in prediction:
                            total_hit += 1
                    return total_hit / len(answer)

                else:
                    return 1.0 if answer.lower() in prediction.lower() else 0.0

            elif data_source in ["squad", "hotpotqa"]:
                # Default: containment score for QA datasets
                if isinstance(answer, list):
                    answer_text = str(answer[0]) if answer else ""
                else:
                    answer_text = (
                        answer.get("text", answer)
                        if isinstance(answer, dict)
                        else str(answer)
                    )

                return 1.0 if answer_text.lower() in prediction.lower() else 0.0
            else:
                # Memory agent bench evaluation for other datasets
                return evaluate_wrt_source({"output": prediction}, answer, data_source)

        if not questions_and_answers:
            return 0.0

        def _iter_qa_pairs(items):
            for item in items:
                if isinstance(item, dict):
                    yield item.get("question"), item.get("answer")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    yield item[0], item[1]

        def _select_top_k(bm25, items, query_tokens, top_k):
            if not items or not bm25:
                return []
            return bm25.get_top_n(query_tokens, items, n=min(top_k, len(items)))

        def _build_filtered_memory(question: str) -> Memory:
            query_tokens = question.split()
            filtered = Memory()
            filtered.core = memory.core

            for mem_id, content in _select_top_k(
                semantic_bm25, semantic_items, query_tokens, self.rag_top_k
            ):
                filtered.semantic[mem_id] = content

            for mem_id, content in _select_top_k(
                episodic_bm25, episodic_items, query_tokens, self.rag_top_k
            ):
                filtered.episodic[mem_id] = content

            return filtered

        def _format_prompt_with_filtered_memory(question: str) -> list[dict]:
            filtered_memory = _build_filtered_memory(question)
            system_prompt = f"""You are a reasoning assistant with access to structured memory. Use the memories below to provide accurate, relevant, and comprehensive responses to user queries.

MEMORY STRUCTURE:
- Core Memory: Fundamental facts about the user (preferences, roles, goals, etc.)
- Semantic Memory: General knowledge, factual or conceptual information
- Episodic Memory: Specific personal experiences or events with time and context

CURRENT MEMORY STATE:

{filtered_memory.export_memory_state()}

INSTRUCTIONS:
- Use the memories above to inform your responses
- If information is available in memory, reference it appropriately
- If memory is insufficient to answer a question, acknowledge this clearly
- Provide helpful and contextual responses based on the available memory
- Be concise but comprehensive in your answers
- The memory state lists semantic/episodic items in a table; memory_id values look like sem_0001 or epi_0001"""

            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]

        async def _score_question(question: str, answer: str | list[str]) -> float:
            messages = _format_prompt_with_filtered_memory(question)
            completion = await self.external_client.chat.completions.create(
                messages=messages,
                max_completion_tokens=self.r1_max_completion_tokens,
                max_total_tokens=16384,
                temperature=0.7,
                store=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            prediction = completion.choices[0].message.content
            return await _compute_reward_wrt_data_source(
                data_source, prediction, answer, question
            )

        tasks = [
            asyncio.create_task(_score_question(question, answer))
            for question, answer in _iter_qa_pairs(questions_and_answers)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        rewards = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Error computing r1 sub-reward: %s", result)
                continue
            rewards.append(result)
        return sum(rewards) / len(rewards) if rewards else 0.0

    def _calculate_r3(self, memory_total_length, total_chunk_length):
        if total_chunk_length == 0:
            return 0.0
        return 1 - memory_total_length / total_chunk_length

    async def _calculate_r4(self, tool_call):

        def _get_analysis_prompt(memory_type: str) -> str:
            """Get analysis prompt template for specific memory type."""
            if memory_type == "core":
                return """You are an expert memory analyst. Analyze the quality of core memory content.

The core memory is invalid if any of the following meets:
(1) The literal content "core memory" appears in the memory such as "This is core memory ...", "The core memory has been updated ...".
(2) The core memory is apparently a placeholder such as "Here we save the summary" while not stating what the "summary" is, "Here are some rules" and not stating what the "rules" are.

Otherwise, the core memory is valid.

Respond ONLY with a JSON code block in this exact format (no extra text).
```json
{
"VALID": true/false,
"REASON": "brief explanation of the assessment"
}
```"""

            elif memory_type == "semantic":
                return """You are an expert memory analyst. Analyze the quality of semantic memory content.

Semantic memory should contain:
- Information or Knowledge about somebody or something
- Definitions, theories, principles, or explanations
- How-to knowledge or procedural information
- Research findings or established facts

Two other memories are Core memory (User Personalities) and Episodic memory (User Experiences). The information not suitable for these two memories should be considered as semantic memory.

Respond ONLY with a JSON code block in this exact format (no extra text).
```json
{
"VALID": true/false,
"REASON": "brief explanation of the assessment"
}
```"""

            elif memory_type == "episodic":
                return """You are an expert memory analyst. Analyze the quality of episodic memory content.

Episodic memory should contain:
- Experiences or events
- Clear temporal information (when it happened)
- Contextual details (what happened)

Respond ONLY with a JSON code block in this exact format (no extra text).
```json
{
"VALID": true/false,
"REASON": "brief explanation of the assessment"
}
```"""

            else:
                logger.warning("Unknown memory type for r4 judge: %s", memory_type)
                return ""

        if tool_call is None:
            return 0.0

        try:
            tool_name = tool_call.get("name")
            tool_arguments = tool_call.get("arguments") or {}
            if not isinstance(tool_arguments, dict):
                return 0.0
            memory_type = tool_arguments.get("memory_type", "")

            if not tool_name or not memory_type:
                return 0.0
            if tool_name not in {"new_memory_insert", "memory_update"}:
                return 0.0
            if memory_type == "core" and tool_name == "new_memory_insert":
                return 0.0

            content = (
                tool_arguments.get("content")
                if tool_name == "new_memory_insert"
                else tool_arguments.get("new_content")
            )
            if not content:
                return 0.0

            analysis_prompt = _get_analysis_prompt(memory_type=memory_type)
            if not analysis_prompt:
                return 0.0
            messages = [
                {
                    "role": "system",
                    "content": analysis_prompt,
                },
                {
                    "role": "user",
                    "content": f"Analyze this {memory_type} memory content:\n\n{content}",
                },
            ]
            completion = await self.external_client.chat.completions.create(
                messages=messages,
                max_completion_tokens=self.r4_max_completion_tokens,
                max_total_tokens=16384,
                temperature=0.1,
                store=False,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw_resp = completion.choices[0].message.content or ""
        except Exception as e:
            logger.warning("Error computing r4 sub-reward: %s", e)
            return 0.0

        def _extract_json_block(text: str) -> dict | None:
            pattern = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)
            match = pattern.search(text)
            candidate = match.group(1) if match else text.strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from r4 response: %s", candidate)
                return None

        def _validate_schema(payload: dict) -> dict | None:
            required_keys = {"VALID", "REASON"}
            if not required_keys <= payload.keys():
                logger.warning("Missing keys in r4 response: %s", payload.keys())
                return None
            valid = payload["VALID"]
            reason = payload["REASON"]
            if not isinstance(valid, bool):
                logger.warning("VALID must be bool, got %s", type(valid))
                return None
            if not isinstance(reason, str):
                logger.warning("REASON must be str, got %s", type(reason))
                return None
            return {
                "valid": valid,
                "reason": reason.strip(),
            }

        payload = _extract_json_block(raw_resp)
        validated = _validate_schema(payload) if payload else None
        if not validated:
            return 0.0

        return 1.0 if validated["valid"] else 0.0

    async def _collect_r4_list(
        self,
        pending_tasks: list[list[asyncio.Task[float]]],
    ) -> list[float]:
        async def _avg_r4(tasks: list[asyncio.Task[float]]) -> float:
            if not tasks:
                return 0.0
            results = await asyncio.gather(*tasks)
            return sum(results) / len(results)

        if not pending_tasks:
            return []
        return await asyncio.gather(*(_avg_r4(tasks) for tasks in pending_tasks))


class MemAlphaWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        external_engine: RemoteSGLangEngine | None = None,
    ):

        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.n_trajs = gconfig.n_samples
        self.adv_estimator = gconfig.agent.adv_estimator
        self.adv_eps = gconfig.agent.adv_eps
        self.adv_lambda = gconfig.agent.adv_lambda
        self.gigpo_gamma = gconfig.agent.gigpo_gamma
        self.rollout_stat_scope = rollout_stat_scope
        self._agent_args = {
            "max_new_tokens": gconfig.max_new_tokens,
            "beta": gconfig.agent.beta,
            "gamma": gconfig.agent.gamma,
            "r1_max_completion_tokens": gconfig.agent.r1_max_completion_tokens,
            "r1_lme_max_completion_tokens": gconfig.agent.r1_lme_max_completion_tokens,
            "r4_max_completion_tokens": gconfig.agent.r4_max_completion_tokens,
        }
        self.external_client = ArealOpenAI(
            engine=external_engine,
            tokenizer=tokenizer,
            tool_call_parser="qwen",
        )

        if dump_dir is not None and not os.path.exists(dump_dir):
            os.makedirs(dump_dir, exist_ok=True)

        self.agent = MemoryAgent(
            external_client=self.external_client,
            **self._agent_args,
            rollout_stat_scope=self.rollout_stat_scope,
        )

    async def arun_episode(
        self, engine: ArealOpenAI, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        instance_id = data["instance_id"]
        clients = [
            ArealOpenAI(
                engine=engine,
                tokenizer=self.tokenizer,
                tool_call_parser="qwen",
            )
            for _ in range(self.n_trajs)
        ]

        run_agent_results = await asyncio.gather(
            *[
                self.agent.run_agent(
                    data=data,
                    client=clients[i],
                )
                for i in range(self.n_trajs)
            ]
        )

        completions_with_reward = {}
        for client in clients:
            completions = client.export_interactions(style="individual")
            completions_with_reward.update(completions)
        if completions_with_reward:
            adv_estimator = self.adv_estimator
            adv_eps = self.adv_eps
            adv_lambda = self.adv_lambda
            gigpo_gamma = self.gigpo_gamma
            traj_interactions = [
                list(client.export_interactions(style="individual").values())
                for client in clients
            ]

            def _grpo_norm(values: list[float]) -> list[float]:
                if not values:
                    return []
                rewards = torch.tensor(values, dtype=torch.float32)
                mean = rewards.mean()
                std = rewards.std(unbiased=False)
                adv = (rewards - mean) / (std + adv_eps)
                return [float(v) for v in adv]

            traj_adv = []
            turn_adv_by_traj = None
            if adv_estimator == "gigpo":
                traj_scores = [
                    float(interactions[0].traj_score)
                    for interactions in traj_interactions
                ]
                traj_adv = _grpo_norm(traj_scores)

                n_turns = len(traj_interactions[0])
                turn_scores_by_traj = []
                for interactions in traj_interactions:
                    scores = [
                        float(interaction.turn_score) for interaction in interactions
                    ]
                    turn_scores_by_traj.append(scores)

                returns_by_traj = []
                for scores in turn_scores_by_traj:
                    running = 0.0
                    returns = [0.0 for _ in range(n_turns)]
                    for idx in reversed(range(n_turns)):
                        running = scores[idx] + gigpo_gamma * running
                        returns[idx] = running
                    returns_by_traj.append(returns)

                turn_adv_by_traj = [
                    [0.0 for _ in range(n_turns)] for _ in traj_interactions
                ]
                for turn_idx in range(n_turns):
                    turn_returns = [
                        returns_by_traj[traj_idx][turn_idx]
                        for traj_idx in range(len(traj_interactions))
                    ]
                    turn_adv = _grpo_norm(turn_returns)
                    for traj_idx, value in enumerate(turn_adv):
                        turn_adv_by_traj[traj_idx][turn_idx] = value

                for traj_idx, interactions in enumerate(traj_interactions):
                    for turn_idx, interaction in enumerate(interactions):
                        interaction.reward = (
                            turn_adv_by_traj[traj_idx][turn_idx]
                            + adv_lambda * traj_adv[traj_idx]
                        )

            if adv_estimator == "grpo":
                rewards = [
                    float(interaction.reward)
                    for interactions in traj_interactions
                    for interaction in interactions
                ]
                adv = _grpo_norm(rewards)
                offset = 0
                for interactions in traj_interactions:
                    for interaction in interactions:
                        interaction.reward = float(adv[offset])
                        offset += 1

            if self.dump_dir is not None:
                version_dir = os.path.join(self.dump_dir, str(engine.get_version()))
                os.makedirs(version_dir, exist_ok=True)
                safe_instance_id = str(instance_id).replace(os.sep, "_")
                for traj_idx, interactions in enumerate(traj_interactions):
                    turn_records = run_agent_results[traj_idx]["turn_records"]
                    final_memory_state = run_agent_results[traj_idx][
                        "final_memory_state"
                    ]
                    final_memory_total_length = run_agent_results[traj_idx][
                        "final_memory_total_length"
                    ]
                    traj_adv_value = None
                    turn_adv_list = None
                    if adv_estimator == "gigpo":
                        traj_adv_value = traj_adv[traj_idx]
                        turn_adv_list = turn_adv_by_traj[traj_idx]
                    elif adv_estimator == "grpo":
                        traj_adv_value = [
                            float(interaction.reward) for interaction in interactions
                        ]
                    dump_payload = {
                        "instance_id": instance_id,
                        "traj_id": traj_idx,
                        "data_source": data.get("data_source"),
                        "chunks_count": len(data.get("chunks", [])),
                        "adv_estimator": adv_estimator,
                        "memory_state": final_memory_state,
                        "memory_total_length": final_memory_total_length,
                        "traj_adv": traj_adv_value,
                        "turn_adv": turn_adv_list,
                        "turns": turn_records,
                    }
                    dump_path = os.path.join(
                        version_dir, f"{safe_instance_id}_traj{traj_idx}.json"
                    )
                    with open(dump_path, "w") as handle:
                        json.dump(dump_payload, handle, ensure_ascii=True, indent=2)
        return completions_with_reward
