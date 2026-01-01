import json
import os
from dataclasses import dataclass, field
from fnmatch import fnmatch

from datasets import load_dataset
import tiktoken
import yaml

from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, InferenceEngineConfig
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils import logging
from areal.utils.launcher import wait_llm_server_addrs
from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger(__name__)


@dataclass
class MemAlphaAgentConfig:
    beta: float = field(
        default=0.05,
        metadata={"help": "Weight of the memory compression reward (r3)."},
    )
    gamma: float = field(
        default=0.1,
        metadata={"help": "Weight of the external judge reward (r4)."},
    )
    adv_estimator: str = field(
        default="grpo",
        metadata={"help": "Advantage estimator: grpo or gigpo."},
    )
    adv_eps: float = field(
        default=1e-8,
        metadata={"help": "Epsilon for normalization."},
    )
    adv_lambda: float = field(
        default=1.0,
        metadata={"help": "Trajectory-level GiGPO weight."},
    )
    gigpo_gamma: float = field(
        default=1.0,
        metadata={"help": "Discount factor for turn returns in GiGPO."},
    )
    r1_max_completion_tokens: int = field(
        default=2048,
        metadata={"help": "Max tokens for r1 answer completions."},
    )
    r1_lme_max_completion_tokens: int = field(
        default=50,
        metadata={"help": "Max tokens for r1 LME judge completions."},
    )
    r4_max_completion_tokens: int = field(
        default=512,
        metadata={"help": "Max tokens for r4 judge completions."},
    )


@dataclass
class MemAlphaGenerationConfig(GenerationHyperparameters):
    agent: MemAlphaAgentConfig = field(
        default_factory=MemAlphaAgentConfig,
        metadata={"help": "MemAlpha reward hyperparameters."},
    )


@dataclass
class MemAlphaConfig(GRPOConfig):
    gconfig: MemAlphaGenerationConfig = field(default_factory=MemAlphaGenerationConfig)
    external_engine: InferenceEngineConfig = field(
        default_factory=InferenceEngineConfig,
        metadata={"help": "External engine for question answering and judging."},
    )


def resolve_external_engine(config: MemAlphaConfig) -> RemoteSGLangEngine:
    if (
        config.external_engine.experiment_name is None
        or config.external_engine.trial_name is None
    ):
        raise ValueError(
            "external_engine.experiment_name and external_engine.trial_name must be set."
        )
    external_addrs = wait_llm_server_addrs(
        experiment_name=config.external_engine.experiment_name,
        trial_name=config.external_engine.trial_name,
    )
    logger.info("External engine addresses: %s", ",".join(external_addrs))
    external_engine = RemoteSGLangEngine(config.external_engine)
    external_engine.config.max_head_offpolicyness = int(1e12)
    external_engine.initialize(addr=external_addrs)
    return external_engine


def workflow_dump_dir(config: MemAlphaConfig, suffix: str) -> str:
    return os.path.join(StatsLogger.get_log_path(config.stats_logger), suffix)


def get_memalpha_dataset(
    path: str,
    split: str,
):
    """return
    chunks,questions_and_answers,data_sources

    prompts would be formatted during rollout(each turn with builtin-templates, chunks and memory)
    """
    dataset = load_dataset(path=path, split=split).select_columns(
        ["instance_id", "chunks", "questions_and_answers", "data_source"]
    )
    prompts_path = os.path.join(
        os.path.dirname(__file__), "prompts_wrt_datasource.yaml"
    )
    with open(prompts_path, "r", encoding="utf-8") as handle:
        prompts_wrt_datasource = yaml.safe_load(handle) or {}

    def _resolve_query_prompt(data_source: str) -> str | None:
        prompt_entry = prompts_wrt_datasource.get(data_source)
        if isinstance(prompt_entry, dict):
            return prompt_entry.get("query_prompt")
        for key, value in prompts_wrt_datasource.items():
            if "*" in key and fnmatch(data_source, key):
                if isinstance(value, dict):
                    return value.get("query_prompt")
        return None

    def _maybe_prefix_questions(example: dict) -> dict:
        qa_list = json.loads(example["questions_and_answers"])
        query_prompt = _resolve_query_prompt(example["data_source"])
        if query_prompt:
            prefixed = []
            for qa in qa_list:
                if isinstance(qa, dict) and "question" in qa and "answer" in qa:
                    prefixed.append(
                        {
                            "question": f"{query_prompt}\n\n{qa['question']}",
                            "answer": qa["answer"],
                        }
                    )
                else:
                    prefixed.append(qa)
            qa_list = prefixed
        return {
            "instance_id": example["instance_id"],
            "chunks": json.loads(example["chunks"]),
            "questions_and_answers": qa_list,
            "data_source": example["data_source"],
        }

    dataset = dataset.map(_maybe_prefix_questions)

    return dataset


@dataclass
class TokenCounter:
    """Reusable token counter that caches a tiktoken encoding."""

    model: str = "gpt-4o-mini"

    def __post_init__(self):
        self._encoding = tiktoken.encoding_for_model(self.model)

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))


MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "new_memory_insert",
            "description": "Infer a new memory and append it to a memory store. Creates a new memory item with a unique ID. Note: Core memory cannot be inserted, only updated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "description": "Type of memory to insert: 'semantic' (general knowledge) or 'episodic' (specific experiences). Core memory cannot be inserted.",
                        "enum": ["semantic", "episodic"],
                    },
                    "content": {
                        "type": "string",
                        "description": "Content of the memory to insert. Creates a new memory item with a unique ID.",
                    },
                },
                "required": ["memory_type", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_update",
            "description": "Update an existing memory. For core memory, replaces the entire paragraph content. If core memory is empty, then directly write into the core memory. For semantic/episodic memories, updates the specific memory item by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "description": "Type of memory to update: 'core' (simple paragraph), 'semantic' (general knowledge), or 'episodic' (specific experiences)",
                        "enum": ["core", "semantic", "episodic"],
                    },
                    "new_content": {
                        "type": "string",
                        "description": "New **combined** content for the memory. For core memory, this replaces the entire paragraph. For semantic/episodic, this replaces the content of the specified memory ID.",
                    },
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory to update. Required for semantic/episodic memories, ignored for core memory. In the memory state display, this appears in brackets like [sem0001] or [epi0001].",
                    },
                },
                "required": ["memory_type", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": "Delete a memory. For core memory, clears the entire paragraph content. For semantic/episodic memories, deletes the specific memory item by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "description": "Type of memory to delete: 'core' (simple paragraph), 'semantic' (general knowledge), or 'episodic' (specific experiences)",
                        "enum": ["core", "semantic", "episodic"],
                    },
                    "memory_id": {
                        "type": "string",
                        "description": "ID of the memory to delete. Required for semantic/episodic memories, ignored for core memory. In the memory state display, this appears in brackets like [sem0001] or [epi0001].",
                    },
                },
                "required": ["memory_type"],
            },
        },
    },
]


class Memory:
    def __init__(self) -> None:
        self.core: str = ""
        self.semantic: Dict[str, str] = {}
        self.episodic: Dict[str, str] = {}
        self._semantic_counter: int = 0
        self._episodic_counter: int = 0

    def insert(
        self, memory_type: str, content: str | None, **kwargs
    ) -> tuple[str, str]:
        """Insert a new semantic/episodic memory. Returns (memory_type, memory_id)."""
        if not content:
            raise ValueError("content is required when inserting a memory item.")
        if memory_type == "semantic":
            self._semantic_counter += 1
            memory_id = f"sem_{self._semantic_counter:04d}"
            self.semantic[memory_id] = content
        elif memory_type == "episodic":
            self._episodic_counter += 1
            memory_id = f"epi_{self._episodic_counter:04d}"
            self.episodic[memory_id] = content
        else:
            raise ValueError("Core memory cannot be inserted, only updated.")
        return memory_type, memory_id

    def update(
        self,
        memory_type: str,
        new_content: str | None,
        memory_id: str | None = None,
        **kwargs,
    ) -> None:
        if not new_content:
            raise ValueError("new_content is required for memory_update.")
        if memory_type == "core":
            self.core = new_content
        elif memory_type == "semantic":
            if not memory_id:
                raise ValueError("memory_id is required when updating semantic memory.")
            if memory_id in self.semantic:
                self.semantic[memory_id] = new_content
            else:
                available_ids = sorted(self.semantic.keys())
                raise ValueError(
                    f"Semantic memory with ID {memory_id} not found. "
                    f"Available IDs: {available_ids}"
                )
        elif memory_type == "episodic":
            if not memory_id:
                raise ValueError("memory_id is required when updating episodic memory.")
            if memory_id in self.episodic:
                self.episodic[memory_id] = new_content
            else:
                available_ids = sorted(self.episodic.keys())
                raise ValueError(
                    f"Episodic memory with ID {memory_id} not found. "
                    f"Available IDs: {available_ids}"
                )
        else:
            raise ValueError("Invalid memory type.")

    def delete(
        self,
        memory_type: str,
        memory_id: str | None = None,
        **kwargs,
    ) -> None:
        if memory_type == "core":
            self.core = ""
        elif memory_type == "semantic":
            if not memory_id:
                raise ValueError("memory_id is required when deleting semantic memory.")
            if self.semantic.pop(memory_id, None) is None:
                available_ids = sorted(self.semantic.keys())
                raise ValueError(
                    f"Semantic memory with ID {memory_id} not found. "
                    f"Available IDs: {available_ids}"
                )
        elif memory_type == "episodic":
            if not memory_id:
                raise ValueError("memory_id is required when deleting episodic memory.")
            if self.episodic.pop(memory_id, None) is None:
                available_ids = sorted(self.episodic.keys())
                raise ValueError(
                    f"Episodic memory with ID {memory_id} not found. "
                    f"Available IDs: {available_ids}"
                )
        else:
            raise ValueError("Invalid memory type.")

    def export_memory_state(self) -> str:
        def _format_block(memory_type: str) -> str:
            if memory_type not in {
                "core",
                "semantic",
                "episodic",
            }:
                raise ValueError("Invalid memory type for block rendering.")

            if memory_type == "core":
                content = self.core.strip()
                if not content:
                    return "<core>\nEmpty.\n</core>"
                return f"<core>\n{content}\n</core>"

            memory_dict = (
                self.semantic if memory_type == "semantic" else self.episodic
            )
            if not memory_dict:
                return f"<{memory_type}>\nEmpty.\n</{memory_type}>"

            header = "| memory_id | content |"
            separator = "| --- | --- |"
            rows = []
            for mem_id, content in memory_dict.items():
                safe_content = content.replace("|", "\\|").replace("\n", " ")
                rows.append(f"| {mem_id} | {safe_content} |")
            body = "\n".join([header, separator, *rows])
            return f"<{memory_type}>\n{body}\n</{memory_type}>"

        blocks = [
            _format_block("core"),
            _format_block("semantic"),
            _format_block("episodic"),
        ]
        return "\n\n".join(blocks)

    def total_length(self, counter: TokenCounter) -> int:
        length = counter.count_tokens(self.core) if self.core else 0
        length += sum(counter.count_tokens(content) for content in self.semantic.values())
        length += sum(counter.count_tokens(content) for content in self.episodic.values())
        return length


MEMORY_TOOL_IMPL = {
    "new_memory_insert": lambda memory, **kwargs: memory.insert(**kwargs),
    "memory_update": lambda memory, **kwargs: memory.update(**kwargs),
    "memory_delete": lambda memory, **kwargs: memory.delete(**kwargs),
}
