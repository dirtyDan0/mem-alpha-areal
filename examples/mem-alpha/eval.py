import os
import re
import sys
from pathlib import Path

import torch.distributed as dist
from tqdm import tqdm

from areal.api.cli_args import load_expr_config
from areal.api.io_struct import FinetuneSpec
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils import seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.printing import tabulate_stats
from areal.utils.stats_logger import StatsLogger

try:  # Package-style relative import (works if executed via -m with package context)
    from .memalpha import MemAlphaWorkflow  # type: ignore
    from .utils import (  # type: ignore
        MemAlphaConfig,
        get_memalpha_dataset,
        resolve_external_engine,
        workflow_dump_dir,
    )
except ImportError:  # Fallback when executed directly (no package parent known)
    module_dir = Path(__file__).parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    from memalpha import MemAlphaWorkflow  # type: ignore  # noqa: E402
    from utils import (  # type: ignore  # noqa: E402
        MemAlphaConfig,
        get_memalpha_dataset,
        resolve_external_engine,
        workflow_dump_dir,
    )

def main(args: list[str]) -> None:
    config, _ = load_expr_config(args, MemAlphaConfig)

    if config.valid_dataset is None:
        raise ValueError("valid_dataset must be set for eval.")

    use_dist = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if use_dist:
        dist.init_process_group("gloo")
        group = dist.new_group()
        rank = int(os.getenv("RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
    else:
        group = None
        rank = 0
        world_size = 1

    seeding.set_random_seed(config.seed, key=f"eval{rank}")
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    valid_dataset = get_memalpha_dataset(config.valid_dataset.path, split="test")
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=rank,
        world_size=world_size,
        dataset_config=config.valid_dataset,
    )

    external_engine = None
    eval_rollout = None
    try:
        external_engine = resolve_external_engine(config)

        eval_rollout = RemoteSGLangEngine(config.rollout)
        eval_rollout.config.max_head_offpolicyness = int(1e12)
        eval_rollout.initialize()

        eval_workflow = MemAlphaWorkflow(
            gconfig=config.gconfig,
            tokenizer=tokenizer,
            rollout_stat_scope="eval-rollout",
            dump_dir=workflow_dump_dir(config, "generated-eval"),
            external_engine=external_engine,
        )

        cnt = 0
        total_items = len(valid_dataset)
        for batch in tqdm(valid_dataloader, total=len(valid_dataloader), desc="Eval"):
            for item in batch:
                eval_rollout.submit(item, eval_workflow)
                cnt += 1
            if cnt >= total_items:
                break
        eval_rollout.wait(cnt, timeout=None)

        eval_stats = stats_tracker.export_all(reduce_group=group)
        print(f"Evaluation results:\n{tabulate_stats(eval_stats)}")
        basename = os.path.basename(config.actor.path)
        epoch_match = re.search(r"epoch(\d+)", basename)
        epoch_step_match = re.search(r"epochstep(\d+)", basename)
        global_step_match = re.search(r"globalstep(\d+)", basename)
        epoch = int(epoch_match.group(1)) if epoch_match else 0
        epoch_step = (
            int(epoch_step_match.group(1)) + 1 if epoch_step_match else 0
        )
        global_step = (
            int(global_step_match.group(1)) + 1 if global_step_match else 0
        )
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=len(valid_dataset),
            train_batch_size=config.valid_dataset.batch_size,
        )
        stats_logger = StatsLogger(config, ft_spec)
        stats_logger.commit(
            epoch=epoch,
            step=epoch_step,
            global_step=global_step,
            data=eval_stats,
        )
        stats_logger.close()
    finally:
        if eval_rollout is not None:
            eval_rollout.destroy()
        if external_engine is not None:
            external_engine.destroy()
        if use_dist:
            dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
