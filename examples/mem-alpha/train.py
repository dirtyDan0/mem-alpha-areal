import sys
from pathlib import Path

from areal.api.cli_args import load_expr_config
from areal.experimental.trainer import PPOTrainer


try:  # Package-style relative import (works if executed via -m with package context)
    from .memalpha import MemAlphaWorkflow  # type: ignore
    from .utils import (
        MemAlphaConfig,
        get_memalpha_dataset,
        resolve_external_engine,
        workflow_dump_dir,
    )  # type: ignore
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
    train_dataset = get_memalpha_dataset(config.train_dataset.path, split="train")
    external_engine = None
    try:
        external_engine = resolve_external_engine(config)
        with PPOTrainer(
            config,
            train_dataset=train_dataset,
            valid_dataset=None,
        ) as trainer:
            workflow = MemAlphaWorkflow(
                gconfig=config.gconfig,
                tokenizer=trainer.tokenizer,
                dump_dir=workflow_dump_dir(config, "generated"),
                external_engine=external_engine,
            )
            trainer.train(workflow, eval_workflow=None)
    finally:
        if external_engine is not None:
            external_engine.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
