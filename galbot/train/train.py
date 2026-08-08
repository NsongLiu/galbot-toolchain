"""Single-dataset LeRobot training entry point for Galbot G1.

Usage::

    PYTHONPATH=. python -m galbot.train.train --config galbot/train/configs/act_example.json

or via the provided runner::

    ./scripts/galbot-run galbot.train.train --config galbot/train/configs/act_example.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from pprint import pformat

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from termcolor import colored
from tqdm import tqdm

# Apply Python 3.12 compatibility patch before any LeRobot policy imports.
from galbot.train._compat import apply_lerobot_compat_patch

apply_lerobot_compat_patch()

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import IMAGENET_STATS, resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.processor import PolicyProcessorPipeline, RenameObservationsProcessorStep
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_training_state,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, init_logging, inside_slurm

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_DIR = "pretrained_model"


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _apply_policy_overrides(pol_cfg: PreTrainedConfig, policy_cfg: dict) -> None:
    """Apply optional policy fields from training JSON (e.g. pi05 dtype, chunk_size)."""
    if "normalization_mapping" in policy_cfg:
        pol_cfg.normalization_mapping = {
            key: NormalizationMode(value)
            for key, value in policy_cfg["normalization_mapping"].items()
        }

    reserved = {"type", "path", "push_to_hub", "normalization_mapping", "tokenizer_path"}
    for key, value in policy_cfg.items():
        if key in reserved:
            continue
        if hasattr(pol_cfg, key):
            setattr(pol_cfg, key, value)


def _save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    policy,
    optimizer,
    scheduler,
    preprocessor,
    postprocessor,
    train_cfg: dict,
) -> None:
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir)

    with open(pretrained_dir / "train_config.json", "w") as f:
        json.dump(train_cfg, f, indent=2)

    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)

    save_training_state(checkpoint_dir, step, optimizer, scheduler)


def train(cfg: dict) -> None:
    dataset_cfg = cfg["dataset"]
    policy_cfg = cfg["policy"]
    train_cfg = cfg["training"]
    use_imagenet_stats = cfg.get("use_imagenet_stats", True)
    video_backend = cfg.get("video_backend", None)

    output_dir = Path(train_cfg["output_dir"])
    steps = train_cfg["steps"]
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg.get("num_workers", 4)
    save_freq = train_cfg.get("save_freq", 20_000)
    log_freq = train_cfg.get("log_freq", 200)
    seed = train_cfg.get("seed", 1000)
    resume = train_cfg.get("resume", False)
    resume_dir = train_cfg.get("resume_dir", None)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )
    init_logging(accelerator=accelerator)
    is_main = accelerator.is_main_process

    if is_main:
        logging.info("Training config:\n%s", pformat(cfg))

    set_seed(seed, accelerator=accelerator)
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # --- dataset ---
    if is_main:
        logging.info("Loading dataset %s from %s", dataset_cfg["repo_id"], dataset_cfg["root"])

    if policy_cfg.get("path"):
        pol_cfg = PreTrainedConfig.from_pretrained(policy_cfg["path"])
    else:
        pol_cfg = make_policy_config(policy_cfg["type"])

    # Apply JSON policy overrides (e.g. pi05 chunk_size, dtype, gradient_checkpointing).
    _apply_policy_overrides(pol_cfg, policy_cfg)

    ds_meta = LeRobotDatasetMetadata(dataset_cfg["repo_id"], root=dataset_cfg["root"])
    delta_timestamps = resolve_delta_timestamps(pol_cfg, ds_meta)

    ds = LeRobotDataset(
        dataset_cfg["repo_id"],
        root=dataset_cfg["root"],
        episodes=dataset_cfg.get("episodes"),
        delta_timestamps=delta_timestamps,
        video_backend=video_backend,
    )

    accelerator.wait_for_everyone()

    if use_imagenet_stats:
        for key in ds.meta.camera_keys:
            for stat_name, stat_val in IMAGENET_STATS.items():
                ds.meta.stats[key][stat_name] = torch.tensor(stat_val, dtype=torch.float32)

    if is_main:
        logging.info("Dataset: %s", ds)

    # --- policy ---
    if is_main:
        logging.info("Creating policy (type=%s)", policy_cfg["type"])

    pol_cfg.push_to_hub = policy_cfg.get("push_to_hub", False)
    if policy_cfg.get("path"):
        pol_cfg.pretrained_path = Path(policy_cfg["path"])

    policy = make_policy(cfg=pol_cfg, ds_meta=ds.meta, rename_map=policy_cfg.get("rename_map"))

    peft_cfg = cfg.get("peft")
    if peft_cfg:
        if is_main:
            logging.info("Wrapping policy with PEFT: %s", pformat(peft_cfg))
        policy = policy.wrap_with_peft(peft_cli_overrides=dict(peft_cfg))

    # --- pre/post processors ---
    processor_stats = ds.meta.stats
    rename_map = policy_cfg.get("rename_map")
    if isinstance(pol_cfg, PI05Config):
        # PI05 processors must be built from code because the upstream pi05_base
        # preprocessor config references steps not present in this lerobot version.
        pol_cfg.device = device.type
        # PaliGemma tokenizer: local dir path (offline) or HF hub name.
        # 以名称（而非对象）传入，使保存 checkpoint 时 tokenizer_name 写入
        # policy_preprocessor.json，部署端可自包含加载。
        tokenizer_path = policy_cfg.get("tokenizer_path", "google/paligemma-3b-pt-224")
        from lerobot.processor import (
            AddBatchDimensionProcessorStep,
            DeviceProcessorStep,
            NormalizerProcessorStep,
            PolicyAction,
            RenameObservationsProcessorStep,
            UnnormalizerProcessorStep,
        )
        from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
        from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
        from lerobot.processor.tokenizer_processor import TokenizerProcessorStep
        from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

        rename_map = rename_map or {}
        input_steps = [
            RenameObservationsProcessorStep(rename_map=rename_map),
            AddBatchDimensionProcessorStep(),
            NormalizerProcessorStep(
                features={**pol_cfg.input_features, **pol_cfg.output_features},
                norm_map=pol_cfg.normalization_mapping,
                stats=processor_stats,
            ),
            Pi05PrepareStateTokenizerProcessorStep(max_state_dim=pol_cfg.max_state_dim),
            TokenizerProcessorStep(
                tokenizer_name=tokenizer_path,
                max_length=pol_cfg.tokenizer_max_length,
                padding_side="right",
                padding="max_length",
            ),
            DeviceProcessorStep(device=device.type),
        ]
        preprocessor = PolicyProcessorPipeline(
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        )
        postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=[
                UnnormalizerProcessorStep(
                    features=pol_cfg.output_features,
                    norm_map=pol_cfg.normalization_mapping,
                    stats=processor_stats,
                ),
                DeviceProcessorStep(device="cpu"),
            ],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
    else:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=pol_cfg,
            pretrained_path=pol_cfg.pretrained_path if hasattr(pol_cfg, "pretrained_path") else None,
            dataset_stats=processor_stats,
        )

    # --- optimizer / scheduler ---
    optimizer_cfg = pol_cfg.get_optimizer_preset()
    scheduler_cfg = pol_cfg.get_scheduler_preset()
    params = policy.get_optim_params() if hasattr(policy, "get_optim_params") else policy.parameters()
    optimizer = optimizer_cfg.build(params)
    lr_scheduler = scheduler_cfg.build(optimizer, steps) if scheduler_cfg else None
    grad_clip_norm = getattr(optimizer_cfg, "grad_clip_norm", 10.0)

    step = 0
    if resume and resume_dir:
        step, optimizer, lr_scheduler = load_training_state(Path(resume_dir), optimizer, lr_scheduler)
        if is_main:
            logging.info("Resumed from step %d", step)

    # --- info ---
    num_learnable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in policy.parameters())
    if is_main:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")
        logging.info("steps=%d (%s)", steps, format_big_number(steps))
        logging.info("num_frames=%d (%s)", ds.num_frames, format_big_number(ds.num_frames))
        logging.info("num_episodes=%d", ds.num_episodes)
        eff_bs = batch_size * accelerator.num_processes
        logging.info("Effective batch size: %d x %d = %d", batch_size, accelerator.num_processes, eff_bs)
        logging.info("num_learnable_params=%d (%s)", num_learnable, format_big_number(num_learnable))
        logging.info("num_total_params=%d (%s)", num_total, format_big_number(num_total))

    # --- dataloader ---
    dataloader = torch.utils.data.DataLoader(
        ds,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)
    policy.train()

    # --- metrics ---
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        batch_size,
        ds.num_frames,
        ds.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    # --- wandb (optional) ---
    wandb_cfg = cfg.get("wandb") or {}
    wandb_run = None
    if wandb_cfg.get("enable") and is_main:
        import wandb

        wandb_run = wandb.init(
            project=wandb_cfg.get("project", "galbot"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name") or output_dir.name,
            notes=wandb_cfg.get("notes"),
            mode=wandb_cfg.get("mode", "online"),
            dir=str(output_dir),
            config=cfg,
        )
        logging.info("Track this run --> %s", wandb_run.get_url())

    if is_main:
        progbar = tqdm(
            total=steps - step,
            desc="Training",
            unit="step",
            disable=inside_slurm(),
            position=0,
            leave=True,
        )
        logging.info("Start training, effective batch size: %d", batch_size * accelerator.num_processes)

    # --- loop ---
    for _ in range(step, steps):
        t0 = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - t0

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        step += 1
        if is_main:
            progbar.update(1)
        train_tracker.step()

        is_log = log_freq > 0 and step % log_freq == 0 and is_main
        is_save = step % save_freq == 0 or step == steps

        if is_log:
            logging.info(train_tracker)
            if wandb_run is not None:
                wandb_run.log({**train_tracker.to_dict(), **(output_dict or {})}, step=step)
            train_tracker.reset_averages()

        if is_save:
            # 仅主进程写盘；所有进程都参与 barrier，否则非主进程在下一个 DDP
            # 集合通信中等待主进程，主进程又在 barrier 等它们，造成 NCCL 超时死锁。
            if is_main:
                logging.info("Saving checkpoint at step %d", step)
                ckpt_dir = get_step_checkpoint_dir(output_dir, steps, step)
                _save_checkpoint(
                    ckpt_dir,
                    step,
                    accelerator.unwrap_model(policy),
                    optimizer,
                    lr_scheduler,
                    preprocessor,
                    postprocessor,
                    cfg,
                )
                update_last_checkpoint(ckpt_dir)
            accelerator.wait_for_everyone()

    if is_main:
        progbar.close()
        if wandb_run is not None:
            wandb_run.finish()
        logging.info("Training complete.")

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Galbot G1 LeRobot training")
    parser.add_argument("--config", required=True, help="Path to training JSON config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
