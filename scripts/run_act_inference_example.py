#!/usr/bin/env python
"""Run one ACT policy inference step on a local LeRobot dataset sample."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = str(
    Path(os.environ.get("ACT_CHECKPOINT", REPO_ROOT / "checkpoints" / "act_100000" / "pretrained_model"))
)
DEFAULT_DATASET_ROOT = str(Path(os.environ.get("DATASET_ROOT", REPO_ROOT / "cup_to_sink")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default="yxzhan/cup_to_sink")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        video_backend="pyav",
    )

    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint
    cfg.device = args.device

    policy = make_policy(cfg, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(checkpoint),
    )
    policy.eval()
    policy.reset()

    item = dataset[args.sample_index]
    observation = {
        key: value.unsqueeze(0).to(args.device)
        for key, value in item.items()
        if key.startswith("observation.")
    }
    observation["task"] = item.get("task", "")

    with torch.inference_mode():
        observation = preprocessor(observation)
        action = policy.select_action(observation)
        action = postprocessor(action)

    predicted = make_robot_action(action, dataset.features)
    print("Predicted action:")
    for name, value in predicted.items():
        print(f"  {name}: {value:.6f}")

    if "action" in item:
        demo_action = make_robot_action(item["action"].unsqueeze(0), dataset.features)
        print("\nDemo action at the same frame:")
        for name, value in demo_action.items():
            print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
