import argparse
import os
import sys
import yaml
import torch
import numpy as np
import random
from pathlib import Path

try:
    from dataset_loader_pheme import DatasetLoader
    from model import Model
    from trainer import Trainer
    from trainer_twstd import TrainerTWStd
    from evaluation import Evaluation
    from utils import set_seed
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)

def load_config(config_path):
    with open(config_path, "r") as f: return yaml.safe_load(f)

def build_tensor_data(df):
    c_ids = torch.stack([x["input_ids"].squeeze(0) for x in df["encoded_content"]])
    c_mask = torch.stack([x["attention_mask"].squeeze(0) for x in df["encoded_content"]])
    if "encoded_comments" in df.columns and df["encoded_comments"].iloc[0] is not None:
        try:
            k_ids = torch.stack([x["input_ids"].squeeze(0) for x in df["encoded_comments"]])
            k_mask = torch.stack([x["attention_mask"].squeeze(0) for x in df["encoded_comments"]])
        except: k_ids, k_mask = None, None
    else: k_ids, k_mask = None, None
    labels = torch.tensor(df["label"].values).long()
    if k_ids is None: k_ids = c_ids; k_mask = c_mask
    return c_ids, c_mask, k_ids, k_mask, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--training_alpha", type=float)
    parser.add_argument("--training_learning_rate_student", type=float)
    parser.add_argument("--training_rl_gamma_reward", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--gpu", type=str)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")

    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["dataset"] = args.dataset
    cfg["seed"] = args.seed

    if "training" not in cfg: cfg["training"] = {}
    if args.training_alpha is not None: cfg["training"]["alpha"] = args.training_alpha
    if args.training_learning_rate_student is not None: cfg["training"]["learning_rate_student"] = args.training_learning_rate_student
    if args.epochs is not None: cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None: cfg["training"]["batch_size"] = args.batch_size

    device_str = f"cuda:{args.gpu}" if args.gpu else args.device
    cfg["training"]["device"] = device_str
    cfg["training"]["save_dir"] = args.save_dir

    print(f"\n=== Running {args.dataset} on {device_str} | Seed: {args.seed} ===")
    set_seed(args.seed)
    loader = DatasetLoader(cfg)
    datasets = loader.load_data()

    if args.dataset not in datasets:
        keys = list(datasets.keys())
        if len(keys) > 0:
            print(f"[Info] Dataset key '{args.dataset}' not found, using '{keys[0]}'")
            train_data = build_tensor_data(datasets[keys[0]]["train"])
            val_data = build_tensor_data(datasets[keys[0]]["validation"])
            test_data = build_tensor_data(datasets[keys[0]]["test"])
        else: raise ValueError("Dataset loader returned empty dict.")
    else:
        train_data = build_tensor_data(datasets[args.dataset]["train"])
        val_data = build_tensor_data(datasets[args.dataset]["validation"])
        test_data = build_tensor_data(datasets[args.dataset]["test"])

    model = Model(cfg).to(torch.device(device_str))
    model.load_teachers_from_config()
    trainer = Trainer(model, cfg, train_data, val_data)
    trainer.train()
    evaluator = Evaluation(model, cfg, test_data)
    res = evaluator.evaluate()
    print(f"\n[Result] {args.dataset} Metrics: {res}")
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    main()
