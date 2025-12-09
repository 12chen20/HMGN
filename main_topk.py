import argparse
import os
import random
import numpy as np
import torch
from HMGN import train

PROFILES = {

    "movie20": dict(
        dataset="movie20", aggregator="sum",
        n_epochs=10, neighbor_sample_size=4, dim=16, n_iter=1,
        batch_size=4096, l2_weight=1e-6, lr=2e-2
    ),
    "restaurant": dict(
        dataset="restaurant", aggregator="sum",
        n_epochs=5, neighbor_sample_size=32, dim=16, n_iter=1,
        batch_size=4096, l2_weight=5e-6, lr=2e-2
    ),
    "music": dict(
        dataset="music", aggregator="concat",
        n_epochs=20, neighbor_sample_size=8, dim=16, n_iter=2,
        batch_size=32, l2_weight=1e-4, lr=5e-4
    ),
}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_parser():
    p = argparse.ArgumentParser(description="PIFSA-GNN runner")

    p.add_argument(
        "--profile",
        type=str,
        choices=list(PROFILES.keys()),
        default="music",
        help="choose a preset hyper-parameter profile"
    )


    p.add_argument(
        "--train_ratio",
        type=float,
        default=1.0,
        help="ratio of base_train actually used for training (0,1]"
    )


    p.add_argument("--dataset", type=str, default="music")
    p.add_argument("--aggregator", type=str, default="concat", choices=["sum", "concat", "mean"])
    p.add_argument("--n_epochs", type=int, default=20)
    p.add_argument("--neighbor_sample_size", type=int, default=8)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--n_iter", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--l2_weight", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=3407)


    p.add_argument("--users_per_eval", type=int, default=1000, help="for Hit@K user sampling")
    p.add_argument("--cand_size", type=int, default=50, help="for Hit@K; record only")
    p.add_argument("--neg_per_user", type=int, default=49, help="negative samples per user for Hit@K")

    return p

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()


    profile_args = PROFILES.get(args.profile, {})
    for k, v in profile_args.items():
        if not hasattr(args, k) or getattr(args, k) == parser.get_default(k):
            setattr(args, k, v)


    args.n_neg_eval = args.neg_per_user

    set_seed(args.seed)
    train(args)

