"""Pre-train SARN (self-supervised) on a dataset and dump the embeddings.

This is the minimal entry point: it skips SARN's downstream-task machinery
(classify/trajsimi/spd), which the emission experiment does not need.  It just
runs the contrastive pre-training and dumps the frozen segment embeddings to
exp/snapshots/<prefix>_SARN_best_embs.pickle.

Usage (local):
    python pretrain_sarn_rome.py --dataset RM --epochs 200

Usage (Colab): set Config.dataset = 'RM' (or pass --dataset RM).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

sys.path.insert(0, '.')

from config import Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="RM")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    Config.dataset = args.dataset
    if args.epochs:
        Config.sarn_epochs = args.epochs
    if args.seed:
        Config.seed = args.seed
    Config.post_value_updates()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
        handlers=[logging.StreamHandler()],
    )

    from model.SARN import SARN  # noqa: E402

    t0 = time.time()
    enc = SARN()
    logging.info("loaded SARN @ %.1fs", time.time() - t0)

    enc.train()
    logging.info("pre-training done @ %.1fs", time.time() - t0)

    embs = enc.get_embeddings(True)
    enc.dump_embeddings(embs)
    logging.info("embeddings dumped @ %.1fs  shape=%s", time.time() - t0, tuple(embs.shape))


if __name__ == "__main__":
    main()
