# main.py

import os
import sys
import argparse
import asyncio
import logging
logging.disable(logging.CRITICAL)




CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CHEFSHAT_SRC_PATH = os.path.join(CURRENT_DIR, "chefshatgym", "src")

if CHEFSHAT_SRC_PATH not in sys.path:
    sys.path.insert(0, CHEFSHAT_SRC_PATH)


from train import train
from evaluate import evaluate


def main():

    parser = argparse.ArgumentParser(
        description="Chef's Hat Sparse / Delayed Reward RL Variant"
    )

    #  Not required anymore
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",   #  DEFAULT MODE
        help="train (default) or eval",
    )

    args = parser.parse_args()

    print(f"Running mode: {args.mode}")

    if args.mode == "train":
        asyncio.run(train())
    elif args.mode == "eval":
        asyncio.run(evaluate())


if __name__ == "__main__":
    main()