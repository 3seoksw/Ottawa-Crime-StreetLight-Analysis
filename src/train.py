import argparse
import torch
from datetime import datetime

from data_module.dataset import AggDataset
from data_module.dataloader import AggDataLoader
from model.attn_model import AttentionModel
from trainer.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Crime prediction model")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f" {device} loaded\n")

    dataset = AggDataset()
    dataloader = AggDataLoader(dataset, args.batch_size)

    model = AttentionModel()

    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer = Trainer(
        dataloader=dataloader,
        model=model,
        lr=args.lr,
        device=device,
        log_dir=f"runs/{time}",
    )

    trainer.fit(max_epochs=args.epochs, verbose=True)


if __name__ == "__main__":
    main()
