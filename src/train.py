from data_module.dataset import AggDataset
from data_module.dataloader import AggDataLoader
from model.attn_model import AttentionModel
from trainer.trainer import Trainer


def main():
    window_size = 3
    batch_size = 64

    dataset = AggDataset(k=window_size)
    dataloader = AggDataLoader(dataset, batch_size)

    model = AttentionModel(k=window_size)

    trainer = Trainer(
        dataloader=dataloader,
        model=model,
    )

    trainer.fit(verbose=True)


if __name__ == "__main__":
    main()
