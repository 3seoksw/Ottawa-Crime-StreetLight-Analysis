import torch
import torch.nn as nn
from data_module.dataloader import AggDataLoader
from model.attn_model import AttentionModel
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        dataloader: AggDataLoader,
        model: AttentionModel,
        lr: float = 1e-3,
        eval_every: int = 20,
        log_dir: str = "runs/",
        device: str = "cpu",
    ):
        self.dataloader = dataloader
        self.train_loader = dataloader.train_loader
        self.val_loader = dataloader.val_loader
        self.test_loader = dataloader.test_loader
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.pos_weight = self.dataloader.pos_weight
        self.eval_every = eval_every
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

    def loss_fn(self, y: torch.Tensor, is_nonzero: torch.Tensor, y_pred: torch.Tensor):
        # Binary Cross Entropy Loss (nonzero or zero crime count)
        y_cls = (y > 0).float()
        pos_weight = self.pos_weight.to(is_nonzero.device)
        loss_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(is_nonzero, y_cls)

        # MSE Loss (crime count when nonzero)
        nonzero_mask = y > 0
        y_log = torch.log1p(y.float())
        if nonzero_mask.any():
            loss_mse = nn.MSELoss()(y_log[nonzero_mask], y_pred[nonzero_mask])
        else:
            loss_mse = torch.zeros((), device=y.device, dtype=torch.float32)
        return loss_cls, loss_mse

    def train_batch(self, batch: tuple[torch.Tensor, torch.Tensor]):
        self.optimizer.zero_grad()
        X, y = self._device_to(batch)
        is_nonzero, pred_y, _ = self.model(X)
        loss_cls, loss_mse = self.loss_fn(y, is_nonzero, pred_y)
        loss = loss_cls + loss_mse
        loss.backward()
        self.optimizer.step()
        return loss_cls, loss_mse

    def fit(self, max_epochs: int = 500, verbose: bool = False):
        for e in range(max_epochs):
            for i, batch in enumerate(self.train_loader):
                self.model.train()
                loss_cls, loss_mse = self.train_batch(batch)
                if i % 20 == 0:
                    self.writer.add_scalar(
                        "loss/loss_cls", loss_cls.item(), self.global_step
                    )
                    self.writer.add_scalar(
                        "loss/loss_mse", loss_mse.item(), self.global_step
                    )
                    if verbose:
                        print(f" [Epoch {e} - {i}] {self.global_step}")
                        print(f"  loss_cls: {loss_cls.item():.3f}")
                        print(f"  loss_mse: {loss_mse.item():.3f}")

                if i % self.eval_every == 0:
                    avg_loss_cls, avg_loss_mse = self.validate()
                    if verbose:
                        print(f"\tval_avg_loss_cls: {avg_loss_cls:.3f}")
                        print(f"\tval_avg_loss_mse: {avg_loss_mse:.3f}")

                self.global_step += 1

    def validate(self):
        self.model.eval()
        total_loss_cls = torch.zeros((), dtype=torch.float32)
        total_loss_mse = torch.zeros((), dtype=torch.float32)

        with torch.no_grad():
            for batch in self.val_loader:
                X, y = self._device_to(batch)
                is_nonzero, count, _ = self.model.predict(X)
                loss_cls, loss_mse = self.loss_fn(y, is_nonzero, count)
                total_loss_cls += loss_cls.detach().cpu()
                total_loss_mse += loss_mse.detach().cpu()

        avg_loss_cls = total_loss_cls / len(self.val_loader)
        avg_loss_mse = total_loss_mse / len(self.val_loader)
        self.writer.add_scalar("val/loss_cls", avg_loss_cls.item(), self.global_step)
        self.writer.add_scalar("val/loss_mse", avg_loss_mse.item(), self.global_step)
        return avg_loss_cls.item(), avg_loss_mse.item()

    def _device_to(self, batch: tuple[torch.Tensor, torch.Tensor]):
        X, y = batch
        return X.to(self.device), y.to(self.device)

    # TODO: TESTING
