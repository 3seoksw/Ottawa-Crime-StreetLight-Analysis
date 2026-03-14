import torch
import torch.nn as nn
import numpy as np
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
        count_loss_type: str = "Poisson",
    ):
        self.dataloader = dataloader
        self.pos_weight = self.dataloader.pos_weight
        self.train_loader = dataloader.train_loader
        self.val_loader = dataloader.val_loader
        self.test_loader = dataloader.test_loader

        self.device = device
        self.model = model
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.count_loss_type = count_loss_type
        self._assign_loss_functions(count_loss_type)

        self.eval_every = eval_every
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

    def _assign_loss_functions(self, count_loss_type: str):
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        if count_loss_type == "MSE":
            self.count_loss_fn = nn.MSELoss()
        elif count_loss_type == "Poisson":
            self.count_loss_fn = nn.PoissonNLLLoss(log_input=True)
        else:
            raise ValueError(f"Unavailable loss type: {self.count_loss_type}")

    def loss_fn(self, y: torch.Tensor, is_nonzero: torch.Tensor, y_pred: torch.Tensor):
        # Binary Cross Entropy Loss (nonzero or zero crime count)
        y_cls = (y > 0).float()
        loss_cls = self.cls_loss_fn(is_nonzero, y_cls)

        # MSE or Poisson NLL Loss (crime count when nonzero)
        nonzero_mask = y > 0
        if self.count_loss_type == "MSE":
            y_log = torch.log1p(y.float())
            if nonzero_mask.any():
                loss_count = self.count_loss_fn(
                    y_pred[nonzero_mask], y_log[nonzero_mask]
                )
            else:
                loss_count = torch.zeros((), device=y.device, dtype=torch.float32)
        else:  # self.count_loss_type == "Poisson"
            loss_count = self.count_loss_fn(y_pred, y.float())
        return loss_cls, loss_count

    def train_batch(self, batch: tuple[torch.Tensor, torch.Tensor]):
        self.optimizer.zero_grad()
        X, y = self._device_to(batch)
        is_nonzero, pred_y, _ = self.model(X)
        loss_cls, loss_count = self.loss_fn(y, is_nonzero, pred_y)
        loss = loss_cls + loss_count
        loss.backward()
        self.optimizer.step()
        return loss_cls, loss_count

    def fit(self, max_epochs: int = 500, verbose: bool = False):
        for e in range(max_epochs):
            for i, batch in enumerate(self.train_loader):
                self.model.train()
                loss_cls, loss_count = self.train_batch(batch)
                if i % 20 == 0:
                    loss_cls = loss_cls.item()
                    loss_count = loss_count.item()
                    loss_sum = loss_cls + loss_count
                    self.writer.add_scalar("loss/loss_cls", loss_cls, self.global_step)
                    self.writer.add_scalar(
                        "loss/loss_count", loss_count, self.global_step
                    )
                    self.writer.add_scalar("loss/loss_sum", loss_sum, self.global_step)
                    if verbose:
                        print(f" [Epoch {e} - {i}] {self.global_step}")
                        print(f"  loss_cls: {loss_cls:.3f}")
                        print(f"  loss_count: {loss_count:.3f}")

                if i % self.eval_every == 0:
                    avg_loss_cls, avg_loss_count = self.validate()
                    if verbose:
                        print(f"\tval_avg_loss_cls: {avg_loss_cls:.3f}")
                        print(f"\tval_avg_loss_count: {avg_loss_count:.3f}")

                self.global_step += 1

    def validate(self):
        self.model.eval()
        losses_cls = []
        losses_count = []
        with torch.no_grad():
            for batch in self.val_loader:
                X, y = self._device_to(batch)
                is_nonzero, count, _ = self.model.predict(X)
                loss_cls, loss_count = self.loss_fn(y, is_nonzero, count)
                losses_cls.append(loss_cls.item())
                losses_count.append(loss_count.item())

        avg_loss_cls = np.mean(losses_cls)
        avg_loss_count = np.mean(losses_count)
        self.writer.add_scalar("val/loss_cls", avg_loss_cls, self.global_step)
        self.writer.add_scalar("val/loss_count", avg_loss_count, self.global_step)
        return avg_loss_cls, avg_loss_count

    def _device_to(self, batch: tuple[torch.Tensor, torch.Tensor]):
        X, y = batch
        return X.to(self.device), y.to(self.device)
