import torch
import torch.nn as nn
import numpy as np
import shap
from data_module.dataloader import AggDataLoader
from model.attn_model import AttentionModel
from model.shap_wrapper import ClsWrapper, CountWrapper
from trainer.utils import (
    plot_confusion_matrix,
    plot_attention_heatmap,
    plot_training_results,
    plot_performance,
    plot_shap,
)
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix


class Trainer:
    def __init__(
        self,
        dataloader: AggDataLoader,
        model: AttentionModel,
        lr: float = 1e-3,
        log_every: int = 50,
        eval_every: int = 100,
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

        self.background = []
        self.test_samples = []

        self.count_loss_type = count_loss_type
        self._assign_loss_functions(count_loss_type)

        self.log_every = log_every
        self.eval_every = eval_every
        self.log_dir = log_dir
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
        y_cls = (y > 0).float()  # class_0 .. zero crime & class_1 .. nonzero_crimes
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

    def compute_accuracy(
        self, y: torch.Tensor, is_nonzero: torch.Tensor, y_pred: torch.Tensor
    ):
        zero_mask = y == 0
        nonzero_mask = y > 0

        pred_cls = is_nonzero > 0

        zero_correct = (pred_cls[zero_mask] == 0).sum()

        pred_count = torch.round(torch.exp(y_pred))
        nonzero_correct = (pred_count[nonzero_mask] == y[nonzero_mask]).sum()

        correct = zero_correct + nonzero_correct
        acc = correct.float() / y.numel()
        return acc.item()

    def train_batch(self, batch: tuple[torch.Tensor, torch.Tensor]):
        self.optimizer.zero_grad()
        X, y = self._device_to(batch)

        if len(self.background) < 100:
            self.background.append(X.detach().cpu())

        is_nonzero, pred_y, _ = self.model(X)
        acc = self.compute_accuracy(y, is_nonzero, pred_y)
        loss_cls, loss_count = self.loss_fn(y, is_nonzero, pred_y)
        loss = loss_cls + loss_count
        loss.backward()
        self.optimizer.step()
        return loss_cls, loss_count, acc

    def fit(self, max_epochs: int = 5, verbose: bool = False):
        for e in range(max_epochs):
            for i, batch in enumerate(self.train_loader):
                self.model.train()
                loss_cls, loss_count, acc = self.train_batch(batch)
                if i % self.log_every == 0:
                    loss_cls = loss_cls.item()
                    loss_count = loss_count.item()
                    loss_sum = loss_cls + loss_count
                    self.writer.add_scalar("train/loss_cls", loss_cls, self.global_step)
                    self.writer.add_scalar(
                        "train/loss_count", loss_count, self.global_step
                    )
                    self.writer.add_scalar("train/loss_sum", loss_sum, self.global_step)
                    self.writer.add_scalar("train/acc", acc, self.global_step)
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
        accuracies = []
        with torch.no_grad():
            for batch in self.val_loader:
                X, y = self._device_to(batch)
                is_nonzero, count, _ = self.model.predict(X)
                acc = self.compute_accuracy(y, is_nonzero, count)
                loss_cls, loss_count = self.loss_fn(y, is_nonzero, count)

                losses_cls.append(loss_cls.item())
                losses_count.append(loss_count.item())
                accuracies.append(acc)

        avg_loss_cls = np.mean(losses_cls)
        avg_loss_count = np.mean(losses_count)
        avg_accuracy = np.mean(accuracies)

        self.writer.add_scalar("val/loss_cls", avg_loss_cls, self.global_step)
        self.writer.add_scalar("val/loss_count", avg_loss_count, self.global_step)
        self.writer.add_scalar("val/acc", avg_accuracy, self.global_step)
        return avg_loss_cls, avg_loss_count

    def test(self):
        self.model.eval()
        trues_list = []
        preds_list = []
        attn_weights_list = []
        accuracies = []
        losses_cls = []
        losses_count = []
        with torch.no_grad():
            for _, batch in enumerate(self.test_loader):
                X, y = self._device_to(batch)
                self.test_samples.append(X.detach().cpu())
                is_nonzero, count, attn_weights = self.model.predict(X)

                acc = self.compute_accuracy(y, is_nonzero, count)
                accuracies.append(acc)

                attn_weights_list.append(attn_weights)

                # Loss
                loss_cls, loss_count = self.loss_fn(y, is_nonzero, count)
                loss_cls = loss_cls.item()
                loss_count = loss_count.item()

                losses_cls.append(loss_cls)
                losses_count.append(loss_count)

                # Confusion Matrix
                probs = torch.sigmoid(is_nonzero)
                pred = (probs > 0.5).int()
                preds_list.append(pred)

                true = (y > 0).int()
                trues_list.append(true)

        # Attention Map
        attn_weights = torch.cat(attn_weights_list, dim=0)
        attn_weights_per_head = attn_weights.mean(dim=0)
        attn_weights_per_head = attn_weights_per_head.cpu().numpy()

        # Loss
        loss_cls = np.mean(losses_cls)
        loss_count = np.mean(losses_count)
        self.writer.add_scalar("test/loss_cls", loss_cls, self.global_step)
        self.writer.add_scalar("test/loss_count", loss_count, self.global_step)

        # Accuracy
        accuracy = np.mean(accuracies)
        self.writer.add_scalar("test/acc", accuracy, self.global_step)

        trues = torch.cat(trues_list).cpu().numpy()
        preds = torch.cat(preds_list).cpu().numpy()
        cm = confusion_matrix(trues, preds)

        self._save_weights(attn_weights_per_head)
        self.writer.close()

        # Plotting
        plot_confusion_matrix(cm, True, self.log_dir)
        plot_confusion_matrix(cm, False, self.log_dir)
        plot_attention_heatmap(
            attn_weights_per_head, self.dataloader.features, self.log_dir
        )
        plot_training_results(self.log_dir)
        plot_performance(self.log_dir)
        self.summarize_shap()

        return cm, attn_weights_per_head

    def summarize_shap(self):
        background = torch.cat(self.background, dim=0)[:100].cpu()
        test_samples = torch.cat(self.test_samples, dim=0).cpu()
        features = self.dataloader.features

        cls_wrap = ClsWrapper(self.model).to("cpu")
        is_nonzero_explainer = shap.GradientExplainer(cls_wrap, background)
        plot_shap(test_samples, is_nonzero_explainer, features, self.log_dir, "cls")

        count_wrap = CountWrapper(self.model).to("cpu")
        count_explainer = shap.GradientExplainer(count_wrap, background)
        plot_shap(test_samples, count_explainer, features, self.log_dir, "count")

    def _save_weights(self, attn_weights_per_head: np.ndarray):
        torch.save(self.model.state_dict(), f"{self.log_dir}/model.pth")
        np.save(f"{self.log_dir}/attn_weights.npy", attn_weights_per_head)

    def _device_to(self, batch: tuple[torch.Tensor, torch.Tensor]):
        X, y = batch
        return X.to(self.device), y.to(self.device)
