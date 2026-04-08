import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from utils import compute_classification_loss

class Trainer:
    def __init__(self, model, config, train_data, val_data):
        self.model = model
        self.config = config
        self.device = torch.device(config["training"]["device"])
        self.train_ds = TensorDataset(*train_data)
        self.val_ds = TensorDataset(*val_data)
        self.train_loader = DataLoader(self.train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
        if len(val_data[0]) > 0:
            self.val_loader = DataLoader(self.val_ds, batch_size=config["training"]["batch_size"], shuffle=False)
        else:
            self.val_loader = None
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(config["training"]["learning_rate_student"]))
        epochs = config["training"].get("epochs", 20)
        warmup_ep = config["training"].get("warmup_epochs", 0)
        num_steps = len(self.train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=len(self.train_loader)*warmup_ep, num_training_steps=num_steps)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch in self.train_loader:
            batch = [t.to(self.device) for t in batch]
            if len(batch) >= 5:
                c_ids, c_mask, k_ids, k_mask, labels = batch[:5]
            else:
                c_ids, c_mask, labels = batch[0], batch[1], batch[-1]
            out = self.model(c_ids, c_mask)
            if isinstance(out, dict): logits = out["logits"]
            else: logits = out
            loss_cls = compute_classification_loss(logits, labels)
            loss_kd = 0.0
            alpha = self.config['training'].get('alpha', 0.0)
            temp = self.config['training'].get('kd_temperature', 1.0)
            if alpha > 0 and hasattr(self.model, 'teacher_encoders') and len(self.model.teacher_encoders) > 0:
                with torch.no_grad():
                    t_logits_list = []
                    for name, t_model in self.model.teacher_encoders.items():
                        t_model = t_model.to(self.device)
                        t_out = t_model(c_ids, c_mask)
                        t_logits_list.append(t_out)
                    if t_logits_list:
                        t_mean = torch.stack(t_logits_list).mean(dim=0)
                        loss_kd = F.kl_div(F.log_softmax(logits / temp, dim=1), F.softmax(t_mean / temp, dim=1), reduction='batchmean') * (temp ** 2)
            loss = (1 - alpha) * loss_cls + alpha * loss_kd
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler: self.scheduler.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def train(self):
        epochs = self.config["training"].get("epochs", 20)
        for epoch in range(epochs):
            loss = self.train_epoch(epoch)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")
