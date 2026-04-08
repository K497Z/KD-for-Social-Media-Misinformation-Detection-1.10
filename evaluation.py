import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_fscore_support
import numpy as np
import torch.nn.functional as F

class Evaluation:
    def __init__(self, model, config, test_data):
        self.model = model
        self.config = config
        self.device = torch.device(config["training"]["device"])
        self.test_loader = DataLoader(TensorDataset(*test_data), batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)

    def evaluate(self):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in self.test_loader:
                batch = [b.to(self.device) for b in batch]
                c_ids, c_mask, k_ids, k_mask, labels = batch
                out = self.model(c_ids, c_mask, k_ids, k_mask)
                if isinstance(out, dict): logits = out["logits"]
                else: logits = out
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        try: auc = roc_auc_score(all_labels, all_probs)
        except: auc = 0.0
        return {"macro_f1": f1_macro, "accuracy": acc, "auc": auc}
