import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset


from transformers import AutoModel, AutoTokenizer, logging
logging.set_verbosity_error()


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

BASE_PATHS = {
    "bert": "/home/share/ZYN/SYB/project/chuanboxue/model_final/model/Bert_chinese_base",
    "roberta": "/home/share/ZYN/SYB/project/chuanboxue/model_final/model/chinese-roberta"
}
TEACHER_WEIGHT_PATHS = {
    "bert": "./model_final/pt/bert_teacher/best_teacher.pt",
    "roberta": "./model_final/pt/roberta_teacher/best_teacher.pt"
}
STUDENT_PATH = "/home/share/ZYN/SYB/project/chuanboxue/model_final/model/chinese-roberta"
DATA_PATH = "/home/share/ZYN/SYB/project/TTFND_repo/data/weibo21/weibo21_std.csv"

BATCH_SIZE = 512
EPOCHS = 5
TEMP = 2.0
ALPHA = 0.5
DEVICE = "cuda"
MAX_LEN = 128
NUM_WORKERS = 8


class DistillDataset(TensorDataset):
    def __init__(self, df, s_tok, t_tok, max_len=128):
        self.text_col = "news_content" if "news_content" in df.columns else "content"
        content = df[self.text_col].fillna("").astype(str).tolist()
        labels = df["label"].values


        s_enc = s_tok(content, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        t_enc = t_tok(content, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")

        self.s_ids = s_enc["input_ids"]
        self.s_mask = s_enc["attention_mask"]
        self.t_ids = t_enc["input_ids"]
        self.t_mask = t_enc["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {
            "s_ids": self.s_ids[idx], "s_mask": self.s_mask[idx],
            "t_ids": self.t_ids[idx], "t_mask": self.t_mask[idx],
            "label": self.labels[idx]
        }

class TeacherNet(nn.Module):
    def __init__(self, ckpt_path, base_path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_path)
        self.fc = nn.Linear(768, 2)

        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location='cpu')
            if isinstance(sd, nn.Module): sd = sd.state_dict()
            new_sd = {}
            for k, v in sd.items():
                k = k.replace("module.", "")
                if "classifier" in k: k = k.replace("classifier", "fc")
                if "logits_proj" in k: k = k.replace("logits_proj", "fc")
                new_sd[k] = v
            self.load_state_dict(new_sd, strict=False)
        else:
            print(f"  ❌ File not found: {ckpt_path}")

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        feat = out.last_hidden_state[:, 0, :]
        return self.fc(feat)

class StudentNet(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        self.fc = nn.Linear(768, 2)
    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        return self.fc(out.last_hidden_state[:, 0, :])

def calculate_metrics(trues, preds, probs):
    acc = accuracy_score(trues, preds)
    try:
        auc = roc_auc_score(trues, probs)
    except:
        auc = 0.0

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(trues, preds, average='macro')
    p_none, r_none, f1_none, _ = precision_recall_fscore_support(trues, preds, average=None)


    return {
        "macro_f1": f1_macro,
        "accuracy": acc,
        "AUC": auc,
        "f1_real": f1_none[0],
        "f1_fake": f1_none[1],
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_fake": p_none[1],
        "recall_fake": r_none[1],
        "precision_real": p_none[0],
        "recall_real": r_none[0]
    }

def print_metrics(metrics, title="Metrics"):
    print(f"\n📊 {title}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("-" * 30)

def evaluate(model, loader, is_teacher=False):
    model.eval()
    preds, trues, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            prefix = "t" if is_teacher else "s"
            ids = batch[f"{prefix}_ids"].to(DEVICE, non_blocking=True)
            mask = batch[f"{prefix}_mask"].to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda'):
                logits = model(ids, mask)
                prob = torch.softmax(logits, dim=1)[:, 1]

            preds.extend(torch.argmax(logits, 1).cpu().numpy())
            trues.extend(batch["label"].numpy())
            probs.extend(prob.cpu().numpy())

    return calculate_metrics(trues, preds, probs)

def run_experiment(t_name):
    print(f"\n{'='*60}\n🚀 Experiment: {t_name.upper()} (Detailed Metrics)\n{'='*60}")


    df = pd.read_csv(DATA_PATH)

    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    s_tok = AutoTokenizer.from_pretrained(STUDENT_PATH)
    t_tok = AutoTokenizer.from_pretrained(BASE_PATHS[t_name])

    train_loader = DataLoader(
        DistillDataset(train_df, s_tok, t_tok),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4, persistent_workers=True
    )
    val_loader = DataLoader(
        DistillDataset(val_df, s_tok, t_tok),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4, persistent_workers=True
    )


    print(f"👉 [Step 1] Checking Teacher Quality ({t_name})...")
    teacher = TeacherNet(TEACHER_WEIGHT_PATHS[t_name], BASE_PATHS[t_name]).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False

    t_metrics = evaluate(teacher, val_loader, is_teacher=True)
    print_metrics(t_metrics, f"Teacher ({t_name}) Performance")


    print(f"👉 [Step 2] Distilling to Student (Roberta)...")
    student = StudentNet(STUDENT_PATH).to(DEVICE)
    opt = torch.optim.AdamW(student.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler('cuda')

    best_f1 = 0
    best_s_metrics = {}

    for ep in range(EPOCHS):
        student.train()
        pbar = tqdm(train_loader, desc=f"Ep {ep+1}/{EPOCHS}", leave=False)

        for batch in pbar:
            s_ids = batch["s_ids"].to(DEVICE, non_blocking=True)
            s_mask = batch["s_mask"].to(DEVICE, non_blocking=True)
            t_ids = batch["t_ids"].to(DEVICE, non_blocking=True)
            t_mask = batch["t_mask"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    t_logits = teacher(t_ids, t_mask)

            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                s_logits = student(s_ids, s_mask)
                loss_ce = nn.functional.cross_entropy(s_logits, labels)
                loss_kd = nn.functional.kl_div(
                    nn.functional.log_softmax(s_logits/TEMP, dim=1),
                    nn.functional.softmax(t_logits/TEMP, dim=1),
                    reduction='batchmean'
                ) * (TEMP**2)
                loss = (1-ALPHA)*loss_ce + ALPHA*loss_kd

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})


        s_metrics = evaluate(student, val_loader, is_teacher=False)


        if s_metrics['macro_f1'] > best_f1:
            best_f1 = s_metrics['macro_f1']
            best_s_metrics = s_metrics

        print(f"   Ep {ep+1} | Student F1: {s_metrics['macro_f1']:.4f} (Best: {best_f1:.4f})")

    print_metrics(best_s_metrics, f"Best Student (taught by {t_name})")
    return t_metrics, best_s_metrics

if __name__ == "__main__":
    print(f"🔥 A40 HIGH-PERFORMANCE MODE | SINGLE TEACHER DETAILED METRICS")

    for t in ["bert", "roberta"]:
        try:
            run_experiment(t)
        except Exception as e:
            print(f"❌ Error running {t}: {e}")
