import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, logging


logging.set_verbosity_error()


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


BATCH_SIZE = 512
LR = 2e-5
ALPHA = 0.5
WARMUP = 4
EPOCHS = 20
DEVICE = "cuda"
NUM_WORKERS = 8


TEACHER_BASE_PATHS = {
    "bert": " model_final/model/Bert",
    "roberta": "model_final/model/  roberta"
}
CHECKPOINT_PATHS = {
    "bert": "model_final/pt/bert_teacher/best_teacher.pt",
    "roberta": "model_final/pt/roberta_teacher/best_teacher.pt"
}
STUDENT_PATH = " model/chinese-roberta"
DATA_PATH = "data/weibo21/weibo21.csv"
MAX_LEN = 128

class SOTADataset(Dataset):
    def __init__(self, df, tokenizer_dict, max_len=128):
        self.df = df
        self.tokenizer_dict = tokenizer_dict
        self.max_len = max_len
        self.text_col = "news_content" if "news_content" in df.columns else "content"
        self.texts = df[self.text_col].fillna("").astype(str).tolist()
        self.labels = df["label"].values

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        batch_data = {"label": torch.tensor(self.labels[idx], dtype=torch.long)}
        text = self.texts[idx]
        for name, tokenizer in self.tokenizer_dict.items():
            enc = tokenizer(text, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
            batch_data[f"{name}_ids"] = enc["input_ids"].squeeze(0)
            batch_data[f"{name}_mask"] = enc["attention_mask"].squeeze(0)
        return batch_data

class TeacherNet(nn.Module):
    def __init__(self, t_name):
        super().__init__()
        base_path = TEACHER_BASE_PATHS[t_name]
        self.encoder = AutoModel.from_pretrained(base_path)
        self.fc = nn.Linear(768, 2)

        ckpt_path = CHECKPOINT_PATHS.get(t_name)
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

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        feat = out.last_hidden_state[:, 0, :]
        return self.fc(feat)

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(STUDENT_PATH)
        self.fc = nn.Linear(768, 2)
    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        return self.fc(out.last_hidden_state[:, 0, :])

class AgentPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )
    def forward(self, s_logits, t_logits_list):
        return self.net(torch.cat([s_logits] + t_logits_list, dim=1))

def calculate_metrics(trues, preds, probs):
    acc = accuracy_score(trues, preds)
    try: auc = roc_auc_score(trues, probs)
    except: auc = 0.0

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(trues, preds, average='macro')
    p_none, r_none, f1_none, _ = precision_recall_fscore_support(trues, preds, average=None)

    return {
        "macro_f1": f1_macro,
        "accuracy": acc,
        "AUC": auc,
        "f1_real": f1_none[0],
        "f1_fake": f1_none[1],
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

def main():
    print(f"\n{'='*60}\n🚀 RUNNING MAIN EXPERIMENT\n{'='*60}")
    print(f"⚙️  Teachers: BERT + RoBERTa")
    print(f"⚙️  Params: LR={LR}, Alpha={ALPHA}, Warmup={WARMUP}")


    df = pd.read_csv(DATA_PATH)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(df, df["label"]))

    toks = {"student": AutoTokenizer.from_pretrained(STUDENT_PATH)}
    for t in ["bert", "roberta"]: toks[t] = AutoTokenizer.from_pretrained(TEACHER_BASE_PATHS[t])

    train_loader = DataLoader(SOTADataset(df.iloc[train_idx], toks, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(SOTADataset(df.iloc[val_idx], toks, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4, persistent_workers=True)


    teachers = {t: TeacherNet(t).to(DEVICE).eval() for t in ["bert", "roberta"]}
    for p in teachers.values():
        for param in p.parameters(): param.requires_grad = False

    student = StudentNet().to(DEVICE)
    agent = AgentPolicy().to(DEVICE)

    opt_s = optim.AdamW(student.parameters(), lr=LR)
    opt_a = optim.AdamW(agent.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    best_f1 = 0.0
    best_metrics = {}


    start_time = time.time()
    for ep in range(EPOCHS):
        student.train(); agent.train()
        pbar = tqdm(train_loader, desc=f"Ep {ep+1}/{EPOCHS}", leave=False)

        for batch in pbar:
            labels = batch["label"].to(DEVICE, non_blocking=True)

            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    t_logits = []
                    for t_name in ["bert", "roberta"]:
                        t_logits.append(teachers[t_name](batch[f"{t_name}_ids"].to(DEVICE, non_blocking=True),
                                                         batch[f"{t_name}_mask"].to(DEVICE, non_blocking=True)))

            opt_s.zero_grad(set_to_none=True); opt_a.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                s_logits = student(batch["student_ids"].to(DEVICE, non_blocking=True),
                                   batch["student_mask"].to(DEVICE, non_blocking=True))


                if ep < WARMUP:
                    weights = torch.ones(s_logits.size(0), 2).to(DEVICE) / 2
                else:
                    weights = agent(s_logits.detach(), t_logits)

                t_stack = torch.stack(t_logits, dim=1)
                weighted_teacher = torch.sum(weights.unsqueeze(-1) * t_stack, dim=1)

                loss_ce = F.cross_entropy(s_logits, labels)
                loss_kd = F.kl_div(F.log_softmax(s_logits/2.0, dim=1), F.softmax(weighted_teacher/2.0, dim=1), reduction='batchmean') * 4.0
                loss = (1 - ALPHA) * loss_ce + ALPHA * loss_kd

            scaler.scale(loss).backward()
            scaler.step(opt_s)
            if ep >= WARMUP: scaler.step(opt_a)
            scaler.update()

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})


        student.eval()
        preds, trues, probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                with torch.amp.autocast('cuda'):
                    logits = student(batch["student_ids"].to(DEVICE, non_blocking=True),
                                     batch["student_mask"].to(DEVICE, non_blocking=True))
                    prob = torch.softmax(logits, dim=1)[:, 1]
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                trues.extend(batch["label"].numpy())
                probs.extend(prob.cpu().numpy())


        m = calculate_metrics(trues, preds, probs)

        if m['macro_f1'] > best_f1:
            best_f1 = m['macro_f1']
            best_metrics = m
            print(f"   Ep {ep+1:02d} | F1: {m['macro_f1']:.4f} | Acc: {m['accuracy']:.4f} 🏆 New Best!")
        else:
            print(f"   Ep {ep+1:02d} | F1: {m['macro_f1']:.4f} | Acc: {m['accuracy']:.4f}")

    print_metrics(best_metrics, "🏆 FINAL RESULT")
    print(f"⏱️ Total Time: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
