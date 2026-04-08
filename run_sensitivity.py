import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, logging

logging.set_verbosity_error()


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

JSON_RESULT_FILE = "pure_sensitivity_results.json"
TXT_LOG_FILE = "pure_sensitivity_log.txt"

TEACHER_BASE_PATHS = {
    "bert": "model_final/model/Bert_chinese_base",
    "roberta": "model_final/model/chinese-roberta"
}
CHECKPOINT_PATHS = {
    "bert": "./model_final/pt/bert_teacher/best_teacher.pt",
    "roberta": "./model_final/pt/roberta_teacher/best_teacher.pt"
}

STUDENT_PATH = "model_final/model/roberta"
DATA_PATH = "data/weibo21/weibo21_std.csv"

MAX_LEN = 128
BATCH_SIZE = 512
NUM_WORKERS = 4
DEVICE = "cuda"
EPOCHS_PER_TRIAL = 15


BASE_LR = 2e-5
BASE_ALPHA = 0.5
BASE_WARMUP = 4
BASE_TEMP = 2.0


EXP_ALPHA = {
    "name": "Alpha Sensitivity",
    "param": "alpha",
    "values": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
}

EXP_TEMP = {
    "name": "Temperature Sensitivity",
    "param": "temp",
    "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
}

experiments = [EXP_ALPHA, EXP_TEMP]


def log_to_txt(exp_name, param_name, param_value, metrics):
    with open(TXT_LOG_FILE, "a") as f:
        line = (
            f"{exp_name:<25} | {param_name:<10} | {param_value:<6} | "
            f"F1={metrics['macro_f1']:.4f} | Acc={metrics['accuracy']:.4f} | "
            f"AUC={metrics['AUC']:.4f}\n"
        )
        f.write(line)

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
        "f1_fake": f1_none[1]
    }


class SOTADataset(Dataset):
    def __init__(self, df, tokenizer_dict):
        self.df = df
        self.tokenizer_dict = tokenizer_dict
        self.text_col = "news_content" if "news_content" in df.columns else "content"
        self.texts = df[self.text_col].fillna("").astype(str).tolist()
        self.labels = df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        batch_data = {"label": torch.tensor(self.labels[idx], dtype=torch.long)}
        text = self.texts[idx]

        for name, tokenizer in self.tokenizer_dict.items():
            enc = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            )
            batch_data[f"{name}_ids"] = enc["input_ids"].squeeze(0)
            batch_data[f"{name}_mask"] = enc["attention_mask"].squeeze(0)

        return batch_data


class TeacherNet(nn.Module):
    def __init__(self, t_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(TEACHER_BASE_PATHS[t_name])
        self.fc = nn.Linear(768, 2)

        ckpt_path = CHECKPOINT_PATHS.get(t_name)
        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location='cpu')
            if isinstance(sd, nn.Module):
                sd = sd.state_dict()
            self.load_state_dict(sd, strict=False)

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        return self.fc(out.last_hidden_state[:, 0, :])

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(STUDENT_PATH)
        self.fc = nn.Linear(768, 2)

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        return self.fc(out.last_hidden_state[:, 0, :])


def run_trial(lr, alpha, warmup, temp):
    df = pd.read_csv(DATA_PATH)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(df, df["label"]))

    toks = {"student": AutoTokenizer.from_pretrained(STUDENT_PATH)}
    for t in ["bert", "roberta"]:
        toks[t] = AutoTokenizer.from_pretrained(TEACHER_BASE_PATHS[t])

    train_loader = DataLoader(
        SOTADataset(df.iloc[train_idx], toks),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        SOTADataset(df.iloc[val_idx], toks),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    teachers = {t: TeacherNet(t).to(DEVICE).eval() for t in ["bert", "roberta"]}
    for p in teachers.values():
        for param in p.parameters():
            param.requires_grad = False

    student = StudentNet().to(DEVICE)
    optimizer = optim.AdamW(student.parameters(), lr=lr)

    best_f1 = 0.0
    best_metrics = {}

    for ep in range(EPOCHS_PER_TRIAL):
        student.train()
        for batch in train_loader:
            labels = batch["label"].to(DEVICE)
            s_logits = student(batch["student_ids"].to(DEVICE),
                               batch["student_mask"].to(DEVICE))

            with torch.no_grad():
                t_logits = [
                    teachers[t](batch[f"{t}_ids"].to(DEVICE),
                                batch[f"{t}_mask"].to(DEVICE))
                    for t in ["bert", "roberta"]
                ]
                t_mean = torch.stack(t_logits).mean(dim=0)

            loss_ce = F.cross_entropy(s_logits, labels)
            loss_kd = F.kl_div(
                F.log_softmax(s_logits/temp, dim=1),
                F.softmax(t_mean/temp, dim=1),
                reduction='batchmean'
            ) * (temp ** 2)

            loss = (1 - alpha) * loss_ce + alpha * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        student.eval()
        preds, trues, probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = student(batch["student_ids"].to(DEVICE),
                                 batch["student_mask"].to(DEVICE))
                prob = torch.softmax(logits, dim=1)[:, 1]
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                trues.extend(batch["label"].numpy())
                probs.extend(prob.cpu().numpy())

        m = calculate_metrics(trues, preds, probs)
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_metrics = m

    torch.cuda.empty_cache()
    return best_metrics


if __name__ == "__main__":
    print("🔥 Running Pure Sensitivity Analysis")

    final_results = {}

    for exp in experiments:
        exp_name = exp["name"]
        param_name = exp["param"]
        print(f"\n=== {exp_name} ===")

        results = []

        for val in exp["values"]:

            lr = BASE_LR
            alpha = BASE_ALPHA
            warmup = BASE_WARMUP
            temp = BASE_TEMP

            if param_name == "alpha":
                alpha = val
            elif param_name == "temp":
                temp = val

            metrics = run_trial(lr, alpha, warmup, temp)

            entry = {
                "param": param_name,
                "value": val,
                "lr": lr,
                "alpha": alpha,
                "warmup": warmup,
                "temp": temp
            }
            entry.update(metrics)

            results.append(entry)
            log_to_txt(exp_name, param_name, val, metrics)

            print(f"{param_name}={val} → F1={metrics['macro_f1']:.4f}")

        final_results[exp_name] = results

    with open(JSON_RESULT_FILE, "w") as f:
        json.dump(final_results, f, indent=4)

    print("\n✅ Sensitivity Analysis Completed!")
