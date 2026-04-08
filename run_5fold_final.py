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


JSON_RESULT_FILE = "5fold_results_final.json"
TXT_LOG_FILE = "5fold_running_log.txt"


TEACHER_BASE_PATHS = {
    "bert": "model_final/model/Bert",
    "roberta": "model_final/model/chinese-roberta"
}
CHECKPOINT_PATHS = {
    "bert": "./model_final/pt/bert_teacher/best_teacher.pt",
    "roberta": "./model_final/pt/roberta_teacher/best_teacher.pt"
}
STUDENT_PATH = "/home/share/ZYN/SYB/project/chuanboxue/model_final/model/roberta"
DATA_PATH = "/home/share/ZYN/SYB/project/TTFND_repo/data/weibo21/weibo21.csv"


BEST_PARAMS = {
    "lr": 2e-5,
    "warmup": 4,
    "alpha": 0.5,
    "temp": 2.0
}

MAX_LEN = 128
BATCH_SIZE = 580
NUM_WORKERS = 8
DEVICE = "cuda"
EPOCHS = 15


def init_log_file():
    with open(TXT_LOG_FILE, "w") as f:
        f.write(f"🔥 5-Fold Cross Validation Log (Start: {time.strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"📌 Params: {BEST_PARAMS}\n")
        f.write("="*120 + "\n")
        header = f"{'Fold':<6} | {'Macro F1':<10} | {'Acc':<8} | {'AUC':<8} | {'F1 Real':<8} | {'F1 Fake':<8} | {'Prec Real':<9} | {'Prec Fake':<9}\n"
        f.write(header)
        f.write("="*120 + "\n")

def log_fold_result(fold, metrics):
    with open(TXT_LOG_FILE, "a") as f:
        line = (f"{fold:<6} | {metrics['macro_f1']:.4f}     | {metrics['accuracy']:.4f}   | {metrics['AUC']:.4f}   | "
                f"{metrics['f1_real']:.4f}   | {metrics['f1_fake']:.4f}   | {metrics['precision_real']:.4f}    | {metrics['precision_fake']:.4f}\n")
        f.write(line)

def log_final_stats(stats):
    with open(TXT_LOG_FILE, "a") as f:
        f.write("\n" + "="*120 + "\n")
        f.write("🏆 FINAL 5-FOLD STATISTICS (Mean ± Std [Var])\n")
        f.write("="*120 + "\n")
        for k, v in stats.items():
            f.write(f"{k:<20}: {v['mean']:.4f} ± {v['std']:.4f}  [Var: {v['var']:.6f}]\n")

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
        return self.fc(out.last_hidden_state[:, 0, :])

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
        self.net = nn.Sequential(nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 2), nn.Softmax(dim=1))
    def forward(self, s_logits, t_logits_list):
        return self.net(torch.cat([s_logits] + t_logits_list, dim=1))

def calculate_metrics(trues, preds, probs):
    acc = accuracy_score(trues, preds)
    try: auc = roc_auc_score(trues, probs)
    except: auc = 0.0
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(trues, preds, average='macro')
    p_none, r_none, f1_none, _ = precision_recall_fscore_support(trues, preds, average=None)
    return {
        "macro_f1": f1_macro, "accuracy": acc, "AUC": auc,
        "f1_real": f1_none[0], "f1_fake": f1_none[1],
        "precision_macro": p_macro, "recall_macro": r_macro,
        "precision_fake": p_none[1], "recall_fake": r_none[1],
        "precision_real": p_none[0], "recall_real": r_none[0],
        "recall_fake": r_none[1]
    }

def run_fold(fold_idx, train_idx, val_idx, df, toks):
    print(f"\n🔄 [Fold {fold_idx+1}/5] Training...")

    train_loader = DataLoader(SOTADataset(df.iloc[train_idx], toks, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(SOTADataset(df.iloc[val_idx], toks, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4, persistent_workers=True)

    teachers = {t: TeacherNet(t).to(DEVICE).eval() for t in ["bert", "roberta"]}
    for p in teachers.values():
        for param in p.parameters(): param.requires_grad = False

    student = StudentNet().to(DEVICE)
    agent = AgentPolicy().to(DEVICE)


    lr = BEST_PARAMS['lr']
    warmup = BEST_PARAMS['warmup']
    alpha = BEST_PARAMS['alpha']
    temp = BEST_PARAMS['temp']

    opt_s = optim.AdamW(student.parameters(), lr=lr)
    opt_a = optim.AdamW(agent.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    best_f1 = 0.0
    best_metrics = {}

    for ep in range(EPOCHS):
        student.train(); agent.train()
        pbar = tqdm(train_loader, desc=f"Ep {ep+1:02d}", leave=False)
        for batch in pbar:
            labels = batch["label"].to(DEVICE, non_blocking=True)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    t_logits = [teachers[t](batch[f"{t}_ids"].to(DEVICE), batch[f"{t}_mask"].to(DEVICE)) for t in ["bert", "roberta"]]

            opt_s.zero_grad(set_to_none=True); opt_a.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                s_logits = student(batch["student_ids"].to(DEVICE), batch["student_mask"].to(DEVICE))
                weights = torch.ones(s_logits.size(0), 2).to(DEVICE)/2 if ep < warmup else agent(s_logits.detach(), t_logits)
                t_stack = torch.stack(t_logits, dim=1)
                weighted_teacher = torch.sum(weights.unsqueeze(-1) * t_stack, dim=1)

                loss_ce = F.cross_entropy(s_logits, labels)
                loss_kd = F.kl_div(F.log_softmax(s_logits/temp, dim=1), F.softmax(weighted_teacher/temp, dim=1), reduction='batchmean') * (temp**2)
                loss = (1 - alpha) * loss_ce + alpha * loss_kd

            scaler.scale(loss).backward()
            scaler.step(opt_s)
            if ep >= warmup: scaler.step(opt_a)
            scaler.update()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})


        student.eval()
        preds, trues, probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                with torch.amp.autocast('cuda'):
                    logits = student(batch["student_ids"].to(DEVICE), batch["student_mask"].to(DEVICE))
                    prob = torch.softmax(logits, dim=1)[:, 1]
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                trues.extend(batch["label"].numpy())
                probs.extend(prob.cpu().numpy())

        m = calculate_metrics(trues, preds, probs)
        if m['macro_f1'] > best_f1:
            best_f1 = m['macro_f1']; best_metrics = m


    del train_loader, val_loader, student, teachers, agent, opt_s, opt_a, scaler
    torch.cuda.empty_cache()

    print(f"   🏆 Fold {fold_idx+1} Best F1: {best_f1:.4f}")
    return best_metrics

if __name__ == "__main__":
    init_log_file()
    print(f"🔥 A40 5-FOLD CV | Batch={BATCH_SIZE} | Best Params: {BEST_PARAMS}")

    df = pd.read_csv(DATA_PATH)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    toks = {"student": AutoTokenizer.from_pretrained(STUDENT_PATH)}
    for t in ["bert", "roberta"]: toks[t] = AutoTokenizer.from_pretrained(TEACHER_BASE_PATHS[t])

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        metrics = run_fold(fold_idx, train_idx, val_idx, df, toks)
        fold_results.append(metrics)
        log_fold_result(f"Fold {fold_idx+1}", metrics)


    final_stats = {}
    keys = fold_results[0].keys()
    print("\n📊 Final 5-Fold Statistics:")
    for k in keys:
        values = [r[k] for r in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        var_val = np.var(values)
        final_stats[k] = {"mean": mean_val, "std": std_val, "var": var_val}
        print(f"{k:<20}: {mean_val:.4f} ± {std_val:.4f} (Var: {var_val:.6f})")

    log_final_stats(final_stats)
    with open(JSON_RESULT_FILE, 'w') as f: json.dump({"folds": fold_results, "stats": final_stats}, f, indent=4)
    print("\n✅ 5-Fold CV Completed Successfully!")
