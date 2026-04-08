import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset


from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, logging
logging.set_verbosity_error()


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

BASE_PATHS = {
    "bert": "/home/share/ZYN/SYB/project/chuanboxue/model_final/model/Bert_chinese_base",
    "roberta": "/home/share/ZYN/SYB/project/chuanboxue/model_final/model/chinese-roberta"
}

SAVE_DIR_ROOT = "/home/share/ZYN/SYB/project/chuanboxue/model_final/pt"
DATA_PATH = "/home/share/ZYN/SYB/project/TTFND_repo/data/weibo21/weibo21_std.csv"

BATCH_SIZE = 512
EPOCHS = 10
WARMUP_EPOCHS = 4
LR = 2e-5
DEVICE = "cuda"
MAX_LEN = 128
NUM_WORKERS = 8


class TeacherNet(nn.Module):
    def __init__(self, base_path):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_path)
        self.fc = nn.Linear(768, 2)

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        feat = out.last_hidden_state[:, 0, :]
        return self.fc(feat)

def prepare_data(tokenizer, data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    def process(sub_df):
        text_col = "news_content" if "news_content" in sub_df.columns else "content"
        texts = sub_df[text_col].fillna("").astype(str).tolist()
        labels = sub_df["label"].values
        enc = tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels, dtype=torch.long))
        return ds

    return process(train_df), process(val_df)

def train_one_teacher(name, base_path):
    print(f"\n{'='*60}\n🚀 Training Teacher: {name.upper()} (Clean Mode)\n{'='*60}")

    save_dir = os.path.join(SAVE_DIR_ROOT, f"{name}_teacher")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_teacher.pt")

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    train_ds, val_ds = prepare_data(tokenizer, DATA_PATH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True, prefetch_factor=4, persistent_workers=True)

    model = TeacherNet(base_path).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


    scaler = torch.amp.GradScaler('cuda')

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()
    best_val_f1 = 0.0

    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        ep_start = time.time()

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}", leave=False)
        for ids, mask, labels in pbar:
            ids, mask, labels = ids.to(DEVICE, non_blocking=True), mask.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()


            with torch.amp.autocast('cuda'):
                logits = model(ids, mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})


        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for ids, mask, labels in val_loader:
                ids, mask = ids.to(DEVICE, non_blocking=True), mask.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits = model(ids, mask)
                preds.extend(torch.argmax(logits, 1).cpu().numpy())
                trues.extend(labels.cpu().numpy())

        val_f1 = f1_score(trues, preds, average='macro')
        avg_loss = total_loss / len(train_loader)
        ep_time = time.time() - ep_start

        print(f"  Ep {epoch+1:02d} | Time: {ep_time:.1f}s | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}", end="")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(" 🏆 Saved!")
        else:
            print("")

    total_time = time.time() - start_time
    print(f"✅ {name} Training Completed. Best F1: {best_val_f1:.4f}")
    print(f"💾 Saved to: {save_path}")

if __name__ == "__main__":
    for t_name, t_path in BASE_PATHS.items():
        try:
            train_one_teacher(t_name, t_path)
        except Exception as e:
            print(f"❌ Error training {t_name}: {e}")
