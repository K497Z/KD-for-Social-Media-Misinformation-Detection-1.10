import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import os

class TeacherModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.fc = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        feat = out.last_hidden_state[:, 0, :]
        return self.fc(feat)

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        s_name = cfg.get("student", {}).get("model_name", cfg.get("student_model_name"))
        if not s_name:
             s_name = "/home/share/ZYN/SYB/project/TTFND_repo/models/roberta"

        self.student_encoder = AutoModel.from_pretrained(s_name, local_files_only=True)
        self.student_hidden = self.student_encoder.config.hidden_size
        self.num_labels = cfg.get("num_labels", cfg.get("model", {}).get("num_classes", 2))

        self.classifier = nn.Sequential(
            nn.Linear(self.student_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_labels)
        )
        self.teacher_encoders = nn.ModuleDict()
        self.teacher_cfgs = cfg.get("model", {}).get("teachers", [])
        self.load_teachers_from_config()

    def load_teachers_from_config(self):
        print("[Info] Loading teachers...")
        for t_cfg in self.teacher_cfgs:
            name = t_cfg["name"]
            path = t_cfg["ckpt_path"]
            if os.path.exists(path):
                try:
                    t_model = TeacherModel(t_cfg['model_name'], self.num_labels)
                    state_dict = torch.load(path, map_location='cpu')
                    if hasattr(state_dict, 'state_dict'):
                        state_dict = state_dict.state_dict()
                    t_model.load_state_dict(state_dict, strict=True)
                    for param in t_model.parameters(): param.requires_grad = False
                    t_model.eval()
                    self.teacher_encoders[name] = t_model
                    print(f"  ✅ [Loaded] {name} successfully")
                except Exception as e:
                    print(f"  ❌ [Error] Failed to load {name}: {e}")
            else:
                print(f"  - [Warn] Teacher {name} path not found: {path}")

    def forward(self, input_ids, attention_mask, k_ids=None, k_mask=None, return_feat=False):
        out = self.student_encoder(input_ids=input_ids, attention_mask=attention_mask)
        s_feat = out.last_hidden_state[:, 0, :]
        logits = self.classifier(s_feat)
        if return_feat: return logits, s_feat
        return {"logits": logits}
