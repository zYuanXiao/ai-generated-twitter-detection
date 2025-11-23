import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


MAX_TEXT_LEN = 77
Image.MAX_IMAGE_PIXELS = None
MODEL_NAME = "openai/clip-vit-large-patch14-336"

TRAIN_CSV = "data/train.csv"
VAL_CSV   = "data/val.csv"

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL   = 64

EPOCHS = 5
LR_HEAD = 1e-4
WEIGHT_DECAY = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)



class TweetMatchDataset(Dataset):
    """
    Read data from csv, format: image_path, text, label
    """

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        text = str(row["text"])
        label = int(row["label"])

        return image_path, text, label


def collate_fn(batch, processor):
    """
    pack(image_path, text, label) to the input of CLIP
    """

    images = []
    texts = []
    labels = []

    for image_path, text, label in batch:
        image = Image.open(image_path).convert("RGB")
        images.append(image)
        texts.append(text)
        labels.append(label)

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_LEN
    )
    labels = torch.tensor(labels, dtype=torch.long)

    return inputs, labels



class MatchClassifier(nn.Module):
    """
    Use:
        - image_embeds
        - text_embeds
        - cosine similarity
    Concatenate them into one vector, then perform binary classification (match / mismatch).
    """

    def __init__(self, embed_dim, hidden_dim=512):
        super().__init__()
        input_dim = embed_dim * 2 + 1  # image + text + sim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # Two classes：0/1
        )

    def forward(self, image_embeds, text_embeds):
        image_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_norm  = text_embeds  / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity
        sim = (image_norm * text_norm).sum(dim=-1, keepdim=True)  # [B,1]

        # Concatenation
        x = torch.cat([image_norm, text_norm, sim], dim=-1)
        logits = self.mlp(x)

        return logits



# Zero-shot baseline：use only CLIP similarity

def zero_shot_baseline(clip_model, processor, csv_path):
    """
    No training. 
    Directly compute CLIP image/text similarity, then sweep a threshold on the val set to find the best one.
    """

    df = pd.read_csv(csv_path)
    sims = []
    labels = []

    clip_model.eval()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Zero-shot baseline"):
        image_path = row["image_path"]
        text = str(row["text"])
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LEN
        ).to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            image_embeds = outputs.image_embeds  # [1, d]
            text_embeds  = outputs.text_embeds   # [1, d]

            image_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_norm  = text_embeds  / text_embeds.norm(dim=-1, keepdim=True)
            sim = (image_norm * text_norm).sum(dim=-1)  # [1]

        sims.append(sim.item())
        labels.append(label)

    sims = np.array(sims)
    labels = np.array(labels)

    # Threshold sweeping
    best_acc = -1
    best_th = None
    for th in [x / 100.0 for x in range(-50, 51)]:  # -0.5 ~ 0.5
        preds = (sims >= th).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_th = th

    preds_best = (sims >= best_th).astype(int)
    best_f1 = f1_score(labels, preds_best)

    print("=== Zero-shot baseline result ===")
    print(f"Best threshold: {best_th:.3f}")
    print(f"Accuracy: {best_acc:.4f}")
    print(f"F1 score: {best_f1:.4f}")
    print("=================================")



# Training + validation functions (freeze CLIP, train only the classifier head)

def train_one_epoch(clip_model, classifier, data_loader, criterion, optimizer, processor):
    clip_model.eval()
    classifier.train()

    all_losses = []
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(data_loader, desc="Train"):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        with torch.no_grad():
            outputs = clip_model(**inputs)
            image_embeds = outputs.image_embeds  # [B, d]
            text_embeds  = outputs.text_embeds   # [B, d]

        logits = classifier(image_embeds, text_embeds)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_losses.append(loss.item())
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = sum(all_losses) / len(all_losses)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")

    return avg_loss, acc, f1


def eval_one_epoch(clip_model, classifier, data_loader, criterion, processor):
    clip_model.eval()
    classifier.eval()

    all_losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Val"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = clip_model(**inputs)
            image_embeds = outputs.image_embeds
            text_embeds  = outputs.text_embeds

            logits = classifier(image_embeds, text_embeds)
            loss = criterion(logits, labels)

            all_losses.append(loss.item())
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = sum(all_losses) / len(all_losses)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")

    return avg_loss, acc, f1




def main():
    print("Loading CLIP model:", MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    #Zero-shot baseline（validation set only）
    if os.path.exists(VAL_CSV):
        zero_shot_baseline(clip_model, processor, VAL_CSV)
    else:
        print(f"[Warning] VAL_CSV {VAL_CSV} not found, skip zero-shot baseline.")

    # Train classifier head
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(VAL_CSV):
        print("[Error] TRAIN_CSV or VAL_CSV not found, cannot train classifier.")
        return

    # Freeze CLIP parameters (train classifier head only)
    for p in clip_model.parameters():
        p.requires_grad = False

    train_ds = TweetMatchDataset(TRAIN_CSV)
    val_ds   = TweetMatchDataset(VAL_CSV)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor)
    )

    embed_dim = clip_model.config.projection_dim
    classifier = MatchClassifier(embed_dim=embed_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=LR_HEAD,
        weight_decay=WEIGHT_DECAY
    )

    best_val_f1 = -1.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")
        train_loss, train_acc, train_f1 = train_one_epoch(
            clip_model, classifier, train_loader, criterion, optimizer, processor
        )
        print(f"[Train] loss={train_loss:.4f}, acc={train_acc:.4f}, f1={train_f1:.4f}")

        val_loss, val_acc, val_f1 = eval_one_epoch(
            clip_model, classifier, val_loader, criterion, processor
        )
        print(f"[Val]   loss={val_loss:.4f}, acc={val_acc:.4f}, f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join("checkpoints", "best_clip_match_classifier.pt")
            torch.save(classifier.state_dict(), save_path)
            print(f">>> New best model saved to {save_path} (F1={best_val_f1:.4f})")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
