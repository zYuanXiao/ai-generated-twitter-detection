import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from clip_tweet_match import MatchClassifier, MODEL_NAME   # ← use training model :contentReference[oaicite:0]{index=0}

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(ckpt_path):
    clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    embed_dim = clip_model.config.projection_dim
    classifier = MatchClassifier(embed_dim=embed_dim).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    clip_model.eval()

    return clip_model, processor, classifier


@torch.no_grad()
def predict_one(clip_model, processor, classifier, image_path, text):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=[text],
        images=[image],
        max_length=77,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    out = clip_model(**inputs)
    img = out.image_embeds
    txt = out.text_embeds

    img_norm = img / img.norm(dim=-1, keepdim=True)
    txt_norm = txt / txt.norm(dim=-1, keepdim=True)
    sim = (img_norm * txt_norm).sum(dim=-1, keepdim=True)

    x = torch.cat([img_norm, txt_norm, sim], dim=-1)
    logits = classifier.mlp(x)
    pred = torch.softmax(logits, dim=-1)[0, 1].item()

    return pred >= 0.5



def evaluate_on_csv(csv_path, clip_model, processor, classifier):
    df = pd.read_csv(csv_path)

    preds = []
    gts = []

    for _, row in df.iterrows():
        pred = predict_one(
            clip_model, processor, classifier,
            image_path=row["image_path"],
            text=str(row["text"])
        )
        preds.append(int(pred))
        gts.append(int(row["label"]))

    preds = np.array(preds)
    gts = np.array(gts)

    cm = confusion_matrix(gts, preds)
    acc = accuracy_score(gts, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gts, preds, labels=[0, 1], zero_division=0
    )

    return cm, acc, precision, recall, f1


def plot_results(cm, acc, precision, recall, f1, out_path="eval_plot.png"):
    classes = ["Real", "Fake"]

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax1
    )
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")

    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(classes))
    width = 0.25

    ax2.bar(x - width, precision, width, label="precision")
    ax2.bar(x,         recall,    width, label="recall")
    ax2.bar(x + width, f1,        width, label="f1-score")

    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.set_ylim(0, 1.05)
    ax2.set_title(f"Classification Metrics by Class\nOverall Accuracy: {acc:.4f}")
    ax2.set_ylabel("Score")
    ax2.legend()

    fig.suptitle("F1 score and Accuracy", fontsize=24, y=1.05)


    plt.figtext(0.5, -0.05, f"(a) Normal prompt Acc ≈ {acc*100:.1f}%", ha="center", fontsize=16)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[Saved] {out_path}")



def main():
    ckpt = "checkpoints/best_clip_match_classifier.pt"
    val_csv = "data/val.csv"

    print("Loading model...")
    clip_model, processor, classifier = load_model(ckpt)

    print("Evaluating on val set...")
    cm, acc, precision, recall, f1 = evaluate_on_csv(
        val_csv, clip_model, processor, classifier
    )

    print("Generating plot...")
    plot_results(cm, acc, precision, recall, f1, out_path="Accuracy_and_F1_score.png")


if __name__ == "__main__":
    main()
