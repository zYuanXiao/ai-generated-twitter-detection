import argparse
import os

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor

from clip_tweet_match import MatchClassifier, MODEL_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import numpy as np

@torch.no_grad()
def compute_modal_contribution(clip_model, processor, classifier, image, text):
    """
    Compute the contributions of:
      - the entire image
      - the entire text
      - the similarity term
    to the final predicted probability.

    Returns:
      p_full: full probability
      d_img : contribution of image (p_full - p_no_img)
      d_txt : contribution of text (p_full - p_no_txt)
      d_sim : contribution of similarity (p_full - p_no_sim)
    """

    # Forward pass to get embeddings
    inputs = processor(
        text=[text],
        images=[image],
        truncation=True,
        padding=True,
        max_length=77,  # or MAX_TEXT_LEN
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = clip_model(**inputs)
    image_embeds = outputs.image_embeds   # [1, D]
    text_embeds  = outputs.text_embeds    # [1, D]

    # Normalize embeddings + cosine similarity (same as training)
    image_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_norm  = text_embeds  / text_embeds.norm(dim=-1, keepdim=True)
    sim = (image_norm * text_norm).sum(dim=-1, keepdim=True)  # [1,1]

    def run(mask_img=1.0, mask_txt=1.0, mask_sim=1.0):
        x = torch.cat(
            [
                image_norm * mask_img,
                text_norm * mask_txt,
                sim * mask_sim,
            ],
            dim=-1
        )
        logits = classifier.mlp(x)
        prob = torch.softmax(logits, dim=-1)[0, 1].item()
        return prob

    # Full prediction
    p_full = run(1.0, 1.0, 1.0)

    # Ablations
    p_no_img = run(0.0, 1.0, 1.0)
    p_no_txt = run(1.0, 0.0, 1.0)
    p_no_sim = run(1.0, 1.0, 0.0)

    d_img = p_full - p_no_img
    d_txt = p_full - p_no_txt
    d_sim = p_full - p_no_sim

    return p_full, d_img, d_txt, d_sim


def compute_text_contribution(clip_model, processor, classifier, text: str, image):
    """
    Compute token-level contribution using occlusion:
      contribution = p_full - p_masked_token

    The higher the value, the more important the token.
    """

    # Baseline full probability
    inputs = processor(
        text=[text],
        images=[image],
        truncation=True,
        padding=True,
        max_length=77,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k,v in inputs.items()}

    with torch.no_grad():
        out = clip_model(**inputs)
        image_embeds = out.image_embeds
        text_embeds = out.text_embeds

        image_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_norm  = text_embeds  / text_embeds.norm(dim=-1, keepdim=True)
        sim = (image_norm * text_norm).sum(dim=-1, keepdim=True)

        logits = classifier.mlp(torch.cat([image_norm, text_norm, sim], dim=-1))
        p_full = torch.softmax(logits, dim=-1)[0, 1].item()

    # Tokenize text
    tokens = processor.tokenizer(text, return_tensors="pt", padding=False, truncation=True)
    input_ids = tokens["input_ids"][0]  # shape [L]
    token_list = processor.tokenizer.convert_ids_to_tokens(input_ids.tolist())

    scores = []
    L = len(input_ids)

    # Mask each token and recompute
    for i in range(L):
        masked_ids = input_ids.clone()
        masked_ids[i] = processor.tokenizer.pad_token_id  # use PAD as mask

        masked = processor.tokenizer.decode(masked_ids, skip_special_tokens=False)

        masked_inputs = processor(
            text=[masked],
            images=[image],
            truncation=True,
            padding=True,
            max_length=77,
            return_tensors="pt"
        )
        masked_inputs = {k: v.to(device) for k,v in masked_inputs.items()}

        with torch.no_grad():
            out2 = clip_model(**masked_inputs)
            img2 = out2.image_embeds
            txt2 = out2.text_embeds

            img2 = img2 / img2.norm(dim=-1, keepdim=True)
            txt2 = txt2 / txt2.norm(dim=-1, keepdim=True)
            sim2 = (img2 * txt2).sum(dim=-1, keepdim=True)

            logits2 = classifier.mlp(torch.cat([img2, txt2, sim2], dim=-1))
            p_masked = torch.softmax(logits2, dim=-1)[0, 1].item()

        score = p_full - p_masked
        scores.append(score)

    return token_list, np.array(scores), p_full


def load_models(ckpt_path: str):
    """Load CLIP and the trained classifier checkpoint."""
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
def predict_prob(clip_model, processor, classifier, image: Image.Image, text: str) -> float:
    """
    Predict match probability using CLIP embeddings + classifier MLP.
    Must be consistent with training-time forward pass.
    """
    inputs = processor(
        text=[text],
        images=[image],
        max_length=77,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = clip_model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    image_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    sim = (image_norm * text_norm).sum(dim=-1, keepdim=True)

    x = torch.cat([image_norm, text_norm, sim], dim=-1)
    logits = classifier.mlp(x)

    prob = torch.softmax(logits, dim=-1)[0, 1].item()
    return prob


def compute_occlusion_map(
    clip_model,
    processor,
    classifier,
    image: Image.Image,
    text: str,
    grid_size: int = 7,
    occ_color=(127, 127, 127),
):
    """
    Divide the image into grid_size x grid_size patches.
    Occlude each patch and recompute probability.
    Heatmap value = (base_prob - occluded_prob).
    """
    w, h = image.size
    base_prob = predict_prob(clip_model, processor, classifier, image, text)

    patch_w = w // grid_size
    patch_h = h // grid_size

    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)

    for i in range(grid_size):
        for j in range(grid_size):
            occluded = image.copy()
            draw = ImageDraw.Draw(occluded)

            left = j * patch_w
            upper = i * patch_h
            right = (j + 1) * patch_w if j < grid_size - 1 else w
            lower = (i + 1) * patch_h if i < grid_size - 1 else h

            draw.rectangle([left, upper, right, lower], fill=occ_color)

            prob_occ = predict_prob(clip_model, processor, classifier, occluded, text)

            heatmap[i, j] = base_prob - prob_occ

    # Normalize heatmap to [0,1]
    min_v = heatmap.min()
    max_v = heatmap.max()
    if max_v > min_v:
        norm_heatmap = (heatmap - min_v) / (max_v - min_v)
    else:
        norm_heatmap = np.zeros_like(heatmap)

    return base_prob, heatmap, norm_heatmap


def overlay_heatmap_on_image(image: Image.Image, norm_heatmap: np.ndarray, out_path: str):
    """
    Overlay the occlusion heatmap on the original image and save it.
    """
    w, h = image.size
    grid_h, grid_w = norm_heatmap.shape

    plt.figure(figsize=(6, 6))
    plt.imshow(image)

    plt.imshow(
        norm_heatmap,
        cmap="jet",
        alpha=0.5,
        extent=(0, w, h, 0),
        interpolation="bilinear",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Classifier checkpoint path, e.g., checkpoints/best_clip_match_classifier.pt")
    parser.add_argument("--image", type=str, required=True,
                        help="Image path for analysis")
    parser.add_argument("--text", type=str, required=True,
                        help="Text content (tweet)")
    parser.add_argument("--grid", type=int, default=7,
                        help="Grid size for occlusion, e.g., 7 means 7x7")
    parser.add_argument("--out", type=str, default="image_contribution.png",
                        help="Output filename for occlusion heatmap")
    args = parser.parse_args()

    clip_model, processor, classifier = load_models(args.ckpt)

    image = Image.open(args.image).convert("RGB")

    # Image occlusion heatmap
    base_prob, heatmap, norm_heatmap = compute_occlusion_map(
        clip_model,
        processor,
        classifier,
        image,
        args.text,
        grid_size=args.grid,
    )

    print(f"Base probability (match=1): {base_prob:.4f}")
    print("Heatmap (raw values, base_prob - prob_occluded):")
    np.set_printoptions(precision=3, suppress=True)
    print(heatmap)

    overlay_heatmap_on_image(image, norm_heatmap, args.out)
    print(f"Saved heatmap overlay to: {args.out}")

    # Text token contribution
    tokens, contrib_tok, p_full_text = compute_text_contribution(
        clip_model, processor, classifier, args.text, image
    )
    print("\n=== Text token contribution ===")
    for tok, sc in zip(tokens, contrib_tok):
        print(f"{tok:15s}  contrib = {sc:.4f}")

    # Modal-level contribution
    p_full, d_img, d_txt, d_sim = compute_modal_contribution(
        clip_model, processor, classifier,
        image=image,
        text=args.text
    )

    print("\n=== Modal-level contribution (image vs text vs similarity) ===")
    print(f"Full probability p_full           = {p_full:.4f}")
    print(f"Δp_image      (remove image)      = {d_img:+.4f}")
    print(f"Δp_text       (remove text)       = {d_txt:+.4f}")
    print(f"Δp_similarity (remove similarity) = {d_sim:+.4f}")

    # Normalized positive contributions
    contrib_vec = np.array([max(d_img, 0), max(d_txt, 0), max(d_sim, 0)])
    if contrib_vec.sum() > 0:
        w = contrib_vec / contrib_vec.sum()
        print("\nApproximate relative importance (normalized positive contributions):")
        print(f"  image      ≈ {w[0]*100:5.1f}%")
        print(f"  text       ≈ {w[1]*100:5.1f}%")
        print(f"  similarity ≈ {w[2]*100:5.1f}%")

if __name__ == "__main__":
    main()
