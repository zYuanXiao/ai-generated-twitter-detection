# AI-Generated Twitter Detection  
*A CLIP-Based Approach to Identifying AI-Generated Tweets*  
**Yuqing Fan & Zhiyuan Xiao — CSC 245 (Fall 2025)**

---

## 1. Motivation

AI-generated text–image content is increasingly common on social media.  
Real and synthetic tweets often appear similar, and manual inspection is slow and inconsistent.  
This project asks:

**Can we detect whether an image–text pair is real by measuring their semantic alignment?**

We develop a scalable multimodal detector using CLIP.

---

## 2. Research Question

- Does the image match the meaning of the text?  
- Do AI-generated tweets show distinct multimodal patterns?  
- Can embeddings reveal inconsistencies invisible to the human eye?

---

## 3. Method Overview

### 🔹 Dataset Construction  
We create real and synthetic tweet pairs.  
AI-generated captions follow a controlled prompt:

> “Look at the image. Do not describe it. Write a tweet-style post (1–2 sentences, <30 words)... Stay concise... No hashtag lists... Output only the tweet.”

This generates natural AI-style captions.

### 🔹 Model Architecture  
We use **CLIP ViT-L/14-336** to extract:

- image embeddings  
- text embeddings  
- cosine similarity  

Feature vector:
[image_emb ∥ text_emb ∥ cosine_sim]

Classifier:  
A 2-layer MLP (hidden size = 512).

Benefits:
- Embeddings retain rich multimodal signals  
- Goes beyond a single similarity score  
- Captures tweet-specific non-linear patterns  

---

## 4. Results

### 🎯 Accuracy: **87.37%**  
### 🎯 F1 Score: **0.88**

<img src="Accuracy_and_F1_score.png" width="650"/>

Model behavior:
- Detects most real/fake tweets accurately  
- Balanced precision, recall, and f1  
- Moderate errors:
  - **False Positive**: real image paired with very short generated text  
  - **False Negative**: artistic images or very long text (>77 tokens)

---

## 5. Interpretability

We analyze:

- **Modal-level contributions** (image vs text vs similarity)  
- **Token contributions** (mask each token)  
- **Image occlusion heatmaps** (patch sensitivity)

Example:

<img src="contrib_0cd5khj3wo5a1.png" width="500"/>

Insights:
- Highlights sensitive image regions  
- Identifies influential words  
- Explains misclassifications  
- Evaluates robustness to text length and visually complex artwork

---

## 6. Method Improvement & Benchmark Plan

### ✔ Performance Comparison
- **Ours (CLIP + MLP)**: 87.37% accuracy  
- **Zero-shot CLIP**: 55.01% accuracy  
- Large improvement in both accuracy and F1.

### ✔ Future Benchmark (Similar to GenEval)
Goal: Evaluate *how real* an AI-generated tweet appears.

Possible benchmark dimensions:
- Semantic consistency  
- Visual artifact detection  
- Text style naturalness  
- Cross-model robustness (Midjourney, SD3, DALL·E, etc.)  
- Reasoning-based mismatch detection  

---

## 7. File Structure
```bash
├── build_csv.py
├── build_csv_adapted.py
├── clip_tweet_match.py
├── plot_eval.py
├── visualize_part_contribution.py
├── requirements.txt
└── data/
```

## 8. Usage

### Install dependencies

```bash
pip install -r requirements.txt
```
Train model
```bash
python clip_tweet_match.py
```
Evaluate
```bash
python plot_eval.py
```
Run interpretability
```bash
python visualize_part_contribution.py \
    --ckpt checkpoints/best_clip_match_classifier.pt \
    --image path/to/image.png \
    --text "tweet text here"
```
