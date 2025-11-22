import os
import csv
import random
from pathlib import Path

random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Fake image + text
FAKE_IMG_DIRS = [
    DATA_DIR / "fakeV2" / "fake-v2",
    DATA_DIR / "real_fake" / "real",
]

FAKE_TXT_DIRS = [
    DATA_DIR / "fakeV2" / "fake_tweet",
    DATA_DIR / "real_fake" / "real_tweet",   # 如果不存在会自动跳过
]

# Real image + text
REAL_IMG_DIR = DATA_DIR / "archive" / "real"
REAL_TXT_DIR = DATA_DIR / "archive" / "img_tweet"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_text(path: Path) -> str:
    """读取 txt 文本并压成一行"""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return " ".join(text.strip().split())


def collect_pairs(img_dir, txt_dir, label):
    """根据 img_dir + txt_dir 生成 (img_path, text, label)"""
    samples = []

    if not txt_dir.exists():
        return samples

    txt_files = {p.stem: p for p in txt_dir.glob("*.txt")}

    for stem, txt_path in txt_files.items():

        # 找对应图片
        image_path = None
        for ext in IMG_EXTS:
            cand = img_dir / f"{stem}{ext}"
            if cand.exists():
                image_path = cand
                break

        if image_path is None:
            continue

        text = read_text(txt_path)

        # image path 相对 data/ 的路径
        rel_img = os.path.join("data", os.path.relpath(image_path, DATA_DIR))

        samples.append((rel_img, text, label))

    return samples


def write_csv(name, rows):
    out_path = DATA_DIR / name
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "text", "label"])
        writer.writerows(rows)
    print(f"> Saved {len(rows)} rows → {out_path}")


def main():
    data = []

    # collect fake
    for img_dir, txt_dir in zip(FAKE_IMG_DIRS, FAKE_TXT_DIRS):
        data += collect_pairs(img_dir, txt_dir, label=0)

    # collect real
    data += collect_pairs(REAL_IMG_DIR, REAL_TXT_DIR, label=1)

    if not data:
        print("No data found. Please check folder paths.")
        return

    random.shuffle(data)

    n = len(data)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    write_csv("train.csv", train_data)
    write_csv("val.csv", val_data)
    write_csv("test.csv", test_data)


if __name__ == "__main__":
    main()
