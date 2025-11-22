import os
from ollama import chat
from tqdm import tqdm

# input and output folder
IMAGE_DIR = "data/real_fake/real"
OUTPUT_DIR = "data/real_fake/real_tweet"
# IMAGE_DIR = "data/fakeV2/fake-v2"
# OUTPUT_DIR = "data/fakeV2/fake_tweet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT = """
Look at the image internally. Do not describe the image. Do not explain the image.

Write a tweet-style post inspired by its mood or message, keeping it extremely short: 1-2 sentences, under 30 words.

If the image shows everyday life, people, casual moments, or real-world scenes, use a natural, conversational, human tone — simple, relaxed, and effortless.

If the image shows art, film, historical scenes, paintings, or anything stylized or symbolic, use a more reflective, philosophical, or poetic tone — but stay concise and avoid over-dramatization.

You may include one or two natural hashtags only if they genuinely fit the tone. Never add hashtag lists or repeated tags.

Avoid descriptions, summaries, or analytical language. Output only the tweet.

""".strip()

def is_image_file(filename: str) -> bool:
    ext = filename.lower().rsplit(".", 1)[-1]
    return ext in ["jpg", "jpeg", "png"]


all_images = sorted(
    [f for f in os.listdir(IMAGE_DIR) if is_image_file(f)]
)[:5000]

existing_txt_stems = {
    os.path.splitext(f)[0]
    for f in os.listdir(OUTPUT_DIR)
    if f.lower().endswith(".txt")
}

images_to_process = [
    f for f in all_images
    if os.path.splitext(f)[0] not in existing_txt_stems
]

print(f"Total images (<=5000): {len(all_images)}")
print(f"Already processed:      {len(all_images) - len(images_to_process)}")
print(f"To process now:         {len(images_to_process)}")

if not images_to_process:
    print("Nothing to do, all images already have txt files.")
else:
    for filename in tqdm(images_to_process, desc="Generating tweets", unit="img"):
        img_path = os.path.join(IMAGE_DIR, filename)
        txt_stem = os.path.splitext(filename)[0]
        txt_path = os.path.join(OUTPUT_DIR, txt_stem + ".txt")

        try:
            response = chat(
                model="qwen3-vl:8b",
                messages=[{
                    "role": "user",
                    "content": PROMPT,
                    "images": [img_path],
                }],
            )

            tweet = response["message"]["content"].strip()

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(tweet)

        except Exception as e:
            tqdm.write(f"[ERROR] {filename}: {e}")

    print("Done.")