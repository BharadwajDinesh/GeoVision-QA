"""
Phase 4 — RSVQA Dataset Preparation
-------------------------------------
Downloads RSVQA-LR dataset, converts Q&A pairs to LLaVA conversation
format, and uploads to GCS for Colab training.

RSVQA-LR uses Sentinel-2 derived imagery — directly compatible with
the Koramangala composites from Phases 2 & 3.

Usage:
    python src/rsvqa_prep.py --bucket geovision-data
"""

import argparse
import json
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm
from google.cloud import storage

# ── RSVQA-LR dataset URLs ────────────────────────────────────────────────────
# Record ID confirmed via: curl -s https://zenodo.org/api/records/6344333
# (redirects to 6344334). Filenames confirmed via API file listing.
_BASE = "https://zenodo.org/api/records/6344334/files"
RSVQA_LR_URLS = {
    "images":     f"{_BASE}/Images_LR.zip/content",
    "train_q":    f"{_BASE}/LR_split_train_questions.json/content",
    "train_a":    f"{_BASE}/LR_split_train_answers.json/content",
    "train_imgs": f"{_BASE}/LR_split_train_images.json/content",
    "val_q":      f"{_BASE}/LR_split_val_questions.json/content",
    "val_a":      f"{_BASE}/LR_split_val_answers.json/content",
    "val_imgs":   f"{_BASE}/LR_split_val_images.json/content",
}

IMAGE_TOKEN = "<image>"

QTYPE_HINTS = {
    "presence":   "Answer with yes or no.",
    "count":      "Answer with a number only.",
    "comparison": "Answer with yes or no.",
    "area":       "Answer with a number only.",
    "rural_urban":"Answer with rural or urban.",
}


# ── Download helpers ──────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return dest
    print(f"  Downloading {dest.name} …")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))
    return dest


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    marker = out_dir / ".extracted"
    if marker.exists():
        print(f"  [skip] {zip_path.name} already extracted")
        return
    print(f"  Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    marker.touch()


# ── LLaVA conversation builder ────────────────────────────────────────────────

def build_llava_entry(image_id, image_filename, question, answer, qtype):
    hint = QTYPE_HINTS.get(qtype.lower(), "")
    human_turn = f"{IMAGE_TOKEN}\n{hint} {question}".strip()
    return {
        "id": f"rsvqa_lr_{image_id}",
        "image": f"Images_LR/{image_filename}",
        "conversations": [
            {"from": "human", "value": human_turn},
            {"from": "gpt",   "value": str(answer)},
        ],
    }


# ── Core conversion ───────────────────────────────────────────────────────────

def convert_split(questions_path, answers_path, images_path, out_jsonl, max_samples=None):
    print(f"\n  Converting → {out_jsonl.name}")

    with open(questions_path) as f:
        questions_data = json.load(f)
    with open(answers_path) as f:
        answers_data = json.load(f)
    with open(images_path) as f:
        images_data = json.load(f)

    # The images JSON key may be "images" or the root list itself
    imgs = images_data.get("images", images_data) if isinstance(images_data, dict) else images_data
    answers_list = answers_data.get("answers", answers_data) if isinstance(answers_data, dict) else answers_data
    questions_list = questions_data.get("questions", questions_data) if isinstance(questions_data, dict) else questions_data

    answer_map = {a["question_id"]: a for a in answers_list if a.get("active", True) and "question_id" in a}
    image_map  = {img["id"]: img for img in imgs}

    entries = []
    for q in tqdm([q for q in questions_list if q.get("active", True) and "question" in q], desc="    Processing"):
        q_id     = q["id"]
        img_id   = q.get("img_id")
        qtype    = q.get("type", "unknown")
        question = q["question"]

        if q_id not in answer_map or img_id is None:
            continue

        answer   = answer_map[q_id]["answer"]
        filename = f"{img_id}.tif"  # images are named by ID

        entries.append(build_llava_entry(str(q_id), filename, question, answer, qtype))

        if max_samples and len(entries) >= max_samples:
            break

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    print(f"    Wrote {len(entries):,} entries → {out_jsonl}")
    return len(entries)


# ── GCS upload ────────────────────────────────────────────────────────────────

def upload_to_gcs(local_path, bucket_name, gcs_prefix):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = f"{gcs_prefix}/{local_path.name}"
    print(f"  Uploading {local_path.name} → gs://{bucket_name}/{blob_name}")
    bucket.blob(blob_name).upload_from_filename(str(local_path))


def upload_images_to_gcs(images_dir, bucket_name, gcs_prefix):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    files  = list(images_dir.glob("*.png")) + list(images_dir.glob("*.tif"))
    print(f"\n  Uploading {len(files)} images to GCS …")
    for f in tqdm(files):
        blob_name = f"{gcs_prefix}/Images_LR/{f.name}"
        blob = bucket.blob(blob_name)
        if not blob.exists():
            blob.upload_from_filename(str(f))


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    data_dir   = Path("data/rsvqa_lr")
    out_dir    = Path("data/llava_format")
    gcs_prefix = "geovision/phase4/rsvqa"
    raw_dir    = data_dir / "raw"

    # Step 1: Download
    print("\n[1/4] Downloading RSVQA-LR dataset …")
    zip_path   = download_file(RSVQA_LR_URLS["images"],     raw_dir / "Images_LR.zip")
    q_train    = download_file(RSVQA_LR_URLS["train_q"],    raw_dir / "train_questions.json")
    a_train    = download_file(RSVQA_LR_URLS["train_a"],    raw_dir / "train_answers.json")
    imgs_train = download_file(RSVQA_LR_URLS["train_imgs"], raw_dir / "train_images.json")
    q_val      = download_file(RSVQA_LR_URLS["val_q"],      raw_dir / "val_questions.json")
    a_val      = download_file(RSVQA_LR_URLS["val_a"],      raw_dir / "val_answers.json")
    imgs_val   = download_file(RSVQA_LR_URLS["val_imgs"],   raw_dir / "val_images.json")

    # Step 2: Extract
    print("\n[2/4] Extracting images …")
    images_dir = data_dir / "Images_LR"
    extract_zip(zip_path, data_dir)

    # Step 3: Convert
    print("\n[3/4] Converting to LLaVA conversation format …")
    n_train = convert_split(q_train, a_train, imgs_train,
                            out_dir / "train.jsonl", args.max_samples)
    n_val   = convert_split(q_val, a_val, imgs_val,
                            out_dir / "val.jsonl",
                            args.max_samples // 10 if args.max_samples else None)
    print(f"\n  Total: {n_train:,} train | {n_val:,} val samples")

    # Step 4: Upload
    if not args.skip_upload:
        print("\n[4/4] Uploading to GCS …")
        upload_to_gcs(out_dir / "train.jsonl", args.bucket, gcs_prefix)
        upload_to_gcs(out_dir / "val.jsonl",   args.bucket, gcs_prefix)
        upload_images_to_gcs(images_dir, args.bucket, gcs_prefix)
        print(f"\n  Done → gs://{args.bucket}/{gcs_prefix}/")
    else:
        print("\n[4/4] Skipping GCS upload")

    print("\n✅ RSVQA preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket",       default="geovision-data")
    parser.add_argument("--max-samples",  type=int, default=None)
    parser.add_argument("--skip-upload",  action="store_true")
    args = parser.parse_args()
    main(args)
