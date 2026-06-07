"""
Phase 4 — LLaVA Inference on Koramangala Satellite Imagery
------------------------------------------------------------
Loads the fine-tuned LLaVA checkpoint and answers natural language
questions over the Koramangala satellite images from Phases 2 & 3.

Also generates a pre-baked Q&A report about detected land-cover changes
to feed into the Phase 4 FastAPI endpoint.

Usage:
    # Interactive Q&A on a single image
    python src/llava_infer.py \
        --bucket geovision-data \
        --image gs://geovision-data/geovision/phase3/koramangala_s2_2024.tif \
        --question "What is the dominant land cover in this area?"

    # Generate full change report (feeds FastAPI)
    python src/llava_infer.py \
        --bucket geovision-data \
        --generate-report
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import torch
from google.cloud import storage
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig


# ── GCS path for fine-tuned checkpoint ───────────────────────────────────────
CHECKPOINT_GCS = "geovision/phase4/checkpoints/best_llava_rsvqa"
BASE_MODEL      = "llava-hf/llava-1.5-7b-hf"

# Questions for the automated change report
REPORT_QUESTIONS = [
    ("2024", "What is the dominant land cover visible in this satellite image?"),
    ("2024", "Are there visible signs of urban development or construction?"),
    ("2024", "How much vegetation cover is present in this area?"),
    ("2024", "Are there any water bodies visible?"),
    ("2015", "What is the dominant land cover visible in this satellite image?"),
    ("2015", "How much of this area appears to be developed or built-up?"),
]

BNBCONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


# ── GCS helpers ───────────────────────────────────────────────────────────────

def download_from_gcs(bucket_name: str, gcs_path: str, local_path: Path) -> Path:
    """Download a file from GCS to a local path."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(gcs_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    return local_path


def gcs_uri_to_parts(uri: str) -> tuple[str, str]:
    """Parse gs://bucket/path → (bucket, path)."""
    assert uri.startswith("gs://"), f"Not a GCS URI: {uri}"
    parts  = uri[5:].split("/", 1)
    return parts[0], parts[1]


def download_checkpoint(bucket_name: str, gcs_prefix: str, local_dir: Path) -> None:
    """Download all files under a GCS prefix to a local directory."""
    if (local_dir / "adapter_config.json").exists():
        print(f"  [skip] Checkpoint already at {local_dir}")
        return

    client  = storage.Client()
    bucket  = client.bucket(bucket_name)
    blobs   = list(client.list_blobs(bucket_name, prefix=gcs_prefix))

    print(f"  Downloading {len(blobs)} checkpoint files …")
    local_dir.mkdir(parents=True, exist_ok=True)
    for blob in blobs:
        rel_path  = blob.name[len(gcs_prefix):].lstrip("/")
        dest      = local_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))


# ── Image preparation ─────────────────────────────────────────────────────────

def geotiff_to_rgb(tif_path: Path) -> Image.Image:
    """
    Convert a multi-band Sentinel-2 GeoTIFF to an 8-bit RGB PIL Image.

    Band mapping (0-indexed):
        B2=0 (Blue), B3=1 (Green), B4=2 (Red), B8=3 (NIR), B11=4, B12=5

    Uses bands 2, 1, 0 → RGB for natural colour composite.
    """
    with rasterio.open(tif_path) as src:
        # Read R, G, B bands (indices 3, 2, 1 in 1-based rasterio notation)
        r = src.read(3).astype(np.float32)
        g = src.read(2).astype(np.float32)
        b = src.read(1).astype(np.float32)

    def normalize(band: np.ndarray) -> np.ndarray:
        """Percentile stretch to [0, 255]."""
        p2, p98 = np.percentile(band[band > 0], [2, 98])
        clipped = np.clip(band, p2, p98)
        return ((clipped - p2) / (p98 - p2) * 255).astype(np.uint8)

    rgb = np.stack([normalize(r), normalize(g), normalize(b)], axis=-1)
    return Image.fromarray(rgb)


# ── Model loader ──────────────────────────────────────────────────────────────

class LLaVAInference:
    """Wraps the fine-tuned LLaVA model for single-image Q&A."""

    def __init__(self, checkpoint_dir: Path, use_finetuned: bool = True):
        print(f"\n  Loading processor …")
        self.processor = AutoProcessor.from_pretrained(str(checkpoint_dir))

        print(f"  Loading base model in 4-bit …")
        base = LlavaForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            quantization_config=BNBCONFIG,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        if use_finetuned:
            print(f"  Loading LoRA adapters …")
            self.model = PeftModel.from_pretrained(base, str(checkpoint_dir))
        else:
            print("  Using base model (no fine-tuning)")
            self.model = base

        self.model.eval()
        print("  Model ready.\n")

    @torch.inference_mode()
    def ask(self, image: Image.Image, question: str, max_new_tokens: int = 128) -> str:
        """
        Ask a natural language question about a PIL image.
        Returns the model's text answer.
        """
        # Build the prompt with the image token
        prompt = f"USER: <image>\n{question} ASSISTANT:"

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy for reproducibility
            temperature=1.0,
        )

        # Decode only the newly generated tokens (skip the prompt)
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()


# ── Change report generator ───────────────────────────────────────────────────

def generate_change_report(
    model: LLaVAInference,
    images: dict[str, Image.Image],
    bucket_name: str,
) -> dict:
    """
    Run a fixed set of questions over the 2015 and 2024 images.
    Returns a structured report dict that the FastAPI endpoint will serve.
    """
    print("\n  Generating change report …")
    report = {
        "area": "Koramangala, Bengaluru",
        "years": ["2015", "2024"],
        "qa_pairs": [],
        "summary": {},
    }

    for year, question in REPORT_QUESTIONS:
        image = images[year]
        print(f"    [{year}] {question}")
        answer = model.ask(image, question)
        print(f"           → {answer}")
        report["qa_pairs"].append({
            "year":     year,
            "question": question,
            "answer":   answer,
        })

    # Upload report to GCS
    report_json = json.dumps(report, indent=2)
    client  = storage.Client()
    bucket  = client.bucket(bucket_name)
    blob    = bucket.blob("geovision/phase4/change_report.json")
    blob.upload_from_string(report_json, content_type="application/json")
    print(f"\n  Report saved → gs://{bucket_name}/geovision/phase4/change_report.json")
    return report


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    local_ckpt = Path("checkpoints/llava_rsvqa_local")
    local_imgs = Path("data/phase3_images")

    # ── Download checkpoint ───────────────────────────────────────────────────
    print("\n[1/3] Loading checkpoint …")
    download_checkpoint(args.bucket, CHECKPOINT_GCS, local_ckpt)
    model = LLaVAInference(local_ckpt, use_finetuned=not args.base_only)

    # ── Load satellite images ─────────────────────────────────────────────────
    print("[2/3] Loading Koramangala images …")
    local_imgs.mkdir(parents=True, exist_ok=True)

    for year, gcs_path in [
        ("2015", "geovision/phase3/koramangala_s2_2015.tif"),
        ("2024", "geovision/phase3/koramangala_s2_2024.tif"),
    ]:
        local_path = local_imgs / f"koramangala_{year}.tif"
        if not local_path.exists():
            download_from_gcs(args.bucket, gcs_path, local_path)

    images = {
        "2015": geotiff_to_rgb(local_imgs / "koramangala_2015.tif"),
        "2024": geotiff_to_rgb(local_imgs / "koramangala_2024.tif"),
    }

    # ── Interactive or report mode ────────────────────────────────────────────
    print("[3/3] Running inference …")

    if args.generate_report:
        report = generate_change_report(model, images, args.bucket)
        print("\n── Change Report ──────────────────────────────────────")
        for qa in report["qa_pairs"]:
            print(f"[{qa['year']}] Q: {qa['question']}")
            print(f"       A: {qa['answer']}\n")

    elif args.question:
        # Single question on the specified year's image
        year  = args.year
        image = images[year]
        print(f"\n  Image : Koramangala {year}")
        print(f"  Q     : {args.question}")
        answer = model.ask(image, args.question)
        print(f"  A     : {answer}\n")

    else:
        # Interactive REPL
        print("\nEntering interactive mode. Type 'exit' to quit.")
        print("Year options: 2015, 2024\n")
        while True:
            year = input("Year (2015/2024): ").strip()
            if year not in images:
                print("  Please enter 2015 or 2024.")
                continue
            q = input("Question: ").strip()
            if q.lower() == "exit":
                break
            a = model.ask(images[year], q)
            print(f"  Answer: {a}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaVA inference on Koramangala imagery")
    parser.add_argument("--bucket",          default="geovision-data")
    parser.add_argument("--question",        default=None,
                        help="Single question to answer")
    parser.add_argument("--year",            default="2024", choices=["2015", "2024"],
                        help="Which composite to query")
    parser.add_argument("--generate-report", action="store_true",
                        help="Generate full change Q&A report and upload to GCS")
    parser.add_argument("--base-only",       action="store_true",
                        help="Use base LLaVA without fine-tuned LoRA weights")
    args = parser.parse_args()
    main(args)
