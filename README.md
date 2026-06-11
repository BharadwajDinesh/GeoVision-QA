# GeoVision QA — Satellite Intelligence Pipeline

An end-to-end pipeline for satellite image analysis, object detection, change detection, and natural language question answering over multispectral satellite imagery.

Built as part of M.Tech research in AI & Data Science at IIIT Kota, targeting applied AI roles in the geospatial/remote sensing domain.

---

## Project Overview

```
Sentinel-2 Satellite Imagery (Google Earth Engine)
        ↓
Phase 1: Data Ingestion & Spectral Analysis      ✅ Complete
        ↓
Phase 2: Object Detection & Segmentation         ✅ Complete
        ↓
Phase 3: Change Detection (SegFormer)            ✅ Complete
        ↓
Phase 4a: VLM Fine-tuning (LLaVA + QLoRA)       ✅ Complete
        ↓
Phase 4b: FastAPI + RAG Deployment              ✅ Complete (live on GPU)
```

---

## Architecture

```
geovision/
├── src/
│   ├── geo_ingest.py        # Google Earth Engine data pipeline
│   ├── band_math.py         # Spectral index computation
│   ├── raster_io.py         # GeoTIFF read/write with rasterio
│   ├── visualize.py         # Band histograms, RGB composites, dashboards
│   ├── osm_labels.py        # OSM feature extraction + rasterization
│   ├── tile_slicer.py       # GeoTIFF → 512x512 patch slicing
│   ├── seg_dataset.py       # PyTorch Dataset for SegFormer training
│   ├── train.py             # SegFormer fine-tuning script
│   ├── phase2_run.py        # Phase 2 end-to-end pipeline runner
│   ├── phase3_run.py        # Pull 2015 Koramangala composite
│   ├── phase3_2024.py       # Pull 2024 Koramangala composite
│   ├── phase3_infer.py      # Change detection inference pipeline
│   ├── rsvqa_prep.py        # RSVQA-LR dataset preparation for LLaVA
│   ├── llava_finetune.py    # LLaVA-1.5 QLoRA fine-tuning script
│   ├── llava_infer.py       # LLaVA inference on Koramangala imagery
│   └── api/                 # Phase 4b — FastAPI serving package
│       ├── main.py          # FastAPI app + all routes
│       ├── model_loader.py  # Loads SegFormer + LLaVA+LoRA from GCS at startup
│       ├── inference.py     # Segmentation, change detection, VQA pipelines
│       ├── rag_context.py   # Builds RAG context from Phase 3 change stats
│       └── observability.py # Langfuse tracing for every request
├── notebooks/
│   └── phase1_data_ingestion.ipynb
├── .github/
│   └── workflows/
│       └── deploy.yml       # CI/CD: lint → build → push → deploy to GPU VM
├── Dockerfile               # CUDA-enabled image for GPU inference
├── docker-compose.yml
├── .gitignore
├── requirements.txt
└── README.md
```

**Infrastructure:**
- API Serving    : GCP VM (g2-standard-4, NVIDIA L4 24GB, asia-south1-b)
- GPU Training   : GCP VM (g2-standard-4, NVIDIA L4 24GB)
- Container Reg. : GCP Artifact Registry (us-central1)
- Storage        : Google Cloud Storage (no local data storage)
- CI/CD          : GitHub Actions (auto build + deploy on push to main)
- Observability  : Langfuse (US region) — per-request tracing
- Region         : us-central1 (registry/storage), asia-south1 (serving)

---

## Phase 1 — Data Ingestion & Spectral Analysis ✅

### What it does
- Connects to Google Earth Engine and pulls Sentinel-2 multispectral
  imagery over Bengaluru (30x30km area)
- Applies cloud masking using QA60 band (bits 10 & 11)
- Builds a median composite from multiple scenes
- Computes spectral indices for land cover analysis
- Exports processed GeoTIFF to Google Cloud Storage

### Output
- **File:** `bengaluru_s2_composite_2024.tif`
- **Size:** 3064 x 3006 px (~30x30 km)
- **Bands:** 6 Sentinel-2 bands (B2, B3, B4, B8, B11, B12)
- **CRS:** EPSG:4326
- **Stored:** `gs://geovision-data/geovision/phase1/`

### Spectral Indices Computed
| Index  | Formula                         | What it measures           |
|--------|---------------------------------|----------------------------|
| NDVI   | (NIR - Red) / (NIR + Red)       | Vegetation health          |
| EVI    | Enhanced Vegetation Index       | Vegetation (robust)        |
| SAVI   | Soil Adjusted Vegetation Index  | Vegetation over bare soil  |
| NDWI   | (Green - NIR) / (Green + NIR)   | Water bodies               |
| MNDWI  | (Green - SWIR) / (Green + SWIR) | Modified water index       |
| NDBI   | (SWIR - NIR) / (SWIR + NIR)     | Built-up areas             |

### Key Results (Bengaluru 2024)
```
NDVI  : mean=0.256  (moderate vegetation — urban area with parks)
SAVI  : mean=0.161  (consistent with NDVI)
NDWI  : mean=-0.328 (more land than water — expected)
NDBI  : mean=0.059  (slight positive — dense urban core)
Urban : 41.8% of pixels classified as urban
```

### Bug Fixed — Double Scaling
Sentinel-2 raw values are integers scaled by 10000. Earth Engine divided
by 10000 during export. Our read_geotiff() was dividing again — causing
values in the 0.000001 range instead of 0.0-0.3. Fixed by adding
as_float=False flag to skip the second division.

---

## Phase 2 — Object Detection & Segmentation ✅

### What it does
- Downloads OpenStreetMap features for Koramangala, Bengaluru
- Rasterizes building footprints, roads, vegetation, and water bodies
  into pixel-level segmentation masks
- Slices the large GeoTIFF into 512x512 patches with 64px overlap
- Fine-tunes SegFormer-b2 on the generated tiles
- All data flows through GCS — nothing stored locally

### Segmentation Classes
| ID | Class      | Source              |
|----|------------|---------------------|
| 0  | Background | —                   |
| 1  | Buildings  | OSM building=*      |
| 2  | Roads      | OSM highway=*       |
| 3  | Vegetation | OSM landuse/natural |
| 4  | Water      | OSM natural=water   |

### Model — SegFormer-b2
- Base model  : `nvidia/segformer-b2-finetuned-ade-512-512`
- Fine-tuned  : On 8 Bengaluru/Koramangala tiles
- Input       : RGB (3 bands), 512x512 patches
- Output      : 5-class segmentation mask
- Optimizer   : AdamW (lr=6e-5, weight_decay=0.01)
- Loss        : Weighted CrossEntropy (handles class imbalance)
- Scheduler   : CosineAnnealingLR over 20 epochs
- Training    : Google Colab T4 GPU

### Training Results
```
Best mIoU  : 0.3380 at epoch 15
Loss trend : 1.64 → 1.22 (steadily decreasing)
mIoU trend : 0.13 → 0.34 (steadily improving)
Checkpoint : gs://geovision-data/geovision/phase2/checkpoints/best_segformer.pt
```

### Pipeline
```
GCS GeoTIFF
    ↓
OSMLabeler — reads local GeoJSON, rasterizes to full-scene mask
    ↓
TileSlicer — slices image + mask into 512x512 patches, uploads to GCS
    ↓
SatelliteSegDataset — PyTorch Dataset streaming tiles from GCS
    ↓
SegFormer fine-tuning on Colab T4 GPU
    ↓
Checkpoint saved to GCS
```

### Key Design Decisions
- **GCS-native pipeline** — no local disk storage, all I/O through GCS
- **64px overlap** — prevents objects at tile boundaries from being cut off
- **min_label_pct filter** — skips tiles that are mostly background
- **Class weights** — [0.5, 2.0, 2.0, 1.5, 3.0] to handle class imbalance
- **Transfer learning** — fine-tune NVIDIA pretrained SegFormer

### GCS Artifacts
```
gs://geovision-data/geovision/phase2/
├── full_mask.npy                          # (3006, 3064) segmentation mask
├── tiles/                                 # 512x512 image patches (.npy)
├── masks/                                 # 512x512 label patches (.npy)
└── checkpoints/best_segformer.pt          # Fine-tuned model (104.5 MB)
```

### Future Improvement
Switch from SegFormer-b2 (RGB, ImageNet pretrained) to
**Prithvi-EO-2.0** (IBM/NASA, 6 Sentinel-2 bands, 4.2M satellite samples)
for native 6-band support and domain-specific pretraining.

---

## Phase 3 — Change Detection ✅

### What it does
- Pulls two Sentinel-2 composites of the same area at different times:
  - 2015 composite (Jun 2015 - Dec 2016, 9 scenes)
  - 2024 composite (Jan 2024 - May 2024)
- Area of interest: Koramangala, Bengaluru (1km buffer)
- Reprojects 2024 image to exactly match 2015 pixel grid
- Runs fine-tuned SegFormer on both images independently
- Compares segmentation masks pixel by pixel to detect changes

### Area of Interest
```
Location  : Koramangala, Bengaluru, India
Center    : 12.9411°N, 77.6158°E
Buffer    : 1km radius
Image size: ~200x200 pixels (10m Sentinel-2 resolution)
```

### Change Detection Approach
```
2015 Sentinel-2 image
        ↓
SegFormer → Segmentation mask 2015
        +
2024 Sentinel-2 image
        ↓
SegFormer → Segmentation mask 2024
        ↓
Pixel-by-pixel comparison
        ↓
Change map with 4 change types
```

### Change Types Detected
| Change Type           | Color  | Meaning                    |
|-----------------------|--------|----------------------------|
| Vegetation → Building | Red    | New construction           |
| Background → Building | Orange | New construction on empty  |
| Building → Vegetation | Green  | Demolition / greening      |
| Building → Background | Gray   | Demolition                 |

### Results (Koramangala 2015 → 2024)
```
Class Distribution:
                  2015      2024      Change
Background      : 30.2%  → 27.8%    -2.4%
Building        : 57.7%  → 44.6%    -13.1%
Vegetation      : 11.5%  → 25.2%    +13.7%
Water           :  0.6%  →  2.4%    +1.8%
```

### Key Engineering Challenges Solved
- **Earth Engine grid snapping** — exports from different dates land on
  slightly different pixel grids. Fixed using rasterio reproject to align
  2024 exactly to 2015 transform.
- **No-data pixels** — reprojection creates black border pixels which
  SegFormer misclassifies as water. Fixed by filling no-data regions with
  per-band mean values before inference.
- **Single scene in 2015** — initial query with tight cloud filter returned
  only 1 scene (nearly unusable). Fixed by widening date range to 18 months
  and relaxing cloud threshold to 30%.

### GCS Artifacts
```
gs://geovision-data/geovision/phase3/
├── koramangala_s2_2015.tif        # 2015 Sentinel-2 composite
├── koramangala_s2_2024.tif        # 2024 Sentinel-2 composite
├── seg_2015.npy                   # Segmentation mask 2015
├── seg_2024.npy                   # Segmentation mask 2024
├── change_map.npy                 # Binary change map
└── change_detection_result.png    # Visualization
```

---

## Phase 4a — VLM Fine-tuning ✅

### What it does
- Downloads RSVQA-LR dataset (Zenodo record 6344334) — 772 Sentinel-2
  satellite images with 77,000 Q&A pairs
- Applies stratified sampling across 4 question types (presence, count,
  comparison, rural_urban) for balanced training
- Converts dataset to LLaVA conversation format (JSONL)
- Fine-tunes LLaVA-1.5-7B using QLoRA (8-bit) on GCP L4 GPU VM
- Uploads LoRA adapter checkpoint to GCS

### Dataset — RSVQA-LR
- **Source:** Zenodo record 6344334
- **Images:** 772 Sentinel-2 satellite tiles (256×256 px, RGB)
- **Q&A pairs:** 57,223 training / 10,005 validation
- **Question types:** presence, count, comparison, rural_urban
- **Stratified sample used:** 20 samples (5 per question type)

### Model — LLaVA-1.5-7B + QLoRA
- Base model    : `llava-hf/llava-1.5-7b-hf`
- Quantization  : 8-bit (bitsandbytes)
- LoRA rank     : 16, alpha 32
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Trainable params: 42,336,256 (0.596% of total)
- Optimizer     : paged_adamw_8bit (lr=2e-4)
- Scheduler     : Cosine annealing
- Epochs        : 3
- Batch size    : 4
- Training VM   : GCP g2-standard-4 (NVIDIA L4 24GB)

### Pipeline
```
Zenodo (RSVQA-LR)
    ↓
rsvqa_prep.py — download, stratified sample, convert to LLaVA JSONL
    ↓
GCS: train.jsonl, val.jsonl, Images_LR/ (772 images)
    ↓
llava_finetune.py — prefetch images, QLoRA fine-tuning on L4 GPU
    ↓
GCS: best_llava_rsvqa/ (LoRA adapter checkpoint)
```

### Key Engineering Challenges Solved
- **Zenodo URL format** — migrated from `/record/` to `/records/{id}/files/{name}/content`
- **Inactive dataset records** — RSVQA JSON contains placeholder entries
  `{id: N, active: False}` — fixed by filtering on `active` flag before building lookup maps
- **Image token mismatch** — LLaVA's `<image>` token expands to 576 patch
  tokens; `apply_chat_template` was stripping it. Fixed by formatting prompts
  manually as `USER: <image>\n{question} ASSISTANT: {answer}`
- **max_length too short** — 256 tokens truncated the 576 image tokens.
  Increased to 768.
- **GCS image download bottleneck** — per-step GCS downloads caused 47s/step.
  Fixed by prefetching all 772 images to local disk before training starts,
  reducing to ~16s/step.
- **OOM on L4** — batch size 8 without gradient checkpointing exceeded 23GB.
  Fixed by re-enabling gradient checkpointing and reducing batch to 4.

### GCS Artifacts
```
gs://geovision-data/geovision/phase4/
├── rsvqa/
│   ├── train.jsonl                    # 57,223 LLaVA-format Q&A pairs
│   ├── val.jsonl                      # 10,005 validation pairs
│   └── Images_LR/                     # 772 satellite image tiles
└── checkpoints/
    └── best_llava_rsvqa/
        ├── adapter_config.json        # LoRA configuration
        ├── adapter_model.safetensors  # Fine-tuned LoRA weights
        ├── tokenizer.json             # Tokenizer
        ├── tokenizer_config.json
        └── processor_config.json
```

---

## Phase 4b — FastAPI + RAG Deployment ✅

### What it does
Serves the entire pipeline as a live REST API on a GPU VM. Natural language
question answering over satellite imagery, semantic segmentation, and change
detection — all behind HTTP endpoints, containerised, auto-deployed via CI/CD,
and fully traced with Langfuse.

### Live API Endpoints
| Method | Endpoint         | Purpose                                            |
|--------|------------------|----------------------------------------------------|
| GET    | `/health`        | Health check + model load status                  |
| GET    | `/rag-context`   | Inspect the RAG knowledge base                    |
| POST   | `/segment`       | Semantic segmentation of an uploaded image        |
| POST   | `/change-detect` | Change detection between two uploaded images      |
| POST   | `/vqa`           | Visual QA (LLaVA + RAG) on an uploaded image      |
| POST   | `/analyze`       | Full pipeline: segmentation + VQA in one call     |

### Deployment Stack
```
FastAPI (serves the API on GPU VM)
    ↓
SegFormer-b2        (segmentation + change detection — Phase 2/3)
LLaVA-1.5-7B + LoRA (VQA — Phase 4a)
RAG layer           (Phase 3 change stats injected into every prompt)
    ↓
Docker (CUDA 12.4 image, --gpus all)
GCP VM (g2-standard-4, NVIDIA L4, asia-south1-b)
    ↓
GitHub Actions CI/CD (lint → build → push → deploy)
Langfuse (per-request tracing)
```

### RAG Layer
On startup the API loads Phase 3 segmentation masks (`seg_2015.npy`,
`seg_2024.npy`, `change_map.npy`) from GCS, computes live class distributions
and change-type breakdowns, and formats them into a knowledge-base text block.
This context is prepended to every LLaVA prompt so answers are grounded in
real change statistics rather than hallucinated numbers. The context is cached
after first build.

### CI/CD Pipeline (GitHub Actions)
Three jobs run on every push to `main`:
1. **Lint & Import Check** — `ruff` + `py_compile` on the API package
2. **Build & Push** — builds the CUDA Docker image, pushes to Artifact Registry
   with layer caching
3. **Deploy** — SSHes into the GPU VM via IAP, authenticates Docker to the
   registry using the VM service-account token, pulls the new image, and
   restarts the container with `--gpus all`

### Observability (Langfuse)
Every API request creates a trace with child spans:
```
vqa (trace)
 ├── rag-context-retrieval  (span)       — context size, latency
 └── llava-vqa              (generation) — full prompt, answer, latency
```
This gives full visibility into the RAG pipeline — what was retrieved, what
prompt was sent to the model, the generated answer, and per-step latency —
for debugging and performance analysis.

### Key Engineering Challenges Solved
- **GDAL build failure in Docker** — pip GDAL failed to compile (`g++ not
  found`). Fixed by adding build tools + GDAL system libraries and pinning
  pip's GDAL to the system version dynamically via `gdal-config --version`.
- **No GPU for 8-bit quantization** — the initial e2-medium serving VM had no
  GPU, so `bitsandbytes` 8-bit loading failed. Migrated serving to an L4 GPU
  VM with manual NVIDIA driver + Container Toolkit install, and switched the
  Docker base image to `nvidia/cuda:12.4.0-runtime` with CUDA-build PyTorch.
- **PEFT version mismatch** — the adapter was saved with PEFT 0.19.1 but the
  container had 0.11.1, which didn't recognise newer config fields. Fixed by
  bumping peft/transformers/accelerate/bitsandbytes to matching versions.
- **Registry auth over SSH** — `docker pull` over non-interactive SSH didn't
  pick up the gcloud credential helper. Fixed by authenticating with the VM's
  service-account access token via the metadata server.
- **Langfuse region mismatch** — keys belonged to a US-region project but the
  host was set to the generic `cloud.langfuse.com`, causing 401s. Fixed by
  setting `LANGFUSE_HOST` to `https://us.cloud.langfuse.com`.
- **Firewall** — port 8080 was not open; added a firewall rule targeting the
  `geovision-api` network tag.

### Example Request
```bash
curl -X POST http://<vm-ip>:8080/vqa \
  -F "image=@satellite.png" \
  -F "question=What land cover types are visible in this satellite image?" \
  -F "use_rag=true"
```
```json
{
  "request_id": "0a028f09",
  "question": "What land cover types are visible in this satellite image?",
  "answer": "In the satellite image, the visible land cover types include buildings, vegetation, roads, and water bodies.",
  "rag_used": true
}
```

### Known Issues (to address before final evaluation)
- **LoRA adapter weights not attaching** — newer transformers renamed LLaVA's
  internal layer keys, so PEFT reports all adapter keys as "missing" and VQA
  currently runs on *base* LLaVA rather than the fine-tuned weights. Requires
  aligning the transformers version with training or re-saving the adapter.
- **SegFormer classifier head** — the checkpoint's classifier layer shape
  (150 ADE classes) doesn't match the 5-class head, so it loads randomly
  initialised; segmentation output needs the head weights remapped.
- **Model re-download on startup** — the LLaVA base + adapter re-download from
  GCS/HuggingFace on every container start; a persistent disk cache would make
  restarts near-instant.

---

## Setup

### Prerequisites
- GCP account with Earth Engine API enabled
- Google Cloud Storage bucket
- Python 3.11+

### Installation
```bash
git clone https://github.com/BharadwajDinesh/GeoVision-QA.git
cd GeoVision-QA
python -m venv geovision-env
source geovision-env/bin/activate
pip install -r requirements.txt
```

### Authentication
```bash
earthengine authenticate --auth_mode=notebook --force
gcloud auth application-default login
```

### Run Phase 2 Pipeline
```bash
python src/phase2_run.py
```

### Run Phase 3 — Pull Data
```bash
python src/phase3_run.py    # Pull 2015 image
python src/phase3_2024.py   # Pull 2024 image
```

### Run Phase 3 — Change Detection
```bash
python src/phase3_infer.py
```

### Run Phase 4 — Prepare RSVQA Dataset
```bash
python src/rsvqa_prep.py --bucket geovision-data
```

### Run Phase 4 — Fine-tune LLaVA
```bash
# Run on GCP GPU VM (g2-standard-4, NVIDIA L4)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 src/llava_finetune.py --bucket geovision-data --epochs 3 --batch-size 4
```

### Run Phase 4 — Inference
```bash
python src/llava_infer.py --bucket geovision-data --generate-report
```

### Run Training (SegFormer)
```bash
python src/train.py
```

### Phase 4b — Deploy the API
Deployment is automated via GitHub Actions on every push to `main`. To run
the API locally for development:
```bash
uvicorn src.api.main:app --reload --port 8080
```

To build and run the container manually:
```bash
docker build -t geovision-api .
docker run --gpus all -p 8080:8080 \
  -e GCS_BUCKET=geovision-data \
  -e LANGFUSE_PUBLIC_KEY=pk-lf-... \
  -e LANGFUSE_SECRET_KEY=sk-lf-... \
  -e LANGFUSE_HOST=https://us.cloud.langfuse.com \
  geovision-api
```

The GitHub Actions workflow requires these repository secrets: `GCP_PROJECT_ID`,
`GCP_SA_KEY`, `GCP_REGION`, `API_VM_NAME`, `API_VM_ZONE`, `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.

---

## Tech Stack

| Component        | Technology                           |
|------------------|--------------------------------------|
| Satellite data   | Google Earth Engine + Sentinel-2     |
| Cloud storage    | Google Cloud Storage                 |
| Compute          | GCP VM (e2-medium, us-central1-b)   |
| GPU Training     | GCP VM (g2-standard-4, NVIDIA L4)   |
| Geospatial       | rasterio, GDAL, geopandas, osmnx    |
| Deep learning    | PyTorch, HuggingFace Transformers   |
| Segmentation     | SegFormer-b2                         |
| Change detection | SegFormer (mask comparison)          |
| VLM              | LLaVA-1.5-7B + QLoRA (8-bit)        |
| VLM fine-tuning  | PEFT, bitsandbytes, TRL              |
| API serving      | FastAPI + Uvicorn                    |
| Containerisation | Docker (CUDA 12.4 base image)        |
| Serving compute  | GCP VM (g2-standard-4, NVIDIA L4)   |
| CI/CD            | GitHub Actions + Artifact Registry  |
| Observability    | Langfuse                             |

---

## Results Summary

| Phase | Task                    | Model              | Metric          | Result  |
|-------|-------------------------|--------------------|-----------------|---------|
| 1     | Spectral analysis       | —                  | NDVI mean       | 0.256   |
| 2     | Semantic segmentation   | SegFormer-b2       | mIoU            | 0.338   |
| 3     | Change detection        | SegFormer-b2       | Change area     | ~30%    |
| 4a    | VLM fine-tuning         | LLaVA-1.5-7B QLoRA | Trainable params | 0.596% |
| 4a    | Visual QA               | LLaVA-1.5-7B       | Acc (planned)   | TBD     |
| 4b    | API deployment          | FastAPI + Docker   | Endpoints live  | 6/6 ✅  |

---

## Research Context

**Research question:** Investigating the application of foundation models
and vision-language models to multispectral satellite imagery for automated
object detection, semantic segmentation, and change detection. This research
explores fine-tuning transformer-based architectures (SegFormer) on
geospatial datasets and developing a retrieval-augmented VQA system over
Sentinel-2 imagery.

**Institution:** IIIT Kota — M.Tech in AI & Data Science
