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
Phase 3: Change Detection (ChangeFormer)         🔜 In Progress
        ↓
Phase 4: VLM + RAG Deployment (LLaVA + FastAPI)  🔜 Coming Soon
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
│   └── phase2_run.py        # Phase 2 end-to-end pipeline runner
├── notebooks/
│   └── phase1_data_ingestion.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

**Infrastructure:**
- Compute  : GCP VM (n2-standard-4, 4 vCPU, 16GB RAM)
- Training : Google Colab (T4 GPU, free tier)
- Storage  : Google Cloud Storage (no local data storage)
- Region   : us-central1

---

## Phase 1 — Data Ingestion & Spectral Analysis ✅

### What it does
- Connects to Google Earth Engine and pulls Sentinel-2 multispectral imagery over Bengaluru
- Applies cloud masking using QA60 band (bits 10 & 11 for opaque cloud + cirrus)
- Builds a median composite from multiple scenes to get a single clean image
- Computes spectral indices for land cover analysis
- Exports processed GeoTIFF to Google Cloud Storage

### Output
- **File:** `bengaluru_s2_composite_2024_indexed.tif`
- **Size:** 3064 × 3006 px (~30×30 km area)
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
- Slices the large GeoTIFF into 512×512 patches with 64px overlap
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
- Input       : RGB (3 bands), 512×512 patches
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
- **Transfer learning** — fine-tune NVIDIA pretrained SegFormer, not from scratch

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

## Phase 3 — Change Detection 🔜

### Goal
Detect what changed between two satellite images of the same area
at different points in time — new buildings, deforestation, urban
growth, flood damage.

### Plan
```
Bengaluru 2020 Sentinel-2 composite (Phase 1 pipeline)
        +
Bengaluru 2024 Sentinel-2 composite (already done)
        ↓
ChangeFormer model
        ↓
Binary/multi-class change map
(what appeared / disappeared between 2020 and 2024)
```

### Model — ChangeFormer
- Siamese transformer architecture
- Takes two images as input, outputs change mask
- Dataset: LEVIR-CD (large building change detection dataset)
- Use cases: Urban growth, deforestation, flood damage detection

---

## Phase 4 — VLM + Deployment 🔜

### Goal
Natural language question answering over satellite imagery.

### Plan
- Fine-tune LLaVA on RSVQA dataset for satellite image Q&A
- FastAPI endpoint on GCP VM
- Docker containerization
- Live endpoint answering questions like:
  - "How many buildings are in this area?"
  - "Has vegetation decreased since 2023?"
  - "What changed in this region after the flood?"

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

### Run Training
```bash
python src/train.py
```

---

## Tech Stack

| Component        | Technology                           |
|------------------|--------------------------------------|
| Satellite data   | Google Earth Engine + Sentinel-2     |
| Cloud storage    | Google Cloud Storage                 |
| Compute          | GCP VM (n2-standard-4)              |
| Training         | Google Colab (T4 GPU)               |
| Geospatial       | rasterio, GDAL, geopandas, osmnx    |
| Deep learning    | PyTorch, HuggingFace Transformers   |
| Segmentation     | SegFormer-b2                         |
| Change detection | ChangeFormer (Phase 3)               |
| VLM              | LLaVA (Phase 4)                      |
| Deployment       | FastAPI + Docker                     |

---

## Results Summary

| Phase | Task                    | Model        | Metric        | Result |
|-------|-------------------------|--------------|---------------|--------|
| 1     | Spectral analysis       | —            | NDVI mean     | 0.256  |
| 2     | Semantic segmentation   | SegFormer-b2 | mIoU          | 0.338  |
| 3     | Change detection        | ChangeFormer | F1 (planned)  | TBD    |
| 4     | Visual QA               | LLaVA        | Acc (planned) | TBD    |

---

## Research Context

**Research question:** Investigating the application of foundation models
and vision-language models to multispectral satellite imagery for automated
object detection, semantic segmentation, and change detection. This research
explores fine-tuning transformer-based architectures (SegFormer, ChangeFormer)
on geospatial datasets and developing a retrieval-augmented VQA system over
Sentinel-2 imagery.

**Institution:** IIIT Kota — M.Tech in AI & Data Science
