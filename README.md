# GeoVision QA — Satellite Intelligence Pipeline

A multimodal AI system for object detection, change detection, and
natural language querying over satellite imagery.

## Structure

```
geovision/
├── notebooks/
│   └── phase1_data_ingestion.ipynb   ← Start here
├── src/
│   ├── geo_ingest.py    Earth Engine ingestion & export
│   ├── band_math.py     Spectral index computation
│   ├── raster_io.py     GeoTIFF read/write
│   └── visualize.py     Plotting utilities
├── data/
│   ├── raw/             Downloaded GeoTIFFs from EE
│   └── processed/       Index stacks, visualisations
└── requirements.txt
```

## Setup (GCP VM)

```bash
pip install -r requirements.txt
earthengine authenticate   # first time only
```

## Phases

| Phase | What | Status |
|-------|------|--------|
| 1 | Geospatial data ingestion & spectral analysis | ✅ |
| 2 | Object detection & segmentation (SAM2/SegFormer) | 🔜 |
| 3 | Change detection (ChangeFormer, LEVIR-CD) | 🔜 |
| 4 | VLM fine-tuning + RAG layer (LLaVA, RSVQA) | 🔜 |

## Data sources

- **Imagery**: Sentinel-2 L2A via Google Earth Engine (`COPERNICUS/S2_SR_HARMONIZED`)
- **Detection labels**: DOTA v2
- **Change detection**: LEVIR-CD
- **VQA**: RSVQA
