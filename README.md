# 🛰️ SVAMITVA Feature Extraction Pipeline
**MoPR × IIT Tirupati — 2026**

> Automatically identify and map land features — buildings, roads, water bodies, and utilities — from high-resolution drone imagery of Indian villages, in support of the Government of India's SVAMITVA property rights scheme.

---

## 🗺️ What Does This Pipeline Do?

Imagine you have aerial photographs of thousands of Indian villages. Manually marking every rooftop, road, pond, and electricity transformer would take years. This pipeline does it automatically.

It takes raw drone images as input and produces GIS-ready map files (GeoPackages) where every detected feature is a labeled polygon — ready for use in land records, urban planning, or infrastructure surveys.

**Input:** High-resolution GeoTIFF drone images of villages  
**Output:** GeoPackage (`.gpkg`) files with labeled feature polygons + analysis charts

---

## 🏷️ What Gets Detected?

The pipeline recognises **9 land feature classes** across every village image:

| # | Feature | Colour | Notes |
|---|---------|--------|-------|
| 1 | 🔴 RCC Roof | Red | Reinforced concrete — permanent structures |
| 2 | 🟠 Tiled Roof | Orange | Most common class (41.3% of all features) |
| 3 | 🟡 Tin Roof | Yellow | Corrugated metal sheets |
| 4 | 🟫 Thatched Roof | Brown | Traditional rural construction |
| 5 | ⬜ Road | Grey | Paved and unpaved paths |
| 6 | 🔵 Waterbody | Blue | Ponds, lakes, rivers |
| 7 | 🟣 Transformer | Purple | Electrical utility boxes |
| 8 | 🟩 Tank | Teal | Water storage tanks |
| 9 | 💚 Well | Green | Wells (rare; only ~3 per village) |

---

## 🧭 Two Ways to Run the Pipeline

There are two scripts — one simpler and one more powerful. Both solve the same problem but at different scales.

---

### 🅰️ Pipeline A — Stage-by-Stage (`--stage`)

The simpler version. Uses a single DuSA U-Net for segmentation and a Faster R-CNN for small utility objects.

```
Input images → Masks → Patches → Augment → Train → Predict → Merge & Report
    Stage 1      Stage 2  Stage 3   Stage 4/5  Stage 6    Stage 7
```

| Stage | What It Does | Where Results Go |
|-------|-------------|-----------------|
| **1** | Converts shapefiles into pixel-level mask images | `data/training/masks_raster/` |
| **2** | Cuts large images into 512×512 tiles (stride 256) | `data/training/patches/` |
| **3** | Creates extra copies of rare classes so the model sees them more (Tank ×5, Well ×65) | Same patch folders |
| **4** | Trains the **DuSA U-Net** segmentation model | `outputs/best_model.pth` |
| **5** | Trains a **Faster R-CNN** specifically for tiny objects (Transformer, Tank, Well) | `outputs/final/faster_rcnn_utilities.pth` |
| **6** | Runs both models on test villages; outputs feature polygons | `outputs/predictions/` + `outputs/rcnn_utilities/` |
| **7** | Merges both models' outputs, evaluates accuracy, saves charts | `outputs/final_predictions/` + PNG charts |

**Quick commands:**
```bash
python svamitva_pipeline.py --stage all       # run everything
python svamitva_pipeline.py --stage 1,2,3     # data prep only
python svamitva_pipeline.py --stage 4,5       # training only
python svamitva_pipeline.py --stage 6,7       # predict + report
python svamitva_pipeline.py --stage 1         # just masks
```

---

### 🅱️ Pipeline B — Phase-by-Phase (`--phase`)

The full-power version. Trains **three models** and combines them into an ensemble for higher accuracy. Also handles ECW format drone files and multi-state datasets (Chhattisgarh + Punjab).

```
data → train → evaluate → tune → predict
```

| Phase | What It Does |
|-------|-------------|
| `data` | Converts ECW → TIF if needed, builds masks from shapefiles, creates 256×256 patches for all villages |
| `train` | Trains all three models (SegFormer, Mask2Former, DuSA U-Net++) separately with full logging |
| `evaluate` | Measures accuracy per class, generates confusion matrices and comparison bar charts |
| `tune` | Grid-searches the best blend ratio between the three models |
| `predict` | Runs the ensemble on test villages → GeoPackage + prediction overlay images |
| `all` | Runs all phases end-to-end |

**Quick commands:**
```bash
python svamitva_pipeline.py --phase all        # run everything
python svamitva_pipeline.py --phase train      # training only
python svamitva_pipeline.py --phase predict    # inference only
```

**Jupyter:** Set `PHASE = "train"` near the bottom of the file and run the cell.

---

## 🧠 The Models Explained

### DuSA U-Net / DuSA U-Net++
*The workhorse — handles all roof types, roads, and water bodies.*

A U-Net style encoder-decoder where every layer has been upgraded with **Dual Attention** — the model learns both *which channels matter* (channel attention) and *where in the image to look* (spatial attention). The bottleneck uses **ASPP** (multiple parallel dilated convolutions) to capture features at different scales simultaneously, which is important when buildings range from tiny huts to large concrete structures.

During training, two extra prediction heads in the middle of the decoder provide additional gradient signals, helping the model learn faster and more robustly.

### SegFormer-B5
*A transformer-based model pretrained on large scene datasets.*

Uses a hierarchical vision transformer encoder that captures long-range context — helpful for understanding that a patch of land is a road because of what surrounds it, not just its local texture.

Fine-tuned with a **lower learning rate on the encoder** (preserves pretrained knowledge) and a **higher rate on the new decoder** (learns the SVAMITVA classes quickly).

### Mask2Former
*A universal segmentation model from Meta AI.*

Treats segmentation as a query-matching problem — it learns a set of "queries" each representing a possible object, then matches them to regions in the image. Useful for detecting well-defined shapes like water tanks.

### Faster R-CNN (Utility Detector)
*Specialised for tiny objects that segmentation models miss.*

Transformers, tanks, and wells are often just a handful of pixels in a large village image. A standard segmentation model struggles with them. Faster R-CNN treats them as object detection (bounding boxes + class labels), with very small anchor sizes `(16, 32, 64, 128, 256 px)` tuned to these tiny targets.

### The Ensemble
The three segmentation models each predict a probability map. These are blended:

```
Final prediction = 0.45 × SegFormer + 0.35 × Mask2Former + 0.20 × DuSA U-Net++
```

The weights are not fixed — the `tune` phase runs a grid search over all combinations to find the best blend for your specific dataset.

---

## ⚙️ Training — Key Design Choices

### Why a Combined Loss Function?
A single cross-entropy loss would just optimise for overall pixel accuracy, which means it would mostly learn the majority class (rooftops) and ignore rare classes (wells, tanks). The pipeline uses four losses together:

| Loss | Weight | Why It's Here |
|------|--------|---------------|
| Label-Smoothing Cross Entropy | 0.30 | Prevents overconfident wrong predictions |
| Dice Loss | 0.40 | Directly optimises overlap — great for imbalanced classes |
| Focal Loss | 0.20 | Down-weights easy pixels, focuses the model on hard/rare regions |
| Boundary Loss | 0.10 | Penalises fuzzy edges — produces cleaner polygon outputs |

### Handling Rare Classes
Tanks and wells appear in very few patches. Two strategies are combined:
- **Inverse-frequency class weights** — rare classes get higher loss weight
- **Targeted augmentation** — Tank patches are duplicated ×5, Well patches ×65, so the model sees proportionally more examples of rare objects

### Training Stability
- **EMA (Exponential Moving Average, decay 0.9998):** Maintains a slowly-updating "shadow" copy of model weights. The final saved model is the EMA version — smoother and more robust than raw training weights
- **Mixed Precision (AMP float16):** Halves GPU memory usage, doubles throughput
- **Gradient Clipping (max norm 1.0):** Prevents unstable weight updates
- **OneCycleLR Scheduler:** Warms up learning rate over the first 10% of training, then anneals down via cosine — avoids local minima early and fine-tunes carefully late

---

## 🔍 Inference — How Prediction Works

Large village images (often 10,000+ pixels wide) cannot be fed into a GPU at full resolution. The pipeline uses a **sliding window** approach with several tricks to keep predictions clean:

1. **Sliding window:** The image is divided into 256×256 tiles with stride 256
2. **Test-Time Augmentation (TTA):** Each tile is predicted 8 times (4 rotations × 2 flips). All 8 predictions are averaged — this typically adds 1-2% accuracy at the cost of 8× inference time
3. **Gaussian blending:** Where tiles overlap, they are blended using a 2D Gaussian weight map (centre pixels trusted more than edges), eliminating seam artefacts
4. **Per-class thresholds:** Each class has a tuned confidence threshold before a pixel is assigned that label (e.g. Transformer needs only 0.25 confidence due to rarity; Waterbody needs 0.42 to avoid false positives)
5. **Morphological cleanup:** Small noise blobs are removed with opening/closing kernels; contours smaller than 100 pixels are deleted

---

## 📊 Actual Results (Test Run)

Across 10 test villages, the pipeline extracted **~1.92 million features**:

| Village | Features Detected |
|---------|-----------------|
| BUTTAR_SIVIYA_AMR | 552,077 |
| CHANABHATA_44547 | 359,495 |
| PARAGAON_444686_OR | 151,741 |
| KARTARPUR_AMRITSAR | 119,903 |
| BADRA_BARNALA_4004 | 225,905 |
| ANAITPURA_FATEHGAR | 202,200 |
| GUDBHELI_443483 | 51,135 |
| *(+ 3 more villages)* | |
| **Total** | **~1,922,916** |

**Feature breakdown:**
- Tiled Roof — 794,989 (41.3%)
- RCC Roof — 732,229 (38.1%)
- Thatched Roof — 255,868 (13.3%)
- Tin Roof — 130,959 (6.8%)
- Road — 4,456 · Transformer — 4,102 · Waterbody — 240 · Tank — 72

**Model split:** DuSA U-Net produced 99.8% of features (1,918,742). Faster R-CNN contributed 4,174 utility detections (Transformers and Tanks).

---

---

## 💡 Frequently Asked Questions

**Q: Why are there two pipelines in one file?**  
Pipeline A (--stage) is the original single-model version. Pipeline B (--phase) is an upgraded ensemble version. Both coexist for backward compatibility. If you're starting fresh, use Pipeline B.

**Q: Why does DuSA U-Net produce 99.8% of features but Faster R-CNN also exists?**  
Faster R-CNN handles three tiny utility classes (Transformer, Tank, Well) that are too small for segmentation to catch reliably. These are merged into the final output at Stage 7.

**Q: The Well class has almost no samples. Is it still useful?**  
Yes — the ×65 augmentation multiplier and high inverse-frequency weight mean the model still learns to recognise wells. The low absolute count (fewer than 100 detections) reflects how rare wells truly are in the test villages.

**Q: How long does training take?**  
Pipeline A (DuSA U-Net, 50 epochs) runs on GPU 5. Pipeline B trains three separate models for 8 epochs each on GPU 6. Expect several hours per model on a single A100/V100.

**Q: What if my shapefiles use a different CRS?**  
The pipeline re-projects all shapefiles to match each TIF's CRS automatically using `gdf.to_crs(crs)`.

<img width="1568" height="733" alt="image" src="https://github.com/user-attachments/assets/8a8ca954-067b-4cc8-bdff-926f66f15dec" />
<img width="1270" height="952" alt="image" src="https://github.com/user-attachments/assets/83c07945-fd99-46dc-af5b-729bd5c32818" />
<img width="1092" height="1104" alt="image" src="https://github.com/user-attachments/assets/69bd17af-7484-4646-a0cf-7bcdcabcbd7c" />
