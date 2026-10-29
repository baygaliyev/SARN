# Run SARN pre-training on Rome (Google Colab)

Pre-train SARN (self-supervised contrastive) on the Rome road network and dump
the segment embeddings, so they can be compared with the supervised GCN.

You need two files from this machine (already prepared):

* `OSM_Rome_raw.zip`  (~17 MB)  — the Rome dataset in SARN's schema
* the fork at `github.com/baygaliyev/SARN` (branch `rome-support`)

## Steps (one Colab cell each)

### 1. Clone the fork

```python
!git clone --branch rome-support https://github.com/baygaliyev/SARN.git
%cd SARN
```

### 2. Install dependencies (modern stack — NOT requirements.txt)

```python
!pip install -q torch_geometric psutil "pandas<3"
# torch is already installed in Colab; keep the runtime's torch.
```

### 3. Upload + unpack the Rome data

```python
from google.colab import files
files.upload()          # select OSM_Rome_raw.zip
!unzip -q OSM_Rome_raw.zip -d data/
!ls data/OSM_Rome_raw_segment data/OSM_Rome_raw_adjsegment
```

### 4. Pre-train SARN (200 epochs)

```python
!python pretrain_sarn_rome.py --dataset RM --epochs 200
```

Embeddings land in `exp/snapshots/OSM_Rome_raw_SARN_best_embs.pickle`.

### 5. Download the embeddings back

```python
from google.colab import files
files.download('exp/snapshots/OSM_Rome_raw_SARN_best_embs.pickle')
```

Then put the pickle in
`~/workspace/unipi/gps2exposure/SARN/exp/snapshots/` on this machine, and I'll run
the frozen-vs-supervised LOCO comparison against the GCN result.

## Notes

* Runtime → Change runtime type → **GPU** (T4) before running step 4.
* ~187k directed segments; expect roughly 30–60 min for 200 epochs on T4.
* If you want a quick sanity check first, use `--epochs 5` in step 4.
* The `rome-support` branch already contains the CPU/NVML fix, the numpy-2x fix,
  the Rome config, and `utils/osm2roadnetwork_rome.py` (not needed on Colab —
  the data is pre-built).
