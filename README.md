# ISD-YOLO

Official source code for **ISD-YOLO: A Lightweight and Efficient Feature Extraction Network for Invoice Seal Detection**.

ISD-YOLO is a lightweight single-class object detector for locating invoice seals. It is implemented on top of Ultralytics YOLO and combines the following components:

- **C3k2_CFC**: a C3k2-ConvFormer-CGLU feature extraction module implemented as `C3k2_ConvFormerCGLU`.
- **C2BRA**: a C2PSA-derived module using bi-level routing attention.
- **DAttention**: deformable attention for adaptive spatial sampling.
- **EfficientHead**: a lightweight detection head implemented as `Detect_Efficient`.

The implementation name `C3k2_ConvFormerCGLU` corresponds to the abbreviated module name `C3k2_CFC` used in the manuscript. The model YAML retains the standard YOLO11 default of `nc: 80`; during training, Ultralytics automatically overrides this value with the single-class definition (`seal`) in the dataset YAML.

## Repository structure

```text
ISD-YOLO/
|-- ultralytics/
|   |-- cfg/models/ISD-YOLO/ISD-YOLO.yaml  # Model architecture
|   |-- nn/extra_modules/                   # ISD-YOLO modules
|   `-- nn/tasks.py                         # Model parser integration
|-- DATA_AVAILABILITY.md
|-- LICENSE
|-- pyproject.toml
|-- train.py
|-- val.py
`-- README.md
```

## Installation

Python 3.8 or later is required. A CUDA-enabled PyTorch installation is recommended for training.

```bash
git clone https://github.com/Qing1815/ISD-YOLO.git
cd ISD-YOLO
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the project in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Dataset configuration

The detector uses one class, `seal`. Prepare an Ultralytics-compatible dataset YAML file, for example:

```yaml
path: /absolute/path/to/seal_dataset
train: images/train
val: images/val
test: images/test
names:
  0: seal
```

The enterprise-provided invoice images used in the associated study are not included in this repository because of confidentiality and third-party restrictions. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for the public data sources and sharing conditions.

## Training

The following command reproduces the main training settings reported in the manuscript. Replace `/path/to/data.yaml` with the local dataset configuration.

```bash
python train.py --data /path/to/data.yaml --device 0
```

By default, `train.py` initializes the custom architecture with `yolo11n.pt` and applies the complete training and augmentation settings reported in the manuscript. Use `--weights ""` to train without pretrained weights.

## Evaluation

Use a trained checkpoint and the fixed test split for evaluation:

```bash
python val.py --weights /path/to/best.pt --data /path/to/data.yaml --device 0
```

Trained weights are not included in this repository.

## Data sources

The study used public seal images from:

- Kaggle StaVer dataset: https://www.kaggle.com/datasets/rtatman/stamp-verification-staver-dataset
- Roboflow StampDet dataset (version 10): https://universe.roboflow.com/marcos-7aslt/stampdet/dataset/10

Users must follow the terms and licenses specified by the original data providers.

## License and acknowledgement

This repository is derived from Ultralytics YOLO 8.3.9 and is distributed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE) for details. Ultralytics YOLO documentation is available at https://docs.ultralytics.com.

## Citation

Citation information will be added after publication of the associated manuscript.
