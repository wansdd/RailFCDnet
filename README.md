# RailFCDnet

RailFCDnet is a railway intrusion detector built on YOLOv5x. This repository is
a compact research release of the final **Structure-Constrained Target-Context
Augmentation (SCTA) + Region-Guided Multi-Scale Feature Refinement (RMFR)**
implementation. It contains the runnable code, model configuration,
training/evaluation launchers, and the logs of the nine final experiments.
Datasets and model weights are intentionally not included.

## Abstract

Railway foreign-object intrusion detection is challenged by domain shifts and
scarce labeled data in newly deployed scenes. We design a few-shot cross-domain
detection framework using only eight labeled target-domain images for
adaptation. To improve adaptation under this constraint, the framework combines
Structure-Constrained Target-Context Augmentation (SCTA) and Region-Guided
Multi-Scale Feature Refinement (RMFR). SCTA constructs
source-object-target-context samples under railway-region constraints, while
RMFR derives a railway-region response from hierarchical backbone features to
refine multi-scale detection features. The two components operate at
complementary levels: SCTA expands target-context coverage at the data level,
whereas RMFR enhances region-related representation learning at the feature
level. Moreover, SCTA is used only during training and therefore introduces no
additional inference-time complexity. Experiments on synthetic-to-real,
cross-site, and cross-weather transfer show that the proposed method achieves
an overall mean mAP50 of 74.9%, outperforming the strongest comparator by 4.6
percentage points in the aggregate. Ablation studies confirm the complementary
contributions of SCTA and RMFR. Further analyses examine the effects of SCTA
training organization and regional supervision design. The code is available
in this [GitHub repository](https://github.com/wansdd/RailFCDnet).

## Final method

The released configuration combines two complementary mechanisms:

1. **Structure-Constrained Target-Context Augmentation (SCTA).** SCTA constructs
   source-object-target-context samples under railway-region constraints. The
   released training pipeline consumes these pre-generated SCTA samples from
   the `middle_domain/` directories, merges them into the source training pool,
   and shuffles the combined pool through the source dataloader. With
   `batch_size=12`, every complete optimization step contains 11 samples drawn
   from the combined source/SCTA pool and one few-shot target-domain sample.
   SCTA samples provide detection labels and railway-region masks for supervised
   training. They are never used for validation, testing, or inference.
2. **Region-Guided Multi-Scale Feature Refinement (RMFR).** RMFR derives a
   railway-region response from hierarchical backbone features and uses it to
   refine P3/P4/P5 before the YOLO detection head. In the implementation, the
   response branch and three refinement blocks retain the legacy internal names
   PGM and D1. The final experiment uses railway perimeter-mask supervision with
   `alpha=0.05` and disables AAM (`--no-aam --beta 0`).

The final architecture is defined by
`models/yolov5x_seg_SAFM.yaml`. The main training entry point is
`train_with_mmd_rail_seg.py`.

## Repository contents

```text
RailFCDnet/
├── models/                         # YOLOv5x and RMFR implementation
├── utils/                          # dataloader, augmentation, loss, metrics
├── data/hyps/rail.yaml             # final training hyperparameters
├── raildatasplit/*.yaml            # source/target config examples; edit paths
├── logs/
│   ├── train/{d1..f3}.log          # final successful training logs
│   ├── test/{d1..f3}.log           # fixed-test evaluation logs
│   └── results/{split}/            # epoch CSV and resolved opt/hyp YAML
├── run_ep300_method.sh             # recommended one-command launcher
├── run_pgm_only.sh                 # low-level train + fixed-test runner
├── run_middle_pool_pgm_d1_9_wait.sh # four-GPU nine-split launcher
├── train_with_mmd_rail_seg.py
├── val.py
└── detect.py
```

Only `README.md` is retained as project documentation. Historical reports,
ablation code, old logs, checkpoints, images, masks, caches, and visualization
outputs are excluded from this compact release.

## Environment

The published experiments used:

- Python 3.8.16
- PyTorch 1.10.1 + CUDA 11.1
- torchvision 0.11.2
- four NVIDIA RTX 3090 GPUs
- conda environment name `yolov5-7.0`

Create or activate a CUDA-enabled environment, then install the Python
dependencies. Install the PyTorch build matching the local CUDA driver first.

```bash
conda activate yolov5-7.0
pip install -r requirements.txt
```

Place the YOLOv5x pretrained checkpoint at the repository root as
`yolov5x.pt`, or set `WEIGHTS=/absolute/path/to/yolov5x.pt` when using the
low-level runner.

## Dataset preparation

No dataset content or image-list file is published. Edit the YAML files in
`raildatasplit/` and create their referenced train/validation/test lists.
Each list is a UTF-8 text file containing one absolute image path per line.

The experiment mapping is:

| Splits | Transfer task | Source YAML | SCTA sample directory |
| --- | --- | --- | --- |
| d1, d2, d3 | S to R | `raildatasplit/A.yaml` | `middle_domain/S_R/images` |
| e1, e2, e3 | A to B | `raildatasplit/B.yaml` | `middle_domain/A_B/images` |
| f1, f2, f3 | G to T | `raildatasplit/C.yaml` | `middle_domain/G_T/images` |

A target split YAML follows this form:

```yaml
train: /absolute/path/to/d1_train.txt
val: /absolute/path/to/d1_val.txt
test: /absolute/path/to/d_test.txt
nc: 2
names:
  0: person
  1: yiwu
```

Use standard YOLO normalized detection labels:

```text
class_id x_center y_center width height
```

For an image at `.../images/example.png`, the loader first looks for its label
at `.../labels/example.txt` and supports a label beside the image as fallback.
The final railway-region mask is read from `.../images_mask/example.png`.
Source and SCTA samples must provide both detection labels and masks. Target
training samples provide detection labels; SCTA samples are absent from target
validation and test lists.

Expected SCTA sample layout (the directory name is retained for compatibility):

```text
middle_domain/
├── S_R/{images,labels,images_mask}/
├── A_B/{images,labels,images_mask}/
└── G_T/{images,labels,images_mask}/
```

Mask images must have the same basename as their images. Foreground perimeter
pixels use value 255 and background pixels use value 0.

## Run the final method

The recommended launcher validates the method/split arguments and invokes
training followed by fixed-test evaluation. The CLI values `pgm-middle` and
`pool` are legacy implementation names corresponding to SCTA + RMFR. This
command reproduces one split:

```bash
CONDA_ENV=yolov5-7.0 \
CONDA_SH=/opt/miniconda/etc/profile.d/conda.sh \
bash run_ep300_method.sh \
  --method pgm-middle \
  --middle-mode pool \
  --splits d1 \
  --batch 12 \
  --workers 8 \
  --img 640 \
  --epochs 300 \
  --patience 400 \
  --seed 42 \
  --train-dev 0,1 \
  --test-dev 0
```

Run all nine splits sequentially on one two-GPU lane:

```bash
bash run_ep300_method.sh \
  --method pgm-middle --middle-mode pool \
  --splits d1,d2,d3,e1,e2,e3,f1,f2,f3 \
  --batch 12 --workers 8 --epochs 300 --patience 400 \
  --train-dev 0,1 --test-dev 0
```

Preview the fully resolved configuration without starting training:

```bash
bash run_ep300_method.sh \
  --method pgm-middle --middle-mode pool --splits d1 --dry-run
```

### Four-GPU launcher

The four-GPU launcher waits until current compute processes finish, then runs
`d1 d2 d3 e1 e2` on GPUs 0/1 and `e3 f1 f2 f3` on GPUs 2/3:

```bash
CONDA_ENV=yolov5-7.0 \
CONDA_SH=/opt/miniconda/etc/profile.d/conda.sh \
bash run_middle_pool_pgm_d1_9_wait.sh
```

Optional PID arguments make the launcher wait for specified local jobs first:

```bash
bash run_middle_pool_pgm_d1_9_wait.sh 12345 12346
```

## Evaluate a trained checkpoint

The launchers automatically evaluate `best.pt` on the fixed target test set.
For manual evaluation:

```bash
python val.py \
  --task test \
  --data raildatasplit/d1.yaml \
  --img 640 \
  --device 0 \
  --weights runs_ep300/middle_source_pool_pgm_d1/ours/d1/weights/best.pt \
  --project runs_ep300/middle_source_pool_pgm_d1/ours \
  --name d1_test \
  --exist-ok
```

## Published results

The metric is target fixed-test mAP50 selected by the best target-validation
checkpoint. All values below are taken directly from `logs/test/*.log`.

| Split | Task | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| d1 | S to R | 92.3 | 81.7 | 84.9 | 40.9 |
| d2 | S to R | 92.1 | 81.2 | 84.6 | 43.4 |
| d3 | S to R | 90.6 | 77.2 | 79.0 | 35.1 |
| e1 | A to B | 96.5 | 78.3 | 82.1 | 42.2 |
| e2 | A to B | 95.1 | 73.8 | 78.4 | 43.4 |
| e3 | A to B | 88.1 | 78.4 | 82.6 | 38.7 |
| f1 | G to T | 94.6 | 48.5 | 57.5 | 36.8 |
| f2 | G to T | 90.7 | 49.6 | 59.7 | 36.9 |
| f3 | G to T | 86.1 | 59.0 | 65.6 | 40.1 |

The nine-split mean mAP50 is **74.9%**. Group means are 82.8% for S to R,
81.0% for A to B, and 60.9% for G to T.

## Output locations

By default, the final launcher writes:

- checkpoints and per-epoch CSV: `runs_ep300/middle_source_pool_pgm_d1/ours/`
- training logs: `logs/ep300_method/pgm_d1_middle_pool/`
- fixed-test log for split `<split>`: `<project>/<split>_test.log`

The checked-in `logs/` directory is a compact copy of the final successful
nine-split run and does not contain model weights or dataset files.

The output directory names retain `middle_source_pool` and `pgm_d1` for
compatibility with the completed experiments; these correspond to SCTA and
RMFR, respectively.

## Notes

- Training uses target-validation performance to select `best.pt`; the target
  test set is only used after training.
- SCTA participates only in training and adds no inference-time complexity.
- The source/SCTA pool is shuffled. It does not force one SCTA sample into every
  batch.
- The code is derived from YOLOv5 and retains the upstream AGPL-3.0 notices in
  the source files.
