# RailFCDnet

RailFCDnet is a railway intrusion detector built on YOLOv5x. This repository is
a compact research release of the final **middle-domain source-pool + railway
perimeter enhancement** implementation. It contains the runnable code, model
configuration, training/evaluation launchers, and the logs of the nine final
experiments. Datasets and model weights are intentionally not included.

## Final method

The released configuration combines two mechanisms:

1. **Middle-domain source pool.** The fixed middle-domain samples are merged
   into the source training pool and shuffled by the source dataloader. With
   `batch_size=12`, every complete optimization step contains 11 samples drawn
   from the combined source/middle pool and one few-shot target-domain sample.
   Middle-domain images have detection labels and perimeter masks and therefore
   participate in supervised training. They are never used for validation or
   testing.
2. **Railway perimeter enhancement (PGM + D1/RMFR).** The PGM branch fuses
   multi-scale railway features into a one-channel perimeter response. Three D1
   blocks use the response to refine P3/P4/P5 before the YOLO detection head.
   The final experiment uses perimeter-mask supervision with `alpha=0.05` and
   disables AAM (`--no-aam --beta 0`).

The final architecture is defined by
`models/yolov5x_seg_SAFM.yaml`. The main training entry point is
`train_with_mmd_rail_seg.py`.

## Repository contents

```text
RailFCDnet/
├── models/                         # YOLOv5x and PGM+D1/RMFR implementation
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

| Splits | Transfer task | Source YAML | Middle-domain directory |
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
The final perimeter mask is read from `.../images_mask/example.png`. Source and
middle-domain samples must provide both detection labels and masks. Target
training samples provide detection labels; middle-domain samples are absent
from target validation and test lists.

Expected middle-domain layout:

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
training followed by fixed-test evaluation. This command reproduces one split:

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

## Notes

- Training uses target-validation performance to select `best.pt`; the target
  test set is only used after training.
- The middle-domain set participates only in training.
- The source/middle pool is shuffled. It does not force one middle sample into
  every batch.
- The code is derived from YOLOv5 and retains the upstream AGPL-3.0 notices in
  the source files.
