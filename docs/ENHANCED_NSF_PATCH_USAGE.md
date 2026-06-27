# Enhanced NSF Patch Usage

This patch adds:

- richer NSF input features: midpoint + length + direction + region id
- lower NSF smoothing strength: default `smooth_weight=0.001`
- an optional simulation-loss fine-tuning wrapper

## Important

These scripts belong in the PhysTwin / NeuSpring repository. A pure VGGT repository does not contain `train_warp.py`, `qqtt`, spring topology files, or PhysTwin checkpoints.

## Copy files to Colab

```bash
%cd /content/PhysTwin
!mkdir -p scripts
# upload files to /content first, or git pull after pushing them
!cp /content/enhanced_neural_spring_field.py scripts/enhanced_neural_spring_field.py
!cp /content/finetune_nsf_simulation_loss.py scripts/finetune_nsf_simulation_loss.py
```

## First experiment: fit enhanced NSF on cand_001

```bash
%cd /content/PhysTwin

!python scripts/finetune_nsf_simulation_loss.py \
  --case-name double_stretch_sloth \
  --base-path /content/PhysTwin/data/different_types \
  --train-frame 134 \
  --topology /content/PhysTwin/results/neuspring_topology_search/double_stretch_sloth/topologies/double_stretch_sloth_cand_001_piecewise_topology.npz \
  --raw-checkpoint /content/PhysTwin/results/neuspring_topology_search/double_stretch_sloth/candidates/cand_001/cand_001_raw_param_opt/iter_199.pth \
  --out-dir /content/PhysTwin/results/neuspring_topology_search/double_stretch_sloth/candidates/cand_001/enhanced_nsf \
  --num-regions 8 \
  --field-fit-steps 2000 \
  --smooth-weight 0.001
```

Output:

```text
.../enhanced_nsf/enhanced_neural_spring_field.pt
.../enhanced_nsf/enhanced_nsf_init_checkpoint.pth
.../enhanced_nsf/enhanced_nsf_summary.json
```

## Simulation-loss fine-tuning

First check train_warp.py arguments:

```bash
%cd /content/PhysTwin
!python train_warp.py --help
```

If it supports `--resume_checkpoint` and `--max_iter`, run:

```bash
%cd /content/PhysTwin

!python scripts/finetune_nsf_simulation_loss.py \
  --case-name double_stretch_sloth \
  --base-path /content/PhysTwin/data/different_types \
  --train-frame 134 \
  --topology /content/PhysTwin/results/neuspring_topology_search/double_stretch_sloth/topologies/double_stretch_sloth_cand_001_piecewise_topology.npz \
  --raw-checkpoint /content/PhysTwin/results/neuspring_topology_search/double_stretch_sloth/candidates/cand_001/cand_001_raw_param_opt/iter_199.pth \
  --out-dir /content/PhysTwin/results/neuspring_topology_search/double_stretch_sloth/candidates/cand_001/enhanced_nsf_sim_ft \
  --num-regions 8 \
  --field-fit-steps 2000 \
  --smooth-weight 0.001 \
  --run-simulation-finetune \
  --train-extra-args "--resume_checkpoint {checkpoint} --max_iter 100"
```

If `train_warp.py` does not support checkpoint resume, add a small loader to `train_warp.py` / `InvPhyTrainerWarp` before training starts.
