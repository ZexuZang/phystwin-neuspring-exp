# NeuSpring Enhanced NSF Patch

This patch implements:

1. Enhanced NSF input features:
   - spring midpoint
   - spring length
   - spring direction
   - region id embedding
2. Lower NSF smoothing strength.
3. Optional simulation-loss fine-tuning wrapper.

Recommended first comparison:

- `cand_001_raw_param_opt`
- `cand_001_enhanced_nsf`
- `cand_001_enhanced_nsf_sim_ft` if simulation fine-tuning is supported by your `train_warp.py`.

The goal is to prevent NSF from becoming only a smoothed, lower-fidelity copy of raw spring parameters.
