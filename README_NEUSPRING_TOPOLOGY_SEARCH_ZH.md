# NeuSpring-style topology search patch for PhysTwin

目标：先不做 pruning，先按照 NeuSpring 思路搭建：

1. piecewise topology candidates
2. inner PhysTwin parameter optimization
3. neural spring field checkpoint smoothing
4. loss-driven topology search
5. 输出 best optimized topology `.npz`

输出目录默认：

```text
results/neuspring_topology_search/double_stretch_sloth/
├── candidate_summary.csv
├── all_method_runs.csv
├── best_topology_manifest.json
├── double_stretch_sloth_best_optimized_topology_no_pruning.npz
├── topologies/
└── candidates/
```

## 放到 VSCode 的位置

把 `scripts/` 里的 3 个文件复制到 PhysTwin 项目根目录的 `scripts/` 文件夹：

```text
scripts/build_piecewise_candidate_topology.py
scripts/neural_spring_field.py
scripts/neuspring_loss_driven_topology_search.py
```

这三个文件依赖你之前包里已有的：

```text
scripts/run_pipeline.py
scripts/eval_geometry.py
scripts/patch_phystwin.py
```

## Colab 运行

```bash
%cd /content/PhysTwin
!python scripts/patch_phystwin.py --project-dir /content/PhysTwin
```

先快速跑 2 个 candidate 测试流程：

```bash
%cd /content/PhysTwin
!python scripts/neuspring_loss_driven_topology_search.py \
  --project-dir /content/PhysTwin \
  --case-name double_stretch_sloth \
  --base-path /content/PhysTwin/data/different_types \
  --original-topology /content/PhysTwin/results/double_stretch_sloth_phystwin_topology_open3d.npz \
  --num-regions 5 \
  --max-candidates 2 \
  --field-fit-steps 200 \
  --test-weight 0.5
```

确认跑通以后，再跑正式版：

```bash
%cd /content/PhysTwin
!python scripts/neuspring_loss_driven_topology_search.py \
  --project-dir /content/PhysTwin \
  --case-name double_stretch_sloth \
  --base-path /content/PhysTwin/data/different_types \
  --original-topology /content/PhysTwin/results/double_stretch_sloth_phystwin_topology_open3d.npz \
  --num-regions 5 \
  --max-candidates 8 \
  --field-fit-steps 800 \
  --test-weight 0.5
```

保存结果：

```bash
!zip -r /content/drive/MyDrive/PhysTwin_Data/double_stretch_sloth_neuspring_topology_search.zip \
  /content/PhysTwin/results/neuspring_topology_search/double_stretch_sloth
```

## 注意

- 这一步不会 pruning。
- changed topology 通常不能安全复用旧 checkpoint，因为 `spring_Y` 长度随 springs 数量变化。
- 旧 checkpoint 可以作为同一 topology 的参考或 baseline，但 loss-driven topology search 里每个 candidate 需要重新训练/适配参数。
