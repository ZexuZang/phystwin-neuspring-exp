# PhysTwin + NeuSpring-style Optimization: VSCode/GitHub/Colab 工作流

这个文件夹是从你的 Colab notebook 清理出来的 **VSCode + GitHub 项目补丁包**。目标是：

1. 把 notebook 里的散乱 cell 转成可维护的 Python scripts；
2. 删除 pruning 相关模块；
3. 先跑出 `OriginalTopology_Adapt` 和 `RegionWiseOptimizedTopology_Adapt` 的优化 / inference / CD-track 结果；
4. 支持复用你已经生成的 `double_stretch_sloth_regionwise_optimization_artifacts` 和 `double_stretch_sloth_checkpoints`，尽量不重复训练 checkpoint；
5. 为后续加入完整 NeuSpring 流程留下清晰接口。

> 重要：这个包没有假装实现官方 NeuSpring 的完整 neural spring field。当前公开的 `GhiXu/NeuSpring` 仓库主要只有 README，没有可直接集成的完整代码。因此这里先把你的 notebook 转成干净、可复现、无 pruning 的 PhysTwin + piecewise topology optimization pipeline。下一步再在 `scripts/neural_spring_fields_todo.py` 对 PhysTwin trainer 做真正的 neural field 参数化改造。

---

## 目录结构

```text
phystwin_neuspring_vscode_package/
├── .gitignore
├── requirements_colab.txt
├── README_ZH.md
├── scripts/
│   ├── unpack_drive_data.py
│   ├── patch_phystwin.py
│   ├── build_regionwise_topology.py
│   ├── run_pipeline.py
│   ├── eval_geometry.py
│   ├── pack_artifacts.py
│   └── neural_spring_fields_todo.py
└── notebooks/
    └── run_no_pruning_optimization_colab.ipynb
```

---

## 第 1 步：在 VSCode 里准备代码

打开 VSCode，然后打开 Terminal：

```bash
cd ~/Desktop
git clone https://github.com/Jianghanxiao/PhysTwin.git phystwin-neuspring-exp
cd phystwin-neuspring-exp
```

把这个补丁包里的文件复制到 PhysTwin 项目根目录：

```bash
# 假设你把本包解压到了 ~/Downloads/phystwin_neuspring_vscode_package
cp -r ~/Downloads/phystwin_neuspring_vscode_package/scripts .
cp -r ~/Downloads/phystwin_neuspring_vscode_package/notebooks .
cp ~/Downloads/phystwin_neuspring_vscode_package/.gitignore .
cp ~/Downloads/phystwin_neuspring_vscode_package/requirements_colab.txt .
cp ~/Downloads/phystwin_neuspring_vscode_package/README_ZH.md .
```

在 VSCode 里你应该看到：

```text
scripts/
notebooks/
requirements_colab.txt
README_ZH.md
```

---

## 第 2 步：上传到你自己的 GitHub

先在 GitHub 网页上新建一个空仓库，例如：

```text
phystwin-neuspring-exp
```

然后在 VSCode terminal 里：

```bash
git checkout -b zexu-no-pruning-opt
git add scripts notebooks requirements_colab.txt README_ZH.md .gitignore
git commit -m "Convert Colab notebook to no-pruning optimization pipeline"

git remote rename origin upstream
git remote add origin https://github.com/zexuzang1/phystwin-neuspring-exp.git
git push -u origin zexu-no-pruning-opt
```

以后你每次修改：

```bash
git add .
git commit -m "update pipeline"
git push
```

---

## 第 3 步：Google Drive 放数据和旧结果

建议 Drive 目录改成无空格：

```text
MyDrive/PhysTwin_Data/
├── data.zip
├── experiments.zip
├── experiments_optimization.zip
├── gaussian_output.zip
├── double_stretch_sloth_regionwise_optimization_artifacts.zip     # 可选
└── double_stretch_sloth_checkpoints.zip                           # 可选
```

如果你不是 zip，而是文件夹，也可以：

```text
MyDrive/PhysTwin_Data/
├── double_stretch_sloth_regionwise_optimization_artifacts/
└── double_stretch_sloth_checkpoints/
```

---

## 第 4 步：Colab 运行

Colab 第一格：

```python
from google.colab import drive
drive.mount("/content/drive")

!nvidia-smi
```

第二格：clone 你的 GitHub repo：

```bash
%cd /content
!rm -rf PhysTwin
!git clone -b zexu-no-pruning-opt https://github.com/zexuzang1/phystwin-neuspring-exp.git PhysTwin
%cd /content/PhysTwin
```

第三格：安装依赖。先用轻量安装，跑不通再补：

```bash
%cd /content/PhysTwin
!python -m pip install -r requirements_colab.txt
```

第四格：解压 Drive 数据：

```bash
%cd /content/PhysTwin
!python scripts/unpack_drive_data.py \
  --drive-dir "/content/drive/MyDrive/PhysTwin_Data" \
  --project-dir "/content/PhysTwin"
```

第五格：打补丁：

```bash
%cd /content/PhysTwin
!python scripts/patch_phystwin.py --project-dir "/content/PhysTwin"
```

第六格：跑无 pruning 的优化 pipeline。先尝试复用旧 artifacts/checkpoints：

```bash
%cd /content/PhysTwin
!python scripts/run_pipeline.py \
  --project-dir "/content/PhysTwin" \
  --case-name double_stretch_sloth \
  --base-path "/content/PhysTwin/data/different_types" \
  --original-topology "/content/PhysTwin/results/double_stretch_sloth_phystwin_topology_open3d.npz" \
  --reuse-artifact-zip "/content/drive/MyDrive/PhysTwin_Data/double_stretch_sloth_regionwise_optimization_artifacts.zip" \
  --reuse-checkpoint-zip "/content/drive/MyDrive/PhysTwin_Data/double_stretch_sloth_checkpoints.zip" \
  --reuse-checkpoints \
  --skip-existing-inference
```

如果你的旧结果是文件夹而不是 zip：

```bash
!python scripts/run_pipeline.py \
  --project-dir "/content/PhysTwin" \
  --case-name double_stretch_sloth \
  --base-path "/content/PhysTwin/data/different_types" \
  --original-topology "/content/PhysTwin/results/double_stretch_sloth_phystwin_topology_open3d.npz" \
  --reuse-artifacts-dir "/content/drive/MyDrive/PhysTwin_Data/double_stretch_sloth_regionwise_optimization_artifacts" \
  --reuse-checkpoints \
  --skip-existing-inference
```

输出会在：

```text
/content/PhysTwin/results/regionwise_optimization_final/double_stretch_sloth/
├── topologies/
├── OriginalTopology_Adapt/
├── RegionWiseOptimizedTopology_Adapt/
├── base_adaptation_summary.csv
├── topology_compare.csv
├── cd_track.csv
└── final_summary.csv
```

打包回 Drive：

```bash
%cd /content/PhysTwin
!python scripts/pack_artifacts.py \
  --adapt-root "/content/PhysTwin/results/regionwise_optimization_final/double_stretch_sloth" \
  --out-zip "/content/drive/MyDrive/PhysTwin_Data/double_stretch_sloth_no_pruning_optimization_results.zip"
```

---

## 这个版本删除了什么

已删除 / 不再运行：

- `RegionWiseOptimized_StiffnessKeep0p5_Adapt`
- `RegionWiseOptimized_RandomKeep0p5_Adapt`
- pruning topology 构建
- pruning 后 adaptation
- keep ratio summary
- notebook 里重复的 render runner cell
- notebook 里重复/临时修补 cell

保留：

- Drive 解压
- PhysTwin patch
- external topology `.npz`
- region-wise topology construction
- train / inference
- checkpoint reuse
- CD / Track Error
- artifact packaging

---

## 后续要做的完整 NeuSpring 化

完整 NeuSpring 不是“多 region topology + pruning”。它至少需要：

1. **loss-driven topology search**：用 CD / Track Error 或训练 loss 选择 topology，而不是纯启发式 KNN。
2. **joint topology-parameter optimization**：topology 搜索时同时考虑 homogeneous spring 参数初始化。
3. **neural spring fields**：把每根 spring 的 stiffness/damping 变成 `S0 + F_theta(x_mid)`，而不是每根 spring 单独学一个自由参数。
4. **再 pruning**：先得到更好的 physical field，再基于 stiffness / effective stiffness / contribution 做剪枝。

本包先把 pruning 删掉、把你现有优化结果整理出来；下一步我们再改 `trainer_warp.py`，让 `spring_Y` 由 CCNN / tri-plane field 生成。
