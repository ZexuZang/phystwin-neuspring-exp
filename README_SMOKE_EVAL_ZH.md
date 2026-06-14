# Smoke Full Evaluation Patch

把 `scripts/smoke_full_compare_pack.py` 复制到 PhysTwin 项目的 `scripts/` 目录。

它会对以下四个方法做完整汇总：

- `cand_000_raw_param_opt`
- `cand_000_nsf_param_opt`
- `cand_001_raw_param_opt`
- `cand_001_nsf_param_opt`

输出：

- `geometry_CD_Track_train_test.csv`
- `render_metrics_train_test.csv`
- `FULL_smoke_metrics_train_test.csv`
- `COMPARE_raw_vs_nsf_train_test.csv`
- `COMPARE_cand000_vs_cand001_train_test.csv`
- 每个 method 的 `GT | Prediction | Abs Error` 视频
- `README_RESULTS.md`
- zip 包

标准渲染目录结构：

```text
/content/PhysTwin/render_eval_smoke/
├── cand_000_raw_param_opt/
│   ├── gt/
│   └── renders/
├── cand_000_nsf_param_opt/
│   ├── gt/
│   └── renders/
├── cand_001_raw_param_opt/
│   ├── gt/
│   └── renders/
└── cand_001_nsf_param_opt/
    ├── gt/
    └── renders/
```
