<!-- 用途：論文重現包（權重＋P1 資產）之上傳與下載說明；由 tools/make_release_bundle.py 產生。 -->

# 重現包上傳與下載說明

本檔由 `tools/make_release_bundle.py` 產生。清單與 sha256 見
`results/release_bundle_manifest.json`。權重與 P1 資產不進 git；下列步驟由作者執行一次，之後重現者依 sha256 核對。

## 內容

- 發布用權重（重現論文所需最小集合，共 1.6 GiB）：
  - `checkpoints/cifar10_cfg.pt`（730.9 MiB，sha256 `571435b1d4024fc6…`）
  - `checkpoints/cifar100_cfg.pt`（731.3 MiB，sha256 `c26373f37ab79f4c…`）
  - `checkpoints/cifar10_judge.pt`（42.7 MiB，sha256 `ff1de40d25728e97…`）
  - `checkpoints/cifar100_judge.pt`（42.9 MiB，sha256 `da8190bebce0ff85…`）
  - `ddpm_mnist.pt`（47.1 MiB，sha256 `82068c06b9a4d38d…`）
  - `mnist_cnn.pt`（2.4 MiB，sha256 `923b175b66589a22…`）
  - `inception-2015-12-05.pt`（91.2 MiB，sha256 `f58cb9b6ec323ed6…`）
- P1 特徵／影像資產（共 3.6 GiB）：
  - `results/p1_assets/`（122 檔，1.8 GiB）
  - `results/p1_assets_cifar100/`（18 檔，1.9 GiB）

## 上傳（作者執行，擇一平台）

### Zenodo
1. 建立新 deposition，標題含論文名與 commit 短雜湊。
2. 上傳上列權重檔與 P1 資產目錄（可先各自 `zip`）。
3. 把 `results/release_bundle_manifest.json` 一併上傳，作為完整性清單。
4. 發佈取得 DOI，回填 README 重現指南的「權重來源」連結。

### Hugging Face Hub
1. `huggingface-cli repo create <name> --type dataset`
2. `git lfs track "*.pt"`，把權重與資產推上去。
3. 於 repo README 貼上本清單的 sha256 表。

## 下載後核對（重現者執行）

```bash
# 依 results/release_bundle_manifest.json 的 sha256 逐檔核對（範例）
uv run python - <<'PY'
import hashlib, json, os
m = json.load(open("results/release_bundle_manifest.json", encoding="utf-8"))
for e in m["checkpoints"]:
    if "sha256" not in e:
        continue
    h = hashlib.sha256(open(e["path"], "rb").read()).hexdigest()
    print("OK " if h == e["sha256"] else "BAD", e["path"])
PY
```

核對通過後，依 `README.md` 的「重現指南」從權重跑到每張表。
