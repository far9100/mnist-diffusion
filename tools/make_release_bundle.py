# 用途：為論文重現打包「權重＋資產清單＋sha256 清單＋上傳說明」。對應任務書 fix_tasks T6a。
#
# 設計說明（給第一次讀的研究生）：
#   論文的每個數字都能指到 results/*.json（已隨 repo 追蹤），但要「從零重現」還需要模型權重與
#   P1 特徵/影像資產——這些檔案太大（權重約 2.4GB、P1 資產約 3.7GB），不進 git。本工具不上傳，
#   只做三件事：(1) 對「發布用」權重逐檔算 sha256 與位元組數，(2) 盤點 P1 資產目錄成清單，
#   (3) 產出一份給作者的 Zenodo/HuggingFace 上傳說明。實際上傳由作者執行。
#
#   為何只選「發布用」權重：checkpoints/ 內含每個 epoch 的快照（約 60GB），重現只需最終權重
#   cifar10/100_cfg.pt、cifar10/100_judge.pt 與 MNIST 的 ddpm_mnist.pt／mnist_cnn.pt（＋選配
#   Inception 參考統計）。本工具預設只打包這份最小集合。
#
# 用法：
#   uv run python tools/make_release_bundle.py                # 算權重 sha256 + P1 資產清單（不 hash 資產）
#   uv run python tools/make_release_bundle.py --hash-assets  # 連 P1 資產也算 sha256（較慢，數 GB）
#   純讀＋只寫 manifest/說明兩檔，不動任何權重、results/*.json 或凍結檔。

import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 發布用權重（重現論文所需的最小集合）。相對 ROOT 的路徑。
RELEASE_CHECKPOINTS = [
    "checkpoints/cifar10_cfg.pt",     # CIFAR-10 CFG backbone（Phase 1）
    "checkpoints/cifar100_cfg.pt",    # CIFAR-100 CFG backbone（Phase 1）
    "checkpoints/cifar10_judge.pt",   # CIFAR-10 judge（TSTR/near-boundary/judge 特徵）
    "checkpoints/cifar100_judge.pt",  # CIFAR-100 judge
    "ddpm_mnist.pt",                  # MNIST 條件式 DDPM（Gen-1 sandbox）
    "mnist_cnn.pt",                   # MNIST judge CNN（MNIST-FID/PRDC 特徵）
]
# 選配：量測正確性錨點（EDM 重現）所需，體積中等。
OPTIONAL_CHECKPOINTS = [
    "inception-2015-12-05.pt",        # clean-fid 用 Inception（重現公開 FID 錨點）
]
# P1 特徵/影像資產目錄（DINOv2 特徵、img_uint8、judge_out 等）。
P1_ASSET_DIRS = ["results/p1_assets", "results/p1_assets_cifar100"]


def sha256_of(path, chunk=1 << 20):
    """逐 1MB 讀檔算 sha256，避免一次載入大檔進記憶體。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def describe_file(rel, do_hash):
    """回傳單檔的 {path, bytes, sha256?}；檔不存在回 None。"""
    ap = os.path.join(ROOT, rel)
    if not os.path.isfile(ap):
        return None
    entry = {"path": rel.replace("\\", "/"), "bytes": os.path.getsize(ap)}
    if do_hash:
        entry["sha256"] = sha256_of(ap)
    return entry


def inventory_dir(rel, do_hash):
    """遞迴盤點一個資產目錄：回傳檔數、總位元組、每檔 (path, bytes[, sha256])。"""
    base = os.path.join(ROOT, rel)
    if not os.path.isdir(base):
        return None
    entries, total = [], 0
    for dirpath, _dirs, files in os.walk(base):
        for name in sorted(files):
            ap = os.path.join(dirpath, name)
            size = os.path.getsize(ap)
            total += size
            e = {"path": os.path.relpath(ap, ROOT).replace("\\", "/"), "bytes": size}
            if do_hash:
                e["sha256"] = sha256_of(ap)
            entries.append(e)
    return {"dir": rel.replace("\\", "/"), "files": len(entries), "bytes": total,
            "entries": sorted(entries, key=lambda e: e["path"])}


def human(n):
    """位元組轉人類可讀（GiB/MiB/KiB）。"""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


UPLOAD_TEMPLATE = """<!-- 用途：論文重現包（權重＋P1 資產）之上傳與下載說明；由 tools/make_release_bundle.py 產生。 -->

# 重現包上傳與下載說明

本檔由 `tools/make_release_bundle.py` 產生。清單與 sha256 見
`{manifest_rel}`。權重與 P1 資產不進 git；下列步驟由作者執行一次，之後重現者依 sha256 核對。

## 內容

- 發布用權重（重現論文所需最小集合，共 {ckpt_bytes_h}）：
{ckpt_list}
- P1 特徵／影像資產（共 {asset_bytes_h}）：
{asset_list}

## 上傳（作者執行，擇一平台）

### Zenodo
1. 建立新 deposition，標題含論文名與 commit 短雜湊。
2. 上傳上列權重檔與 P1 資產目錄（可先各自 `zip`）。
3. 把 `{manifest_rel}` 一併上傳，作為完整性清單。
4. 發佈取得 DOI，回填 README 重現指南的「權重來源」連結。

### Hugging Face Hub
1. `huggingface-cli repo create <name> --type dataset`
2. `git lfs track "*.pt"`，把權重與資產推上去。
3. 於 repo README 貼上本清單的 sha256 表。

## 下載後核對（重現者執行）

```bash
# 依 {manifest_rel} 的 sha256 逐檔核對（範例）
uv run python - <<'PY'
import hashlib, json, os
m = json.load(open("{manifest_rel}", encoding="utf-8"))
for e in m["checkpoints"]:
    if "sha256" not in e:
        continue
    h = hashlib.sha256(open(e["path"], "rb").read()).hexdigest()
    print("OK " if h == e["sha256"] else "BAD", e["path"])
PY
```

核對通過後，依 `README.md` 的「重現指南」從權重跑到每張表。
"""


def main():
    p = argparse.ArgumentParser(description="打包論文重現包的權重清單與上傳說明（T6a）。")
    p.add_argument("--hash-assets", action="store_true",
                   help="連 P1 資產也算 sha256（較慢，數 GB）；預設只算權重、資產僅盤點大小。")
    p.add_argument("--include-optional", action=argparse.BooleanOptionalAction, default=True,
                   help="納入選配權重（Inception 參考統計）；預設納入。")
    p.add_argument("--manifest", default="results/release_bundle_manifest.json")
    p.add_argument("--instructions", default="docs/release_bundle.md")
    args = p.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ckpt_rels = list(RELEASE_CHECKPOINTS) + (OPTIONAL_CHECKPOINTS if args.include_optional else [])
    checkpoints, missing = [], []
    for rel in ckpt_rels:
        e = describe_file(rel, do_hash=True)
        (checkpoints if e else missing).append(e if e else rel)
    assets = [inv for rel in P1_ASSET_DIRS if (inv := inventory_dir(rel, args.hash_assets))]

    ckpt_bytes = sum(e["bytes"] for e in checkpoints)
    asset_bytes = sum(a["bytes"] for a in assets)

    manifest = {
        "purpose": "論文重現包清單（權重 sha256 + P1 資產盤點）",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root_note": "路徑相對 repo 根目錄",
        "checkpoints": checkpoints,
        "missing_checkpoints": missing,
        "p1_assets": assets,
        "totals": {"checkpoint_bytes": ckpt_bytes, "asset_bytes": asset_bytes,
                   "checkpoint_human": human(ckpt_bytes), "asset_human": human(asset_bytes)},
        "assets_hashed": args.hash_assets,
        "argv": " ".join(sys.argv),
    }
    os.makedirs(os.path.join(ROOT, os.path.dirname(args.manifest)), exist_ok=True)
    with open(os.path.join(ROOT, args.manifest), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    ckpt_list = "\n".join(f"  - `{e['path']}`（{human(e['bytes'])}，sha256 `{e['sha256'][:16]}…`）"
                          for e in checkpoints)
    asset_list = "\n".join(f"  - `{a['dir']}/`（{a['files']} 檔，{human(a['bytes'])}）" for a in assets)
    instructions = UPLOAD_TEMPLATE.format(
        manifest_rel=args.manifest.replace("\\", "/"),
        ckpt_bytes_h=human(ckpt_bytes), asset_bytes_h=human(asset_bytes),
        ckpt_list=ckpt_list or "  - （無）", asset_list=asset_list or "  - （無）")
    os.makedirs(os.path.join(ROOT, os.path.dirname(args.instructions)), exist_ok=True)
    with open(os.path.join(ROOT, args.instructions), "w", encoding="utf-8") as f:
        f.write(instructions)

    print("=" * 72)
    print("  重現包清單")
    print("=" * 72)
    for e in checkpoints:
        print(f"  {e['path']:<34} {human(e['bytes']):>10}  sha256 {e['sha256'][:16]}…")
    if missing:
        print(f"  缺檔（略過）: {missing}")
    for a in assets:
        print(f"  {a['dir'] + '/':<34} {human(a['bytes']):>10}  {a['files']} 檔"
              f"{' (hashed)' if args.hash_assets else ''}")
    print("-" * 72)
    print(f"  權重合計 {human(ckpt_bytes)}｜P1 資產合計 {human(asset_bytes)}")
    print(f"  Wrote {args.manifest}")
    print(f"  Wrote {args.instructions}")


if __name__ == "__main__":
    main()
