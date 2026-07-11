"""CIFAR-100 base-model FID gate（D 包 D9／D10 第一閘）。

以自訓 CFG 模型 checkpoints/cifar100_cfg.pt 在代表性設定（w=1 純條件、steps=50、eta=0）平衡生成
50k 樣本，量標準 Inception clean-fid，對 D9 凍結門檻 clean-fid ≤ 20 判定 base model 是否堪用。

為什麼要這一閘：後續整條 CIFAR-100 管線（judge、scout、confirmatory）都建立在這個 backbone 上；
若 backbone 生成品質不夠（FID 太高），下游的 coverage/TSTR 訊號會被生成瑕疵污染，量再多也無意義。
所以先用一個便宜、單點的 FID gate 確認 backbone 落在堪用帶，再投入昂貴的下游。

clean-fid 的參考統計：clean-fid 內建 CIFAR-10 的預算 stats，但沒有 CIFAR-100（實測遠端 404）。
因此本 driver 先從真實 CIFAR-100 train 全 50k 影像，用 clean-fid 自己的 resizer 建一份 clean-mode
自訂參考統計（make_custom_stats，會快取到 venv，之後重跑免重算），再讓生成樣本走同一條 clean 路徑
比對，確保 real 與 fake 的量測口徑一致（apples-to-apples）。

用法（先小樣本 smoke，再全量）：
    uv run python cifar100_base_gate.py --smoke
    uv run python cifar100_base_gate.py --num 50000
"""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

import torch

# 重用既有、已驗證的生成與量測輔助，避免重複實作、且不改動 CIFAR-10 既有路徑。
from cifar_cfg_sample import load_cfg_model, generate_balanced
from fid_clean import _patch_scipy_sqrtm, _dump_pngs
from datasets.cifar import load_cifar


def _env_versions():
    """記錄可重現所需的環境版本（依 claude.md §5.2）。"""
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def build_real_reference(ref_name, real_num, device, batch_size=64):
    """從真實 CIFAR-100 train 建 clean-fid 自訂參考統計；已快取則略過。

    real_num=0 表示用全部 train（50000 張，gate 正式口徑）。small 值供 smoke 測試。
    """
    from cleanfid import fid as cfid
    _patch_scipy_sqrtm()
    if cfid.test_stats_exists(ref_name, "clean"):
        print(f"[ref] custom stats '{ref_name}' 已存在，略過重建", flush=True)
        return
    print("[ref] 載入真實 CIFAR-100 train ...", flush=True)
    real, _ = load_cifar("cifar100", train=True)          # [-1,1], (N,3,32,32)
    if real_num and real_num < real.size(0):
        real = real[:real_num]
    real01 = (real + 1) / 2                                # -> [0,1]
    tmp = tempfile.mkdtemp(prefix="c100_real_")
    try:
        print(f"[ref] 寫出 {real01.size(0)} 張真實 PNG 並建 clean-fid 參考 ...", flush=True)
        _dump_pngs(real01, tmp)
        print(f"[ref] 實際落盤 PNG 檔數：{len([f for f in os.listdir(tmp) if f.endswith('.png')])}",
              flush=True)
        # num_workers=0：Windows 的 spawn worker 無法 pickle clean-fid 的區域 resizer closure。
        cfid.make_custom_stats(ref_name, tmp, mode="clean", num_workers=0, batch_size=batch_size)
    finally:
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)
    print(f"[ref] 參考統計 '{ref_name}' 建立完成", flush=True)


def clean_fid_gen_vs_ref(gen01, ref_name):
    """生成樣本（[0,1]）相對於自訂 CIFAR-100 參考的 clean-fid。"""
    from cleanfid import fid as cfid
    _patch_scipy_sqrtm()
    tmp = tempfile.mkdtemp(prefix="c100_gen_")
    try:
        _dump_pngs(gen01, tmp)
        print(f"[fid] 生成 PNG 落盤檔數：{len([f for f in os.listdir(tmp) if f.endswith('.png')])}",
              flush=True)
        return float(cfid.compute_fid(tmp, dataset_name=ref_name, mode="clean",
                                      dataset_split="custom", num_workers=0))
    finally:
        for f in os.listdir(tmp):
            os.remove(os.path.join(tmp, f))
        os.rmdir(tmp)


def main():
    p = argparse.ArgumentParser(description="CIFAR-100 base-model FID gate (clean-fid, D9/D10).")
    p.add_argument("--ckpt", default="checkpoints/cifar100_cfg.pt")
    p.add_argument("--num", type=int, default=50000, help="生成張數（平均分到 100 類）")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--guidance", type=float, default=1.0, help="w=1 純條件，gate 口徑")
    p.add_argument("--batch", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gate", type=float, default=20.0, help="D9 凍結門檻：clean-fid ≤ gate")
    p.add_argument("--ref-name", default="cifar100_train_clean32")
    p.add_argument("--real-num", type=int, default=0, help="建參考用的真實張數，0=全 train(50k)")
    p.add_argument("--output", default="results/cifar100_base_fid.json")
    p.add_argument("--smoke", action="store_true",
                   help="小樣本煙霧測試：num=200、real-num=200、獨立 ref-name，不污染正式參考")
    args = p.parse_args()

    if args.smoke:
        args.num, args.real_num = 200, 200
        args.ref_name, args.output = "cifar100_smoke_clean32", "results/cifar100_base_fid_smoke.json"

    start_ts = datetime.now(timezone.utc).isoformat()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)
    print(f"device={device} start={start_ts}", flush=True)

    # 1) 真實參考統計（快取）
    build_real_reference(args.ref_name, args.real_num, device)

    # 2) 生成
    model, schedule, hp = load_cfg_model(args.ckpt, device)
    num_classes = hp.get("num_classes", 100)
    per_class = args.num // num_classes
    print(f"[gen] 生成 {per_class*num_classes} 張（{num_classes} 類 × {per_class}）"
          f" steps={args.steps} eta={args.eta} w={args.guidance} ...", flush=True)
    gen, _ = generate_balanced(model, schedule, per_class, device, args.steps, args.eta,
                               args.guidance, num_classes=num_classes, batch=args.batch,
                               seed=args.seed)
    gen01 = (gen + 1) / 2

    # 3) clean-fid 與 gate 判定
    print("[fid] 量 clean-fid vs 自訂 CIFAR-100 參考 ...", flush=True)
    cfid = clean_fid_gen_vs_ref(gen01, args.ref_name)
    passed = bool(cfid <= args.gate)
    end_ts = datetime.now(timezone.utc).isoformat()

    print("\n" + "=" * 60)
    print(f"  CIFAR-100 base-model FID gate (w={args.guidance} steps={args.steps} eta={args.eta})")
    print("=" * 60)
    print(f"  clean-fid       : {cfid:.3f}")
    print(f"  gate (<= {args.gate}) : {'PASS 通過' if passed else 'FAIL 未過'}")
    print("=" * 60, flush=True)

    out = {
        "gate": "cifar100_base_model_fid",
        "ckpt": args.ckpt, "epoch": hp.get("epoch"), "num_classes": num_classes,
        "num": per_class * num_classes, "per_class": per_class,
        "steps": args.steps, "eta": args.eta, "guidance": args.guidance,
        "seed": args.seed, "batch": args.batch,
        "ref_name": args.ref_name, "real_num_for_ref": (args.real_num or 50000),
        "clean_fid": cfid, "gate_threshold": args.gate, "passed": passed,
        "start_timestamp": start_ts, "end_timestamp": end_ts,
        "argv": sys.argv, "env": _env_versions(),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
