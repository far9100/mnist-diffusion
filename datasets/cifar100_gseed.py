# 用途：CIFAR-100 之無碰撞生成種子公式（D9，修 CIFAR-10 gseed 反對角線碰撞 R-2026-07-06-05 §1.12）。
# hash 派生：gseed(seed, w) = int(sha256(f"{seed}_{w:g}").hexdigest()[:15], 16)。附全網格枚舉唯一性驗證。
import hashlib


def gseed(seed, w):
    """(seed, w) → 無碰撞生成種子。sha256 前 15 hex（60 bit）避免 CIFAR-10 之 (seed+w)×1e7 退化碰撞。"""
    h = hashlib.sha256(f"{seed}_{w:g}".encode()).hexdigest()
    return int(h[:15], 16)


def verify_unique(seeds, ws):
    """對 seeds × ws 全網格枚舉，驗證 gseed 兩兩相異。回傳 (unique_bool, n_cells, n_distinct)。"""
    vals = {}
    for s in seeds:
        for w in ws:
            g = gseed(s, w)
            vals.setdefault(g, []).append((s, w))
    n_cells = len(seeds) * len(ws)
    n_distinct = len(vals)
    collisions = {g: cells for g, cells in vals.items() if len(cells) > 1}
    return n_distinct == n_cells, n_cells, n_distinct, collisions


if __name__ == "__main__":
    # 驗證於一個寬鬆的候選網格（CIFAR-100 實際網格於 D10 scout 定；此處證公式本身無碰撞）
    seeds = list(range(0, 100))
    ws = [round(1.0 + 0.1 * i, 1) for i in range(0, 91)]  # w∈[1.0, 10.0] 步 0.1
    ok, n_cells, n_distinct, coll = verify_unique(seeds, ws)
    print(f"grid: {len(seeds)} seeds × {len(ws)} ws = {n_cells} cells")
    print(f"distinct gseeds: {n_distinct}")
    print(f"collision-free: {ok}")
    if coll:
        print(f"collisions: {list(coll.items())[:5]}")
    # 對照 CIFAR-10 舊公式之碰撞（記錄用）
    def old(s, w):
        return s * 10_000_000 + int(w * 1000) * 10_000
    old_c10 = {}
    for s in [10, 11, 12]:
        for w in [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8]:
            old_c10.setdefault(old(s, w), []).append((s, w))
    old_collisions = {g: c for g, c in old_c10.items() if len(c) > 1}
    print(f"\n舊公式於 CIFAR-10 confirmatory 網格（30 cells）碰撞組數: {len(old_collisions)}（{sum(len(c) for c in old_collisions.values())} cells 涉入）")
