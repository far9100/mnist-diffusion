"""用途：路徑墊片。讓 src/ 下依類別分資料夾的腳本沿用扁平 import，並輸出專案根路徑 ROOT。

原理：本專案未打包成 package（無 __init__.py／build-system／pip install -e），扁平 import
（例如 `import metrics_features`、`from selector import ...`）能成立，唯一靠「被 import 的模組
所在資料夾在 sys.path 上」。過去所有腳本平放專案根、從根執行，sys.path[0] 剛好是根，故互相
找得到。把檔案依類別搬進 src/<類別>/ 後，sys.path[0] 變成該子資料夾，扁平 import 會失效。

本墊片在被 import 時，把「專案根」與「每個 src/<類別>/」都補回 sys.path，扁平 import 即照舊
可解析（根供 datasets/ 這類套件，var/ 已停放 attic/；各 src 子資料夾供扁平模組——所有模組 basename 全域唯一，
不會碰撞）。每個搬移檔在 docstring 之後、其他 import 之前 `import _pathfix` 一次即可。
"""

import os
import sys

# 本檔位於 <root>/src/_pathfix.py，往上兩層即專案根。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(ROOT, "src")

# 把根目錄與每個 src 子資料夾補進 sys.path（insert(0, ...) 使其優先於其他路徑）。
for _p in [ROOT] + [os.path.join(_SRC, _d) for _d in sorted(os.listdir(_SRC))]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# 把 argv[0] 收斂為裸檔名。driver 會把 " ".join(sys.argv) 寫進 results/*.json 作為對帳欄位；
# 檔案搬進子資料夾後改以 `python src/experiments/run_xxx.py` 執行，argv[0] 會含資料夾前綴。
# 正規化為裸檔名，使搬移後重跑的 argv 與既凍結產物逐字元相符（CLAUDE.md §5.2 之慣例層補強）。
sys.argv[0] = os.path.basename(sys.argv[0])
