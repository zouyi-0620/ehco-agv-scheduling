# -*- coding: utf-8 -*-
"""随包数据（`results/`）路径解析 —— 统一入口。

**为什么需要它**（2026-09-23 修复的实现缺陷）
------------------------------------------------
原来各实验模块把输入路径写死成单一路径，导致两种布局下**都不可能正确**：

| 布局 | 代码位置 | 随包数据实际位置 | 原解析结果 |
|---|---|---|---|
| 本地开发 | `<repo>/sim/experiments/X.py` | `<repo>/sim/results/…` | ✅（仅当写的是 `../results`） |
| 发布仓库 | `<repo>/src/sim/experiments/X.py` | `<repo>/results/…` | ❌（`src/sim/results`、`src/results` 都不存在） |

后果：
- 写 `"..","..","results"` 的模块（`e_c19`/`e_c21`/`e_c22`）在本地会解析到
  **项目根 `results/`** —— 那里存在一棵**陈旧的重复树**（2026-09-02, pre-slack），
  静默顶替了正确的 `sim/results/e4`，使参考臂取错值；
- 在发布布局下该路径**根本不存在**，`_load_published` 又对不存在的路径
  静默返回空 ⇒ 运行结果静默变成空表。

**解析顺序**（对 `rel="e4/e4_raw.csv"`，`here=<...>/experiments`）
1. `<here>/../results/…`     本地首选：`<repo>/sim/results/…`
2. `<here>/../../results/…`  本地 = 仓库根 / 发布 = `<repo>/src/results/…`
3. `<here>/../../../results/…` 发布布局的仓库根：`<repo>/results/…`

⚠️ 若多个候选同时存在（例如工作区里残留一棵重复树），**取优先级最高的**，
但在选中输出与忽略项之间**打印告警**，避免又一次静默取错。

用法（保持模块级变量名不变，便于外部脚本 monkey-patch）::

    from . import _paths
    PUBLISHED_E4 = _paths.resolve_file("e4/e4_raw.csv")

外部想改指向（如生成 31–60 独立批）仍然直接赋值即可::

    C19.PUBLISHED_E4 = os.path.join("sim", "results", "e4_indep", "e4_raw.csv")
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_UPS = (("..",), ("..", ".."), ("..", "..", ".."))


def candidates(rel: str) -> list[str]:
    """返回 `<results>/rel` 在各布局下的候选绝对路径，按优先级降序。"""
    rel = rel.replace("\\", "/").lstrip("/")
    out = []
    for up in _UPS:
        p = os.path.normpath(os.path.join(_HERE, *up, "results", *rel.split("/")))
        if p not in out:
            out.append(p)
    return out


def resolve_dir(rel: str = "", warn: bool = True) -> str:
    """解析 `results/` 下的子路径（文件或目录）；全不存在时返回首选候选。"""
    cands = candidates(rel)
    for c in cands:
        if os.path.exists(c):
            if warn:
                shadowed = [o for o in cands if o != c and os.path.exists(o)]
                if shadowed:
                    print("[paths] WARNING: 选中 %s；同名副本被忽略：%s"
                          % (c, shadowed), flush=True)
            return c
    if warn:
        print("[paths] WARNING: 未找到随包数据 %r；已尝试：%s"
              % (rel, cands), flush=True)
    return cands[0]


def resolve_file(rel: str, warn: bool = True) -> str:
    """同 `resolve_dir`，语义上表示「文件」（不存在时不报错，便于 import 期调用）。"""
    return resolve_dir(rel, warn=warn)


def require_file(path: str, what: str = "published baseline") -> str:
    """在真正读取前调用：路径不存在就**显式报错**，绝不静默降级为空。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            "[paths] %s not found: %s\n"
            "        searched: %s\n"
            "        (run from the repository root, or keep `results/` next to the code)"
            % (what, path, candidates('')))
    return path
