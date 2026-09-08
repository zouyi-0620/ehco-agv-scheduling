# e4_agg.csv — 口径说明（2026-09-05）

`e4_agg.csv` 由 `sim/experiments/e4_dynamic.py` 的 `aggregate()` 生成，各 metric 的 `n` 语义不同，复算时请注意：

| metric | n 语义 |
|---|---|
| `deg_pct` / `event_fired_rate` / `replan_applied_rate` / `detection_rate` 等 | 全部事件行：5 注入实例 × 30 seeds = 150 /（scenario, strategy）|
| `response_ms` | mean/std 只对 **replan_applied=1（实际执行 replan）且有响应值** 的行计算；`n` 字段记录的是该子集大小（congestion EHCO: 95、fault EHCO: 90、urgent EHCO: 150），与正文 §3.3（P148）"n=95/90/150 runs that replanned" 口径一致 |

背景：2026-09-05 修正——此前 `response_ms` 的 `n` 误记为总行数（150），且 mean 混入了"触发 replan 但被 guard 拒绝（未应用）"的事件（congestion 49 行、耗时约 125 ms 的 guard 评估开销），导致 mean 120.6 虚高；现改为 applied 子集后 mean 118.1±25.0（n=95），与论文 §3.3/Table 9/Figure 6(a)/Figure 8(b) 一致。

生成脚本：`sim/experiments/e4_dynamic.py`（aggregate 段）。
