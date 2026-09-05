# 论文数据表目录（2026-09-04 同步版）

所有归档表均基于 E1/E2/E6/E-fig4/E9 真实重实现数据生成，旧稿合成数据已全部废弃。
**2026-09-04 同步**：数值对齐 M7-slack/W5 全量重跑后的当前口径（`sim/results/e{1,2,6,...}` 当前 CSV），与稿件 Table 3 / Table 5 / Table S2 / Table S4 / Table S6 及重建后的补充材料表一致。旧版归档在 `backup_pre_20260904/`。

| 文件 | 对应论文位置 | 数据来源 | 状态 |
|------|-------------|----------|------|
| table1_main_comparison.md | **Table 3**（静态 8 算法主比较） | E1 (8 算法 × 30 种子) | 2026-09-04 重建 |
| table3_ablation.md | **Table 5**（消融） | E2 (7 变体 × 30 种子) | 2026-09-04 重建 |
| tableS2_sensitivity.md | Table S2 | E6 (14 扰动 × 30 种子) | 2026-09-04 重建 |
| tableS4_taskload.md | Table S4 | E-fig4 (3 变体 × 30 种子) | 2026-09-04 核对（已最新） |
| tableS6_correlation.md | Table S6 | E9 (AW-NSGA-II 前沿 1,202 点) | 2026-09-04 核对（已最新） |

> 注意：历史版本曾按旧稿件编号写作"Table 1 / Table 3"，现稿件 Table 1=时间槽表、Table 2=参数表，E1/E2 主表已分别落在 Table 3 / Table 5——归档 md 的"对应论文位置"以上表为准。
> 2026-09-04 路线 B 已执行：`sim/results/e_fig4/`、正文 P190/P192/P193、补充 Table S4、md tableS4 与 Figure 5 图均与当前代码 30 seeds 重跑产物同步（Full ρ = 0.123 ± 0.307，hard-only −0.129 ± 0.301，GA-SA-multiSoH 0.053 ± 0.368；Bonferroni ×3 后 Full / hard-only / GA-SA-multiSoH 的 p = 0.0839 / 0.0578 / 1.00，none remains significant，正文"静态 SoH–负载机制不活跃"结论保留）。路线 A 留下的官方 0.067 版备份于 `e_fig4/backup_release_v1.0.1_20260904/`，07:56 重跑版备份于 `e_fig4/backup_20260904_0756_rerun/`，可追溯。

## 关键结论速览（当前口径）

- **Table 3 (E1)**：MOEA/D 静态全面占优——最高 HV-E1（0.790 ± 0.145，+70.6% over AW-NSGA-II，p = 1.3 × 10⁻⁸）、最短 f1（285.0 ± 12.9 s）；AW-NSGA-II 的 Gini(Δh) 无可分优势（最低 0.358 为 LWC，两两最小 p = 0.105，Holm 后全 ns）。seeds 31–60 独立复现 MOEA/D 仍最高（0.734 vs 0.530，p = 1.3 × 10⁻⁷，Table S9）。
- **Table 5 (E2)**：移除健康约束（−health ≡ hard-only）使 f2 显著下降 −12.1%（2152.0 vs Full 2449.0 CNY）；−impA*/−closedLoop ≡ Full（比特级恒等）；HV-E2 Full 0.282 ± 0.097 vs −health 0.611 ± 0.166（f4 ≡ 0 的结构性后果，如实披露）；−adaptW 不显著（p = 0.30）。
- **Table S2**：h_safe/α/退化阈值结构性不敏感（0.0%）；FLC 边界 ±10% → Δf1 −1.8%/+1.2%、ΔHV −4.6%/+1.7%；deadline 乘子最敏感（Δf1 −3.7%）；β/AHP 只改变 SoH 读数（ΔΔh ±25.4% / ±169.8%，解不变）。
- **Table S4**：静态场景 SoH–任务负载 Spearman ρ 校正后均不显著（n = 30 seeds；Full ρ = 0.123 ± 0.307 / p = 0.028；hard-only ρ = −0.129 ± 0.301 / p = 0.0193；GA-SA-multiSoH ρ = 0.053 ± 0.368 / p = 0.431；Bonferroni ×3 后 Full / hard-only / GA-SA-multiSoH 的 p = 0.0839 / 0.0578 / 1.00，none remains significant）。
- **Table S6**：f1–f3 中等正相关（ρ = 0.45/0.53/0.59）；PCA 仅 λ1 = 1.9852 (66.2%) > 1（Kaiser 有效维数 1）；f4 结构性恒 0。
