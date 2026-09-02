# 论文数据表目录（#280 生成）

所有表格均基于 E1/E2/E6/E-fig4/E-fig6 真实重实现数据生成，旧稿合成数据已全部废弃。

| 文件 | 对应论文位置 | 数据来源 |
|------|-------------|----------|
| table1_main_comparison.md | Table 1 | E1 (8 算法 × 30 种子) |
| table3_ablation.md | Table 3 | E2 (7 变体 × 30 种子) |
| tableS2_sensitivity.md | Table S2 | E6 (14 扰动 × 30 种子) |
| tableS4_taskload.md | Table S4 | E-fig4 (3 变体 × 30 种子) |
| tableS6_correlation.md | Table S6 | E9 (AW-NSGA-II 前沿 1,202 点) |

## 关键结论速览

- **Table 1**: MOEA/D 静态全面占优（f1=287.0, HV=0.765）；AW-NSGA-II 以 Gini(Δh)=0.363 健康公平性最优
- **Table 3**: −health 使 f2 显著下降 −13.7%（1947 vs 2256）；−adaptW 不显著；−impA*/−closedLoop ≡ Full
- **Table S2**: h_safe/α/β/AHP/退化阈值结构性不敏感；FLC 边界 ±10% → f1 ±3.1%；截止期乘子最敏感 → HV −6.1%
- **Table S4**: 静态场景 SoH-任务负载相关性均不显著（|ρ|<0.07）
- **Table S6**: f1–f3 中等正相关（ρ=0.45–0.59）；PCA 有效维数 ≈1；f4 结构性恒 0
