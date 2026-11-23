# -*- coding: utf-8 -*-
"""
电商折扣效率与利润质量诊断
数据：commerce_order_2018.csv（9,906 行 × 22 字段，2017-2018 年订单明细）

分析主线：
  1. 经营总览：GMV、利润、利润率、退货率、履约时效
  2. 折扣分层：不同折扣区间对利润的边际影响（核心）
  3. 品类结构：17 个子类别的利润率分化
  4. 区域与客户细分：利润质量的空间/客群差异
  5. 销售经理绩效：11 项指标体系落地 + 折扣率与利润率的相关性

用法：python analysis.py
输出：results/*.csv, figures/*.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 路径：相对本脚本定位，可移植 ----------
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figures")
for d in (RES, FIG):
    os.makedirs(d, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 读取与清洗 ----------
def load():
    path = os.path.join(DATA, "commerce_order_2018.csv")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")

    # 金额字段带千分位，需转数值
    for c in ("利润", "销售额"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")
    df["订单日期"] = pd.to_datetime(df["订单日期"], errors="coerce", format="mixed")
    df["年"] = df["订单日期"].dt.year

    # 履约：实际发货时间 <= 计划发货天数 视为及时
    df["及时发货"] = df["实际发货时间"] <= df["计划发货天数"]
    df["延误天数"] = (df["实际发货时间"] - df["计划发货天数"]).clip(lower=0)
    df["亏损"] = df["利润"] < 0
    return df.dropna(subset=["销售额", "利润"])


# ---------- 1. 经营总览 ----------
def overview(df):
    gmv = df["销售额"].sum()
    profit = df["利润"].sum()
    delay = df.loc[df["延误天数"] > 0, "延误天数"]
    row = {
        "订单行数": len(df),
        "客户数": df["客户 ID"].nunique(),
        "SKU数": df["产品id"].nunique(),
        "覆盖城市": df["城市"].nunique(),
        "GMV(元)": round(gmv, 2),
        "利润合计(元)": round(profit, 2),
        "整体利润率(%)": round(profit / gmv * 100, 2),
        "退货率(%)": round(df["是否退回"].mean() * 100, 2),
        "亏损行占比(%)": round(df["亏损"].mean() * 100, 2),
        "折扣使用率(%)": round((df["折扣"] > 0).mean() * 100, 2),
        "平均折扣(%)": round(df["折扣"].mean() * 100, 2),
        "发货及时率(%)": round(df["及时发货"].mean() * 100, 2),
        "延误行均延误天数": round(delay.mean(), 2) if len(delay) else 0,
    }
    out = pd.DataFrame([row]).T.rename(columns={0: "值"})
    out.to_csv(os.path.join(RES, "01_经营总览.csv"), encoding="utf-8-sig")
    return row


# ---------- 2. 折扣分层（核心）----------
def discount_tiers(df):
    bins = [-0.001, 0.0001, 0.1, 0.2, 0.3, 0.4001]
    labels = ["无折扣", "0-10%", "10-20%", "20-30%", "30-40%"]
    df = df.copy()
    df["折扣档"] = pd.cut(df["折扣"], bins=bins, labels=labels)

    t = df.groupby("折扣档", observed=True).agg(
        订单行数=("行 ID", "size"),
        销售额=("销售额", "sum"),
        利润=("利润", "sum"),
        亏损行占比=("亏损", "mean"),
        退货率=("是否退回", "mean"),
    )
    t["利润率(%)"] = (t["利润"] / t["销售额"] * 100).round(2)
    t["销售额占比(%)"] = (t["销售额"] / df["销售额"].sum() * 100).round(2)
    t["亏损行占比(%)"] = (t["亏损行占比"] * 100).round(2)
    t["退货率(%)"] = (t["退货率"] * 100).round(2)
    t[["销售额", "利润"]] = t[["销售额", "利润"]].round(2)
    t = t[["订单行数", "销售额", "销售额占比(%)", "利润", "利润率(%)", "亏损行占比(%)", "退货率(%)"]]
    t.to_csv(os.path.join(RES, "02_折扣分层.csv"), encoding="utf-8-sig")

    # 相关性：折扣 vs 利润率（行级）
    corr = np.corrcoef(df["折扣"], df["利润"] / df["销售额"].replace(0, np.nan))[0, 1]
    return df, t, corr


# ---------- 3. 品类结构 ----------
def category(df):
    sub = df.groupby("子类别").agg(
        销售额=("销售额", "sum"), 利润=("利润", "sum"),
        订单行数=("行 ID", "size"), 退货率=("是否退回", "mean"),
        平均折扣=("折扣", "mean"),
    )
    sub["利润率(%)"] = (sub["利润"] / sub["销售额"] * 100).round(2)
    sub["销售占比(%)"] = (sub["销售额"] / df["销售额"].sum() * 100).round(2)
    sub["退货率(%)"] = (sub["退货率"] * 100).round(2)
    sub["平均折扣(%)"] = (sub["平均折扣"] * 100).round(2)
    sub = sub.round(2).sort_values("利润率(%)", ascending=False)
    sub.to_csv(os.path.join(RES, "03_子类别利润率.csv"), encoding="utf-8-sig")

    cat = df.groupby("类别").agg(销售额=("销售额", "sum"), 利润=("利润", "sum"))
    cat["利润率(%)"] = (cat["利润"] / cat["销售额"] * 100).round(2)
    cat["销售占比(%)"] = (cat["销售额"] / df["销售额"].sum() * 100).round(2)
    cat.round(2).to_csv(os.path.join(RES, "03b_大类利润率.csv"), encoding="utf-8-sig")
    return sub, cat


# ---------- 4. 区域与客户细分 ----------
def region_segment(df):
    reg = df.groupby("地区").agg(
        GMV=("销售额", "sum"), 利润=("利润", "sum"),
        客户数=("客户 ID", "nunique"), 退货率=("是否退回", "mean"),
        平均折扣=("折扣", "mean"),
    )
    reg["利润率(%)"] = (reg["利润"] / reg["GMV"] * 100).round(2)
    reg["GMV占比(%)"] = (reg["GMV"] / df["销售额"].sum() * 100).round(2)
    reg["退货率(%)"] = (reg["退货率"] * 100).round(2)
    reg["平均折扣(%)"] = (reg["平均折扣"] * 100).round(2)
    reg.round(2).sort_values("GMV", ascending=False).to_csv(
        os.path.join(RES, "04_区域.csv"), encoding="utf-8-sig")

    seg = df.groupby("细分").agg(
        GMV=("销售额", "sum"), 利润=("利润", "sum"),
        客户数=("客户 ID", "nunique"), 订单行数=("行 ID", "size"),
    )
    seg["利润率(%)"] = (seg["利润"] / seg["GMV"] * 100).round(2)
    seg["客户均消费"] = (seg["GMV"] / seg["客户数"]).round(2)
    seg["单行均额"] = (seg["GMV"] / seg["订单行数"]).round(2)
    seg.round(2).to_csv(os.path.join(RES, "05_客户细分.csv"), encoding="utf-8-sig")
    return reg, seg


# ---------- 5. 销售经理绩效（指标体系落地）----------
def manager(df):
    m = df.groupby("销售经理").agg(
        销售额=("销售额", "sum"), 利润=("利润", "sum"),
        客户数=("客户 ID", "nunique"),
        亏损行占比=("亏损", "mean"),
        退货率=("是否退回", "mean"),
        平均折扣=("折扣", "mean"),
        及时率=("及时发货", "mean"),
    )
    m["利润率(%)"] = (m["利润"] / m["销售额"] * 100).round(2)
    m["亏损行占比(%)"] = (m["亏损行占比"] * 100).round(2)
    m["退货率(%)"] = (m["退货率"] * 100).round(2)
    m["平均折扣(%)"] = (m["平均折扣"] * 100).round(2)
    m["及时率(%)"] = (m["及时率"] * 100).round(2)
    m["客户均消费"] = (m["销售额"] / m["客户数"]).round(2)

    # 简化评分：业绩 35% / 利润质量 30% / 客户经营 25% / 履约 10%
    def score(s):
        return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0

    m["综合得分"] = (
        0.35 * score(m["销售额"])
        + 0.30 * score(m["利润率(%)"])
        + 0.25 * score(m["客户均消费"])
        + 0.10 * score(m["及时率(%)"])
    ).round(4)
    m = m.round(2).sort_values("综合得分", ascending=False)
    m.to_csv(os.path.join(RES, "06_销售经理绩效.csv"), encoding="utf-8-sig")

    r = np.corrcoef(m["平均折扣(%)"], m["利润率(%)"])[0, 1]
    return m, r


# ---------- 绘图 ----------
def fig_discount(t):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(t))
    bars = ax1.bar(x, t["销售额占比(%)"], width=0.55, color="#c9d6e5", label="销售额占比")
    ax1.set_ylabel("销售额占比 (%)", color="#5b6b7c")
    ax1.tick_params(axis="y", labelcolor="#5b6b7c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(t.index, fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(x, t["利润率(%)"], marker="o", linewidth=2.2, color="#2f6fd0", label="利润率")
    ax2.axhline(0, color="#999", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("利润率 (%)", color="#2f6fd0")
    ax2.tick_params(axis="y", labelcolor="#2f6fd0")
    for i, v in enumerate(t["利润率(%)"]):
        ax2.annotate(f"{v:.1f}%", (i, v), textcoords="offset points",
                     xytext=(0, 10 if v >= 0 else -18), ha="center",
                     fontsize=9, color="#c0392b" if v < 0 else "#2f6fd0")
    lo, hi = t["利润率(%)"].min(), t["利润率(%)"].max()
    ax2.set_ylim(lo - (hi - lo) * 0.18, hi + (hi - lo) * 0.20)
    ax1.set_ylim(0, t["销售额占比(%)"].max() * 1.18)
    ax1.set_title("折扣档位与利润率的剪刀差：高折扣贡献 GMV 但不贡献利润", fontsize=12.5, pad=12)
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "01_折扣与利润率.png"), dpi=160)
    plt.close(fig)


def fig_subcategory(sub):
    s = sub.sort_values("利润率(%)")
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    colors = ["#c0392b" if v < 0 else "#2f6fd0" for v in s["利润率(%)"]]
    ax.barh(s.index, s["利润率(%)"], color=colors, height=0.66)
    for i, v in enumerate(s["利润率(%)"]):
        ax.text(v + (0.5 if v >= 0 else -0.5), i, f"{v:.1f}%",
                va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.axvline(0, color="#555", linewidth=0.9)
    ax.set_xlabel("利润率 (%)")
    ax.set_title("17 个子类别利润率分化：围巾 27.5% vs 短裙 −15.5%", fontsize=12.5, pad=12)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "02_子类别利润率.png"), dpi=160)
    plt.close(fig)


def fig_region(reg):
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    r = reg.sort_values("利润率(%)", ascending=False)
    x = np.arange(len(r))
    ax.bar(x, r["利润率(%)"], color="#2f6fd0", width=0.5, label="利润率")
    ax2 = ax.twinx()
    ax2.plot(x, r["退货率(%)"], marker="s", color="#e67e22", linewidth=1.8, label="退货率")
    ax.set_xticks(x); ax.set_xticklabels(r.index)
    ax.set_ylabel("利润率 (%)"); ax2.set_ylabel("退货率 (%)")
    ax.axhline(0, color="#999", linewidth=0.8)
    for i, v in enumerate(r["利润率(%)"]):
        ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_title("区域利润质量：华北最高 17.7%，西南最低 7.5%", fontsize=12.5, pad=12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "03_区域利润与退货.png"), dpi=160)
    plt.close(fig)


def fig_manager(m):
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.scatter(m["平均折扣(%)"], m["利润率(%)"], s=m["销售额"] / 20000,
               color="#2f6fd0", alpha=0.75, edgecolor="white")
    for name, r in m.iterrows():
        ax.annotate(name, (r["平均折扣(%)"], r["利润率(%)"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    z = np.polyfit(m["平均折扣(%)"], m["利润率(%)"], 1)
    xs = np.linspace(m["平均折扣(%)"].min(), m["平均折扣(%)"].max(), 50)
    ax.plot(xs, np.polyval(z, xs), "--", color="#c0392b", linewidth=1.4,
            label=f"拟合线（r={np.corrcoef(m['平均折扣(%)'], m['利润率(%)'])[0,1]:.2f}）")
    ax.set_xlabel("平均折扣率 (%)")
    ax.set_ylabel("利润率 (%)")
    ax.set_title("折扣率与利润率负相关：让利换量的直接代价", fontsize=12.5, pad=12)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "04_经理折扣与利润率.png"), dpi=160)
    plt.close(fig)


def main():
    df = load()
    ov = overview(df)
    df2, t, corr = discount_tiers(df)
    sub, cat = category(df)
    reg, seg = region_segment(df)
    m, r_mgr = manager(df)

    fig_discount(t)
    fig_subcategory(sub)
    fig_region(reg)
    fig_manager(m)

    print("=" * 58)
    print("经营总览")
    for k, v in ov.items():
        print(f"  {k:<16} {v}")
    print("\n折扣分层")
    print(t.to_string())
    print(f"\n行级 折扣-利润率 相关系数: {corr:.4f}")
    print("\n子类别利润率 Top5 / Bottom3")
    print(sub["利润率(%)"].head(5).to_string())
    print(sub["利润率(%)"].tail(3).to_string())
    print("\n销售经理（按综合得分）")
    print(m[["销售额", "利润率(%)", "客户均消费", "平均折扣(%)", "综合得分"]].to_string())
    print(f"\n经理 平均折扣 vs 利润率 相关系数: {r_mgr:.4f}")
    print("\n已输出:", RES, "|", FIG)


if __name__ == "__main__":
    main()
