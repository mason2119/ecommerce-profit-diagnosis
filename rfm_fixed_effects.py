# -*- coding: utf-8 -*-
"""
电商折扣诊断 · 增强分析：RFM 客群分层 + 客户固定效应回归

用法 A：RFM 分群 × 折扣分层 —— 回答「哪类客户在高折扣档被侵蚀」
用法 B：客户固定效应回归 —— 回答「控制客户自身特征后，折扣对利润率的净效应」

用法：python rfm_fixed_effects.py
输出：results/07_*.csv 08_*.csv 09_*.csv 10_*.csv；figures/05_*.png 06_*.png 07_*.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import statsmodels.formula.api as smf

BASE = os.path.dirname(os.path.abspath(__file__))
DATA, RES, FIG = (os.path.join(BASE, x) for x in ("data", "results", "figures"))
for d in (RES, FIG):
    os.makedirs(d, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 读取与清洗 ----------
def load():
    p = os.path.join(DATA, "commerce_order_2018.csv")
    try:
        df = pd.read_csv(p, encoding="utf-8", thousands=",")
    except UnicodeDecodeError:
        df = pd.read_csv(p, encoding="gb18030", thousands=",")
    df["订单日期"] = pd.to_datetime(df["订单日期"], errors="coerce", format="mixed")
    df["年"] = df["订单日期"].dt.year
    df["亏损"] = df["利润"] < 0
    df["行级利润率"] = df["利润"] / df["销售额"].replace(0, np.nan)
    return df.dropna(subset=["销售额", "利润", "客户 ID"])


def qscore(s, higher_better=True):
    """按秩五分位打分，避免重复值导致切分失败。higher_better=False 时值越小分越高。"""
    r = s.rank(method="first")
    labels = [1, 2, 3, 4, 5] if higher_better else [5, 4, 3, 2, 1]
    return pd.qcut(r, 5, labels=labels).astype(int)


# ---------- 用法 A：RFM 客群分层 ----------
def build_rfm(df):
    anchor = df["订单日期"].max()
    g = df.groupby("客户 ID")
    rfm = pd.DataFrame({
        "R_最近购买距今天数": (anchor - g["订单日期"].max()).dt.days,
        "F_购买次数": g["订单日期"].apply(lambda s: s.dt.normalize().nunique()),
        "M_销售额": g["销售额"].sum(),
        "利润贡献": g["利润"].sum(),
        "订单行数": g.size(),
    }).reset_index()

    rfm["R_score"] = qscore(rfm["R_最近购买距今天数"], higher_better=False)
    rfm["F_score"] = qscore(rfm["F_购买次数"])
    rfm["M_score"] = qscore(rfm["M_销售额"])

    X = StandardScaler().fit_transform(rfm[["R_score", "F_score", "M_score"]])
    km = KMeans(n_clusters=4, n_init=10, random_state=42)
    rfm["cluster"] = km.fit_predict(X)

    cent = rfm.groupby("cluster")[["R_score", "F_score", "M_score"]].mean()

    def name(c):
        R, F, M = cent.loc[c, ["R_score", "F_score", "M_score"]]
        if R >= 3.5 and F >= 3.5 and M >= 3.5:
            return "高价值核心客户"
        if R < 3.5 and F >= 3.5 and M >= 3.5:
            return "高价值待唤醒客户"
        if R >= 3.5 and F < 3.5 and M < 3.5:
            return "新客与低频客户"
        if R < 3.5 and F < 3.5 and M < 3.5:
            return "低价值沉睡客户"
        if M >= 4:
            return "高消费维持客户"
        return "一般维持客户"

    names = {c: name(c) for c in cent.index}
    # 若出现重名，按综合分加序号区分
    seen = {}
    for c in sorted(cent.index, key=lambda x: -(cent.loc[x].sum())):
        base = names[c]
        seen[base] = seen.get(base, 0) + 1
        names[c] = base if seen[base] == 1 else "%s②" % base

    rfm["客群"] = rfm["cluster"].map(names)
    rfm["RFM总分"] = rfm[["R_score", "F_score", "M_score"]].sum(axis=1)

    cols = ["客户 ID", "客群", "R_最近购买距今天数", "F_购买次数", "M_销售额", "利润贡献",
            "订单行数", "R_score", "F_score", "M_score", "RFM总分"]
    rfm[cols].round(2).to_csv(os.path.join(RES, "07_RFM客户分群.csv"),
                              encoding="utf-8-sig", index=False)
    return rfm, cent, names


def discount_bins(df):
    # 完整覆盖：数据实际存在 60%/80% 折扣（208 行，全部亏损），原主分析表切到 40% 会丢弃它们
    bins = [-0.001, 0.0001, 0.1, 0.2, 0.3, 0.4001, 1.0]
    labels = ["无折扣", "0-10%", "10-20%", "20-30%", "30-40%", ">40%"]
    df = df.copy()
    df["折扣档"] = pd.cut(df["折扣"], bins=bins, labels=labels)
    return df


def cross_rfm_discount(df, rfm):
    df = df.merge(rfm[["客户 ID", "客群", "RFM总分"]], on="客户 ID", how="left")
    df = discount_bins(df)
    df["高折扣GMV"] = np.where(df["折扣"] >= 0.3, df["销售额"], 0.0)

    g = df.groupby(["客群", "折扣档"], observed=True)
    cross = g.agg(
        订单行数=("行 ID", "size"),
        GMV=("销售额", "sum"),
        利润=("利润", "sum"),
        亏损行占比=("亏损", "mean"),
        平均折扣=("折扣", "mean"),
    )
    cross["利润率(%)"] = (cross["利润"] / cross["GMV"] * 100).round(2)
    cross["亏损行占比(%)"] = (cross["亏损行占比"] * 100).round(2)
    cross["平均折扣(%)"] = (cross["平均折扣"] * 100).round(2)
    cross["GMV(万元)"] = (cross["GMV"] / 10000).round(2)
    keep = ["订单行数", "GMV(万元)", "利润率(%)", "亏损行占比(%)", "平均折扣(%)"]
    cross[keep].round(2).to_csv(os.path.join(RES, "08_客群折扣交叉.csv"), encoding="utf-8-sig")

    seg = df.groupby("客群").agg(
        客户数=("客户 ID", "nunique"),
        订单行数=("行 ID", "size"),
        GMV=("销售额", "sum"),
        利润=("利润", "sum"),
        平均折扣=("折扣", "mean"),
        高折扣GMV=("高折扣GMV", "sum"),
    )
    seg["利润率(%)"] = (seg["利润"] / seg["GMV"] * 100).round(2)
    seg["高折扣GMV占比(%)"] = (seg["高折扣GMV"] / seg["GMV"] * 100).round(2)
    seg["客均GMV"] = (seg["GMV"] / seg["客户数"]).round(2)
    seg["平均折扣(%)"] = (seg["平均折扣"] * 100).round(2)
    seg.sort_values("利润率(%)", ascending=False).round(2).to_csv(
        os.path.join(RES, "08b_客群总览.csv"), encoding="utf-8-sig")
    return df, cross, seg


# ---------- 用法 B：客户固定效应回归 ----------
def fixed_effects(df):
    d = df[["客户 ID", "折扣", "行级利润率", "子类别", "年"]].rename(columns={
        "客户 ID": "customer", "折扣": "discount", "行级利润率": "margin",
        "子类别": "category", "年": "year"}).dropna()
    lo, hi = d["margin"].quantile(0.01), d["margin"].quantile(0.99)
    d["margin_w"] = d["margin"].clip(lo, hi)

    cluster = {"groups": d["customer"]}
    pool = smf.ols("margin_w ~ discount", data=d).fit(
        cov_type="cluster", cov_kwds=cluster)
    fe = smf.ols("margin_w ~ discount + C(customer) + C(category) + C(year)", data=d).fit(
        cov_type="cluster", cov_kwds=cluster)

    # within R²：对客户内去均值后的两列做单变量回归
    dm = d[["customer", "margin_w", "discount"]].copy()
    for c in ("margin_w", "discount"):
        dm[c + "_dm"] = dm[c] - dm.groupby("customer")[c].transform("mean")
    within = smf.ols("margin_w_dm ~ discount_dm", data=dm).fit()
    r2w = within.rsquared

    # 稳健性 1：只用有跨折扣变动的客户
    n_disc = d.groupby("customer")["discount"].nunique()
    varying = set(n_disc[n_disc > 1].index)
    dv = d[d["customer"].isin(varying)]
    fe_v = smf.ols("margin_w ~ discount + C(customer) + C(category) + C(year)", data=dv).fit(
        cov_type="cluster", cov_kwds={"groups": dv["customer"]})

    # 稳健性 2：未缩尾利润率
    fe_raw = smf.ols("margin ~ discount + C(customer) + C(category) + C(year)", data=d).fit(
        cov_type="cluster", cov_kwds=cluster)

    def row(model, label, note):
        ci = model.conf_int().loc["discount"]
        return {
            "模型": label, "样本行数": int(model.nobs),
            "折扣系数": round(model.params["discount"], 4),
            "标准误(聚类)": round(model.bse["discount"], 4),
            "t值": round(model.tvalues["discount"], 2),
            "p值": "<0.001" if model.pvalues["discount"] < 0.001 else round(model.pvalues["discount"], 4),
            "95%CI下限": round(ci[0], 4), "95%CI上限": round(ci[1], 4),
            "折扣+10pp对应利润率变化(pp)": round(model.params["discount"] * 0.1 * 100, 2),
            "R²": round(model.rsquared, 4), "说明": note,
        }

    res = pd.DataFrame([
        row(pool, "① 混合OLS（无控制）", "折扣与利润率的原始相关，未控制客户与品类"),
        row(fe, "② 客户固定效应", "控制客户自身特征 + 子类别 + 年份；标准误按客户聚类"),
        row(fe_v, "③ 稳健性：跨折扣客户", "仅保留内部有折扣变动的客户 %d 个" % len(varying)),
        row(fe_raw, "④ 稳健性：未缩尾", "因变量用原始行级利润率，未做 1%/99% 缩尾"),
    ])
    res.to_csv(os.path.join(RES, "09_固定效应回归.csv"), encoding="utf-8-sig", index=False)

    # 客户内折扣档 → 利润率变化（描述性印证）
    dv2 = d.copy()
    dv2["折扣档"] = pd.cut(dv2["discount"], [-0.001, 0.0001, 0.1, 0.2, 0.3, 0.4001, 1.0],
                            labels=["无折扣", "0-10%", "10-20%", "20-30%", "30-40%", ">40%"])
    cwin = dv2.groupby("折扣档", observed=True)["margin_w"].agg(["mean", "size"]).round(4)
    # 客户内去均值后的档位均值：剔除了客户层面固定差异
    dv2["margin_dm"] = dv2["margin_w"] - dv2.groupby("customer")["margin_w"].transform("mean")
    cwin["客户内去均值利润率"] = dv2.groupby("折扣档", observed=True)["margin_dm"].mean().round(4)
    cwin.to_csv(os.path.join(RES, "10_客户内折扣档利润率.csv"), encoding="utf-8-sig")
    return res, cwin, r2w, pool, fe


# ---------- 绘图 ----------
def fig_rfm(seg, cent, names):
    s = seg.sort_values("利润率(%)")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(s))
    bars = ax.bar(x, s["利润率(%)"], width=0.55, color="#2f6fd0", alpha=0.85)
    ax2 = ax.twinx()
    ax2.plot(x, s["高折扣GMV占比(%)"], marker="s", color="#c0392b", linewidth=2,
             label="高折扣(≥30%)GMV占比")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}\n({int(n)}人)" for i, n in zip(s.index, s["客户数"])], fontsize=9)
    ax.set_ylabel("客群利润率 (%)", color="#2f6fd0")
    ax2.set_ylabel("高折扣GMV占比 (%)", color="#c0392b")
    ax.axhline(0, color="#999", linewidth=0.8)
    for i, v in enumerate(s["利润率(%)"]):
        ax.text(i, v + (0.4 if v >= 0 else -0.8), f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_title("RFM 客群：利润质量与高折扣依赖度（高价值客户也在吃高折扣）", fontsize=12.5, pad=12)
    ax2.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "05_RFM客群利润率.png"), dpi=160)
    plt.close(fig)


def fig_heatmap(cross):
    p = cross["利润率(%)"].unstack("折扣档")
    p = p.reindex(index=[i for i in p.index if isinstance(i, str)],
                  columns=["无折扣", "0-10%", "10-20%", "20-30%", "30-40%", ">40%"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    data = p.values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", vmin=-40, vmax=35, aspect="auto")
    ax.set_xticks(range(p.shape[1])); ax.set_xticklabels(p.columns, fontsize=10)
    ax.set_yticks(range(p.shape[0])); ax.set_yticklabels(p.index, fontsize=10)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=9,
                        color="#111" if -15 < v < 15 else "#fff")
    ax.set_title("客群 × 折扣档利润率：同一折扣下客群分化，同一客群内折扣侵蚀", fontsize=12.5, pad=12)
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("利润率 (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "06_客群折扣热力图.png"), dpi=160)
    plt.close(fig)


def fig_fe(res, r2w):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    r = res.iloc[:2][::-1]
    y = np.arange(len(r))
    coef = r["折扣系数"].values * 10 * 100  # 每 10pp 折扣的利润率变化(pp)
    lo = r["95%CI下限"].values * 10 * 100
    hi = r["95%CI上限"].values * 10 * 100
    ax.errorbar(coef, y, xerr=[coef - lo, hi - coef], fmt="o", color="#1f4e79",
                capsize=5, markersize=8, linewidth=2)
    ax.axvline(0, color="#999", linewidth=0.9, linestyle="--")
    ax.set_yticks(y); ax.set_yticklabels(r["模型"], fontsize=10)
    ax.set_xlabel("折扣提高 10 个百分点 → 利润率变化（百分点）")
    ax.set_title(f"控制客户自身特征后负效应几乎不变：效应是客户内生的（within R²={r2w:.3f}）", fontsize=11.5, pad=12)
    for i, v in enumerate(coef):
        ax.text(v, i + 0.16, f"{v:.1f}pp", ha="center", fontsize=9.5, color="#c0392b")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "07_固定效应对比.png"), dpi=160)
    plt.close(fig)


def main():
    df = load()
    rfm, cent, names = build_rfm(df)
    df2, cross, seg = cross_rfm_discount(df, rfm)
    res, cwin, r2w, pool, fe = fixed_effects(df)

    fig_rfm(seg, cent, names)
    fig_heatmap(cross)
    fig_fe(res, r2w)

    print("=" * 64)
    print("RFM 聚类中心（1-5 分，R 越高=越近期）")
    print(cent.round(2).to_string())
    print("\n客群命名:", names)
    print("\n客群总览")
    print(seg[["客户数", "订单行数", "GMV", "利润率(%)", "平均折扣(%)", "高折扣GMV占比(%)"]].to_string())
    print("\n客群 × 折扣档（利润率%）")
    print(cross["利润率(%)"].unstack("折扣档").to_string())
    print("\n固定效应回归")
    print(res[["模型", "折扣系数", "标准误(聚类)", "p值", "折扣+10pp对应利润率变化(pp)", "R²"]].to_string(index=False))
    print("\nwithin R²:", round(r2w, 4))
    print("\n客户内折扣档利润率")
    print(cwin.to_string())
    print("\n输出完成")


if __name__ == "__main__":
    main()
