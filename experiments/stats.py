"""
stats.py — Statistical procedures of the Chapter 4 evaluation.

Purpose : One implementation of every statistic reported in Chapter 4, written out
          in full (not hidden behind library defaults) so each value can be traced
          to its equation.
Function : desc() descriptive statistics with the t-based 95% CI of the mean;
          wilcoxon() paired signed-rank test with tie correction, Z, and r = Z/sqrt(N);
          holm() step-down adjusted p-values; mcnemar() exact test on paired rates;
          wilson() score interval; shapiro() normality of paired differences;
          friedman(); kruskal(); chi2(); mannwhitney() with rank-biserial r;
          stratified_mean() population-weighted estimator of a stratified sample.
Notes   : Missing values (NaN) are dropped pairwise. Significance level 0.05.
"""
import math

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


def _num(v):
    return pd.to_numeric(pd.Series(v), errors="coerce").to_numpy(float)


def desc(v):
    """n, mean, SD (n-1), median, Q1, Q3, IQR, min, max, and the 95% CI half-width of the mean."""
    v = _num(v)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"n": 0}
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    q1, q3 = float(np.percentile(v, 25)), float(np.percentile(v, 75))
    return {"n": int(v.size), "mean": float(v.mean()), "sd": sd, "median": float(np.median(v)),
            "q1": q1, "q3": q3, "iqr": q3 - q1, "min": float(v.min()), "max": float(v.max()),
            "ci95": float(stats.t.ppf(0.975, v.size - 1) * sd / math.sqrt(v.size)) if v.size > 1 else 0.0}


def wilcoxon(x, y):
    """
    Paired Wilcoxon signed-rank test of y against x (d = y - x). Zero differences are
    dropped; |d| is ranked with average ranks for ties.
      Z = (W+ - n'(n'+1)/4) / sqrt(n'(n'+1)(2n'+1)/24 - sum(t^3 - t)/48)
    Two-sided p from N(0, 1); r = Z / sqrt(N), N = complete pairs (Fritz et al., 2012).
    """
    x, y = _num(x), _num(y)
    ok = ~(np.isnan(x) | np.isnan(y))
    d = (y - x)[ok]
    N = int(ok.sum())
    nz = d[d != 0]
    n = nz.size
    out = {"N": N, "n_nonzero": int(n), "n_pos": int((nz > 0).sum()), "n_neg": int((nz < 0).sum()),
           "n_zero": int(N - n), "median_diff": float(np.median(d)) if N else None,
           "mean_diff": float(d.mean()) if N else None}
    if n == 0:
        return {**out, "W_plus": 0.0, "W_minus": 0.0, "E_W": 0.0, "SD_W": 0.0, "tie_corr": 0.0,
                "Z": 0.0, "p": 1.0, "r": 0.0}
    ranks = stats.rankdata(np.abs(nz))
    w_plus = float(ranks[nz > 0].sum())
    w_minus = float(ranks[nz < 0].sum())
    _, t = np.unique(np.abs(nz), return_counts=True)
    tie = float(((t ** 3) - t).sum()) / 48
    e_w = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - tie
    z = (w_plus - e_w) / math.sqrt(var) if var > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    return {**out, "W_plus": w_plus, "W_minus": w_minus, "E_W": e_w, "SD_W": math.sqrt(max(var, 0)),
            "tie_corr": tie, "Z": float(z), "p": p, "r": float(z / math.sqrt(N))}


def holm(ps):
    """Holm (1979) step-down adjusted p-values, returned in input order."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj


def mcnemar(a, b):
    """
    Exact McNemar test on paired booleans: a = reference, b = compared.
    b10 = only the reference has the outcome, b01 = only the compared one;
    p = min(1, 2 * P[Binomial(b10 + b01, 0.5) <= min(b10, b01)]).
    """
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    p = float(stats.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
    return {"N": int(a.size), "both": int((a & b).sum()), "neither": int((~a & ~b).sum()),
            "only_ref": n10, "only_cmp": n01, "rate_ref": 100 * float(a.mean()) if a.size else None,
            "rate_cmp": 100 * float(b.mean()) if b.size else None, "p": p}


def wilson(k, n, z=1.959964):
    """Wilson (1927) score interval for k successes in n, in percent."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (c - h), 100 * (c + h))


def shapiro(x, y=None):
    """Shapiro–Wilk W of x, or of the paired differences y - x when y is given."""
    v = _num(x) if y is None else _num(y) - _num(x)
    v = v[~np.isnan(v)]
    if v.size < 3 or np.ptp(v) == 0:
        return {"n": int(v.size), "W": None, "p": None}
    w = stats.shapiro(v)
    return {"n": int(v.size), "W": float(w.statistic), "p": float(w.pvalue)}


def friedman(mat):
    """Friedman (1937) test on a slices x configurations DataFrame (higher = better)."""
    mat = mat.dropna()
    fr = stats.friedmanchisquare(*[mat[c] for c in mat.columns])
    ranks = mat.rank(axis=1, ascending=False)
    k, n = mat.shape[1], len(mat)
    return {"n": n, "k": k, "chi2": float(fr.statistic), "df": k - 1, "p": float(fr.pvalue),
            "mean_rank": {c: float(v) for c, v in ranks.mean().items()},
            "kendall_w": float(fr.statistic / (n * (k - 1)))}


def kruskal(groups):
    g = [np.asarray(v, float)[~np.isnan(np.asarray(v, float))] for v in groups]
    g = [v for v in g if v.size]
    if len(g) < 2:
        return None
    h = stats.kruskal(*g)
    n = sum(v.size for v in g)
    return {"H": float(h.statistic), "df": len(g) - 1, "p": float(h.pvalue), "n": n,
            "epsilon2": float(h.statistic / (n - 1))}


def chi2(table):
    """Chi-square test of independence on a list of [hits, misses] rows, with Cramér's V."""
    t = np.asarray(table, float)
    try:
        c, p, dof, exp = stats.chi2_contingency(t, correction=False)
    except ValueError:
        return None
    n = t.sum()
    return {"chi2": float(c), "df": int(dof), "p": float(p), "n": int(n),
            "cramers_v": float(math.sqrt(c / (n * (min(t.shape) - 1)))), "min_expected": float(exp.min())}


def mannwhitney(a, b):
    """Mann–Whitney U of a vs b with rank-biserial r = 2U/(n_a n_b) - 1 (Kerby, 2014)."""
    a, b = _num(a), _num(b)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if not a.size or not b.size:
        return None
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"U": float(u.statistic), "p": float(u.pvalue), "n_a": int(a.size), "n_b": int(b.size),
            "r_rb": float(2 * u.statistic / (a.size * b.size) - 1)}


def stratified_mean(df, col, strata, N_h):
    """
    Population-weighted (stratified) estimate of a mean (Cochran, 1977):
      ybar_st = sum_h W_h ybar_h,  W_h = N_h / N
      SE      = sqrt( sum_h W_h^2 (1 - f_h) s_h^2 / n_h ),  f_h = n_h / N_h
    df holds one row per slice with column `strata`; N_h maps stratum -> population size.
    """
    N = sum(N_h.values())
    est, var = 0.0, 0.0
    for h, Nh in N_h.items():
        v = _num(df.loc[df[strata] == h, col])
        v = v[~np.isnan(v)]
        if not v.size:
            continue
        W, f = Nh / N, v.size / Nh
        est += W * v.mean()
        if v.size > 1:
            var += W * W * (1 - f) * v.var(ddof=1) / v.size
    se = math.sqrt(var)
    return {"estimate": est, "se": se, "ci95": (est - 1.959964 * se, est + 1.959964 * se)}


def fmt_p(p):
    """APA style: p < .001, otherwise three decimals without the leading zero."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    if p < 0.001:
        return "< .001"
    return f"{p:.3f}".lstrip("0") if p < 1 else "1.000"
