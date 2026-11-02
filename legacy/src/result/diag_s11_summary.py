"""진단 s11 요약: diag_s11_plan.json과 diag_s11_units/에서만 표를 만든다 (Rule-continuation c_i).
(1) c_i 잡음, (2) own feature가 비슷하고 context 하나가 다른 자연 짝의 c 차이, (3) 섭동별 Δc_i, (4) context 구간별 c,
(5) own-only와 own + context 선형 모형(ridge, λ 고정)의 반복 단위 교차검증 성능, (6) 상태 내 순위 일치를 낸다.
"""
import json
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from result.diag_s11_context import PLAN, UNITS, CASES
from result import diag

OUT = diag.RUNS / "diag_s11_context.json"
CSV = diag.RUNS / "diag_s11_context_obs.csv"
OWN = ("cap", "n_items", "cost_mean", "cost_min", "hs", "rs", "tenure")
MATCH = ("cap", "n_items", "cost_mean", "hs", "rs")
CALIPER = 0.5
LAM = 1.0
B = 2000
# 자연 짝의 context 기준: (변수, 희소한 쪽 판정, 최소 차이)
REL = {"alt_min": (lambda a, b: a < b, 2), "ratio_max": (lambda a, b: a > b, 0.5), "hi_max": (lambda a, b: a > b, 0.15),
       "zero_if_removed": (lambda a, b: a > b, 1), "n_pend": (lambda a, b: a < b, 1)}
EXPECT = {"alt_minus": +1, "alt_plus": -1, "slow": +1, "dem_hi": +1, "dem_lo": -1}


# 평균과 표준오차
def ms(x):
  x = np.asarray(x)
  return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


# 부트스트랩 평균 CI
def boot(x, seed=0):
  x = np.asarray(x)
  g = np.random.default_rng(seed)
  m = g.choice(x, (B, len(x))).mean(1)
  return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# 관측 행: 상태·공급자별 c, se, own, ctx
def load():
  P = json.loads(PLAN.read_text(encoding="utf-8"))
  vals, pert = defaultdict(dict), []
  for f in UNITS.glob("*.json"):
    u = json.loads(f.read_text(encoding="utf-8"))
    if u["pert"]:
      pert.append(u)
    else:
      vals[(u["pol"], u["rep"], u["t"])].update({k: np.array(v) for k, v in u["values"].items()})
  rows, futs = [], {}
  for s in P:
    V = vals[(s["pol"], s["rep"], s["t"])]
    full = V[""]
    for x in s["sup"]:
      d = full - V[str(x["sid"])]
      futs[(s["pol"], s["rep"], s["t"], x["sid"])] = d
      m, se = ms(d)
      rows.append(dict(pol=s["pol"], rep=s["rep"], t=s["t"], sid=x["sid"], c=m, se=se, **x["own"], **x["ctx"]))
  return P, rows, futs, pert


# own feature가 비슷하고 (같은 품목 수, 표준화 거리 ≤ CALIPER) context 변수 v가 기준 이상 다른 짝을 탐욕적으로 고른다 (비복원)
def matched(rows, v):
  better, gap = REL[v]
  X = np.array([[r[k] for k in MATCH] for r in rows], float)
  Z = (X - X.mean(0)) / X.std(0)
  cand = []
  for a in range(len(rows)):
    for b in range(len(rows)):
      if a == b or rows[a]["n_items"] != rows[b]["n_items"]:
        continue
      if better(rows[a][v], rows[b][v]) and abs(rows[a][v] - rows[b][v]) >= gap:
        dist = float(np.sqrt(np.mean((Z[a] - Z[b]) ** 2)))
        if dist <= CALIPER:
          cand.append((dist, a, b))
  used, out = set(), []
  for dist, a, b in sorted(cand):
    if a in used or b in used:
      continue
    used |= {a, b}
    out.append((a, b, dist))
  return out


# ridge (표준화, 절편 무벌점)
def ridge(Xtr, ytr, Xte):
  mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
  A, Bm = (Xtr - mu) / sd, (Xte - mu) / sd
  w = np.linalg.solve(A.T @ A + LAM * np.eye(A.shape[1]), A.T @ (ytr - ytr.mean()))
  return Bm @ w + ytr.mean(), w


# 반복 단위 교차검증 (같은 반복의 Myopic·Rule 상태를 함께 뺀다)
def cv(rows, keys):
  X = np.array([[r[k] for k in keys] for r in rows], float)
  y = np.array([r["c"] for r in rows])
  g = np.array([r["rep"] for r in rows])
  pred = np.zeros(len(y))
  for rep in np.unique(g):
    tr, te = g != rep, g == rep
    pred[te], _ = ridge(X[tr], y[tr], X[te])
  st = defaultdict(list)
  for i, r in enumerate(rows):
    st[(r["pol"], r["rep"], r["t"])].append(i)
  rho = [spearmanr(pred[ix], y[ix])[0] for ix in st.values() if len(ix) >= 3]
  top = [len(set(np.array(ix)[np.argsort(-pred[ix])[:3]]) & set(np.array(ix)[np.argsort(-y[ix])[:3]])) / 3 for ix in st.values() if len(ix) > 3]
  _, w = ridge(X, y, X)
  return dict(r2=float(1 - np.mean((pred - y) ** 2) / np.var(y)), mae=float(np.mean(np.abs(pred - y))), spearman_within=float(np.nanmean(rho)),
              top3=float(np.mean(top)), coef=dict(zip(keys, w.tolist())), pred=pred)


if __name__ == "__main__":
  P, rows, futs, pert = load()
  CTX = [k for k in rows[0] if k not in OWN and k not in ("pol", "rep", "t", "sid", "c", "se")]
  y, se = np.array([r["c"] for r in rows]), np.array([r["se"] for r in rows])
  res = dict(note="PROC15H, Rule-continuation c_i (M = 40, vf_H = 10). 정책 성능이 아니다.", n_states=len(P),
             n_obs=len(rows), by_pol={p: sum(r["pol"] == p for r in rows) for p in ("Myopic", "Rule")},
             noise=dict(c_mean=float(y.mean()), c_sd=float(y.std()), se_mean=float(se.mean()), reliability=float(1 - np.mean(se ** 2) / np.var(y))))
  res["matched"] = {}
  for v in REL:
    pr = matched(rows, v)
    if not pr:
      res["matched"][v] = dict(n=0)
      continue
    d = np.array([rows[a]["c"] - rows[b]["c"] for a, b, _ in pr])
    dse = np.sqrt([rows[a]["se"] ** 2 + rows[b]["se"] ** 2 for a, b, _ in pr])
    other = {k: float(np.mean([rows[a][k] - rows[b][k] for a, b, _ in pr])) for k in CTX + list(OWN)}
    lo, hi = boot(d)
    res["matched"][v] = dict(n=len(pr), mean=float(d.mean()), lo=lo, hi=hi, pos=float((d > 0).mean()), effect=float(d.mean() / d.std(ddof=1)),
                             mc_se_mean=float(dse.mean()), sig_pos=float((d / dse > 1.96).mean()), sig_neg=float((d / dse < -1.96).mean()),
                             own_dist=float(np.mean([x for _, _, x in pr])), other_diff=other,
                             pairs=[dict(a=rows[a], b=rows[b], d=float(rows[a]["c"] - rows[b]["c"])) for a, b, _ in pr[:3]])
  res["pert"] = {}
  base = {(r["pol"], r["rep"], r["t"], r["sid"]): r for r in rows}
  for case in CASES:
    D, rel, skip = [], [], 0
    for u in pert:
      if u["pert"][1] != case:
        continue
      if u["info"] is None:
        skip += 1
        continue
      sid = u["pert"][0]
      cp = np.array(u["values"][""]) - np.array(u["values"][str(sid)])
      c0 = futs[(u["pol"], u["rep"], u["t"], sid)]
      dm, dse = ms(cp - c0)
      D.append(dict(dc=dm, se=dse, c0=float(c0.mean()), c1=float(cp.mean()), pol=u["pol"], info=u["info"]))
    if not D:
      res["pert"][case] = dict(n=0, skipped=skip)
      continue
    dc = np.array([x["dc"] for x in D])
    lo, hi = boot(dc)
    res["pert"][case] = dict(n=len(D), skipped=skip, mean=float(dc.mean()), lo=lo, hi=hi, expect=EXPECT[case],
                             agree=float((np.sign(dc) == EXPECT[case]).mean()), sig_agree=float(np.mean([x["dc"] / x["se"] * EXPECT[case] > 1.96 for x in D])),
                             sig_against=float(np.mean([x["dc"] / x["se"] * EXPECT[case] < -1.96 for x in D])),
                             mean_abs=float(np.abs(dc).mean()), c0_abs=float(np.mean([abs(x["c0"]) for x in D])), mean_se=float(np.mean([x["se"] for x in D])), rows=D)
  res["bins"] = {}
  for v, edges in (("alt_min", (0, 1, 2, 3, 99)), ("ratio_max", None), ("hi_max", None), ("n_pend", (0, 1, 99)), ("zero_if_removed", (0, 1, 2))):
    x = np.array([r[v] for r in rows], float)
    e = np.quantile(x, [0, 1 / 3, 2 / 3, 1]) if edges is None else np.array(edges, float)
    out = []
    for lo_, hi_ in zip(e[:-1], e[1:]):
      m = (x >= lo_) & ((x < hi_) if hi_ < e[-1] or edges is not None else (x <= hi_))
      if m.sum():
        out.append(dict(lo=float(lo_), hi=float(hi_), n=int(m.sum()), c=float(y[m].mean()), se=float(y[m].std(ddof=1) / np.sqrt(m.sum())) if m.sum() > 1 else None))
    res["bins"][v] = out
  A, Bm = cv(rows, list(OWN)), cv(rows, list(OWN) + CTX)
  res["models"] = {k: {kk: vv for kk, vv in m.items() if kk != "pred"} for k, m in (("own", A), ("own_ctx", Bm))}
  d = np.abs(A["pred"] - y) - np.abs(Bm["pred"] - y)
  g = np.array([r["rep"] for r in rows])
  reps = np.unique(g)
  gb = np.random.default_rng(1)
  bs = [np.mean(np.concatenate([d[g == r] for r in gb.choice(reps, len(reps))])) for _ in range(B)]
  res["models"]["mae_gain"] = dict(mean=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
  res["ctx_keys"] = CTX
  OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
  with CSV.open("w", encoding="utf-8") as f:
    keys = list(rows[0])
    f.write(",".join(keys) + "\n")
    for r in rows:
      f.write(",".join(str(r[k]) for k in keys) + "\n")
  n = res["noise"]
  print(f"상태 {len(P)}, 공급자 관측 {len(rows)} {res['by_pol']} | c 평균 {n['c_mean']:.1f} sd {n['c_sd']:.1f} MC se {n['se_mean']:.1f} 신뢰도 {n['reliability']:.2f}")
  print("\n[자연 짝: 희소한 쪽 c − 덜 희소한 쪽 c]")
  for v, x in res["matched"].items():
    if x["n"]:
      print(f"  {v}: 짝 {x['n']} 평균 {x['mean']:+.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}] 양수 {x['pos']:.2f} effect {x['effect']:+.2f} 유의 양/음 {x['sig_pos']:.2f}/{x['sig_neg']:.2f} own 거리 {x['own_dist']:.2f}")
    else:
      print(f"  {v}: 짝 없음")
  print("\n[섭동 Δc = c(바뀐 context) − c(원래)]")
  for c, x in res["pert"].items():
    if x["n"]:
      print(f"  {c}: n {x['n']} (적용 불가 {x['skipped']}) 평균 {x['mean']:+.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}] 기대 부호 {x['expect']:+d} 일치 {x['agree']:.2f} 유의 일치/반대 {x['sig_agree']:.2f}/{x['sig_against']:.2f} 평균|Δc| {x['mean_abs']:.1f} vs 평균|c| {x['c0_abs']:.1f} (se {x['mean_se']:.1f})")
    else:
      print(f"  {c}: 적용 가능 대상 없음 ({x['skipped']})")
  print("\n[context 구간별 평균 c]")
  for v, bins in res["bins"].items():
    print(f"  {v}: " + " | ".join(f"[{b['lo']:.2f}, {b['hi']:.2f}) n {b['n']} c {b['c']:.1f}" for b in bins))
  print("\n[선형 모형, 반복 단위 교차검증]")
  for k in ("own", "own_ctx"):
    m = res["models"][k]
    print(f"  {k}: R² {m['r2']:.3f} MAE {m['mae']:.1f} 상태 내 Spearman {m['spearman_within']:.3f} top3 {m['top3']:.2f}")
  g_ = res["models"]["mae_gain"]
  print(f"  MAE 개선 (own − own+ctx): {g_['mean']:+.2f} [{g_['lo']:+.2f}, {g_['hi']:+.2f}], 신뢰도 상한 R² {n['reliability']:.2f}")
  print("  own+ctx 표준화 계수 상위:", sorted(res["models"]["own_ctx"]["coef"].items(), key=lambda x: -abs(x[1]))[:8])
  print(f"\nsaved {OUT}, {CSV}")
