"""실험 7 요약: 단위 원자료(result/runs/exp7_units/)에서만 표를 만든다.
칸·정책별 누적 이윤(30기간, 기간 0~19)·충족률·재참여율·지급 잉여·공급자 잉여 집중도를 내고, 반복 단위 짝지은 차이의 95% CI(t 구간, 반복 bootstrap)를 낸다.
파일럿이면 사전 등록한 규칙으로 본 실험 반복 수를 계산하고, 칸마다 세 경우 중 어디에 해당하는지 판정한다.
"""
import json
import numpy as np
from scipy.stats import t as tdist
from result import diag
from result.exp5_fair import cell_cfg, tag
from result.exp7_oracle import UNITS, CELLS, POLS, C_POLS

OUT = diag.RUNS / "exp7_summary.json"
KEY = (("Uniform", "Myopic-sup"), ("Oracle", "Uniform"), ("Oracle", "Shuffled"))
REF = (("Oracle", "Myopic-sup"), ("Oracle", "Rule"), ("Oracle", "Myopic"), ("Rule", "Myopic-sup"), ("Myopic", "Myopic-sup"))
TOP = 3
H_FRAC = 0.02


# 단위 하나의 반복 지표 (top_* 는 c를 잰 정책에서만: 상태 내 c 상위 3 공급자의 잉여 비중과 재참여율)
def rep_stats(u):
  R = u["rows"]
  s = lambda k, a=0, b=None: float(sum(r[k] for r in R[a:b]))
  ss = [np.array(r["ss"]) for r in R]
  hhi = [float(((x / x.sum()) ** 2).sum()) for x in ss if x.sum() > 0]
  out = dict(profit=s("profit"), profit20=s("profit", 0, 20), fill=s("n_ok") / s("n_buy"),
             ret_buy=s("stay_buy") / s("n_buy"), ret_sup=s("stay_sup") / s("n_sup"),
             act_buy=s("n_buy") / len(R), act_sup=s("n_sup") / len(R), new_sup=s("new_sup"),
             sb=s("sb"), ss=float(sum(x.sum() for x in ss)), paid_sup=float(np.mean([(x > 0).sum() for x in ss])),
             hhi=float(np.mean(hhi)) if hhi else float("nan"))
  if R[0]["c"] is not None:
    sh, st_top, st_rest = [], [], []
    for r, x in zip(R, ss):
      c, stay = np.array(r["c"]), np.array(r["stay"])
      if len(c) <= TOP:
        continue
      top = np.argsort(-c)[:TOP]
      rest = np.setdiff1d(np.arange(len(c)), top)
      if x.sum() > 0:
        sh.append(x[top].sum() / x.sum())
      st_top.append(stay[top].mean())
      st_rest.append(stay[rest].mean())
    out.update(top_share=float(np.mean(sh)) if sh else float("nan"), ret_top=float(np.mean(st_top)), ret_rest=float(np.mean(st_rest)),
               c_mean=float(np.mean([np.mean(r["c"]) for r in R])))
  return out


# 칸의 정책별 반복 지표 (반복이 모두 있는 정책만)
def load(cell):
  cfg = cell_cfg(*cell)
  out = {}
  for pol in POLS:
    fs = sorted(UNITS.glob(f"{tag(cfg)}_{pol}_r*.json"), key=lambda f: int(f.stem.rsplit("_r", 1)[1]))
    if fs:
      us = [json.loads(f.read_text(encoding="utf-8")) for f in fs]
      out[pol] = {u["rep"]: rep_stats(u) for u in us}
  return cfg, out


# 짝지은 차이 a − b: 평균, t 95% CI, 반복 bootstrap 95% CI, 표준편차 (공통 반복만)
def paired(cfg, D, a, b, k="profit"):
  reps = sorted(set(D[a]) & set(D[b]))
  d = np.array([D[a][r][k] - D[b][r][k] for r in reps])
  n, m, sd = len(d), float(d.mean()), float(d.std(ddof=1))
  h = tdist.ppf(0.975, n - 1) * sd / np.sqrt(n)
  bs = np.random.default_rng(cfg.seed).choice(d, (cfg.boot_B, n)).mean(1)
  return dict(n=n, diff=m, sd=sd, lo=m - h, hi=m + h, blo=float(np.percentile(bs, 2.5)), bhi=float(np.percentile(bs, 97.5)))


# 세 경우 판정 (CI가 0을 벗어나면 차이 있음)
def verdict(P):
  up = lambda x: x["lo"] > 0
  same = lambda x: x["lo"] <= 0 <= x["hi"]
  om, ou = P["Oracle-Myopic-sup"], P["Oracle-Uniform"]
  if same(om):
    return "Oracle ≈ Myopic-sup"
  if up(om) and same(ou):
    return "Oracle > Myopic-sup, Uniform ≈ Oracle"
  if up(om) and up(ou):
    return "Oracle > Uniform, Myopic-sup"
  return "기타 (Oracle < Myopic-sup 또는 Oracle < Uniform)"


if __name__ == "__main__":
  res = dict(note="Oracle-c는 진정한 최적 oracle이 아니라 Rule을 10기간 이어 간다고 보고 잰 MC marginal value(M = 10)를 매 기간 쓰는 benchmark.", cells={})
  for cell in CELLS:
    cfg, D = load(cell)
    if not all(p in D for p in POLS):
      continue
    pols = {p: {k: float(np.nanmean([v[k] for v in D[p].values()])) for k in next(iter(D[p].values()))} for p in POLS}
    P = {f"{a}-{b}": paired(cfg, D, a, b) for a, b in KEY + REF}
    P20 = {f"{a}-{b}": paired(cfg, D, a, b, "profit20") for a, b in KEY + REF}
    n = len(D["Myopic-sup"])
    h = H_FRAC * pols["Myopic-sup"]["profit"]
    sd = max(P[f"{a}-{b}"]["sd"] for a, b in KEY)
    need = int(np.clip(np.ceil((tdist.ppf(0.975, n - 1) * sd / h) ** 2), 10, 30))
    res["cells"][tag(cfg)] = dict(reps=n, pols=pols, paired=P, paired20=P20, verdict=verdict(P),
                                  reps_rule=dict(h=h, sd_max=sd, need=need))
    print(f"\n{tag(cfg)} (반복 {n}) 판정: {verdict(P)} | 본 실험 반복 수 규칙: h {h:.0f}, sd {sd:.0f} → {need}")
    for p in POLS:
      x = pols[p]
      print(f"  {p:10s} 이윤 {x['profit']:8.0f} (0~19 {x['profit20']:7.0f}) 충족 {x['fill']:.3f} 재참여 주문자 {x['ret_buy']:.3f} 공급자 {x['ret_sup']:.3f} "
            f"활동 공급자 {x['act_sup']:.1f} 잉여 주문자 {x['sb']:6.0f} 공급자 {x['ss']:6.0f} 받은 공급자 {x['paid_sup']:.1f} HHI {x['hhi']:.2f}"
            + (f" | 상위3 비중 {x['top_share']:.2f} 재참여 상위3 {x['ret_top']:.2f} 나머지 {x['ret_rest']:.2f}" if p in C_POLS else ""))
    for k, x in P.items():
      print(f"  {k:22s} {x['diff']:+8.0f} t[{x['lo']:+.0f}, {x['hi']:+.0f}] boot[{x['blo']:+.0f}, {x['bhi']:+.0f}] | 0~19 {P20[k]['diff']:+.0f} [{P20[k]['lo']:+.0f}, {P20[k]['hi']:+.0f}]")
  OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
  print(f"\nsaved {OUT}")
