"""진단 s12 보조: Context MLP 오차 분석과 학습 곡선 (diag_s12_pred.csv와 diag s11 상태에서만, 모형 입력은 바꾸지 않는다).
(1) 학습 반복 수 2·4·6에서 Own·Context MLP 시험 성능 (같은 분할 규칙, 같은 설정 예산의 일부: 은닉 64, wd 0.01 고정)으로 남은 차이가 데이터 양 때문인지 본다.
(2) 입력에 없는 사후 구조 지표(빼면 생기는 용량 부족, AON 노출, 단독 공급 품목 수)와 Context MLP 잔차의 관계, 큰 오차 사례, 입력이 거의 같은 짝의 c 차이를 본다.
"""
import csv
import json
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
from scipy.stats import spearmanr
from vfa.mlp import MLP
from result.diag_s11_context import state
from result.diag_s12_mlp import BASE, SEEDS, OWN, CTX, PRED, folds
from result import diag

OUT = diag.RUNS / "diag_s12_errors.json"
NTR = (2, 4, 6)


# 학습 곡선 작업: (입력 이름, 시험 반복, 학습 반복 수)
def lc(a):
  name, te, va, ntr, X, y, g, reps = a
  rest = [r for r in reps if r not in (te, va)]
  tr = np.isin(g, rest[:ntr])
  pt = []
  for s in SEEDS:
    m = MLP(replace(BASE, nn_hidden=64, nn_seed=s), 0.01).fit(X[tr], y[tr], X[g == va], y[g == va])
    pt.append(m.predict(X[g == te]))
  return name, te, ntr, np.mean(pt, 0)


# 상태 하나의 공급자별 사후 구조 지표
# short_gap: 빼면 늘어나는 품목 용량 부족 합 Σ max(0, D − (C − cap_i)) − max(0, D − C)
# aon_exposed: 빼면 묶음의 어떤 품목이 요구량을 못 채우게 되는 주문자의 묶음 전체 요구량 (빼기 전에는 채울 수 있던 주문자만)
# sole: i가 유일한 활동 공급자인 품목 수
def structure(env):
  o = env.o
  C, D = o.cap.sum(0), o.q.sum(0)
  out = {}
  for k, s in enumerate(o.sid):
    C2 = C - o.cap[k]
    gap = float(np.maximum(D - C2, 0).sum() - np.maximum(D - C, 0).sum())
    ok_before = ((o.q <= C[None]) | (o.q == 0)).all(1)
    ok_after = ((o.q <= C2[None]) | (o.q == 0)).all(1)
    exp_ = float(o.q[ok_before & ~ok_after].sum())
    sole = int(((o.cap > 0).sum(0)[o.cap[k] > 0] == 1).sum())
    out[s] = dict(short_gap=gap, aon_exposed=exp_, sole=sole)
  return out


if __name__ == "__main__":
  rows = list(csv.DictReader(PRED.open(encoding="utf-8")))
  num = lambda k: np.array([float(r[k]) for r in rows])
  y, se, g = num("c"), num("se"), num("rep").astype(int)
  reps = sorted(set(g.tolist()))
  F = folds(reps)
  Xo, Xc = np.c_[[num(k) for k in OWN]].T, np.c_[[num(k) for k in OWN + CTX]].T
  jobs = [(n, te, va, k, X, y, g, reps) for n, X in (("own", Xo), ("ctx", Xc)) for te, va in F for k in NTR]
  P = {}
  with Pool(12) as p:
    for n, te, k, pt in p.imap_unordered(lc, jobs):
      P.setdefault((n, k), np.zeros(len(y)))[g == te] = pt
  curve = {f"{n}_{k}": float(1 - np.mean((P[(n, k)] - y) ** 2) / np.var(y)) for n, k in P}
  pc = num("mlp_ctx")
  res_ = y - pc
  st = {}
  for key in sorted({(r["pol"], int(r["rep"]), int(r["t"])) for r in rows}):
    st[key] = structure(state(*key))
  S = np.array([[st[(r["pol"], int(r["rep"]), int(r["t"]))][int(r["sid"])][k] for k in ("short_gap", "aon_exposed", "sole")] for r in rows])
  corr = {k: dict(spearman_resid=float(spearmanr(S[:, j], res_)[0]), spearman_c=float(spearmanr(S[:, j], y)[0]))
          for j, k in enumerate(("short_gap", "aon_exposed", "sole"))}
  # 사후 지표를 더하면 잔차가 줄어드는가: 잔차를 사후 지표로 선형 회귀한 반복 단위 교차검증 R²
  Z = np.c_[S, np.ones(len(y))]
  pr = np.zeros(len(y))
  for te in reps:
    w = np.linalg.lstsq(Z[g != te], res_[g != te], rcond=None)[0]
    pr[g == te] = Z[g == te] @ w
  resid_r2 = float(1 - np.mean((res_ - pr) ** 2) / np.var(res_))
  big = np.argsort(-np.abs(res_))[:15]
  cases = [dict(pol=rows[i]["pol"], rep=int(rows[i]["rep"]), t=int(rows[i]["t"]), sid=int(rows[i]["sid"]), c=float(y[i]), se=float(se[i]),
                pred=float(pc[i]), **{k: float(rows[i][k]) for k in ("cap", "n_items", "hs", "alt_min", "alt_mean", "zero_if_removed", "ratio_max", "hi_max", "n_buy", "hb_mean", "n_pend")},
                **dict(zip(("short_gap", "aon_exposed", "sole"), S[i].tolist()))) for i in big]
  Zs = (Xc - Xc.mean(0)) / (Xc.std(0) + 1e-9)
  near = []
  for i in range(len(y)):
    d = np.sqrt(((Zs - Zs[i]) ** 2).mean(1))
    d[i] = np.inf
    j = int(np.argmin(d))
    if i < j or np.argmin(np.where(np.arange(len(y)) == j, np.inf, np.sqrt(((Zs - Zs[j]) ** 2).mean(1)))) != i:
      near.append((float(d[j]), i, j))
  near = sorted(set((round(a, 6), min(i, j), max(i, j)) for a, i, j in near))[:100]
  z = np.array([abs(y[i] - y[j]) / np.sqrt(se[i] ** 2 + se[j] ** 2) for _, i, j in near])
  out = dict(note="Context MLP 오차 분석·학습 곡선 (사후 분석, 모형 입력 변경 없음).", curve=curve, struct_corr=corr, resid_r2_struct=resid_r2,
             cases=cases, near=dict(n=len(near), mean_dist=float(np.mean([a for a, _, _ in near])), z_gt2=float((z > 2).mean()), z_median=float(np.median(z)),
                                   d_mean=float(np.mean([abs(y[i] - y[j]) for _, i, j in near]))))
  OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
  print("학습 곡선 (시험 R², 학습 반복 수):", {k: round(v, 3) for k, v in sorted(curve.items())})
  print("사후 구조 지표와 c, Context MLP 잔차의 Spearman:", corr)
  print(f"잔차를 사후 구조 지표로 설명하는 교차검증 R²: {resid_r2:.3f}")
  print(f"입력이 가장 가까운 짝 {len(near)}개: 평균 거리 {out['near']['mean_dist']:.3f}, |Δc| 평균 {out['near']['d_mean']:.1f}, MC 잡음 대비 z > 2 비율 {out['near']['z_gt2']:.2f} (잡음만이면 약 0.05), z 중앙값 {out['near']['z_median']:.2f}")
  for c in cases[:8]:
    print("  ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()})
  print(f"saved {OUT}")
