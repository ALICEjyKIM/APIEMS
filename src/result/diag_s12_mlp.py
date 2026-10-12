"""진단 s12: own feature와 network context 요약만으로 MLP가 공급자 c_i를 얼마나 설명하는가 (PROC15H, diag s11의 c_i 929개 재사용).
반복 단위 중첩 분할: 시험 반복 1개, 검증 반복 1개(시험 다음 반복), 학습 6개. 모든 모형이 같은 분할·같은 설정 예산(9개 × 초기화 seed 3개)을 쓴다.
Own MLP와 Context MLP의 차이는 context 요약 포함 여부뿐이다. context 그룹 제거, thin-market 부분집합, 오차 사례를 함께 낸다. c_i는 Rule-continuation 값이다.
"""
import json
import sys
import time
from dataclasses import replace
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from scipy.stats import spearmanr
from utils.params import Cfg
from vfa.mlp import MLP
from result.diag_s11_summary import load, OWN
from result import diag

OUT = diag.RUNS / "diag_s12_mlp.json"
PRED = diag.RUNS / "diag_s12_pred.csv"
HID = (32, 64, 128)
WDS = (0.0, 1e-3, 1e-2)
SEEDS = (0, 1, 2)
LAMS = tuple(float(x) for x in np.logspace(-2, 3, 9))
GROUPS = {"A_competition": ("alt_min", "alt_mean", "altcap_min", "altcap_mean", "share_max", "zero_if_removed", "deg_min"),
          "B_tightness": ("dem_sum", "ratio_max", "ratio_mean", "hi_max", "hi_mean", "bundle_min_deg", "bundle_bottleneck"),
          "C_buyer": ("n_buy", "buy_dem", "hb_mean", "hb_min"),
          "D_recovery": ("pend_cap", "n_pend", "wait_min")}
CTX = tuple(k for g in GROUPS.values() for k in g)
B = 2000
WORKERS = 12
BASE = replace(Cfg(), nn_epochs=Cfg().es_epochs)


# 모형별 입력 열
def sets():
  out = {"own": OWN, "ctx": OWN + CTX}
  for g, ks in GROUPS.items():
    out[f"ctx-{g}"] = OWN + tuple(k for k in CTX if k not in ks)
  return out


# 바깥 fold: 시험 반복 r, 검증 반복 (r + 1) mod 8, 학습은 나머지
def folds(reps):
  return [(r, reps[(i + 1) % len(reps)]) for i, r in enumerate(reps)]


# 작업 하나: (모형, 시험 반복, 설정) → 검증 MSE와 시험 예측 (seed 평균)
def job(a):
  name, cols, te, va, hid, wd, X, y, g = a
  tr = (g != te) & (g != va)
  vm, pt = [], []
  for s in SEEDS:
    m = MLP(replace(BASE, nn_hidden=hid, nn_seed=s), wd).fit(X[tr][:, cols], y[tr], X[g == va][:, cols], y[g == va])
    vm.append(float(np.mean((m.predict(X[g == va][:, cols]) - y[g == va]) ** 2)))
    pt.append(m.predict(X[g == te][:, cols]))
  return name, te, (hid, wd), float(np.mean(vm)), np.mean(pt, 0)


# ridge (학습 기준 표준화)
def ridge(Xtr, ytr, Xte, lam):
  mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
  A, Bm = (Xtr - mu) / sd, (Xte - mu) / sd
  w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ytr - ytr.mean()))
  return Bm @ w + ytr.mean()


# 지표: R², MAE, RMSE, 상태 내 Spearman, 상위 3, 상태 내 상위 20% 적중 (예측·실제 개수가 같아 precision = recall)
def metrics(pred, y, st, idx=None):
  idx = np.arange(len(y)) if idx is None else idx
  p, t = pred[idx], y[idx]
  out = dict(n=len(idx), r2=float(1 - np.mean((p - t) ** 2) / np.var(t)), mae=float(np.mean(np.abs(p - t))), rmse=float(np.sqrt(np.mean((p - t) ** 2))))
  groups = defaultdict(list)
  for i in idx:
    groups[st[i]].append(i)
  rho = [spearmanr(pred[ix], y[ix])[0] for ix in groups.values() if len(ix) >= 3 and np.ptp(y[ix]) > 0]
  top = [len(set(np.array(ix)[np.argsort(-pred[ix])[:3]]) & set(np.array(ix)[np.argsort(-y[ix])[:3]])) / 3 for ix in groups.values() if len(ix) > 3]
  hit = []
  for ix in groups.values():
    k = int(np.ceil(0.2 * len(ix)))
    if len(ix) >= 5:
      hit.append(len(set(np.array(ix)[np.argsort(-pred[ix])[:k]]) & set(np.array(ix)[np.argsort(-y[ix])[:k]])) / k)
  out.update(spearman=float(np.nanmean(rho)) if rho else float("nan"), top3=float(np.mean(top)) if top else float("nan"),
             top20=float(np.mean(hit)) if hit else float("nan"), pooled_spearman=float(spearmanr(p, t)[0]))
  return out


# 반복 블록 부트스트랩 한 번: 뽑힌 반복의 행을 이어 붙이고 상태 키에 뽑힌 순번을 붙인다 (같은 반복이 두 번 뽑혀도 상태 안 지표가 섞이지 않게)
def draw(rng, idx, g, st):
  reps = np.unique(g[idx])
  ix, sk = [], []
  for b, r in enumerate(rng.choice(reps, len(reps))):
    m = idx[g[idx] == r]
    ix.append(m)
    sk += [(st[i], b) for i in m]
  return np.concatenate(ix), sk


# 두 모형 차이의 반복 블록 부트스트랩 (R², MAE, 상태 내 Spearman)
def diff(pa, pb, y, g, st, idx=None):
  idx = np.arange(len(y)) if idx is None else idx
  rng = np.random.default_rng(0)
  full = {k: metrics(pa, y, st, idx)[k] - metrics(pb, y, st, idx)[k] for k in ("r2", "mae", "spearman")}
  draws = defaultdict(list)
  for _ in range(B):
    ix, sk = draw(rng, idx, g, st)
    ma, mb = metrics(pa[ix], y[ix], sk), metrics(pb[ix], y[ix], sk)
    for k in full:
      draws[k].append(ma[k] - mb[k])
  return {k: dict(diff=v, lo=float(np.nanpercentile(draws[k], 2.5)), hi=float(np.nanpercentile(draws[k], 97.5))) for k, v in full.items()}


if __name__ == "__main__":
  t0 = time.time()
  P, rows, futs, _ = load()
  keys = list(OWN + CTX)
  X = np.array([[r[k] for k in keys] for r in rows], float)
  y = np.array([r["c"] for r in rows])
  se = np.array([r["se"] for r in rows])
  g = np.array([r["rep"] for r in rows])
  st = [(r["pol"], r["rep"], r["t"]) for r in rows]
  reps = sorted(np.unique(g).tolist())
  F = folds(reps)
  S = {n: [keys.index(k) for k in ks] for n, ks in sets().items()}
  jobs = [(n, te, va, h, w, X, y, g) for n in S for te, va in F for h in HID for w in WDS]
  jobs = [(n, S[n], te, va, h, w, X, y, g) for n, te, va, h, w, X, y, g in jobs]
  res_jobs = defaultdict(dict)
  with Pool(WORKERS) as p:
    for i, (n, te, cfg, vm, pt) in enumerate(p.imap_unordered(job, jobs), 1):
      res_jobs[(n, te)][cfg] = (vm, pt)
      if i % 50 == 0:
        print(f"[{i}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
  pred, picks = {}, defaultdict(dict)
  for n in S:
    pr = np.zeros(len(y))
    for te, va in F:
      cfg = min(res_jobs[(n, te)], key=lambda c: res_jobs[(n, te)][c][0])
      picks[n][te] = cfg
      pr[g == te] = res_jobs[(n, te)][cfg][1]
    pred[f"mlp_{n}"] = pr
  for n in ("own", "ctx"):
    cols = S[n]
    fixed, sel = np.zeros(len(y)), np.zeros(len(y))
    for te, va in F:
      fixed[g == te] = ridge(X[g != te][:, cols], y[g != te], X[g == te][:, cols], 1.0)
      tr = (g != te) & (g != va)
      lam = min(LAMS, key=lambda l: np.mean((ridge(X[tr][:, cols], y[tr], X[g == va][:, cols], l) - y[g == va]) ** 2))
      sel[g == te] = ridge(X[tr][:, cols], y[tr], X[g == te][:, cols], lam)
    pred[f"ridge_{n}"], pred[f"ridgesel_{n}"] = fixed, sel
  thin = np.where(np.array([r["alt_min"] for r in rows]) <= 1)[0]
  rel = lambda ix: float(1 - np.mean(se[ix] ** 2) / np.var(y[ix]))
  res = dict(note="PROC15H, diag s11 c_i (Rule-continuation, M = 40) 재사용. 정책 성능이 아니다.", n=len(y), reps=reps, folds=F,
             target=dict(skew=float(((y - y.mean()) ** 3).mean() / y.std() ** 3), q=np.percentile(y, [1, 25, 50, 75, 99]).tolist()),
             ceiling=dict(all=rel(np.arange(len(y))), thin=rel(thin)), grid=dict(hidden=HID, wd=WDS, seeds=SEEDS, lams=LAMS),
             picks={n: {str(k): list(v) for k, v in d.items()} for n, d in picks.items()},
             metrics={k: metrics(v, y, st) for k, v in pred.items()},
             thin={k: metrics(v, y, st, thin) for k, v in pred.items() if k in ("mlp_own", "mlp_ctx", "ridge_own", "ridge_ctx")})
  res["diffs"] = {"ridge_ctx-ridge_own": diff(pred["ridge_ctx"], pred["ridge_own"], y, g, st),
                  "mlp_own-ridge_own": diff(pred["mlp_own"], pred["ridge_own"], y, g, st),
                  "mlp_ctx-mlp_own": diff(pred["mlp_ctx"], pred["mlp_own"], y, g, st),
                  "mlp_ctx-ridge_ctx": diff(pred["mlp_ctx"], pred["ridge_ctx"], y, g, st),
                  "thin:mlp_ctx-mlp_own": diff(pred["mlp_ctx"], pred["mlp_own"], y, g, st, thin)}
  res["ablation"] = {gname: {k: res["metrics"][f"mlp_ctx-{gname}"][k] - res["metrics"]["mlp_ctx"][k] for k in ("r2", "mae", "spearman", "top3")}
                     for gname in GROUPS}
  OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
  with PRED.open("w", encoding="utf-8") as f:
    cols = ["pol", "rep", "t", "sid", "c", "se"] + keys + list(pred)
    f.write(",".join(cols) + "\n")
    for i, r in enumerate(rows):
      f.write(",".join([str(r[k]) for k in ["pol", "rep", "t", "sid", "c", "se"] + keys] + [f"{pred[k][i]:.4f}" for k in pred]) + "\n")
  print(f"상한 R² 전체 {res['ceiling']['all']:.3f}, thin {res['ceiling']['thin']:.3f} (thin n {len(thin)})")
  for k, m in res["metrics"].items():
    print(f"  {k:24s} R² {m['r2']:.3f} MAE {m['mae']:.1f} RMSE {m['rmse']:.1f} Spearman {m['spearman']:.3f} top3 {m['top3']:.3f} top20% {m['top20']:.3f}")
  print("thin (대체 공급자 ≤ 1)")
  for k, m in res["thin"].items():
    print(f"  {k:24s} R² {m['r2']:.3f} MAE {m['mae']:.1f} Spearman(상태 내) {m['spearman']:.3f} pooled {m['pooled_spearman']:.3f}")
  for k, d in res["diffs"].items():
    print(f"  {k:24s} " + " ".join(f"{m} {x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]" for m, x in d.items()))
  for gname, d in res["ablation"].items():
    print(f"  제거 {gname:15s} ΔR² {d['r2']:+.3f} ΔMAE {d['mae']:+.2f} ΔSpearman {d['spearman']:+.3f}")
  print("고른 설정", {n: res["picks"][n] for n in ("own", "ctx")})
  print(f"saved {OUT} ({time.time() - t0:.0f}s)")
