"""실험 6 요약: retention value 정확도 (원자료 exp5_<칸>.json의 모델 c와 exp6_units의 Monte Carlo c만 읽는다).
모델별·종류별로 RMSE·MAE·편향·Pearson·Spearman(전체·상태 내)·상태 내 상위 3 공급자 적중률을 가드 전후로 내고,
MC c의 잡음 바닥(표준오차)과 반쪽 일치도를 함께 보고하며 GNN − MLP 등을 시험 반복 블록 부트스트랩으로 비교한다.
"""
import json
import numpy as np
from scipy.stats import spearmanr, pearsonr
from result import diag
from result.exp5_fair import CELLS, NAMES, PAIRS, cell_cfg, tag
from result.exp6_coef import UNITS, exp5

OUT = diag.RUNS / "exp6_summary.json"
TOP = 3


# MILP에 들어가는 가드: 종류 평균 쪽 shrink 후 [0, coef_hi × max(V, 0)]로 자른다 (vfa/coef.coefs와 같은 식)
def guard(cfg, c, v):
  c = np.asarray(c, float)
  return np.clip((1 - cfg.shrink) * c + cfg.shrink * c.mean(), 0, cfg.coef_hi * max(v, 0)) if len(c) else c


# 칸 하나의 짝 자료: 상태마다 (MC c, MC 표준오차, 반쪽 c 두 개, 모델별 seed 평균 c 가드 전·후)
def pairs(cell):
  cfg, src = cell_cfg(*cell), exp5(cell)
  st = {"sup": [], "buy": []}
  for rep in range(cfg.test_rep0, cfg.test_rep0 + cfg.test_reps):
    f = UNITS / f"{tag(cfg)}_r{rep}.json"
    if not f.exists():
      continue
    u = json.loads(f.read_text(encoding="utf-8"))
    for row in u["rows"]:
      i = row["idx"]
      for kind, key in (("sup", "s"), ("buy", "b")):
        if kind not in row["diffs"]:
          continue
        D = np.array(row["diffs"][kind])  # 참여자 × 미래
        e = dict(rep=rep, t=row["t"], mc=D.mean(1), se=D.std(1, ddof=1) / np.sqrt(D.shape[1]),
                 h1=D[:, ::2].mean(1), h2=D[:, 1::2].mean(1), raw={}, grd={})
        for n in NAMES:
          seeds = src["models"][n]["test_coef"]
          e["raw"][n] = np.mean([s[i][key] for s in seeds], 0)
          e["grd"][n] = np.mean([guard(cfg, s[i][key], s[i]["v"]) for s in seeds], 0)
        st[kind].append(e)
  return st


# 지표 묶음 (상태 목록 S, 예측 키 mode, 모델 n)
def metrics(S, mode, n):
  y = np.concatenate([s["mc"] for s in S])
  p = np.concatenate([s[mode][n] for s in S])
  ws = [spearmanr(s[mode][n], s["mc"])[0] for s in S if len(s["mc"]) >= 3 and np.ptp(s[mode][n]) > 0 and np.ptp(s["mc"]) > 0]
  top = [len(set(np.argsort(-s[mode][n])[:TOP]) & set(np.argsort(-s["mc"])[:TOP])) / TOP for s in S if len(s["mc"]) > TOP]
  return dict(rmse=float(np.sqrt(np.mean((p - y) ** 2))), mae=float(np.mean(np.abs(p - y))), bias=float(np.mean(p - y)),
              pearson=float(pearsonr(p, y)[0]) if np.ptp(p) > 0 else float("nan"),
              spearman=float(spearmanr(p, y)[0]) if np.ptp(p) > 0 else float("nan"),
              spearman_within=float(np.nanmean(ws)) if ws else float("nan"), top3=float(np.mean(top)) if top else float("nan"),
              pred_mean=float(p.mean()), mc_mean=float(y.mean()))


# MC c 자체의 잡음 바닥: RMSE 바닥, 신뢰도, 반쪽 c 사이 상위 3 일치도·상태 내 Spearman
def floor(S):
  y = np.concatenate([s["mc"] for s in S])
  se2 = np.concatenate([s["se"] ** 2 for s in S])
  top = [len(set(np.argsort(-s["h1"])[:TOP]) & set(np.argsort(-s["h2"])[:TOP])) / TOP for s in S if len(s["mc"]) > TOP]
  ws = [spearmanr(s["h1"], s["h2"])[0] for s in S if len(s["mc"]) >= 3 and np.ptp(s["h1"]) > 0 and np.ptp(s["h2"]) > 0]
  return dict(rmse_floor=float(np.sqrt(se2.mean())), reliability=float(1 - se2.mean() / y.var()),
              split_half_top3=float(np.mean(top)) if top else float("nan"),
              split_half_spearman_within=float(np.nanmean(ws)) if ws else float("nan"), n=len(y), states=len(S))


# 상태 하나의 지표 재료: 제곱오차 합, 절대오차 합, 개수, 상태 내 Spearman, 상위 3 적중률 (정의 안 되면 nan)
def parts(s, mode, n):
  p, y = s[mode][n], s["mc"]
  ws = spearmanr(p, y)[0] if len(y) >= 3 and np.ptp(p) > 0 and np.ptp(y) > 0 else np.nan
  top = len(set(np.argsort(-p)[:TOP]) & set(np.argsort(-y)[:TOP])) / TOP if len(y) > TOP else np.nan
  return np.array([np.sum((p - y) ** 2), np.sum(np.abs(p - y)), len(y), ws, top])


# 반복별 재료 합 (nan 지표는 합과 개수를 따로 센다): 열 = sse, sae, cnt, ws합, ws개수, top합, top개수
def rep_sums(S, reps, mode, n):
  A = np.zeros((len(reps), 7))
  pos = {r: i for i, r in enumerate(reps)}
  for s in S:
    q = parts(s, mode, n)
    A[pos[s["rep"]]] += [q[0], q[1], q[2], np.nan_to_num(q[3]), ~np.isnan(q[3]), np.nan_to_num(q[4]), ~np.isnan(q[4])]
  return A


# 재료 합에서 지표 (행렬이면 행마다)
def agg(A):
  return dict(rmse=np.sqrt(A[..., 0] / A[..., 2]), mae=A[..., 1] / A[..., 2],
              spearman_within=A[..., 3] / A[..., 4], top3=A[..., 5] / A[..., 6])


# 시험 반복 블록 부트스트랩: 모델 a − b의 지표 차이 (RMSE·MAE는 음수면 a가 낫다). 반복을 뽑은 횟수로 반복별 합에 가중한다
def boot(cfg, S, mode, a, b, keys=("rmse", "mae", "spearman_within", "top3")):
  reps, rng = sorted({s["rep"] for s in S}), np.random.default_rng(cfg.seed)
  W = np.stack([np.bincount(rng.choice(len(reps), len(reps)), minlength=len(reps)) for _ in range(cfg.boot_B)])
  Aa, Ab = rep_sums(S, reps, mode, a), rep_sums(S, reps, mode, b)
  fa, fb, da, db = agg(Aa.sum(0)), agg(Ab.sum(0)), agg(W @ Aa), agg(W @ Ab)
  return {k: dict(diff=float(fa[k] - fb[k]), lo=float(np.nanpercentile(da[k] - db[k], 2.5)),
                  hi=float(np.nanpercentile(da[k] - db[k], 97.5))) for k in keys}


# 요인 수준별 칸 평균과 유의 칸 수 (낮을수록 좋은 지표는 better = CI 상한 < 0)
def margins(cells, kind, mode, pair, key):
  low = key in ("rmse", "mae")
  out = {}
  for fac in ("n_alt", "conc", "k"):
    lv = sorted({r[fac] for r in cells.values()})
    out[fac] = {}
    for v in lv:
      bs = [r[kind][f"boot_{mode}"][pair][key] for r in cells.values() if r[fac] == v]
      out[fac][str(v)] = dict(mean=float(np.mean([b["diff"] for b in bs])), n=len(bs),
                              better=sum((b["hi"] < 0) if low else (b["lo"] > 0) for b in bs),
                              worse=sum((b["lo"] > 0) if low else (b["hi"] < 0) for b in bs))
  return out


if __name__ == "__main__":
  res = dict(cells={})
  for cell in CELLS:
    cfg = cell_cfg(*cell)
    st = pairs(cell)
    if not st["sup"]:
      continue
    r = dict(n_alt=cell[0], conc=cell[1], k=cell[2], reps=len({e["rep"] for e in st["sup"]}))
    for kind in ("sup", "buy"):
      S = st[kind]
      r[kind] = dict(floor=floor(S),
                     raw={n: metrics(S, "raw", n) for n in NAMES}, grd={n: metrics(S, "grd", n) for n in NAMES},
                     boot_raw={f"{a}-{b}": boot(cfg, S, "raw", a, b) for a, b in PAIRS},
                     boot_grd={f"{a}-{b}": boot(cfg, S, "grd", a, b) for a, b in PAIRS})
    res["cells"][tag(cfg)] = r
    s = r["sup"]
    print(f"{tag(cfg)} 공급자 | RMSE 선형 {s['raw']['선형']['rmse']:.0f} MLP {s['raw']['MLP']['rmse']:.0f} GNN {s['raw']['GNN']['rmse']:.0f} "
          f"(바닥 {s['floor']['rmse_floor']:.0f}) | 상태내 ρ {s['raw']['선형']['spearman_within']:.2f}/{s['raw']['MLP']['spearman_within']:.2f}/{s['raw']['GNN']['spearman_within']:.2f} "
          f"(반쪽 {s['floor']['split_half_spearman_within']:.2f}) | top3 {s['raw']['선형']['top3']:.2f}/{s['raw']['MLP']['top3']:.2f}/{s['raw']['GNN']['top3']:.2f} "
          f"(반쪽 {s['floor']['split_half_top3']:.2f})", flush=True)
  res["margins"] = {f"{kind}_{mode}_{pair[0]}-{pair[1]}_{key}": margins(res["cells"], kind, mode, f"{pair[0]}-{pair[1]}", key)
                    for kind in ("sup", "buy") for mode in ("raw", "grd") for pair in PAIRS
                    for key in ("rmse", "mae", "spearman_within", "top3")}
  OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
  for name, m in res["margins"].items():
    if "GNN-MLP" in name and "_raw_" in name:
      print(name + " | " + " | ".join(f"{fac} " + ", ".join(f"{v}: {x['mean']:+.3f} (+{x['better']}/−{x['worse']})" for v, x in lv.items())
                                      for fac, lv in m.items()))
  print(f"saved {OUT}")
