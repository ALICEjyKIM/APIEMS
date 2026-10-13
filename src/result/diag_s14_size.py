"""진단 s14 Part I: SUM-GNN의 우위가 파라미터 수 때문인가 (PROC15H, 96상태, Rule-continuation c_i, s13 보조와 같은 분할).
Large Context MLP는 SUM-GNN의 (은닉, 층) 6개마다 같은 층 수에서 파라미터 수가 가장 가까운 폭을 고르고 wd 3개를 곱해 18설정이다 (입력은 Context MLP와 같음).
final: Part II용으로 반복 0~13 학습, 14~15 검증으로 SUM-GNN과 Context MLP의 설정을 18개 중 고르고 seed 3개 모형을 저장한다.
"""
import json
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
from vfa.hgnn import HGNN
from vfa.mlp import MLP
from result.diag_s13_gnn import states, init, job, collect, diff, ci, metrics, folds, CACHE, HID, LAY, WDS, SEEDS, D
from result.diag_s13_mc import PLAN as PLAN13, UNITS as U13
from result.diag_s12_mlp import BASE
from result import diag

OUT = diag.RUNS / "diag_s14_size.json"
CACHE96 = diag.RUNS / "diag_s14_cache96.pkl"
MODELS = diag.RUNS.parent / "models" / "s14"
FINAL_TR, FINAL_VA = tuple(range(14)), (14, 15)


# 같은 층 수에서 파라미터 수가 SUM-GNN (은닉, 층)에 가장 가까운 MLP 폭 (입력 28)
def width(hid, lay, gnn_params):
  mp = lambda w: 29 * w + (lay - 1) * (w * w + w) + w + 1
  return min(range(8, 600), key=lambda w: abs(mp(w) - gnn_params[(hid, lay)]))


# 96상태 데이터 (s13 48상태 캐시 + 반복 8~15)
def data96():
  if CACHE96.exists():
    return pickle.loads(CACHE96.read_bytes())
  S, _ = pickle.loads(CACHE.read_bytes())
  P13 = json.loads(PLAN13.read_text(encoding="utf-8"))
  vals = defaultdict(dict)
  for f in U13.glob("*_c*.json"):
    u = json.loads(f.read_text(encoding="utf-8"))
    vals[(u["pol"], u["rep"], u["t"])].update({k: np.array(v) for k, v in u["values"].items()})
  rows = []
  for s in P13:
    V = vals[(s["pol"], s["rep"], s["t"])]
    for x in s["sup"]:
      d = V[""] - V[str(x["sid"])]
      rows.append(dict(pol=s["pol"], rep=s["rep"], t=s["t"], sid=x["sid"], c=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), **x["own"], **x["ctx"]))
  S2 = S + states(P13, rows)
  CACHE96.write_bytes(pickle.dumps(S2))
  return S2


# final 작업: (모형, 설정) → 검증 MSE와 seed 3개 모형
def final_job(a):
  model, (hid, lay, wd) = a
  S = D["S"]
  itr = [i for i, s in enumerate(S) if s["key"][1] in FINAL_TR]
  iva = [i for i, s in enumerate(S) if s["key"][1] in FINAL_VA]
  yv = np.concatenate([S[i]["y"] for i in iva])
  ms, vm = [], []
  for sd in SEEDS:
    if model == "gnn":
      m = HGNN(hid, lay, wd, sd).fit([S[i]["g"] for i in itr], [S[i]["y"] for i in itr], [S[i]["g"] for i in iva], [S[i]["y"] for i in iva])
      vp = np.concatenate(m.predict([S[i]["g"] for i in iva]))
    else:
      X = lambda ii: np.concatenate([S[i]["X"] for i in ii])
      m = MLP(replace(BASE, nn_hidden=hid, nn_layers=lay, nn_seed=sd), wd).fit(X(itr), np.concatenate([S[i]["y"] for i in itr]), X(iva), yv)
      vp = m.predict(X(iva))
    vm.append(float(np.mean((vp - yv) ** 2)))
    ms.append(m)
  return model, (hid, lay, wd), float(np.mean(vm)), ms


if __name__ == "__main__":
  t0 = time.time()
  S = data96()
  for x in S:
    x["grp"] = x["key"][1] // 2
  gp = {}
  for k, v in json.loads((diag.RUNS / "diag_s13_gnn.json").read_text(encoding="utf-8"))["params"]["gnn"].items():
    h, l, _ = eval(k)
    gp[(h, l)] = v
  print(f"상태 {len(S)}, 공급자 {sum(len(s['y']) for s in S)} ({time.time() - t0:.0f}s)", flush=True)
  if "size" in sys.argv:
    grps = sorted({x["grp"] for x in S})
    F = folds(grps)
    jobs = []
    for te, va in F:
      rest = [r for r in grps if r not in (te, va)]
      jobs += [(m, te, va, rest, (h, l, w), False) for m in ("gnn", "mlp") for h in HID for l in LAY for w in WDS]
      jobs += [("mlpL", te, va, rest, (width(h, l, gp), l, w), False) for h in HID for l in LAY for w in WDS]
    out = defaultdict(dict)
    with Pool(12, initializer=init, initargs=(dict(S=S, Q=[]),)) as p:
      for i, r in enumerate(p.imap_unordered(job, jobs), 1):
        out[(r[0], r[1], r[2], r[3])][r[4]] = r[5:]
        if i % 50 == 0:
          print(f"  [{i}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
    y = np.concatenate([s["y"] for s in S])
    g = np.concatenate([[s["grp"]] * len(s["y"]) for s in S])
    st = [s["key"] for s in S for _ in s["y"]]
    thin = np.where(np.concatenate([s["alt"] for s in S]) <= 1)[0]
    P, picks, npar = {}, {}, {}
    for m in ("gnn", "mlp", "mlpL"):
      P[m], _, picks[m] = collect(out, S, [], F, m)
      npar[m] = {str(c): v[-1] for c, v in out[(m, False, F[0][0], len(grps) - 2)].items()}
    res = dict(note="96상태, 2반복 묶음 8개 중첩 분할, Rule-continuation c_i.", params=npar, picks={m: {str(k): list(v) for k, v in d.items()} for m, d in picks.items()},
               metrics={m: metrics(P[m], y, st) for m in P}, ci={m: ci(P[m], y, g, st) for m in P},
               thin={m: metrics(P[m], y, st, thin) for m in P},
               diffs={"gnn-mlpL": diff(P["gnn"], P["mlpL"], y, g, st), "gnn-mlp": diff(P["gnn"], P["mlp"], y, g, st), "mlpL-mlp": diff(P["mlpL"], P["mlp"], y, g, st),
                      "thin:gnn-mlpL": diff(P["gnn"], P["mlpL"], y, g, st, thin)})
    res["per_group_r2"] = {m: [float(1 - np.mean((P[m][g == k] - y[g == k]) ** 2) / np.var(y[g == k])) for k in grps] for m in P}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    np.savez(diag.RUNS / "diag_s14_pred.npz", y=y, g=g, **P)
    for m, x in res["metrics"].items():
      print(f"  {m:5s} R² {x['r2']:.3f} MAE {x['mae']:.1f} RMSE {x['rmse']:.1f} Spearman {x['spearman']:.3f} top3 {x['top3']:.3f} top20 {x['top20']:.3f}")
    for k, d in res["diffs"].items():
      print(f"  {k:14s} " + " ".join(f"{a} {x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]" for a, x in d.items()))
    print("  묶음별 R²", {m: np.round(v, 2).tolist() for m, v in res["per_group_r2"].items()})
    print("  고른 설정", res["picks"])
  if "final" in sys.argv:
    jobs = [(m, (h, l, w)) for m in ("gnn", "mlp") for h in HID for l in LAY for w in WDS]
    best = {}
    with Pool(12, initializer=init, initargs=(dict(S=S, Q=[]),)) as p:
      for m, cfg, vm, ms in p.imap_unordered(final_job, jobs):
        if m not in best or vm < best[m][1]:
          best[m] = (cfg, vm, ms)
    MODELS.mkdir(parents=True, exist_ok=True)
    for m, (cfg, vm, ms) in best.items():
      (MODELS / f"{m}.pkl").write_bytes(pickle.dumps(dict(cfg=cfg, val_mse=vm, models=ms, train=FINAL_TR, val=FINAL_VA)))
      print(f"  final {m}: 설정 {cfg}, 검증 MSE {vm:.0f}")
  print(f"done ({time.time() - t0:.0f}s)")
