"""진단 s13: Buyer–Item–Supplier SUM-GNN이 Context MLP보다 공급자 c_i를 더 잘 설명하는가 (PROC15H, Rule-continuation c_i, 정책 성능 아님).
main: diag s11의 48상태·c_i 929개, s12와 같은 반복 단위 중첩 분할. SUM-GNN과 Context MLP 모두 은닉 {32, 64, 128} × 층 {2, 3} × wd {0, 1e-3, 1e-2} = 18설정 × seed 3.
같은 fold 모형으로 섭동 상태(대체 공급자 제거, 수요 1.5·0.5배, 연결 재배치)의 ĉ 변화를 예측하고, 학습 곡선과 품목 degree 명시 버전(공정성 점검)을 함께 낸다. expand: 반복 8~15 상태를 더한 보조 진단.
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
from result.diag_s11_context import state, feats, perturb, PLAN as PLAN11, UNITS as U11
from result.diag_s11_summary import load, OWN
from result.diag_s12_mlp import CTX, BASE, metrics, folds, draw
from result.diag_s13_graph import graph
from result.diag_s13_mc import swap, PLAN as PLAN13, UNITS as U13
from result import diag

OUT = diag.RUNS / "diag_s13_gnn.json"
CACHE = diag.RUNS / "diag_s13_cache.pkl"
HID, LAY, WDS, SEEDS = (32, 64, 128), (2, 3), (0.0, 1e-3, 1e-2), (0, 1, 2)
CASES = ("alt_minus", "dem_hi", "dem_lo", "swap")
NTR = (2, 4, 6)
LC = (64, 2, 0.01)
B = 2000
D = {}


# 작업자 공유 데이터
def init(data):
  D.update(data)


# 상태 목록의 그래프·MLP 입력·목표 (s11 계획 순서, 공급자 순서 = 계획의 sup 순서)
def states(P, rows):
  by = defaultdict(dict)
  for r in rows:
    by[(r["pol"], r["rep"], r["t"])][r["sid"]] = r
  out = []
  for s in P:
    env = state(s["pol"], s["rep"], s["t"])
    g, gd = graph(env), graph(env, deg=True)
    sids = [x["sid"] for x in s["sup"]]
    assert g["sid"] == sids
    R = [by[(s["pol"], s["rep"], s["t"])][i] for i in sids]
    out.append(dict(key=(s["pol"], s["rep"], s["t"]), g=g, gd=gd, X=np.array([[r[k] for k in OWN + CTX] for r in R], float),
                    y=np.array([r["c"] for r in R]), se=np.array([r["se"] for r in R]), alt=np.array([r["alt_min"] for r in R]), sids=sids))
  return out


# 섭동 상태: s11 섭동 대상마다 case별 바뀐 상태의 그래프와 대상 공급자의 MLP 입력
def perts(P):
  out = []
  for s in P:
    for sid in s["pert"]:
      for case in CASES:
        env = state(s["pol"], s["rep"], s["t"])
        info = swap(env, sid) if case == "swap" else perturb(env, sid, case)
        if info is None:
          continue
        k = env.o.sid.index(sid)
        own, ctx = feats(env, k)
        out.append(dict(key=(s["pol"], s["rep"], s["t"]), sid=sid, case=case, g=graph(env), gd=graph(env, deg=True), k=k,
                        x=np.array([{**own, **ctx}[f] for f in OWN + CTX], float)))
  return out


# 작업: (모형, 시험 반복, 검증 반복, 학습 반복들, 설정, deg) → 검증 MSE, 시험 예측, 섭동 예측 (seed 평균)
def job(a):
  model, te, va, tr, (hid, lay, wd), deg = a
  S, Q = D["S"], D["Q"]
  idx = lambda reps: [i for i, s in enumerate(S) if s["grp"] in reps]
  itr, iva, ite = idx(tr), idx([va]), idx([te])
  qte = [i for i, q in enumerate(Q) if q["grp"] == te]
  gk = "gd" if deg else "g"
  vm, pt, pq = [], [], []
  for sd in SEEDS:
    if model == "gnn":
      m = HGNN(hid, lay, wd, sd).fit([S[i][gk] for i in itr], [S[i]["y"] for i in itr], [S[i][gk] for i in iva], [S[i]["y"] for i in iva])
      vp = np.concatenate(m.predict([S[i][gk] for i in iva]))
      pt.append(np.concatenate(m.predict([S[i][gk] for i in ite])))
      pq.append(np.array([p[Q[i]["k"]] for p, i in zip(m.predict([Q[i][gk] for i in qte]), qte)]) if qte else np.zeros(0))
      npar = m.n_params()
    else:
      X = lambda ii: np.concatenate([S[i]["X"] for i in ii])
      Y = lambda ii: np.concatenate([S[i]["y"] for i in ii])
      m = MLP(replace(BASE, nn_hidden=hid, nn_layers=lay, nn_seed=sd), wd).fit(X(itr), Y(itr), X(iva), Y(iva))
      vp = m.predict(X(iva))
      pt.append(m.predict(X(ite)))
      pq.append(m.predict(np.array([Q[i]["x"] for i in qte])) if qte else np.zeros(0))
      npar = sum(p.numel() for p in m.net.parameters())
    vm.append(float(np.mean((vp - np.concatenate([S[i]["y"] for i in iva])) ** 2)))
  return model, deg, te, len(tr), (hid, lay, wd), float(np.mean(vm)), np.mean(pt, 0), np.mean(pq, 0), qte, npar


# 차이 a − b의 반복 블록 부트스트랩 (R², MAE, 상태 내 Spearman, 상위 3)
def diff(pa, pb, y, g, st, idx=None):
  idx = np.arange(len(y)) if idx is None else idx
  rng = np.random.default_rng(0)
  ks = ("r2", "mae", "spearman", "top3")
  full = {k: metrics(pa, y, st, idx)[k] - metrics(pb, y, st, idx)[k] for k in ks}
  dr = defaultdict(list)
  for _ in range(B):
    ix, sk = draw(rng, idx, g, st)
    ma, mb = metrics(pa[ix], y[ix], sk), metrics(pb[ix], y[ix], sk)
    for k in ks:
      dr[k].append(ma[k] - mb[k])
  return {k: dict(diff=v, lo=float(np.nanpercentile(dr[k], 2.5)), hi=float(np.nanpercentile(dr[k], 97.5))) for k, v in full.items()}


# 지표의 반복 블록 부트스트랩 CI
def ci(p, y, g, st):
  rng = np.random.default_rng(0)
  dr = defaultdict(list)
  for _ in range(B):
    ix, sk = draw(rng, np.arange(len(y)), g, st)
    for k, v in metrics(p[ix], y[ix], sk).items():
      if k != "n":
        dr[k].append(v)
  return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in dr.items()}


# 실험 한 벌: 격자 선택(main) 또는 고정 설정(학습 곡선)으로 fold별 시험·섭동 예측
def run(S, Q, reps, grid=True, lc=False, workers=12):
  F = folds(reps)
  jobs = []
  for te, va in F:
    rest = [r for r in reps if r not in (te, va)]
    if grid:
      jobs += [(m, te, va, rest, (h, l, w), False) for m in ("gnn", "mlp") for h in HID for l in LAY for w in WDS]
    if lc:
      jobs += [(m, te, va, rest[:n], LC, False) for m in ("gnn", "mlp") for n in NTR]
  out = defaultdict(dict)
  t0 = time.time()
  with Pool(workers, initializer=init, initargs=(dict(S=S, Q=Q),)) as p:
    for i, r in enumerate(p.imap_unordered(job, jobs), 1):
      out[(r[0], r[1], r[2], r[3])][r[4]] = r[5:]
      if i % 50 == 0:
        print(f"  [{i}/{len(jobs)}] {time.time() - t0:.0f}s", flush=True)
  return out, F


# fold별 선택(검증 MSE 최소) → 모은 시험 예측, 섭동 예측, 고른 설정
def collect(out, S, Q, F, model, deg=False, ntr=None):
  g = np.concatenate([[s["grp"]] * len(s["y"]) for s in S])
  pred, pq, picks = np.zeros(len(g)), np.full(len(Q), np.nan), {}
  for te, va in F:
    n = ntr if ntr else len([r for r in set(g.tolist()) if r not in (te, va)])
    res = out[(model, deg, te, n)]
    cfg = min(res, key=lambda c: res[c][0])
    picks[te] = cfg
    _, pt, pp, qte, npar = res[cfg]
    pred[g == te] = pt
    pq[qte] = pp
  return pred, pq, picks


if __name__ == "__main__":
  t0 = time.time()
  P11 = json.loads(PLAN11.read_text(encoding="utf-8"))
  _, rows, futs, _ = load()
  if CACHE.exists():
    S, Q = pickle.loads(CACHE.read_bytes())
  else:
    S, Q = states(P11, rows), perts(P11)
    CACHE.write_bytes(pickle.dumps((S, Q)))
  print(f"상태 {len(S)}, 공급자 {sum(len(s['y']) for s in S)}, 섭동 상태 {len(Q)} ({time.time() - t0:.0f}s)", flush=True)
  if "main" in sys.argv:
    for x in S + Q:
      x["grp"] = x["key"][1]
    reps = sorted({s["key"][1] for s in S})
    out, F = run(S, Q, reps, grid=True, lc=True)
    y = np.concatenate([s["y"] for s in S])
    g = np.concatenate([[s["key"][1]] * len(s["y"]) for s in S])
    st = [s["key"] for s in S for _ in s["y"]]
    alt = np.concatenate([s["alt"] for s in S])
    thin = np.where(alt <= 1)[0]
    P = {}
    for m in ("gnn", "mlp"):
      P[m], P[f"{m}_pert"], P[f"{m}_picks"] = collect(out, S, Q, F, m)
    # 공정성 점검: GNN이 fold별로 고른 설정 그대로 품목 degree·용량·요구량을 명시한 그래프로 다시 학습
    djobs = [("gnn", te, va, [r for r in reps if r not in (te, va)], P["gnn_picks"][te], True) for te, va in F]
    with Pool(12, initializer=init, initargs=(dict(S=S, Q=Q),)) as p:
      for r in p.imap_unordered(job, djobs):
        out[(r[0], r[1], r[2], r[3])][r[4]] = r[5:]
    P["gnn_deg"], _, _ = collect(out, S, Q, F, "gnn", deg=True)
    lc = {f"{m}_{n}": metrics(collect(out, S, Q, F, m, ntr=n)[0], y, st)["r2"] for m in ("gnn", "mlp") for n in NTR}
    s12 = {}
    import csv
    pr = list(csv.DictReader((diag.RUNS / "diag_s12_pred.csv").open(encoding="utf-8")))
    order = {(r["pol"], int(r["rep"]), int(r["t"]), int(r["sid"])): i for i, r in enumerate(pr)}
    keys = [(s["key"][0], s["key"][1], s["key"][2], sid) for s in S for sid in s["sids"]]
    for k in ("mlp_own", "mlp_ctx", "ridge_own", "ridge_ctx"):
      s12[k] = np.array([float(pr[order[x]][k]) for x in keys])
    assert np.allclose([float(pr[order[x]]["c"]) for x in keys], y)
    npar = {m: {str(c): v[-1] for c, v in out[(m, False, F[0][0], len(reps) - 2)].items()} for m in ("gnn", "mlp")}
    res = dict(note="PROC15H, diag s11 c_i (Rule-continuation, M = 40). 정책 성능이 아니다.", n=len(y), folds=F,
               grid=dict(hidden=HID, layers=LAY, wd=WDS, seeds=SEEDS), params=npar,
               picks={m: {str(k): list(v) for k, v in P[f"{m}_picks"].items()} for m in ("gnn", "mlp")},
               metrics={k: metrics(v, y, st) for k, v in {"gnn": P["gnn"], "mlp18": P["mlp"], "gnn_deg": P["gnn_deg"], **s12}.items()},
               ci={k: ci(v, y, g, st) for k, v in {"gnn": P["gnn"], "mlp18": P["mlp"]}.items()},
               thin={k: metrics(v, y, st, thin) for k, v in {"gnn": P["gnn"], "mlp18": P["mlp"], "mlp_ctx": s12["mlp_ctx"], "mlp_own": s12["mlp_own"]}.items()},
               diffs={"gnn-mlp18": diff(P["gnn"], P["mlp"], y, g, st), "gnn-mlp_ctx(s12)": diff(P["gnn"], s12["mlp_ctx"], y, g, st),
                      "mlp18-mlp_ctx(s12)": diff(P["mlp"], s12["mlp_ctx"], y, g, st), "gnn-mlp_own(s12)": diff(P["gnn"], s12["mlp_own"], y, g, st),
                      "gnn_deg-gnn": diff(P["gnn_deg"], P["gnn"], y, g, st), "thin:gnn-mlp18": diff(P["gnn"], P["mlp"], y, g, st, thin)},
               curve=lc)
    # 섭동: 예측 Δĉ = ĉ(섭동) − ĉ(원래), 참 Δc (s11·s13 MC)
    base_pred = {}
    i = 0
    for s in S:
      for j, sid in enumerate(s["sids"]):
        base_pred[(s["key"], sid)] = (P["gnn"][i + j], P["mlp"][i + j])
      i += len(s["y"])
    pert = defaultdict(list)
    for q, pg, pm in zip(Q, P["gnn_pert"], P["mlp_pert"]):
      pol, rep, t = q["key"]
      f = (U13 / f"{pol}_r{rep}_t{t}_swap_s{q['sid']}.json") if q["case"] == "swap" else (U11 / f"{pol}_r{rep}_t{t}_{q['case']}_s{q['sid']}.json")
      if not f.exists():
        continue
      u = json.loads(f.read_text(encoding="utf-8"))
      if not u["values"]:
        continue
      c1 = np.array(u["values"][""]) - np.array(u["values"][str(q["sid"])])
      d = c1 - futs[(pol, rep, t, q["sid"])]
      bg, bm = base_pred[(q["key"], q["sid"])]
      pert[q["case"]].append(dict(true=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), gnn=float(pg - bg), mlp=float(pm - bm)))
    res["pert"] = {}
    for case, L in pert.items():
      t_, gn, ml = (np.array([x[k] for x in L]) for k in ("true", "gnn", "mlp"))
      cor = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else float("nan")
      res["pert"][case] = dict(n=len(L), true_mean=float(t_.mean()), gnn_mean=float(gn.mean()), mlp_mean=float(ml.mean()),
                               gnn_sign=float(np.mean(np.sign(gn) == np.sign(t_))), mlp_sign=float(np.mean(np.sign(ml) == np.sign(t_))),
                               gnn_corr=cor(gn, t_), mlp_corr=cor(ml, t_), mlp_abs=float(np.abs(ml).mean()), gnn_abs=float(np.abs(gn).mean()),
                               true_abs=float(np.abs(t_).mean()), true_se=float(np.mean([x["se"] for x in L])), rows=L)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    np.savez(diag.RUNS / "diag_s13_pred.npz", y=y, g=g, alt=alt, keys=np.array([f"{k[0]}|{k[1]}|{k[2]}|{sid}" for s in S for k, sid in [(s["key"], x) for x in s["sids"]]]),
             **{k: v for k, v in {"gnn": P["gnn"], "mlp18": P["mlp"], "gnn_deg": P["gnn_deg"], **s12}.items()})
    for k, m in res["metrics"].items():
      print(f"  {k:12s} R² {m['r2']:.3f} MAE {m['mae']:.1f} RMSE {m['rmse']:.1f} Spearman {m['spearman']:.3f} top3 {m['top3']:.3f} top20 {m['top20']:.3f}")
    for k, d in res["diffs"].items():
      print(f"  {k:20s} " + " ".join(f"{a} {x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]" for a, x in d.items()))
    for k, m in res["thin"].items():
      print(f"  thin {k:8s} R² {m['r2']:.3f} MAE {m['mae']:.1f} Spearman {m['spearman']:.3f}")
    print("  학습 곡선", {k: round(v, 3) for k, v in lc.items()})
    for c, x in res["pert"].items():
      print(f"  섭동 {c}: n {x['n']} 참 Δc {x['true_mean']:+.1f} (|Δc| {x['true_abs']:.1f}, se {x['true_se']:.1f}) GNN {x['gnn_mean']:+.1f} (부호 {x['gnn_sign']:.2f}, 상관 {x['gnn_corr']:+.2f}) MLP {x['mlp_mean']:+.1f} (부호 {x['mlp_sign']:.2f}, 상관 {x['mlp_corr']:+.2f})")
    print("  고른 설정", res["picks"])
    print(f"saved {OUT} ({time.time() - t0:.0f}s)")
  if "expand" in sys.argv:
    # 보조: 반복 8~15 상태를 더해 16반복을 2개씩 8묶음으로 나누고 같은 격자·같은 규칙으로 다시 비교 (학습 묶음 3·6 = 반복 6·12의 학습 곡선 포함)
    P13 = json.loads(PLAN13.read_text(encoding="utf-8"))
    vals = defaultdict(dict)
    for f in U13.glob("*_c*.json"):
      u = json.loads(f.read_text(encoding="utf-8"))
      vals[(u["pol"], u["rep"], u["t"])].update({k: np.array(v) for k, v in u["values"].items()})
    rows13 = []
    for s in P13:
      V = vals[(s["pol"], s["rep"], s["t"])]
      for x in s["sup"]:
        d = V[""] - V[str(x["sid"])]
        rows13.append(dict(pol=s["pol"], rep=s["rep"], t=s["t"], sid=x["sid"], c=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), **x["own"], **x["ctx"]))
    S2 = S + states(P13, rows13)
    for x in S2:
      x["grp"] = x["key"][1] // 2
    grps = sorted({x["grp"] for x in S2})
    NTR = (3, 6)
    out, F = run(S2, [], grps, grid=True, lc=True)
    y = np.concatenate([s["y"] for s in S2])
    g = np.concatenate([[s["grp"]] * len(s["y"]) for s in S2])
    st = [s["key"] for s in S2 for _ in s["y"]]
    alt = np.concatenate([s["alt"] for s in S2])
    thin = np.where(alt <= 1)[0]
    P = {m: collect(out, S2, [], F, m)[0] for m in ("gnn", "mlp")}
    se = np.concatenate([s["se"] for s in S2])
    res = dict(note="보조 진단: 반복 0~15 (상태 96개), 2반복 묶음 8개로 같은 중첩 분할.", n=len(y), states=len(S2),
               ceiling=float(1 - np.mean(se ** 2) / np.var(y)), metrics={k: metrics(v, y, st) for k, v in P.items()},
               thin={k: metrics(v, y, st, thin) for k, v in P.items()}, diffs={"gnn-mlp": diff(P["gnn"], P["mlp"], y, g, st), "thin:gnn-mlp": diff(P["gnn"], P["mlp"], y, g, st, thin)},
               curve={f"{m}_{n * 2}reps": metrics(collect(out, S2, [], F, m, ntr=n)[0], y, st)["r2"] for m in ("gnn", "mlp") for n in NTR})
    (diag.RUNS / "diag_s13_expand.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"확장: 상태 {len(S2)}, 공급자 {len(y)}, 상한 {res['ceiling']:.3f}")
    for k, m in res["metrics"].items():
      print(f"  {k} R² {m['r2']:.3f} MAE {m['mae']:.1f} Spearman {m['spearman']:.3f} top3 {m['top3']:.3f}")
    for k, d in res["diffs"].items():
      print(f"  {k} " + " ".join(f"{a} {x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]" for a, x in d.items()))
    print("  학습 곡선", {k: round(v, 3) for k, v in res["curve"].items()})
