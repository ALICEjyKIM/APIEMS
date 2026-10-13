"""진단 s14 Part II Step A: 한 기간 결정 품질 (PROC15H). 같은 상태에서 공급자 c 출처만 다른 네 정책이 고른 행동을 같은 미래 키로 이어 간다.
정책: Myopic-sup (공급자 c 없음), Exact-MC Rule-c (그 상태에서 MC + CRN으로 잰 Rule-continuation c, M = 40), Context-MLP-c, SUM-GNN-c (s14 final 모형, seed 3개 평균).
모두 주문자 잉여는 몫 sh_buy로 고정하고 MILP·재참여 PWL은 같다. 상태는 학습에 쓰지 않은 평가 seed 반복 16~23의 Myopic-sup trajectory 기간 6·12·18.
"""
import copy
import json
import pickle
import sys
import time
from multiprocessing import Pool
import numpy as np
from env.platform import Env, settle, worth
from match.milp_solve import Policy
from bench.rule_split import rule
from result.diag_s11_context import CFG, value, feats
from result.diag_s11_summary import OWN
from result.diag_s12_mlp import CTX
from result.diag_s13_graph import graph
from result.diag_s14_size import MODELS
from result import diag

UNITS = diag.RUNS / "diag_s14_units"
OUT = diag.RUNS / "diag_s14_onestep.json"
REPS = range(16, 24)
TS = (6, 12, 18)
POLS = ("Myopic-sup", "Exact-MC", "Context-MLP", "SUM-GNN")
CONTS = ("Rule", "Myopic-sup")
M = 40
KEY0 = 7_000_000
WORKERS = 12


# Myopic-sup (주문자 몫 고정, 공급자 c 없음)
def msup(cfg):
  return Policy(cfg, split=(cfg.sh_buy, None))


# 평가 상태: Myopic-sup로 반복 rep을 기간 t까지 걸은 상태
def state(rep, t):
  env, p = Env(CFG, rep), msup(CFG)
  env.reset()
  for _ in range(t):
    env.step(p.act(env.o))
  return env


# 공급자 c를 쓰는 정책의 행동 (c = None이면 Myopic-sup)
def act(env, c):
  P = msup(CFG) if c is None else Policy(CFG, split=(CFG.sh_buy, None), coef=lambda o: (np.zeros(len(o.bid)), np.asarray(c, float)))
  return P.act(env.o)


# 이어 가기: 미래 키 k마다 행동을 적용한 뒤 후속 정책으로 vf_H기간
def cont(env, dec, cname):
  out = []
  for k in range(M):
    e = copy.deepcopy(env)
    e.rep = KEY0 + (env.rep * CFG.T + env.t) * 100 + k
    r0 = e.step(dec)
    pol = rule(CFG) if cname == "Rule" else msup(CFG)
    fut, ok, nb, act_s, zero = 0.0, r0["n_ok"], r0["n_buy"], [], []
    for _ in range(CFG.vf_H):
      o = e.o
      act_s.append(len(o.sid))
      zero.append(float(((o.cap > 0).sum(0) == 0).mean()))
      r = e.step(pol.act(o))
      fut += r["profit"]
      ok, nb = ok + r["n_ok"], nb + r["n_buy"]
    out.append(dict(imm=r0["profit"], fut=fut, fill=ok / max(nb, 1), ret_sup=r0["stay_sup"] / max(r0["n_sup"], 1), act_sup=float(np.mean(act_s)), zero=float(np.mean(zero))))
  return out


# 작업: ("mc", 반복, 기간, 묶음, 제거 집합들) 또는 ("cont", 반복, 기간, 정책, 행동, 후속 정책)
def unit(job):
  t0 = time.time()
  if job[0] == "mc":
    _, rep, t, c, drops = job
    out = UNITS / f"mc_r{rep}_t{t}_c{c}.json"
    if out.exists():
      return out.name, "skip"
    env = state(rep, t)
    res = dict(rep=rep, t=t, values={",".join(map(str, d)): value(env, d) for d in drops})
  else:
    _, rep, t, pol, dec, cname = job
    out = UNITS / f"cont_r{rep}_t{t}_{pol}_{cname}.json"
    if out.exists():
      return out.name, "skip"
    env = state(rep, t)
    res = dict(rep=rep, t=t, pol=pol, cont=cname, rows=cont(env, (np.array(dec[0]), np.array(dec[1]), np.array(dec[2])), cname))
  res["sec"] = time.time() - t0
  UNITS.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(res), encoding="utf-8")
  return out.name, f"{res['sec']:.0f}s"


# 학습된 모형의 상태별 공급자 ĉ (seed 3개 평균)
def predict(env, name):
  d = pickle.loads((MODELS / f"{name}.pkl").read_bytes())
  if name == "gnn":
    g = graph(env)
    return np.mean([m.predict([g])[0] for m in d["models"]], 0)
  X = np.array([[{**a, **b}[f] for f in OWN + CTX] for a, b in (feats(env, k) for k in range(len(env.o.sid)))], float)
  return np.mean([m.predict(X) for m in d["models"]], 0)


if __name__ == "__main__":
  t0 = time.time()
  keys = [(rep, t) for rep in REPS for t in TS]
  if "mc" in sys.argv:
    jobs = []
    for rep, t in keys:
      env = state(rep, t)
      drops = [[]] + [[int(s)] for s in env.o.sid]
      jobs += [("mc", rep, t, c, drops[i:i + 7]) for c, i in enumerate(range(0, len(drops), 7))]
    print(f"MC 작업 {len(jobs)}", flush=True)
    with Pool(WORKERS, maxtasksperchild=4) as p:
      for i, (name, msg) in enumerate(p.imap_unordered(unit, jobs), 1):
        print(f"[{i}/{len(jobs)}] {name} {msg} ({time.time() - t0:.0f}s)", flush=True)
  if "cont" in sys.argv:
    A, jobs = {}, []
    for rep, t in keys:
      env = state(rep, t)
      V = {}
      for f in UNITS.glob(f"mc_r{rep}_t{t}_c*.json"):
        V.update({k: np.array(v) for k, v in json.loads(f.read_text(encoding="utf-8"))["values"].items()})
      cmc = np.array([(V[""] - V[str(s)]).mean() for s in env.o.sid])
      C = {"Myopic-sup": None, "Exact-MC": cmc, "Context-MLP": predict(env, "mlp"), "SUM-GNN": predict(env, "gnn")}
      o, f = env.o, env.resp
      rec = dict(rep=rep, t=t, sid=[int(s) for s in o.sid], c={k: (None if v is None else v.tolist()) for k, v in C.items()}, act={})
      for pol, c in C.items():
        dec = act(env, c)
        r = settle(CFG, o, dec)
        alloc = (dec[0] * r["ok"][:, None, None]).sum((0, 2)) > 0
        ps = f.prob("sup", r["ss"], worth(o)[1], alloc, o.hs)
        rec["act"][pol] = dict(acc=int(r["ok"].sum()), alloc=alloc.astype(int).tolist(), ss=r["ss"].tolist(), sb=float(r["sb"].sum()), profit=float(r["profit"]),
                               contrib={k: (None if v is None else float(np.asarray(v) @ ps)) for k, v in C.items()})
        jobs += [("cont", rep, t, pol, [dec[0].tolist(), dec[1].tolist(), dec[2].tolist()], cn) for cn in CONTS]
      A[f"{rep}_{t}"] = rec
    (diag.RUNS / "diag_s14_actions.json").write_text(json.dumps(A), encoding="utf-8")
    print(f"이어 가기 작업 {len(jobs)}", flush=True)
    with Pool(WORKERS, maxtasksperchild=4) as p:
      for i, (name, msg) in enumerate(p.imap_unordered(unit, jobs), 1):
        print(f"[{i}/{len(jobs)}] {name} {msg} ({time.time() - t0:.0f}s)", flush=True)
  if "summary" in sys.argv:
    A = json.loads((diag.RUNS / "diag_s14_actions.json").read_text(encoding="utf-8"))
    R = {}
    for f in UNITS.glob("cont_*.json"):
      u = json.loads(f.read_text(encoding="utf-8"))
      R[(u["rep"], u["t"], u["pol"], u["cont"])] = u["rows"]
    res = dict(note="Step A. Exact-MC는 Rule-continuation c (optimal·oracle 아님). 가치 = 이번 기간 이윤 + 이후 10기간 이윤 (할인 없음).", n_states=len(keys), actions={}, cont={}, pairs={}, pred={})
    # 행동 요약과 c 예측 정확도
    for pol in POLS:
      res["actions"][pol] = {k: float(np.mean([v(A[f"{r}_{t}"]["act"][pol]) for r, t in keys])) for k, v in
                             (("accepted", lambda a: a["acc"]), ("alloc_sup", lambda a: sum(a["alloc"])), ("sup_surplus", lambda a: sum(a["ss"])),
                              ("paid_sup", lambda a: sum(x > 1e-6 for x in a["ss"])), ("profit", lambda a: a["profit"]))}
    for pol in ("Context-MLP", "SUM-GNN"):
      cm = np.concatenate([A[f"{r}_{t}"]["c"]["Exact-MC"] for r, t in keys])
      cp = np.concatenate([A[f"{r}_{t}"]["c"][pol] for r, t in keys])
      from scipy.stats import spearmanr
      res["pred"][pol] = dict(r2=float(1 - np.mean((cp - cm) ** 2) / np.var(cm)), spearman_within=float(np.mean([spearmanr(A[f"{r}_{t}"]["c"][pol], A[f"{r}_{t}"]["c"]["Exact-MC"])[0] for r, t in keys])))
    agree = {}
    for a in POLS:
      for b in POLS:
        if a < b:
          agree[f"{a}|{b}"] = float(np.mean([np.mean(np.array(A[f"{r}_{t}"]["act"][a]["alloc"]) == np.array(A[f"{r}_{t}"]["act"][b]["alloc"])) for r, t in keys]))
    res["alloc_agree"] = agree
    rng = np.random.default_rng(0)
    for cn in CONTS:
      res["cont"][cn] = {}
      for pol in POLS:
        rows = [R[(r, t, pol, cn)] for r, t in keys]
        res["cont"][cn][pol] = {k: float(np.mean([np.mean([x[k] for x in rr]) for rr in rows])) for k in ("imm", "fut", "fill", "ret_sup", "act_sup", "zero")}
        res["cont"][cn][pol]["total"] = res["cont"][cn][pol]["imm"] + res["cont"][cn][pol]["fut"]
      res["pairs"][cn] = {}
      for a, b in (("Exact-MC", "Myopic-sup"), ("SUM-GNN", "Exact-MC"), ("SUM-GNN", "Context-MLP"), ("SUM-GNN", "Myopic-sup"), ("Context-MLP", "Myopic-sup")):
        d = np.array([np.mean([x["imm"] + x["fut"] - y["imm"] - y["fut"] for x, y in zip(R[(r, t, a, cn)], R[(r, t, b, cn)])]) for r, t in keys])
        reps = np.array([r for r, t in keys])
        bs = [np.mean(np.concatenate([d[reps == q] for q in rng.choice(np.unique(reps), len(np.unique(reps)))])) for _ in range(2000)]
        same = np.array([np.array_equal(A[f"{r}_{t}"]["act"][a]["alloc"], A[f"{r}_{t}"]["act"][b]["alloc"]) and np.allclose(A[f"{r}_{t}"]["act"][a]["ss"], A[f"{r}_{t}"]["act"][b]["ss"]) for r, t in keys])
        res["pairs"][cn][f"{a}-{b}"] = dict(mean=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), pos=float((d > 0).mean()), same_action=float(same.mean()))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("c 예측 (이 상태들에서 Exact-MC 대비):", res["pred"])
    for pol, x in res["actions"].items():
      print(f"  행동 {pol:12s} 수락 {x['accepted']:.1f} 배정 공급자 {x['alloc_sup']:.1f} 잉여 받은 공급자 {x['paid_sup']:.1f} 공급자 잉여 {x['sup_surplus']:.0f} 이번 기간 이윤 {x['profit']:.0f}")
    print("  배정 일치율", {k: round(v, 3) for k, v in agree.items()})
    for cn in CONTS:
      print(f"[후속 정책 {cn}]")
      for pol, x in res["cont"][cn].items():
        print(f"  {pol:12s} 이번 {x['imm']:.0f} 이후 {x['fut']:.0f} 합 {x['total']:.0f} 충족 {x['fill']:.3f} 공급자 잔류 {x['ret_sup']:.3f} 활동 공급자 {x['act_sup']:.1f} 0명 품목 {x['zero']:.3f}")
      for k, x in res["pairs"][cn].items():
        print(f"  {k:24s} {x['mean']:+.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}] 양수 {x['pos']:.2f} 같은 행동 {x['same_action']:.2f}")
  print(f"done ({time.time() - t0:.0f}s)")
