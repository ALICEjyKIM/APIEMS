"""진단 s9b: 공급자 가치의 non-additivity (Rule-continuation interaction value, optimal·oracle 가치 아님).
상태 S는 PROC15B에서 Rule로 걸은 반복의 기간 8·16. 같은 미래 키(CRN)로 V(S), V(S∖i), V(S∖{i,j})와 무작위 3·6명 제거를 Rule로 vf_H기간 돌려 잰다.
c_i = V(S) − V(S∖i), I_ij = V(S) − V(S∖i) − V(S∖j) + V(S∖{i,j}). 기준 칸은 모든 쌍, 주변 조건은 유형별 층화 표본. 단위 결과는 runs/diag_s9_units/.
"""
import copy
import json
import sys
import time
from dataclasses import replace
from itertools import combinations
from multiprocessing import Pool
import numpy as np
from env.platform import Env
from bench.rule_split import rule
from utils.params import Cfg, PROC15B
from result import diag

UNITS = diag.RUNS / "diag_s9_units"
PLAN = diag.RUNS / "diag_s9_plan.json"
OUT = diag.RUNS / "diag_s9_interaction.json"
CSV = diag.RUNS / "diag_s9_interaction_pairs.csv"
BASE = replace(Cfg(), **PROC15B)
CONDS = {"base": BASE, "k1": replace(BASE, items_per_order=1), "k3": replace(BASE, items_per_order=3),
         "het0": replace(BASE, het=0.0), "het1": replace(BASE, het=1.0), "L1": replace(BASE, onboard=1), "L4": replace(BASE, onboard=4)}
M = 20
TS = (8, 16)
REPS = {"base": range(4)}
REPS_SIDE = range(2)
PER_TYPE = 15
SETS = (3, 3, 3, 6, 6, 6)
CHUNK = 20
WORKERS = 12
KEY0 = 3_000_000


# 미래 키 (반복, 기간, k): 실험 5~7 키(tgt_rep0, mc_rep0)와 겹치지 않는 영역
def key(cfg, rep, t, k):
  return KEY0 + (rep * cfg.T + t) * 200 + k


# 반복 rep을 Rule로 기간 t까지 걸은 상태
def state(cfg, rep, t):
  env, pol = Env(cfg, rep), rule(cfg)
  env.reset()
  for _ in range(t):
    env.step(pol.act(env.o))
  return env


# 공급자 ID 집합 drop을 뺀 상태의 미래 M개 Rule 이윤 합 (같은 키라 drop만 다르다)
def value(env, drop, m=M):
  cfg, out = env.cfg, []
  for k in range(m):
    e = copy.deepcopy(env)
    e.rep = key(cfg, env.rep, env.t, k)
    for s in drop:
      e.drop("sup", e.o.sid.index(s))
    pol = rule(cfg)
    out.append(sum(e.step(pol.act(e.o))["profit"] for _ in range(cfg.vf_H)))
  return out


# 쌍 분류: A = 공통 품목, B = 공통 품목 없이 같은 주문자 묶음의 서로 다른 품목을 나눠 공급, C = 둘 다 아님
# 꼬리표: D = 둘 중 하나라도 활동 공급자 ≤ 2인 품목을 공급, E = 둘의 모든 품목이 활동 공급자 ≥ 4
def classify(o):
  have = o.cap > 0
  deg = have.sum(0)
  need = o.q > 0
  out = {}
  for a, b in combinations(range(len(o.sid)), 2):
    sh = have[a] & have[b]
    co = int(sum((need[r] & have[a] & ~have[b]).any() and (need[r] & have[b] & ~have[a]).any() for r in range(len(o.bid))))
    both = have[a] | have[b]
    typ = "A" if sh.any() else ("B" if co else "C")
    out[(o.sid[a], o.sid[b])] = dict(type=typ, D=bool(deg[both].min() <= 2), E=bool(deg[both].min() >= 4), shared=int(sh.sum()),
                                     min_deg_shared=int(deg[sh].min()) if sh.any() else None, coneed=co, min_deg=int(deg[both].min()))
  return out


# 상태 계획: 평가할 제거 집합 (전체, 단일, 쌍, 무작위 집합)
def plan_state(cond, cfg, rep, t):
  env = state(cfg, rep, t)
  o = env.o
  cls = classify(o)
  g = np.random.default_rng([cfg.seed, rep, t, 9])
  if cond == "base":
    pairs = list(cls)
    sets = [sorted(int(x) for x in g.choice(o.sid, n, replace=False)) for n in SETS if n <= len(o.sid)]
  else:
    pairs = []
    for typ in "ABC":
      ps = [p for p in cls if cls[p]["type"] == typ]
      pairs += [ps[i] for i in sorted(g.choice(len(ps), min(PER_TYPE, len(ps)), replace=False))]
    sets = []
  drops = [[]] + [[int(s)] for s in o.sid] + [[int(a), int(b)] for a, b in pairs] + sets
  return dict(cond=cond, rep=rep, t=t, sid=[int(s) for s in o.sid], bid=[int(b) for b in o.bid], cap=o.cap.tolist(), q=o.q.tolist(),
              rs=o.rs.tolist(), wait=o.wait.tolist(), pend=o.pend.tolist(),
              pairs=[dict(i=int(a), j=int(b), **cls[(a, b)]) for a, b in pairs], sets=sets, drops=drops)


# 작업 단위 (조건, 반복, 기간, 묶음 번호, 제거 집합들)
def unit(job):
  cond, rep, t, c, drops = job
  out = UNITS / f"{cond}_r{rep}_t{t}_c{c}.json"
  if out.exists():
    return out.name, "skip"
  t0 = time.time()
  env = state(CONDS[cond], rep, t)
  res = {",".join(map(str, d)): value(env, d) for d in drops}
  UNITS.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(dict(cond=cond, rep=rep, t=t, values=res, sec=time.time() - t0)), encoding="utf-8")
  return out.name, f"{time.time() - t0:.0f}s"


# 모든 상태 계획과 작업 목록
def plan():
  P = []
  for cond, cfg in CONDS.items():
    for rep in REPS.get(cond, REPS_SIDE):
      for t in TS:
        P.append(plan_state(cond, cfg, rep, t))
  PLAN.write_text(json.dumps(P), encoding="utf-8")
  return P


if __name__ == "__main__":
  if "smoke" in sys.argv:
    s = plan_state("base", BASE, 0, 8)
    print({k: len(v) for k, v in s.items() if isinstance(v, list)}, {t: sum(p["type"] == t for p in s["pairs"]) for t in "ABC"})
    env = state(BASE, 0, 8)
    print("V", np.mean(value(env, [], 2)))
  if "run" in sys.argv:
    P = plan()
    jobs = [(s["cond"], s["rep"], s["t"], c, s["drops"][i:i + CHUNK]) for s in P for c, i in enumerate(range(0, len(s["drops"]), CHUNK))]
    print(f"상태 {len(P)}, 제거 집합 {sum(len(s['drops']) for s in P)}, 작업 {len(jobs)}", flush=True)
    t0 = time.time()
    with Pool(WORKERS, maxtasksperchild=4) as p:
      for i, (name, msg) in enumerate(p.imap_unordered(unit, jobs), 1):
        print(f"[{i}/{len(jobs)}] {name} {msg} (경과 {time.time() - t0:.0f}s)", flush=True)
