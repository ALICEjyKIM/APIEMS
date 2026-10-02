"""실험 7: 공급자별 c의 의사결정 가치 (oracle benchmark, 모델 학습 없음, 목적함수 profit + Σ c_i p_i 그대로).
Oracle-c는 진정한 최적 oracle이 아니라, 매 기간 자기 경로의 상태에서 Rule을 vf_H기간 이어 간다고 보고 잰 MC marginal value(M = val_M, 키 tkey)를 쓰는 benchmark다.
주문자 잉여는 몫 sh_buy로 고정하고 공급자 c만 바꾼다. 작업 단위 (칸, 정책, 반복)마다 기간별 원자료를 result/runs/exp7_units/에 남긴다.
"""
import json
import sys
import time
from multiprocessing import Pool
import numpy as np
from env.platform import Env
from match.milp_solve import Policy
from match.mc_value import mc_value
from bench.rule_split import rule
from utils.common import rng
from vfa.train import tkey
from result.exp5_fair import cell_cfg, tag
from result import diag

UNITS = diag.RUNS / "exp7_units"
WORKERS = 12
BASE = (4, 0.5, 2)
CELLS = [BASE, (3, 1.0, 3), (3, 1.0, 1), (3, 0.0, 3), (4, 0.0, 1)]
PILOT_REPS = range(10)
C_POLS = ("Uniform", "Oracle", "Shuffled")  # 매 기간 MC c를 재는 정책
POLS = C_POLS + ("Myopic-sup", "Rule", "Myopic")


# 공급자 MC c: c_k = 평균(V(S) − V(S∖k)), 같은 미래 키 (R = val_M, 키 tkey)
def mc_c(env, R):
  full = mc_value(env, R=R, key=tkey)
  return np.array([(full - mc_value(env, ("sup", k), R, tkey)).mean() for k in range(len(env.o.sid))])


# MILP에 넣을 공급자 c: Uniform = 상태 내 평균, Oracle = 그대로, Shuffled = 상태 내 무작위 순열 (perm 스트림, 키 (반복, 기간))
def use_c(cfg, pol, c, rep, t):
  if pol == "Uniform":
    return np.full(len(c), c.mean())
  if pol == "Shuffled":
    return c[rng(cfg, rep, "perm", t).permutation(len(c))]
  return c


# 정책 이름 → MILP 정책과 매 기간 c 함수 (c를 재지 않는 정책은 None)
def make(cfg, pol):
  if pol == "Rule":
    return rule(cfg)
  if pol == "Myopic":
    return Policy(cfg)
  return Policy(cfg, split=(cfg.sh_buy, None))


# 작업 단위 (칸, 정책, 반복): 평가 seed 반복을 T기간 돌리며 기간별 원자료를 모은다 (periods·R은 점검용 축소)
def unit(job, periods=None, R=None):
  cell, pol, rep = job
  cfg = cell_cfg(*cell)
  out = UNITS / f"{tag(cfg)}_{pol}_r{rep}.json"
  if out.exists() and periods is None:
    return out.name, "skip"
  t0 = time.time()
  env, P = Env(cfg, rep), make(cfg, pol)
  env.reset()
  P.reset(rep)
  rows = []
  for t in range(periods or cfg.T):
    o, t1 = env.o, time.time()
    c = mc_c(env, R or cfg.val_M) if pol in C_POLS else None
    if c is not None:
      cu = use_c(cfg, pol, c, rep, t)
      P.coef = lambda o, cu=cu: (np.zeros(len(o.bid)), cu)
    r = env.step(P.act(o))
    rows.append(dict(t=t, profit=r["profit"], rev=r["rev"], ship=r["ship"], opp=r["opp"], n_ok=int(r["n_ok"]),
                     n_buy=r["n_buy"], stay_buy=r["stay_buy"], n_sup=r["n_sup"], stay_sup=r["stay_sup"],
                     new_sup=len(set(env.o.sid) - set(o.sid)), sb=float(r["sb"].sum()),
                     sid=[int(i) for i in o.sid], ss=r["ss"].tolist(), ps=r["ps"].tolist(),
                     stay=[int(i in set(env.o.sid)) for i in o.sid],
                     c=None if c is None else c.tolist(), cu=None if c is None else cu.tolist(), sec=time.time() - t1))
  if periods is None:
    UNITS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(tag=tag(cfg), pol=pol, rep=rep, seed=cfg.seed, rows=rows, sec=time.time() - t0)), encoding="utf-8")
  return out.name, f"{time.time() - t0:.0f}s"


# 작업 목록을 병렬로 돌린다 (c를 재는 비싼 정책 먼저)
def run(jobs):
  jobs = sorted(jobs, key=lambda j: j[1] not in C_POLS)
  t0 = time.time()
  with Pool(WORKERS, maxtasksperchild=1) as p:
    for i, (name, msg) in enumerate(p.imap_unordered(unit, jobs), 1):
      print(f"[{i}/{len(jobs)}] {name} {msg} (경과 {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
  if "smoke" in sys.argv:
    for pol in POLS:
      print("smoke", unit((BASE, pol, 0), periods=2, R=2))
  if "pilot" in sys.argv:
    run([(BASE, pol, rep) for pol in POLS for rep in PILOT_REPS])
