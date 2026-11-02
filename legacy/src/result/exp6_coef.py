"""실험 6: retention value 정확도 (실험 5의 저장 모델·시험 rollout 재사용, 모델 수정·재학습 없음).
시험 상태에서 참여자 i를 뺀 상태를 저장된 시험 미래와 같은 키(공통 난수)로 다시 돌려 Monte Carlo 기준 c_i = 평균(V_full − V_drop)을 잰다.
공급자는 시험 반복 18개 × 기간 {5, 15}의 전원, 주문자는 기간 5의 전원. 작업 단위는 (칸, 반복)이며 단위마다 파일을 남겨 재시작할 수 있다.
"""
import json
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
import numpy as np
from env.platform import Env
from bench.rule_split import rule
from match.mc_value import mc_value
from vfa.train import tkey
from result.exp5_fair import CELLS, cell_cfg, tag
from result import diag

UNITS = diag.RUNS / "exp6_units"
T_SUP = (5, 15)  # 공급자 c를 재는 기간
T_BUY = (5,)  # 주문자 c를 재는 기간
WORKERS = 12


# 칸의 실험 5 원자료
def exp5(cell):
  return json.loads((diag.RUNS / f"exp5_{tag(cell_cfg(*cell))}.json").read_text(encoding="utf-8"))


# 작업 단위 (칸, 반복): 반복을 규칙 기반으로 걸어 지정 기간에서 참여자별 V_drop 미래 val_M개를 잰다
# 같은 상태의 V_full은 다시 재서 저장된 시험 미래와 같은지 확인한다 (공통 난수 무결성)
def unit(job):
  cell, rep, limit = job
  cfg = cell_cfg(*cell)
  out = UNITS / f"{tag(cfg)}_r{rep}.json"
  if out.exists() and limit is None:
    return out.name, "skip"
  t0 = time.time()
  src = exp5(cell)
  n_per = cfg.T - cfg.vf_H + 1
  tc = replace(cfg, seed=cfg.tune_seed)
  env, pol = Env(tc, rep), rule(tc)
  env.reset()
  rows = []
  for t in range(max(T_SUP + T_BUY) + 1):
    if t in T_SUP or t in T_BUY:
      idx = (rep - cfg.test_rep0) * n_per + t
      assert src["test_g"][idx] == rep and src["test_t"][idx] == t
      assert [int(i) for i in env.o.bid] == src["test_ids"][idx]["bid"] and [int(i) for i in env.o.sid] == src["test_ids"][idx]["sid"]
      full = mc_value(env, R=cfg.val_M, key=tkey)
      same = bool(np.allclose(full, src["futures"]["test"][idx]))
      assert same
      d = {}
      for kind, ts in (("sup", T_SUP), ("buy", T_BUY)):
        if t not in ts:
          continue
        n = len(env.o.sid if kind == "sup" else env.o.bid)
        ks = range(n if limit is None else min(n, limit))
        d[kind] = [(full - mc_value(env, (kind, k), cfg.val_M, tkey)).tolist() for k in ks]
      rows.append(dict(idx=idx, rep=rep, t=t, full=full.tolist(), crn_ok=same, diffs=d))
    env.step(pol.act(env.o))
  if limit is None:
    UNITS.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(tag=tag(cfg), rep=rep, rows=rows, sec=time.time() - t0)), encoding="utf-8")
  return out.name, f"{time.time() - t0:.0f}s"


if __name__ == "__main__":
  if "smoke" in sys.argv:
    name, msg = unit((CELLS[0], cell_cfg(*CELLS[0]).test_rep0, 2))
    print("smoke", name, msg)
  if "run" in sys.argv:
    jobs = [(cell, r, None) for r in range(cell_cfg().test_rep0, cell_cfg().test_rep0 + cell_cfg().test_reps) for cell in CELLS]
    t0 = time.time()
    with Pool(WORKERS, maxtasksperchild=4) as p:
      for i, (name, msg) in enumerate(p.imap_unordered(unit, jobs), 1):
        print(f"[{i}/{len(jobs)}] {name} {msg} (경과 {time.time() - t0:.0f}s)", flush=True)
