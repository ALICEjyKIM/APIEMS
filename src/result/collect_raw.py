"""원재료 수집 (질문 3, ref/decisions.md "질문 3 비교 범위"): 행동 정책 경로의 상태마다 피처 대신 입찰 전 관측(PreObs)과 후보 m별 예상 청산(value.feat.plan)을 저장한다.
adv: 후보 12개로 분기해 끝까지 돌린 남은 누적 이윤 (collect.py와 같은 반복·난수). td: 후보 12개를 한 기간만 진행한 Π와 다음 상태의 (관측, 예상 청산) (collect_td.py와 같음).
반복마다 result/data/raw/{adv|td}/{rep}.pkl.gz에 저장하고 이미 있으면 건너뛴다. 실행: src/에서 python -m result.collect_raw adv|td
"""
from copy import deepcopy
from multiprocessing import Pool
import gzip
import os
import pickle
import sys
import time
import numpy as np
from utils.params import Cfg
from env.sim import Sim, cands
from value.feat import plan, ready
from result.collect import behave

DIR = "result/data/raw"


# 저장용 관측·예상 청산: 배정 y는 int16, Q는 int32로 줄인다
def pack(cfg, o):
  P = plan(cfg, o)
  for p in P.values():
    p.y, p.Q = p.y.astype(np.int16), p.Q.astype(np.int32)
  return o, P


# 반복 하나: 상태마다 (관측, 예상 청산)과 adv는 후보별 남은 누적 이윤, td는 후보별 (Π, 다음 관측, 다음 예상 청산)
def one(args):
  cfg, mode, rep = args
  path = f"{DIR}/{mode}/{rep}.pkl.gz"
  if os.path.exists(path):
    return rep
  t0, C = time.time(), cands(cfg)
  assert C[0] == (1.0, 0.0)
  s, pol, rows = Sim(cfg, rep), behave(cfg, rep), []
  while s.t < cfg.T:
    o = s.pre()
    if o.t > 0 and ready(o):
      row = dict(t=o.t, obs=pack(cfg, o), pi=[], nxt=[])
      for a in C:
        b = deepcopy(s)
        r = b.step(*a)["Pi"]
        if mode == "adv":
          while b.t < cfg.T:
            r += b.step(*pol(b.pre()))["Pi"]
        else:
          row["nxt"].append(pack(cfg, b.pre()) if b.t < cfg.T else None)
        row["pi"].append(r)
      rows.append(row)
    s.step(*pol(o))
  pickle.dump(dict(rep=rep, mode=mode, rows=rows), gzip.open(path, "wb"))
  print(rep, len(rows), round(time.time() - t0), flush=True)
  return rep


def main():
  cfg, mode = Cfg(), sys.argv[1]
  os.makedirs(f"{DIR}/{mode}", exist_ok=True)
  r0, n = (cfg.rep0_train, cfg.n_train) if mode == "adv" else (cfg.rep0_td, cfg.n_td)
  with Pool(cfg.n_proc) as p:
    p.map(one, [(cfg, mode, r) for r in range(r0, r0 + n)], chunksize=1)


if __name__ == "__main__":
  main()
