"""TD 학습 자료 (N5 2차): 행동 정책(고정 padding, 확률 eps로 후보 무작위)의 경로에서 기간 t ≥ 1마다 시장을 복사해 후보 12개를 한 기간씩만 진행한다.
후보마다 이번 기간 이윤 Π와 다음 상태의 Q̂ 입력(마지막 기간은 0)을 기록한다. 끝까지 돌리지 않아 미래 한 번의 운이 목표에 섞이지 않는다.
반복마다 result/data/td/{rep}.npz에 저장하고 이미 있으면 건너뛴다. 실행: src/에서 python -m result.collect_td
"""
from copy import deepcopy
from multiprocessing import Pool
import os
import time
import numpy as np
from utils.params import Cfg
from env.sim import Sim, cands
from value.feat import feats, ready
from result.collect import behave

DIR = "result/data/td"


# 반복 하나: 상태마다 입력 X, 후보별 이윤 P, 후보별 다음 상태 입력 Xn
def one(args):
  cfg, rep = args
  path = f"{DIR}/{rep}.npz"
  if os.path.exists(path):
    return rep
  t0, C = time.time(), cands(cfg)
  s, pol, X, P, Xn, T = Sim(cfg, rep), behave(cfg, rep), [], [], [], []
  while s.t < cfg.T:
    o = s.pre()
    if o.t > 0 and ready(o):
      x, pi, xn = feats(cfg, o), [], []
      for a in C:
        b = deepcopy(s)
        pi.append(b.step(*a)["Pi"])
        xn.append(feats(cfg, b.pre()) if b.t < cfg.T else np.zeros_like(x))
      X.append(x)
      P.append(pi)
      Xn.append(xn)
      T.append(o.t)
    s.step(*pol(o))
  np.savez_compressed(path, X=np.array(X, np.float32), P=np.array(P), Xn=np.array(Xn, np.float32), t=np.array(T), rep=np.full(len(T), rep))
  print(rep, len(T), round(time.time() - t0), flush=True)
  return rep


def main():
  cfg = Cfg()
  os.makedirs(DIR, exist_ok=True)
  with Pool(cfg.n_proc) as p:
    p.map(one, [(cfg, r) for r in range(cfg.rep0_td, cfg.rep0_td + cfg.n_td)], chunksize=1)


if __name__ == "__main__":
  main()
