"""Q̂ 학습 자료 (N5): 행동 정책(고정 padding, 확률 eps로 후보 무작위)의 경로에서 기간 t ≥ 1마다 시장을 복사해 후보 12개로 분기한다.
분기 뒤는 같은 공통 난수로 행동 정책을 T까지 따르고, 목표는 고정 padding 대비 남은 누적 이윤 차이 A다 (ref/design.md 5장).
결과는 result/data/adv.npz (X: 상태 × 후보 × 입력, Y: 상태 × 후보, rep, t, R0). 실행: src/에서 python -m result.collect
"""
from copy import deepcopy
from multiprocessing import Pool
import os
import time
import numpy as np
from utils.params import Cfg
from utils.rng import rng
from env.sim import Sim, cands
from value.feat import feats, ready


# 행동 정책: 확률 eps로 후보 균등 무작위, 나머지는 고정 padding (난수는 반복·기간으로 고정)
def behave(cfg, rep):
  C = cands(cfg)
  def pol(o):
    g = rng(cfg, rep, "explore", o.t)
    return C[g.integers(len(C))] if g.random() < cfg.eps else C[0]
  return pol


# 반복 하나: 상태마다 입력과 후보별 남은 누적 이윤
def one(args):
  cfg, rep = args
  t0, C = time.time(), cands(cfg)
  assert C[0] == (1.0, 0.0)
  s, pol, X, Y, T, R0 = Sim(cfg, rep), behave(cfg, rep), [], [], [], []
  while s.t < cfg.T:
    o = s.pre()
    if o.t > 0 and ready(o):
      R = []
      for a in C:
        b, r = deepcopy(s), 0.0
        r += b.step(*a)["Pi"]
        while b.t < cfg.T:
          r += b.step(*pol(b.pre()))["Pi"]
        R.append(r)
      X.append(feats(cfg, o))
      Y.append(np.array(R) - R[0])
      T.append(o.t)
      R0.append(R[0])
    s.step(*pol(o))
  print(rep, len(T), round(time.time() - t0), flush=True)
  return np.array(X), np.array(Y), np.full(len(T), rep), np.array(T), np.array(R0)


def main():
  cfg = Cfg()
  with Pool(20) as p:
    out = p.map(one, [(cfg, r) for r in range(cfg.rep0_train, cfg.rep0_train + cfg.n_train)], chunksize=1)
  os.makedirs("result/data", exist_ok=True)
  np.savez_compressed("result/data/adv.npz", **{k: np.concatenate([o[i] for o in out]) for i, k in enumerate(("X", "Y", "rep", "t", "R0"))})


if __name__ == "__main__":
  main()
