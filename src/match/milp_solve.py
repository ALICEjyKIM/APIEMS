"""MILP 배정 정책.
매 기간 관측으로 MILP를 만들어 풀고, 최적해의 배정 수량을 정수 배열로 돌려준다.
모든 정책은 이 MILP를 공유하고 유지 가치만 다르다 (슬라이스 1은 유지 가치 0).
"""
import numpy as np
from gurobipy import GRB
from match.milp_build import build


class Policy:
  """MILP 배정 정책.
  reset으로 반복을 시작하고, act로 기간 결정을 내리고, observe로 정산 결과를 받는다.
  obj에는 직전 act의 MILP 목적함수 값을 남긴다.
  """

  def __init__(self, cfg):
    self.cfg = cfg

  # 반복 rep 시작
  def reset(self, rep):
    self.rep = rep

  # 기간 결정: 최적해의 배정 수량 x[b,j,i]
  def act(self, o):
    m, x, _ = build(self.cfg, o)
    m.optimize()
    assert m.Status == GRB.OPTIMAL
    self.obj = m.ObjVal
    return np.rint(x.X).astype(int)

  # 정산 결과 관찰 (슬라이스 1은 쓰지 않음)
  def observe(self, r):
    pass
