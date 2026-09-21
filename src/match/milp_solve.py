"""MILP 정책.
매 기간 관측으로 MILP를 만들어 풀고, 최적해의 배정 수량과 잉여배분을 돌려준다.
근시안(split 없음, 유지 가치 0)과 규칙 기반(split = 고정 비율)이 같은 MILP를 쓴다.
"""
import numpy as np
from gurobipy import GRB
from match.milp_build import build


class Policy:
  """MILP 정책.
  reset으로 반복을 시작하고, act로 기간 결정을 내리고, observe로 정산 결과를 받는다.
  obj에는 직전 act의 MILP 목적함수 값을 남긴다.
  """

  def __init__(self, cfg, split=None):
    self.cfg, self.split = cfg, split

  # 반복 rep 시작
  def reset(self, rep):
    self.rep = rep

  # 기간 결정: 최적해의 (배정 수량 x[b,j,i], 주문자 잉여, 공급자 잉여)
  def act(self, o):
    m, x, u, v = build(self.cfg, o, self.split)
    m.optimize()
    assert m.Status == GRB.OPTIMAL
    self.obj = m.ObjVal
    return np.rint(x.X).astype(int), u.X, v.X

  # 정산 결과 관찰 (아직 쓰지 않음)
  def observe(self, r):
    pass
