"""MILP 정책.
매 기간 관측으로 MILP를 만들어 풀고, 최적해의 배정 수량과 잉여배분을 돌려준다.
근시안(유지 가치 0), 규칙 기반(split = 고정 비율), 가치 근사(coef = 유지 가치)가 같은 MILP를 쓴다.
"""
import numpy as np
from gurobipy import GRB
from match.milp_build import build
from env.response import Logistic


class Policy:
  """MILP 정책.
  reset으로 반복을 시작하고, act로 기간 결정을 내리고, observe로 정산 결과를 받는다.
  resp는 MILP가 쓰는 학습용 반응 함수, coef(o)는 (주문자 유지 가치, 공급자 유지 가치). obj에는 직전 목적함수 값을 남긴다.
  """

  def __init__(self, cfg, resp=None, split=None, coef=None):
    self.cfg, self.resp, self.split, self.coef = cfg, resp or Logistic(cfg), split, coef

  # 반복 rep 시작
  def reset(self, rep):
    self.rep = rep

  # 기간 결정: 최적해의 (배정 수량 x[b,j,i], 주문자 잉여, 공급자 잉여)
  # 잉여는 반올림한 x의 마진에 맞춘다 (규칙 기반은 고정 비율 그대로, 나머지는 [0, 마진]으로 자름): 솔버 허용오차(1e-5) 정리
  def act(self, o):
    m, x, u, v = build(self.cfg, o, self.split, self.resp, self.coef and self.coef(o))
    m.optimize()
    assert m.Status == GRB.OPTIMAL
    self.obj = m.ObjVal
    x = np.rint(x.X).astype(int)
    mg = (o.p[:, None] - o.c[None]) * x
    g, mj = mg.sum((1, 2)), mg.sum((0, 2))
    if self.split:
      return x, self.split[0] * g, self.split[1] * mj
    return x, np.clip(u.X, 0, g), np.clip(v.X, 0, mj)

  # 정산 결과 관찰 (아직 쓰지 않음)
  def observe(self, r):
    pass
