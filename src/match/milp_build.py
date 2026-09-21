"""주문 수락, 품목별 공급자·수량 배정, 잉여배분 MILP.
한 주문의 한 품목은 여러 공급자가 나눠 공급할 수 있다 (허브에서 모아 통합 배송).
목적함수 = 플랫폼 이윤(마진 − 배송 고정비 − 배분 잉여). 모든 정책이 이 MILP를 공유한다.
"""
import gurobipy as gp
from gurobipy import GRB
import numpy as np

ENV = gp.Env(params={"OutputFlag": 0})


# 관측 o의 MILP: x[b,j,i] = 공급자 j가 주문 b의 품목 i에 보내는 수량, y[b] = 주문 b 성립, u·v = 주문자·공급자 잉여
# split = (주문자 몫, 공급자 몫)이면 잉여를 마진의 고정 비율로 묶고(규칙 기반), None이면 [0, 자기 거래 마진]에서 고른다
def build(cfg, o, split=None):
  m = gp.Model(env=ENV)
  m.Params.OutputFlag = 0
  m.Params.Seed = cfg.grb_seed
  m.Params.Threads = cfg.grb_threads
  m.Params.MIPGap = cfg.mip_gap
  x = m.addMVar((len(o.q), len(o.cap), o.q.shape[1]), vtype=GRB.INTEGER, ub=np.minimum(o.q[:, None], o.cap[None]))
  y = m.addMVar(len(o.q), vtype=GRB.BINARY)
  u, v = m.addMVar(len(o.q)), m.addMVar(len(o.cap))
  mg = (o.p[:, None] - o.c[None]) * x
  g, mj = mg.sum(2).sum(1), mg.sum(2).sum(0)
  m.addConstr(x.sum(1) == o.q * y[:, None])
  m.addConstr(x.sum(0) <= o.cap)
  if split:
    m.addConstr(u == split[0] * g)
    m.addConstr(v == split[1] * mj)
  else:
    m.addConstr(u <= g)
    m.addConstr(v <= mj)
  prof = mg.sum() - cfg.f_ship * y.sum() - u.sum() - v.sum()
  m.addConstr(prof >= 0)
  m.setObjective(prof, GRB.MAXIMIZE)
  return m, x, u, v
