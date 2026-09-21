"""주문 수락과 품목별 공급자·수량 배정 MILP.
한 주문의 한 품목은 여러 공급자가 나눠 공급할 수 있다 (허브에서 모아 통합 배송).
목적함수 = 고정 비율의 플랫폼 몫 − 성립 주문당 배송 고정비 (슬라이스 1: 유지 가치 0).
"""
import gurobipy as gp
from gurobipy import GRB
import numpy as np

ENV = gp.Env(params={"OutputFlag": 0})


# 관측 o의 MILP: x[b,j,i] = 공급자 j가 주문 b의 품목 i에 보내는 수량, y[b] = 주문 b 성립
def build(cfg, o):
  m = gp.Model(env=ENV)
  m.Params.OutputFlag = 0
  m.Params.Seed = cfg.grb_seed
  m.Params.Threads = cfg.grb_threads
  m.Params.MIPGap = cfg.mip_gap
  x = m.addMVar((len(o.q), len(o.cap), o.q.shape[1]), vtype=GRB.INTEGER, ub=np.minimum(o.q[:, None], o.cap[None]))
  y = m.addMVar(len(o.q), vtype=GRB.BINARY)
  m.addConstr(x.sum(1) == o.q * y[:, None])
  m.addConstr(x.sum(0) <= o.cap)
  sh = 1 - cfg.sh_buy - cfg.sh_sup
  m.setObjective((sh * (o.p[:, None] - o.c[None]) * x).sum() - cfg.f_ship * y.sum(), GRB.MAXIMIZE)
  return m, x, y
