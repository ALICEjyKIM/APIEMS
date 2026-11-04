"""GNN Q̂ 테스트: 노드 순서를 바꿔도 값이 같은지, 개수에 민감한 목표(공급자 수)를 배우는지, 나눠 계산해도·pickle 왕복해도 같은지.
그래프는 기준 시장 상태(value.graph)와 난수로 만든 작은 그래프를 쓴다.
학습 테스트는 에폭을 줄인 Cfg로 돌린다.
"""
import pickle
from dataclasses import replace
import numpy as np
import pytest
import torch
from utils.params import Cfg
from env.sim import Sim, cands
from value.feat import feats, plan
from value.graph import graph
from value.fit import scaler
from value.gnn import G, Net, stats, gnn

torch.set_num_threads(2)
CFG = replace(Cfg(), T=6)
K, NM = len(cands(CFG)), len(CFG.m_grid)


# 기준 시장 기간 2의 (표 입력, 그래프)
def _state():
  s = Sim(CFG, 0)
  s.step(1.0, 0.0)
  s.step(1.0, 0.0)
  o = s.pre()
  P = plan(CFG, o)
  return feats(CFG, o, P), graph(CFG, o, P)


# 공급자·주문자 순서를 바꾼 그래프 (엣지 번호도 바꿈)
def _perm(g, r):
  h = dict(g)
  for t, k, pos in (("s", "si", 0), ("b", "bi", 0)):
    p = r.permutation(len(g[t]))
    inv = np.argsort(p)
    h[t] = g[t][p]
    e = list(g[k])
    e[pos] = inv[e[pos]]
    h[k] = tuple(e)
  return h


# 노드 순서를 바꿔도 후보별 값이 같다 (합·평균으로 모음)
def test_permutation_invariant():
  torch.manual_seed(0)
  X, gs = _state()
  mu, sd = scaler(X)
  q = G(mu, sd, 1.0, stats(gs), Net(X.shape[1], 2, 16).eval(), K, NM)
  r = np.random.default_rng(0)
  assert q((X, [_perm(g, r) for g in gs])) == pytest.approx(q((X, gs)), abs=1e-5)


# 난수 그래프: 주문자 nb명(품목 2개씩), 공급자 ns명(품목 1개), 품목 15
def _rand(r, nb, ns):
  f = lambda *s: r.normal(size=s).astype(np.float32)
  bc, sc = r.integers(15, size=2 * nb), r.integers(15, size=ns)
  return dict(b=f(nb, 8), i=f(15, 12), s=f(ns, 8), bi=(np.repeat(np.arange(nb), 2), bc, f(2 * nb, 1)), si=(np.arange(ns), sc, f(ns, 3)),
              ii=(bc[0::2], bc[1::2], f(nb, 1)))


# 목표가 그래프의 공급자 수(개수)면 GNN이 평균 예측보다 훨씬 잘 맞힌다. 나눠 계산·pickle 왕복해도 같은 값
def test_learns_count():
  r = np.random.default_rng(1)
  n = 24
  gs = [_rand(r, int(r.integers(3, 8)), int(r.integers(3, 20))) for _ in range(n * NM)]
  X = r.normal(size=(n * K, 3))
  Y = np.array([len(gs[(i // K) * NM + (i % K) // (K // NM)]["s"]) for i in range(n * K)], float)
  mu, sd = scaler(X)
  q = gnn(replace(CFG, mlp_epochs=40), X, gs, Y, mu, sd, Y.std(), K, 2, 16, 1e-2)
  v = q((X, gs))
  assert ((v - Y) ** 2).mean() < 0.1 * Y.var()
  q.chunk = 7
  assert q((X, gs)) == pytest.approx(v, abs=1e-4)
  assert np.array_equal(pickle.loads(pickle.dumps(q))((X, gs)), q((X, gs)))
