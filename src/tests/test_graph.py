"""GNN 입력 테스트: 노드·엣지 차원이 design.md 5장 표와 같은지, 엣지가 입찰 전 관측의 수량·용량과 맞는지, 후보 m에 따라 바뀌는 부분.
입찰가를 바꿔도 그래프와 구조 요약이 같아야 한다 (1단계는 입찰 전 관측만 받는다).
엔진은 기준 시장 반복 0을 두 기간 돌린다.
"""
from dataclasses import replace
import numpy as np
from utils.params import Cfg
from env.sim import Sim, cands
from value.feat import feats, plan
from value.graph import DIM, graph, summ, feats_s

CFG = replace(Cfg(), T=6)


# 상태 하나 (기간 2)
def _state():
  s = Sim(CFG, 0)
  s.step(1.0, 0.0)
  s.step(1.0, 0.0)
  return s


# 차원·엣지 수·품목–품목 대칭·단일 품목 공급자, m이 커지면 Q_c/용량이 줄지 않는다
def test_graph_shape():
  s = _state()
  o = s.pre()
  G = graph(CFG, o, plan(CFG, o))
  assert len(G) == len(CFG.m_grid)
  for g in G:
    assert g["b"].shape == (len(o.bid), DIM["b"]) and g["i"].shape == (CFG.n_items, DIM["i"]) and g["s"].shape == (len(o.slot), DIM["s"])
    assert all(g[k][2].shape[1] == DIM[k] for k in ("bi", "si", "ii"))
    assert len(g["bi"][0]) == (o.q > 0).sum() and len(g["si"][0]) == len(o.slot)
    a, b, w = g["ii"]
    assert {(x, y, float(v)) for x, y, v in zip(a, b, w[:, 0])} == {(y, x, float(v)) for x, y, v in zip(a, b, w[:, 0])}
    assert (g["s"][:, 4] == 1).all() and g["i"][:, 2].sum() == len(o.slot)
  assert all((G[k + 1]["i"][:, 9] >= G[k]["i"][:, 9]).all() for k in range(len(G) - 1))


# 구조 요약 66개, MLP + 구조 요약 입력은 표 입력 옆에 후보 m의 요약, 입찰가를 바꿔도 같다
def test_summary_bid_invariant():
  s = _state()
  o = s.pre()
  P = plan(CFG, o)
  X, G = feats_s(CFG, o, P), graph(CFG, o, P)
  assert X.shape == (len(cands(CFG)), feats(CFG, o, P).shape[1] + 66)
  ms = list(CFG.m_grid)
  assert all(np.array_equal(x[-66:], summ(G[ms.index(m)])) for x, (m, _) in zip(X, cands(CFG)))
  s.W, s.g = s.W * 3 + 1, s.g * 0.2
  o2 = s.pre()
  assert np.array_equal(X, feats_s(CFG, o2, plan(CFG, o2)))
