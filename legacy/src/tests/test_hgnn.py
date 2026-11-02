"""Buyer–Item–Supplier SUM-GNN (vfa/hgnn) 테스트.
그래프 묶음의 노드 offset, 공급자별 출력 개수, 노드 순서를 바꿔도 같은 공급자 예측이 같은지(SUM 집계), 한 그래프만 넣은 예측과 묶음 예측이 같은지 확인한다.
"""
import numpy as np
import pytest
from vfa.hgnn import HGNN, batch


# 작은 무작위 그래프: 주문자 nb, 품목 ni, 공급자 ns, 연결선은 무작위
def toy(seed, nb=4, ni=3, ns=3):
  g = np.random.default_rng(seed)
  bb, bi = np.nonzero(g.random((nb, ni)) < 0.6)
  ss, si = np.nonzero(g.random((ns, ni)) < 0.6)
  return dict(xb=g.random((nb, 2)), xi=g.random((ni, 2)), xs=g.random((ns, 2)), ebi=g.random((len(bb), 2)), esi=g.random((len(ss), 2)),
              bi_b=bb, bi_i=bi, si_s=ss, si_i=si)


# 묶음: 두 번째 그래프의 연결선 끝 번호에 첫 그래프의 노드 수가 더해진다
def test_batch_offsets():
  a, b = toy(0), toy(1)
  B = batch([a, b])
  assert len(B["xb"]) == len(a["xb"]) + len(b["xb"])
  assert np.array_equal(B["bi_b"][len(a["bi_b"]):], b["bi_b"] + len(a["xb"]))
  assert np.array_equal(B["si_i"][len(a["si_i"]):], b["si_i"] + len(a["xi"]))
  assert np.array_equal(B["si_s"][len(a["si_s"]):], b["si_s"] + len(a["xs"]))


# 공급자마다 예측 하나, 그래프 하나만 넣어도 묶음에서와 같은 값, 공급자 순서를 바꾸면 예측도 같은 순서로 바뀐다
@pytest.mark.parametrize("layers", [2, 3])
def test_outputs_and_invariance(layers):
  G = [toy(s) for s in range(4)]
  y = [np.random.default_rng(s).random(len(g["xs"])) for s, g in enumerate(G)]
  m = HGNN(8, layers, 0.0, 0, epochs=30, every=10).fit(G[:3], y[:3], G[3:], y[3:])
  p = m.predict(G)
  assert [len(x) for x in p] == [len(g["xs"]) for g in G]
  assert np.allclose(m.predict([G[1]])[0], p[1], atol=1e-5)
  g = G[0]
  perm = np.array([2, 0, 1])
  inv = np.argsort(perm)
  h = dict(g, xs=g["xs"][perm], si_s=inv[g["si_s"]])
  assert np.allclose(m.predict([h])[0], p[0][perm], atol=1e-5)
