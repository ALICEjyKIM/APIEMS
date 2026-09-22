"""거래 연결 구조와 GNN 테스트 (슬라이스 5).
편중도 요약과 그래프 입력의 손계산 값, GNN의 노드 순서 불변성과 학습, 세 모델의 같은 학습 데이터를 확인한다.
유지 가치 변환은 모델의 입력 변환(enc)을 따라 계산된다.
"""
from dataclasses import replace
import numpy as np
import pytest
from utils.params import Cfg
from env.platform import Env
from env.graph import conc_summary, phi_conc, graph
from utils.features import phi, drop
from vfa.train import collect, collect_obs
from vfa.gnn import GNN, select as gnn_select
from vfa.coef import raw
from tests.test_vfa import two_obs

CFG = Cfg()


# 편중도 요약: 활동 공급자별 공급 품목 수의 최댓값·분산, MLP + 편중도 요약 입력 = 지표 34개 + 2개
def test_conc_summary():
  o = two_obs()
  assert conc_summary(o).tolist() == [1.0, 0.0]
  assert np.array_equal(phi_conc(o), np.r_[phi(o), 1.0, 0.0])
  assert len(phi_conc(Env(CFG, 0).reset())) == 36


# 그래프 입력: 노드 특징·활성 마스크·연결선(주문자가 원하는 품목을 공급자가 공급하면 1)
def test_graph_encoding():
  cfg = replace(CFG, n_items=2, gnn_nb=3, n_sup=2)
  o = two_obs()
  g = graph(cfg, o)
  n, d = 5, 6
  x, mask, A = g[:n * d].reshape(n, d), g[n * d:n * d + n], g[n * d + n:].reshape(3, 2)
  assert x[1].tolist() == [4, 2, 30.0, 25.0, 0.7, 0] and x[3].tolist() == [6, 0, 10.0, 0, 0.6, 1]
  assert mask.tolist() == [1, 1, 0, 1, 1]
  assert A.tolist() == [[1, 1], [1, 1], [0, 0]]
  o.cap = np.array([[0, 6], [6, 0]])
  A = graph(cfg, o)[n * d + n:].reshape(3, 2)
  assert A.tolist() == [[0, 1], [1, 1], [0, 0]]


# 세 모델이 같은 데이터로 학습한다: 관측 목록의 지표 변환이 기존 학습 데이터와 같다
def test_collect_obs_same_data():
  cfg = replace(CFG, T=6, vf_H=2, vf_reps=2)
  O, y, g = collect_obs(cfg)
  X, y2, g2 = collect(cfg)
  assert np.array_equal(np.array([phi(o) for o in O]), X) and np.array_equal(y, y2) and np.array_equal(g, g2)


# GNN: 같은 seed면 같은 예측, 주문자 순서를 바꿔도 같은 가치, 파라미터 수(6h² + 19h + 1 = 6356)가 MLP(6465)의 2% 이내
def test_gnn_invariant_and_size():
  cfg = replace(CFG, T=6, vf_H=2, vf_reps=3, nn_epochs=30)
  O, y, g = collect_obs(cfg)
  X = np.array([graph(cfg, o) for o in O])
  m = GNN(cfg, 0.0).fit(X, y)
  assert np.array_equal(m.predict(X), GNN(cfg, 0.0).fit(X, y).predict(X))
  assert abs(sum(p.numel() for p in m.net.parameters()) - 6465) < 0.02 * 6465
  o = O[-1]
  perm = np.arange(len(o.bid))[::-1]
  o2 = replace(o, q=o.q[perm], p=o.p[perm], rb=o.rb[perm], bid=[o.bid[i] for i in perm])
  assert m.predict(graph(cfg, o)[None])[0] == pytest.approx(m.predict(graph(cfg, o2)[None])[0], rel=1e-5)


# GNN 유지 가치: raw가 모델의 입력 변환을 따라 노드를 지운 그래프로 다시 예측한다
def test_gnn_raw_coef():
  cfg = replace(CFG, T=6, vf_H=2, vf_reps=5, nn_epochs=20)
  O, y, g = collect_obs(cfg)
  m, mse = gnn_select(cfg, np.array([graph(cfg, o) for o in O]), y, g)
  assert set(mse) == set(cfg.nn_wds)
  o = O[3]
  cb, cs, v = raw(m, o)
  assert len(cb) == len(o.bid) and len(cs) == len(o.sid)
  assert cs[0] == pytest.approx(v - m.predict(graph(cfg, drop(o, "sup", 0))[None])[0], abs=1e-3)  # float32 배치·단건 차이
