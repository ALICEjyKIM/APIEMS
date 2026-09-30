"""거래 연결 구조와 GNN 테스트 (슬라이스 5).
편중도 요약과 그래프 입력의 손계산 값, GNN의 노드 순서·자리 수 불변성과 학습, 세 모델의 같은 학습 데이터를 확인한다.
유지 가치 변환은 모델의 입력 변환(enc)을 따라 계산되고, 종류별 표준화가 실제로 분리되는지도 본다.
"""
from dataclasses import replace
import numpy as np
import pytest
from utils.params import Cfg
from env.platform import Env
from env.graph import conc_summary, phi_conc, graph, dims, size, split
from utils.features import phi, drop
from vfa.train import collect, collect_obs, val_mask
from vfa.gnn import GNN, select as gnn_select, n_params, mlp_params, hidden
from vfa.coef import raw
from tests.test_vfa import two_obs

CFG = Cfg()


# 편중도 요약: 활동 공급자별 공급 품목 수의 최댓값·분산, MLP + 편중도 요약 입력 = 지표 34개 + 2개
def test_conc_summary():
  o = two_obs()
  assert conc_summary(o).tolist() == [1.0, 0.0]
  assert np.array_equal(phi_conc(o), np.r_[phi(o), 1.0, 0.0])
  assert len(phi_conc(Env(CFG, 0).reset())) == 36


# 그래프 입력: 종류별 노드 특징 블록·활성 마스크·연결선(주문자가 원하는 품목을 공급자가 공급하면 1)
def test_graph_encoding():
  cfg = replace(CFG, n_items=2, gnn_nb=3, n_sup=2)
  o = two_obs()
  g = graph(cfg, o)
  assert dims(cfg) == 5 and len(g) == size(cfg)
  xb, xs, kb, ks, A = (a[0] for a in split(cfg, g[None]))
  assert xb[1].tolist() == [4, 2, 30.0, 25.0, 0.7] and xs[0].tolist() == [6, 0, 10.0, 0, 0.6]
  assert xb[2].tolist() == [0] * 5
  assert kb.tolist() == [1, 1, 0] and ks.tolist() == [1, 1]
  assert A.tolist() == [[1, 1], [1, 1], [0, 0]]
  o.cap = np.array([[0, 6], [6, 0]])
  A = split(cfg, graph(cfg, o)[None])[4][0]
  assert A.tolist() == [[0, 1], [1, 1], [0, 0]]


# 세 모델이 같은 데이터로 학습한다: 관측 목록의 지표 변환이 기존 학습 데이터와 같다
def test_collect_obs_same_data():
  cfg = replace(CFG, T=6, vf_H=2, vf_reps=2)
  O, y, g = collect_obs(cfg)
  X, y2, g2 = collect(cfg)
  assert np.array_equal(np.array([phi(o) for o in O]), X) and np.array_equal(y, y2) and np.array_equal(g, g2)


# GNN: 같은 seed면 같은 예측, 주문자 순서를 바꿔도 같은 가치, 파라미터 수가 MLP의 5% 이내 (정수 은닉 크기라 정확히 같을 수는 없다)
def test_gnn_invariant_and_size():
  cfg = replace(CFG, T=6, vf_H=2, vf_reps=3, nn_epochs=30)
  O, y, g = collect_obs(cfg)
  X = np.array([graph(cfg, o) for o in O])
  m = GNN(cfg, 0.0).fit(X, y)
  assert np.array_equal(m.predict(X), GNN(cfg, 0.0).fit(X, y).predict(X))
  got = sum(p.numel() for p in m.net.parameters())
  assert got == n_params(cfg, m.h) and abs(got - mlp_params(cfg)) < 0.05 * mlp_params(cfg)
  assert m.h == hidden(cfg) and hidden(replace(cfg, gnn_hidden=17)) == 17
  o = O[-1]
  perm = np.arange(len(o.bid))[::-1]
  o2 = replace(o, q=o.q[perm], p=o.p[perm], rb=o.rb[perm], bid=[o.bid[i] for i in perm])
  assert m.predict(graph(cfg, o)[None])[0] == pytest.approx(m.predict(graph(cfg, o2)[None])[0], rel=1e-5)


# 자리 수 불변: 빈 주문자 자리를 늘려도 같은 상태의 예측이 같다 (활성 마스크와 평균 풀링이 제대로 동작)
def test_gnn_slot_invariant():
  cfg = replace(CFG, T=6, vf_H=2, vf_reps=3, nn_epochs=20)
  O, y, g = collect_obs(cfg)
  m = GNN(cfg, 0.0).fit(np.array([graph(cfg, o) for o in O]), y)
  wide = replace(cfg, gnn_nb=cfg.gnn_nb + 17)
  m2 = GNN(wide, 0.0)
  for k, v in m.__dict__.items():
    if k not in ("cfg", "enc"):
      setattr(m2, k, v)
  o = O[-1]
  assert m.predict(graph(cfg, o)[None])[0] == pytest.approx(m2.predict(graph(wide, o)[None])[0], rel=1e-5)


# 종류별 표준화: 주문자·공급자 통계가 따로 잡히고, 활성 노드만으로 계산된다
def test_gnn_type_standardization():
  cfg = replace(CFG, T=6, vf_H=2, vf_reps=3, nn_epochs=5)
  O, y, g = collect_obs(cfg)
  X = np.array([graph(cfg, o) for o in O])
  m = GNN(cfg, 0.0).fit(X, y)
  xb, xs, kb, ks, _ = split(cfg, X)
  assert m.mu_b == pytest.approx(xb[kb.astype(bool)].mean(0)) and m.mu_s == pytest.approx(xs[ks.astype(bool)].mean(0))
  assert not np.allclose(m.mu_b, m.mu_s)  # 같은 열이라도 종류별 규모가 다르다
  assert m.mu_n == pytest.approx(np.c_[kb.sum(1), ks.sum(1)].mean(0))
  z = m._unpack(X)
  assert float(z[0][~kb.astype(bool)].abs().max()) == 0 and float(z[1][~ks.astype(bool)].abs().max()) == 0


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
