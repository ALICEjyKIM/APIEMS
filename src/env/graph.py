"""거래 연결 구조와 구조 지표.
공급 편중도 요약(활동 공급자별 공급 품목 수의 최댓값·분산)과 "MLP + 편중도 요약" 입력, GNN 입력인 거래 연결 그래프를 만든다.
그래프는 주문자·공급자 블록을 따로 두고(같은 열을 공유하지 않는다) 활성 마스크와 연결선을 이어 한 줄로 펴서 돌려준다.
"""
import numpy as np
from utils.features import phi


# 공급 편중도 요약: 활동 공급자별 공급 품목 수(공급가능량 > 0)의 최댓값·분산 (공급자가 없으면 0)
def conc_summary(o):
  d = (o.cap > 0).sum(1)
  return np.array([d.max(), d.var()], float) if len(d) else np.zeros(2)


# MLP + 편중도 요약 입력: 시장 요약 지표 34개 + 편중도 요약 2개
def phi_conc(o):
  return np.r_[phi(o), conc_summary(o)]


# 종류별 노드 특징 차원: 품목별 수량·가격 + 직전 재참여확률 (종류 표시는 두지 않는다. 종류별 임베딩이 구분한다)
def dims(cfg):
  return 2 * cfg.n_items + 1


# 그래프 입력 한 줄의 길이
def size(cfg):
  d, nb, ns = dims(cfg), cfg.gnn_nb, cfg.n_sup
  return (nb + ns) * d + nb + ns + nb * ns


# 그래프 입력 (한 줄): 주문자 특징 (gnn_nb × d), 공급자 특징 (n_sup × d), 주문자 마스크, 공급자 마스크, 연결선 (gnn_nb × n_sup)
# 주문자 노드 = [요구수량, 제안가격, 직전 재참여확률], 공급자 노드 = [공급가능량, 공급가격, 직전 재참여확률]
# 연결선 = 주문자가 원하는 품목을 공급자가 하나 이상 공급할 수 있으면 1 (1단계: 거래 가능 여부만)
def graph(cfg, o):
  NB, NS, B, J, d = cfg.gnn_nb, cfg.n_sup, len(o.bid), len(o.sid), dims(cfg)
  assert B <= NB and J <= NS
  xb, xs = np.zeros((NB, d)), np.zeros((NS, d))
  kb, ks, A = np.zeros(NB), np.zeros(NS), np.zeros((NB, NS))
  if B:
    xb[:B] = np.c_[o.q, o.p, o.rb]
    kb[:B] = 1
  if J:
    xs[:J] = np.c_[o.cap, o.c, o.rs]
    ks[:J] = 1
  if B and J:
    A[:B, :J] = (o.q > 0).astype(float) @ (o.cap > 0).T.astype(float) > 0
  return np.r_[xb.ravel(), xs.ravel(), kb, ks, A.ravel()]


# 그래프 한 줄 → (주문자 특징, 공급자 특징, 주문자 마스크, 공급자 마스크, 연결선)
def split(cfg, X):
  d, nb, ns, i = dims(cfg), cfg.gnn_nb, cfg.n_sup, 0
  xb, i = X[:, i:i + nb * d].reshape(-1, nb, d), i + nb * d
  xs, i = X[:, i:i + ns * d].reshape(-1, ns, d), i + ns * d
  mb, i = X[:, i:i + nb], i + nb
  ms, i = X[:, i:i + ns], i + ns
  return xb, xs, mb, ms, X[:, i:].reshape(-1, nb, ns)
