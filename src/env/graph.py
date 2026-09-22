"""거래 연결 구조와 구조 지표.
공급 편중도 요약(활동 공급자별 공급 품목 수의 최댓값·분산)과 "MLP + 편중도 요약" 입력, GNN 입력인 거래 연결 그래프를 만든다.
그래프는 최대 참여자 수만큼 자리를 둔 고정 크기 배열(노드 특징, 활성 마스크, 연결선)을 한 줄로 펴서 돌려준다.
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


# 그래프 입력 (한 줄): 노드 특징 ((gnn_nb + n_sup) × (2 × 품목 수 + 2)), 활성 마스크, 연결선 (gnn_nb × n_sup)
# 주문자 노드 = [요구수량, 제안가격, 직전 재참여확률, 종류 0], 공급자 노드 = [공급가능량, 공급가격, 직전 재참여확률, 종류 1]
# 연결선 = 주문자가 원하는 품목을 공급자가 하나 이상 공급할 수 있으면 1 (1단계: 거래 가능 여부만)
def graph(cfg, o):
  I, NB, NS, B, J = cfg.n_items, cfg.gnn_nb, cfg.n_sup, len(o.bid), len(o.sid)
  assert B <= NB and J <= NS
  x, mask, A = np.zeros((NB + NS, 2 * I + 2)), np.zeros(NB + NS), np.zeros((NB, NS))
  x[:B] = np.c_[o.q, o.p, o.rb, np.zeros(B)]
  x[NB:NB + J] = np.c_[o.cap, o.c, o.rs, np.ones(J)]
  mask[:B], mask[NB:NB + J] = 1, 1
  A[:B, :J] = (o.q > 0).astype(float) @ (o.cap > 0).T.astype(float) > 0
  return np.r_[x.ravel(), mask, A.ravel()]
