"""GNN 입력과 구조 요약 (질문 3, ref/design.md 5장 표, ref/decisions.md "GNN 입력"): 입찰 전 관측과 예상 청산(value.feat.plan)으로 후보 m마다 그래프 하나를 만든다.
노드는 주문자(8)·품목(12)·공급자(8), 엣지는 주문자–품목(1)·공급자–품목(3)·품목–품목(1). 개인의 지난 신고가(W0, g0)와 재참여 확률 값은 쓰지 않는다.
구조 요약은 노드 종류별·엣지 종류별 피처의 합·평균이고, MLP + 구조 요약 입력에 더해 GNN과 같은 정보를 준다.
"""
import numpy as np
from env.sim import cands
from value.feat import feats

DIM = dict(b=8, i=12, s=8, bi=1, si=3, ii=1)


# 기록 없는 품목 값은 기준가 × 다른 품목의 평균 비율 (policy.base.est와 같은 규칙)
def _fill(v, base):
  return np.where(np.isnan(v), np.nanmean(v / base) * base, v)


# 후보 m마다 그래프 (m_grid 순): 노드 행렬 b, i, s와 엣지 bi, si, ii = (출발 번호, 도착 번호, 엣지 피처)
def graph(cfg, o, P):
  q, K, base = o.q, o.K, o.base
  cap, kmax = K.sum(0), K.max(0, initial=0)
  c1 = np.maximum(cap, 1)
  vb, cb = _fill(o.vbar, base) / base, _fill(o.cbar, base) / base
  b0 = np.c_[o.tb == 0, o.hb, np.nan_to_num(o.rb), np.isnan(o.rb), (q > 0).sum(1), o.eb, o.tb]
  s = np.c_[o.ts == 0, o.hs, np.nan_to_num(o.rs), np.isnan(o.rs), (K > 0).sum(1), K.sum(1) / max(cap.sum(), 1), o.es, o.ts]
  i0 = np.c_[q.sum(0) / c1, kmax / c1, (K > 0).sum(0), o.hi, (o.pend > 0).sum(0), vb, cb, vb - cb, (cap + o.pend.sum(0)) / c1]
  bb, bc = np.nonzero(q)
  sj, sc = np.nonzero(K)
  co = (q > 0).astype(int).T @ (q > 0).astype(int)
  np.fill_diagonal(co, 0)
  ia, ib = np.nonzero(co)
  bi, ii = (bb, bc, (q[bb, bc] / c1[bc])[:, None]), (ia, ib, co[ia, ib][:, None].astype(float))
  si0 = np.c_[K[sj, sc] / c1[sc], K[sj, sc] == kmax[sc]]
  f = lambda x: np.asarray(x, np.float32)
  out = []
  for m in cfg.m_grid:
    p = P[m]
    i = np.c_[i0, p.Q / c1, q[p.Ub].sum(0) / c1, p.Q > cap]
    out.append(dict(b=f(np.c_[b0, p.win]), i=f(i), s=f(s), bi=(bi[0], bi[1], f(bi[2])), si=(sj, sc, f(np.c_[si0, p.y[sj, sc] / K[sj, sc]])),
                    ii=(ii[0], ii[1], f(ii[2]))))
  return out


# 구조 요약: 노드 종류별·엣지 종류별 피처의 합과 평균 (2 × (8 + 12 + 8 + 1 + 3 + 1) = 66개)
def summ(g):
  f = lambda x: np.r_[x.sum(0), x.mean(0) if len(x) else np.zeros(x.shape[1])]
  return np.concatenate([f(g[k]) for k in ("b", "i", "s")] + [f(g[k][2]) for k in ("bi", "si", "ii")])


# MLP + 구조 요약 입력: 표 입력(value.feat.feats) 옆에 그 후보 m 그래프의 구조 요약
def feats_s(cfg, o, P):
  G, ms = [summ(g) for g in graph(cfg, o, P)], list(cfg.m_grid)
  return np.c_[feats(cfg, o, P), np.array([G[ms.index(m)] for m, _ in cands(cfg)])]
