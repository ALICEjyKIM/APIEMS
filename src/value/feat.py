"""Q̂ 입력: 입찰 전 관측(PreObs) 요약 17개와 후보 (m, θ)별 예상 청산 요약 11개, 그리고 m × 상태, θ × 상태.
예상 청산은 policy.base.est의 추정 입찰가로 m마다 Chu IPB를 한 번 풀고, θ는 예상 쿠폰액에만 쓴다.
선형·MLP는 같은 입력을 받는다 (ref/design.md 5장, ref/decisions.md "Q̂ 입력").
"""
import numpy as np
from env.sim import cands
from mech.chu import clear
from policy.base import est


# 평균 (빈 배열·전부 nan이면 0)
def _mu(x):
  x = np.asarray(x, float)
  return float(np.nanmean(x)) if np.isfinite(x).any() else 0.0


# 상태 요약: 기간, 주문자(수, 신규, 이력, 기준점, 기준점 없음, 지난 제외), 공급자(같은 항목, 신규), onboard 대기, 미지급 쿠폰, 부족 이력, 수요/용량
def state(cfg, o):
  nan = lambda x: float(np.isnan(x).mean()) if len(x) else 0.0
  return np.array([o.t / cfg.T, len(o.bid), _mu(o.tb == 0), _mu(o.hb), _mu(o.rb), nan(o.rb), _mu(o.eb),
                   len(o.slot), _mu(o.hs), _mu(o.rs), nan(o.rs), _mu(o.es), _mu(o.ts == 0), len(o.wait), o.owe, _mu(o.hi),
                   o.q.sum() / max(o.K.sum(), 1)])


# 가격 기록이 있어 예상 청산을 할 수 있는지 (기간 0은 없음)
def ready(o):
  return not (np.isnan(o.vbar).all() or np.isnan(o.cbar).all())


# 후보별 입력 행렬 (cands 순서): [상태, 후보·예상 청산, m × 상태, θ × 상태]
def feats(cfg, o):
  s, (W, g) = state(cfg, o), est(o)
  P = {m: clear(cfg, o.q, W, o.K, g, m) for m in cfg.m_grid}
  wb, ws, cap = o.q @ o.base, o.K @ o.base, o.K.sum(0)
  rows = []
  for m, th in cands(cfg):
    p = P[m]
    tot = wb[p.Ub].sum() + ws[p.Uj].sum()
    f = th * p.G / tot if tot > 0 else 0.0
    k = np.minimum(f * wb, cfg.cbar * wb)[p.Ub].sum() + np.minimum(f * ws, cfg.cbar * ws)[p.Uj].sum()
    a = np.array([m, th, _mu(p.win), p.G, p.sw, p.Ub.sum(), p.Uj.sum(), wb[p.Ub].sum(), ws[p.Uj].sum(), k, (p.Q > cap).sum()])
    rows.append(np.concatenate([s, a, m * s, th * s]))
  return np.array(rows)
