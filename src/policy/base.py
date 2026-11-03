"""1단계 비교 정책: 고정 padding·최선 고정값(fixed), 한 기간 최적(myopic)과 예상 청산에 쓰는 추정 입찰가(est).
모든 정책은 PreObs만 받아 같은 후보 집합(env.sim.cands)에서 (m, θ)를 고른다. 이번 기간 입찰가는 읽지 않는다.
예상 청산은 추정 입찰가로 Chu IPB(mech.chu.clear)를 그대로 푼다 (ref/design.md 5·6장).
"""
import numpy as np
from env.sim import cands
from mech.chu import clear


# 늘 같은 (m, θ)를 고르는 정책
def fixed(m, th):
  return lambda o: (m, th)


# 추정 입찰가 (시장 통계만): 주문자 Ŵ_b = Σ_c q_bc vbar_c, 공급자 ĝ_jc = cbar_c (자격 품목만). 기록 없는 품목은 기준가 × 다른 품목의 평균 비율
def est(o):
  fill = lambda v: np.where(np.isnan(v), np.nanmean(v / o.base) * o.base, v)
  return o.q @ fill(o.vbar), np.where(o.K > 0, fill(o.cbar), 0.0)


# 한 기간 최적: 후보 m마다 예상 청산의 플랫폼 몫이 가장 큰 m (동점이면 작은 m), θ = 0 (쿠폰은 다음 기간 지급). 가격 기록이 없으면 고정 padding
def myopic(cfg):
  def pol(o):
    if np.isnan(o.vbar).all() or np.isnan(o.cbar).all():
      return 1.0, 0.0
    W, g = est(o)
    ms = sorted({m for m, _ in cands(cfg)})
    G = [clear(cfg, o.q, W, o.K, g, m).G for m in ms]
    return ms[int(np.argmax(G))], 0.0
  return pol
