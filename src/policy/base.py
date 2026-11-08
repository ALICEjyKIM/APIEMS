"""1단계 비교 정책: 고정 padding·최선 고정값(fixed), 한 기간 최적(myopic)과 예상 청산에 쓰는 추정 입찰가(est), 시나리오 입찰가(scen)와 시나리오 한 기간 최적(saa).
모든 정책은 PreObs만 받아 같은 후보 집합(env.sim.cands)에서 (m, θ)를 고른다. 이번 기간 입찰가는 읽지 않는다.
예상 청산은 추정 입찰가로 Chu IPB(mech.chu.clear)를 그대로 푼다 (ref/design.md 5·6장, ref/decisions.md "예측 점검").
"""
import numpy as np
from env.sim import cands
from mech.chu import clear
from utils.rng import rng


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


# 시나리오 입찰가 K개 (경험분포 복원추출): 주문자 Ŵ_b = u × (q_b · base), u ~ uh. 공급자 ĝ_jc ~ 품목 c의 과거 원가 (표본 < saa_min이면 전체 원가 비율 × base_c). 난수는 (t, 주문자 수, 자리 수)로 고정
def scen(cfg, o, K):
  r, wb, pool = rng(cfg, 0, "saa", o.t, len(o.bid), len(o.slot)), o.q @ o.base, o.ch / o.base[o.ci]
  src = [o.ch[o.ci == c] if (o.ci == c).sum() >= cfg.saa_min else pool * o.base[c] for c in range(cfg.n_items)]
  out = []
  for _ in range(K):
    g = np.zeros(o.K.shape)
    for c, j in enumerate(o.K.T > 0):
      g[j, c] = r.choice(src[c], j.sum())
    out.append((r.choice(o.uh, len(o.bid)) * wb, g))
  return out


# 참고용 시나리오: 재참여자는 지난 신고가(W0, g0)를 그대로 쓰고 신규만 시나리오 (개인 신고가를 쓰므로 정책에는 쓰지 않음)
def scen_ind(cfg, o, K):
  return [(np.where(np.isnan(o.W0), W, o.W0), np.where(np.isnan(o.g0) | (o.K == 0), g, o.g0)) for W, g in scen(cfg, o, K)]


# 후보 m별 예상 이번 기간 플랫폼 몫: 평균 추정 입찰가(mean) 또는 시나리오 K개 평균(saa)
def gains(cfg, o, ms, kind, K=None):
  if kind == "mean":
    W, g = est(o)
    return np.array([clear(cfg, o.q, W, o.K, g, m).G for m in ms])
  S = scen(cfg, o, K or cfg.saa_k)
  return np.array([np.mean([clear(cfg, o.q, W, o.K, g, m).G for W, g in S]) for m in ms])


# 장기 표 규칙 (ref/decisions.md "×2 장기 표 규칙 실험"): 예상 이번 기간 이득 + 시점 구간별 평균 이후 손실 Lbar[구간][m]이 가장 큰 m, θ = 0. 구간 경계 bins는 기간 t의 왼쪽 끝. 기록이 없으면 고정 padding
def table(cfg, Lbar, bins, kind="saa"):
  def pol(o):
    if not len(o.uh) or np.isnan(o.vbar).all() or np.isnan(o.cbar).all():
      return 1.0, 0.0
    ms = sorted({m for m, _ in cands(cfg)})
    G = gains(cfg, o, ms, kind)
    v = G - G[0] + np.asarray(Lbar[max(np.searchsorted(bins, o.t, side="right") - 1, 0)])
    return ms[int(np.argmax(v))], 0.0
  return pol


# 시나리오 한 기간 최적: 후보 m마다 시나리오 K개 청산의 플랫폼 몫 평균이 가장 큰 m (동점이면 작은 m), θ = 0. 기록이 없으면 고정 padding
def saa(cfg, K=None):
  K = K or cfg.saa_k
  def pol(o):
    if not len(o.uh):
      return 1.0, 0.0
    ms, S = sorted({m for m, _ in cands(cfg)}), scen(cfg, o, K)
    G = [np.mean([clear(cfg, o.q, W, o.K, g, m).G for W, g in S]) for m in ms]
    return ms[int(np.argmax(G))], 0.0
  return pol
