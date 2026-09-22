"""참여자별 유지 가치 변환.
가치 모델 V로 c_i = V(φ(o)) − V(φ(o에서 i 제외))를 계산한다 (주문자·공급자 모두). MILP에는 가드를 거친 값을 넣는다.
가드: 같은 종류 평균 쪽으로 shrink한 뒤 [0, coef_hi × max(V(φ(o)), 0)]로 자른다. 변환 오차는 시뮬레이션 기준치 유지 가치와 비교한다.
"""
import numpy as np
from utils.features import phi, phi_drops
from match.mc_value import mc_value
from match.milp_solve import Policy


# 가드 전 유지 가치 (주문자 배열, 공급자 배열)와 현재 상태 가치
def raw(model, o):
  v = model.predict(phi(o)[None])[0]
  c = v - model.predict(phi_drops(o))
  return c[:len(o.bid)], c[len(o.bid):], v


# MILP용 유지 가치: 종류별 평균 쪽 shrink(κ = shrink) 후 [0, coef_hi × max(V, 0)]로 자른다
def coefs(cfg, model, o):
  cb, cs, v = raw(model, o)
  guard = lambda c: np.clip((1 - cfg.shrink) * c + cfg.shrink * c.mean(), 0, cfg.coef_hi * max(v, 0)) if len(c) else c
  return guard(cb), guard(cs)


# 시뮬레이션 기준치 유지 가치: rollout R개(None이면 mc_R)의 V_mc(s) − V_mc(s에서 i 제외) 평균과 그 표준오차 (같은 미래 키끼리 짝지음)
# → (주문자 c, 공급자 c, 상태 기준치 배열, 주문자 c 표준오차, 공급자 c 표준오차)
def mc_coefs(env, R=None):
  v = mc_value(env, R=R)
  d = {kind: [v - mc_value(env, (kind, k), R) for k in range(n)] for kind, n in (("buy", len(env.o.bid)), ("sup", len(env.o.sid)))}
  c = {kind: np.array([x.mean() for x in d[kind]]) for kind in d}
  se = {kind: np.array([x.std(ddof=1) / np.sqrt(len(x)) for x in d[kind]]) for kind in d}
  return c["buy"], c["sup"], v, se["buy"], se["sup"]


# 변환 오차: 모델 유지 가치(가드 전)와 기준치 유지 가치의 평균 절대 차이
def conv_err(c, c_mc):
  return float(np.abs(np.asarray(c) - np.asarray(c_mc)).mean())


# 가치 근사 정책: 같은 MILP에 가치 모델의 유지 가치(가드 후)를 넣는다 (잉여는 자유, 학습용 반응 = Cfg 로지스틱)
def vfa_policy(cfg, model):
  return Policy(cfg, coef=lambda o: coefs(cfg, model, o))
