"""시뮬레이션 기준치.
환경 상태를 복사하고 미래 난수 키만 바꾼 mc_R개 rollout을 후속 정책(규칙 기반, Cfg 몫)으로 vf_H기간 돌려 이윤 합을 잰다.
같은 상태와 참여자 하나를 뺀 상태는 같은 미래 키를 써서 공통 난수로 비교한다.
"""
import copy
import numpy as np
from bench.rule_split import rule


# 미래 키: 반복 rep의 기간 t 상태에서 k번째 rollout (평가·튜닝 반복 번호와 겹치지 않음)
def fkey(cfg, rep, t, k):
  return cfg.mc_rep0 + (rep * cfg.T + t) * cfg.mc_R + k


# 환경 env의 현재 상태에서 mc_R개 미래 각각의 후속 정책 vf_H기간 이윤 합 (drop = (종류, 행)이면 그 참여자를 뺀 상태에서). env는 바꾸지 않는다
def mc_value(env, drop=None):
  cfg, out = env.cfg, []
  for k in range(cfg.mc_R):
    e = copy.deepcopy(env)
    e.rep = fkey(cfg, env.rep, env.t, k)
    if drop:
      e.drop(*drop)
    pol = rule(cfg)
    out.append(sum(e.step(pol.act(e.o))["profit"] for _ in range(cfg.vf_H)))
  return np.array(out)
