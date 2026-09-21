"""정책 평가 경로.
rollout은 한 반복 동안 정책과 환경을 번갈아 돌려 누적 이윤과 성과 지표를 모은다.
공통 난수를 쓰므로 같은 rep에서는 정책이 달라도 같은 도착·변동을 본다.
"""
from env.platform import Env

KEYS = ("profit", "rev", "ship", "opp", "n_ok", "n_buy", "stay_buy", "n_sup", "stay_sup")


# 한 반복 rollout → 누적 이윤·이윤 구성요소, 주문 충족률, 참여자 유지율·평균 활동 참여자 수(주문자·공급자)
# resp는 평가용 반응 함수 (None이면 Cfg의 로지스틱)
def rollout(cfg, policy, rep, resp=None):
  env = Env(cfg, rep, resp)
  env.reset()
  policy.reset(rep)
  s = dict.fromkeys(KEYS, 0)
  for _ in range(cfg.T):
    r = env.step(policy.act(env.o))
    policy.observe(r)
    for k in KEYS:
      s[k] += r[k]
  return dict(profit=s["profit"], rev=s["rev"], ship=s["ship"], opp=s["opp"], fill=s["n_ok"] / s["n_buy"],
              ret_buy=s["stay_buy"] / s["n_buy"], ret_sup=s["stay_sup"] / s["n_sup"],
              act_buy=s["n_buy"] / cfg.T, act_sup=s["n_sup"] / cfg.T)
