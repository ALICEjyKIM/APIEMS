"""기간별 도착과 재참여 주문·공급 생성.
신규 도착 난수는 기간(과 공급자 자리)으로, 재참여 변동 난수는 참여자·기간으로 키를 잡는다.
그래서 정책이 달라도 같은 기간의 도착과 같은 참여자의 변동은 같다 (공통 난수).
"""
import numpy as np
from utils.common import rng, KINDS
from utils.instance import Prof, new_buy, new_sup


# 기간 t의 신규 주문자 (Poisson 도착)
def new_buyers(cfg, inst, rep, t):
  g = rng(cfg, rep, "arr", 0, t)
  return new_buy(cfg, inst, g, g.poisson(cfg.lam_buy))


# 기간 t에 빈 공급자 자리 j를 확률 p_sup_new로 새 공급자가 채운다
def new_sups(cfg, inst, rep, t, vacant):
  out = {}
  for j in vacant:
    g = rng(cfg, rep, "arr", 1, t, j)
    if g.random() < cfg.p_sup_new:
      out[j] = new_sup(cfg, inst, g, j)
  return out


# 재참여 참여자의 기간 t 주문·공급: 프로필 수량·가격에 작은 상대 변동을 더한다
def perturb(cfg, rep, kind, pid, t, prof):
  g = rng(cfg, rep, "noise", KINDS[kind], pid, t)
  n = len(prof.items)
  qty = np.maximum(1, np.round(prof.qty * (1 + cfg.noise_qty * g.standard_normal(n)))).astype(int)
  price = prof.price * (1 + cfg.noise_price * g.standard_normal(n))
  return Prof(prof.items.copy(), qty * prof.items, price * prof.items)
