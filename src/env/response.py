"""재참여 반응 함수.
재참여 확률은 잉여율(배분 잉여 / 제안 금액)의 로지스틱이며, 잉여 0이면 ret_p0다.
평가용(Env)과 학습용(Policy의 MILP) 반응 함수를 따로 넘겨 지정한다.
"""
import numpy as np


class Logistic:
  """로지스틱 반응 p = σ(a + b_kind · s / w).
  a = logit(ret_p0)는 주문자·공급자 공통, 기울기 b는 종류별 (Cfg의 b_buy, b_sup).
  s는 배분 잉여, w는 제안 금액 (주문자 Σ 제안가격×수량, 공급자 Σ 공급가격×공급가능량), 배열 가능.
  """

  def __init__(self, cfg):
    self.a = np.log(cfg.ret_p0 / (1 - cfg.ret_p0))
    self.b = {"buy": cfg.b_buy, "sup": cfg.b_sup}

  # 종류 kind 참여자의 재참여 확률
  def prob(self, kind, s, w):
    return 1 / (1 + np.exp(-(self.a + self.b[kind] * np.asarray(s) / w)))

  # 재참여 확률 p가 되는 잉여율
  def rate(self, kind, p):
    return (np.log(p / (1 - p)) - self.a) / self.b[kind]
