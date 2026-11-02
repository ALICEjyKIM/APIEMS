"""재참여 확률 함수 (ref/design.md 4장).
p = σ(β0 + β2 e + β1 x + β3 (h − h0) + e G(x − r) − δ u): 거래 경험 e, 잉여율 x, 이력 h, 기준점 r, padding 제외 u.
근거: Popescu & Wu 2007, Aflaki & Popescu 2014, Bolton 1998, Brown et al. 2024, Backus et al. 2015.
"""
import numpy as np


# logit
def lg(p):
  return np.log(p / (1 - p))


class Ret:
  """재참여 확률 계산기.
  β0, β2, β1은 세 점 (r2_p0, r2_p1, r2_p2)과 기준점 잉여율(주문자 r2_rate_buy, 공급자 r2_rate_sup)로 정하고, β3 = h_w β2.
  G(d)는 기준점 대비 이득 η β1 d, 손실 max(η λ β1 d, −β2). 기준점이 없으면(nan) G 항은 0, δ = logit(0.5) − logit(ex_p).
  """

  def __init__(self, cfg):
    self.cfg = cfg
    self.b0, self.b2 = lg(cfg.r2_p0), lg(cfg.r2_p1) - lg(cfg.r2_p0)
    self.b1 = {"buy": (lg(cfg.r2_p2) - lg(cfg.r2_p1)) / cfg.r2_rate_buy, "sup": (lg(cfg.r2_p2) - lg(cfg.r2_p1)) / cfg.r2_rate_sup}
    self.b3, self.d = cfg.h_w * self.b2, -lg(cfg.ex_p)

  # 종류 kind 참여자의 재참여 확률 (배열 가능)
  def prob(self, kind, e, x, h, r, u):
    c, b1 = self.cfg, self.b1[kind]
    e, d = np.asarray(e, float), np.nan_to_num(np.asarray(x, float) - r)
    G = np.where(d >= 0, c.eta * b1 * d, np.maximum(c.eta * c.lam_loss * b1 * d, -self.b2))
    z = self.b0 + self.b2 * e + b1 * x + self.b3 * (np.asarray(h) - c.h0) + e * G - self.d * np.asarray(u, float)
    return 1 / (1 + np.exp(-z))
