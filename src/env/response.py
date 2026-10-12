"""재참여 반응 함수.
legacy: 잉여율(배분 잉여 / 제안 금액)의 로지스틱이며 잉여 0이면 ret_p0다. v2: 주문 성립·거래 배정 경험이 잉여율과 따로 들어간다.
평가용(Env)과 학습용(Policy의 MILP) 반응 함수를 따로 넘겨 지정하고, 생략하면 response(cfg)가 시장 버전에 맞춰 고른다.
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


class Retention:
  """경험 의존 재참여 p = σ(β0 + β2 · e + β1 · s / w) (조달 시장 v2).
  e는 경험 지표 (주문자: 주문 성립, 공급자: 성립 주문에서 거래 배정을 받음), s / w는 잉여율 (분모는 legacy와 같은 제안 금액).
  β는 세 점 (e = 0, s = 0) → r2_p0, (e = 1, s = 0) → r2_p1, (e = 1, s / w = r2_rate) → r2_p2로 정한다 (주문자·공급자 공통).
  """

  def __init__(self, cfg):
    lg = lambda p: np.log(p / (1 - p))
    self.a = lg(cfg.r2_p0)
    self.e = lg(cfg.r2_p1) - lg(cfg.r2_p0)
    self.b = (lg(cfg.r2_p2) - lg(cfg.r2_p1)) / cfg.r2_rate

  # 종류 kind 참여자의 재참여 확률 (e를 생략하면 경험 있음)
  def prob(self, kind, s, w, e=1.0):
    return 1 / (1 + np.exp(-(self.a + self.e * np.asarray(e, float) + self.b * np.asarray(s) / w)))

  # 경험 있는 참여자가 재참여 확률 p가 되는 잉여율 (MILP 꺾인 점)
  def rate(self, kind, p):
    return (np.log(p / (1 - p)) - self.a - self.e) / self.b

  # 잉여 0에서 경험 유무에 따른 재참여 확률 차이 (MILP의 성립·배정 항)
  def gap(self, kind):
    return float(self.prob(kind, 0.0, 1.0, 1.0) - self.prob(kind, 0.0, 1.0, 0.0))


class HistRetention(Retention):
  """이력 의존 재참여 p = σ(β0 + β2 · e + β1 · s / w + β3 · (h − h0)) (조달 시장 v2 + 거래 이력).
  h는 참여자의 누적 경험 (주문자 성립 이력 hB, 공급자 배정 이력 hS), β3 = h_w · β2. h = h0이면 Retention과 같다.
  h를 생략하면 h0으로 본다. MILP는 hist(o, kind)로 참여자별 h를 읽어 참여자별 꺾인 점과 gap을 쓴다.
  """

  def __init__(self, cfg):
    super().__init__(cfg)
    self.w, self.h0 = cfg.h_w * self.e, cfg.h0

  # 종류 kind 참여자의 재참여 확률
  def prob(self, kind, s, w, e=1.0, h=None):
    d = 0.0 if h is None else self.w * (np.asarray(h, float) - self.h0)
    return 1 / (1 + np.exp(-(self.a + d + self.e * np.asarray(e, float) + self.b * np.asarray(s) / w)))

  # 경험 있는 참여자(이력 h)가 재참여 확률 p가 되는 잉여율
  def rate(self, kind, p, h=None):
    d = 0.0 if h is None else self.w * (h - self.h0)
    return (np.log(p / (1 - p)) - self.a - d - self.e) / self.b

  # 잉여 0에서 경험 유무에 따른 재참여 확률 차이 (이력 h)
  def gap(self, kind, h=None):
    return float(self.prob(kind, 0.0, 1.0, 1.0, h) - self.prob(kind, 0.0, 1.0, 0.0, h))

  # 관측의 참여자별 이력
  def hist(self, o, kind):
    return o.hb if kind == "buy" else o.hs


# 시장 버전에 맞는 기본 반응 함수 (legacy = Logistic, v2 = Retention, v2 + 이력 = HistRetention)
def response(cfg):
  if cfg.market != "v2":
    return Logistic(cfg)
  return HistRetention(cfg) if cfg.hist else Retention(cfg)
