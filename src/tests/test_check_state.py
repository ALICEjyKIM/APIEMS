"""상태별 이후 손실 점검(result.check.state)의 요약 계산 확인.
summ: 이후 손실이 주문자 수에 비례하는 가짜 상태에서 반분 신뢰도 · 교차 검증 · 결정이 그 차이를 잡는지. rel: 상태마다 다른 이후 손실이 있으면 신뢰도가 높고, 미래 잡음뿐이면 상관이 0 근처인지.
MILP 없이 summ과 rel만 검사한다.
"""
import numpy as np
from result.check.state import summ, rel
from result.common import COM


# 가짜 상태: L_m = −4 × m번호 × (주문자 수 − 10) + 잡음, 예상 이번 기간 이득은 m번호 × 40 (주문자 20명 미만이면 올리는 게 이득)
def fake():
  r, rows = np.random.default_rng(0), []
  for rep in range(10):
    for t in range(5, 30):
      nb = r.integers(10, 31)
      Lm = -4.0 * np.arange(4) * (nb - 10)
      I = r.normal(0, 5, 4)
      R = I + Lm + r.normal(0, 5, (8, 4))
      f = {k: float(r.normal()) for k in COM}
      f["nb"] = float(nb)
      rows.append(dict(rep=rep, t=t, I=I.tolist(), R=R.tolist(), G_saa=[0.0, 40.0, 80.0, 120.0], G_mean=[0.0, 40.0, 80.0, 120.0], ex=[0, 1, 2, 3], **f))
  return rows


# 가짜 상태 (제외 연쇄 점검 형식): L_m = s × 상태값 × m번호 + 미래 잡음, I는 미래 8개에 같은 값
def fake_chain(s):
  r, rows = np.random.default_rng(0), []
  for rep in range(10):
    for t in range(5, 30):
      I = np.tile(r.normal(0, 5, 4), (8, 1))
      R = I + s * r.normal() * np.arange(4) + r.normal(0, 5, (8, 4))
      rows.append(dict(rep=rep, t=t, I=I.tolist(), R=R.tolist()))
  return rows


# 상태마다 다른 이후 손실을 피처 회귀가 시점 표보다 잘 맞히고 더 나은 m을 고른다
def test_summ_detects_state_signal():
  out = summ(fake())
  for rl, pred in zip(out["rel"], out["pred"]):
    assert rl["res"]["mean"] > 0.8 and rl["res"]["mean"] - rl["res"]["ci"] > 0
    assert pred["r2_feat"] > pred["r2_time"] + 0.5 and pred["gain"]["mean"] - pred["gain"]["ci"] > 0
    assert pred["coef"]["nb"] < 0 and pred["same_sign"]["nb"] == 10
  d = out["dec"]["saa"]
  assert d["feat_vs_table"]["mean"] - d["feat_vs_table"]["ci"] > 0


# 상태 신호가 있으면 신뢰도가 높고 필요한 미래 수가 적으며, 잡음뿐이면 상관이 0 근처
def test_rel_signal_and_noise():
  for x in rel(fake_chain(20.0)):
    assert x["res"]["mean"] - x["res"]["ci"] > 0.8 and x["sb"] > 0.9 and x["need"] < 1
  for x in rel(fake_chain(0.0)):
    assert abs(x["res"]["mean"]) < 0.2
