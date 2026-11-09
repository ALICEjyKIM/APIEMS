"""상태별 이후 손실 점검의 요약 계산 확인.
이후 손실이 주문자 수에 비례하는 가짜 상태에서 반분 신뢰도 · 교차 검증 · 결정이 그 차이를 잡는지 본다.
MILP 없이 summ만 검사한다.
"""
import numpy as np
from result.check_state import summ, COM


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


# 상태마다 다른 이후 손실을 피처 회귀가 시점 표보다 잘 맞히고 더 나은 m을 고른다
def test_summ_detects_state_signal():
  out = summ(fake())
  for rel, pred in zip(out["rel"], out["pred"]):
    assert rel["res"]["mean"] > 0.8 and rel["res"]["mean"] - rel["res"]["ci"] > 0
    assert pred["r2_feat"] > pred["r2_time"] + 0.5 and pred["gain"]["mean"] - pred["gain"]["ci"] > 0
    assert pred["coef"]["nb"] < 0 and pred["same_sign"]["nb"] == 10
  d = out["dec"]["saa"]
  assert d["feat_vs_table"]["mean"] - d["feat_vs_table"]["ci"] > 0
