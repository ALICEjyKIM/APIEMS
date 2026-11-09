"""상태 신호 점검의 반분 신뢰도 계산 확인.
상태마다 다른 이후 손실이 있으면 신뢰도가 높고, 미래 잡음뿐이면 상관이 0 근처인지 본다.
MILP 없이 rel만 검사한다.
"""
import numpy as np
from result.check_rel import rel


# 가짜 상태 (제외 연쇄 점검 형식): L_m = s × 상태값 × m번호 + 미래 잡음, I는 미래 8개에 같은 값
def fake(s):
  r, rows = np.random.default_rng(0), []
  for rep in range(10):
    for t in range(5, 30):
      I = np.tile(r.normal(0, 5, 4), (8, 1))
      R = I + s * r.normal() * np.arange(4) + r.normal(0, 5, (8, 4))
      rows.append(dict(rep=rep, t=t, I=I.tolist(), R=R.tolist()))
  return rows


# 상태 신호가 있으면 신뢰도가 높고 필요한 미래 수가 적으며, 잡음뿐이면 상관이 0 근처
def test_rel_signal_and_noise():
  for x in rel(fake(20.0)):
    assert x["res"]["mean"] - x["res"]["ci"] > 0.8 and x["sb"] > 0.9 and x["need"] < 1
  for x in rel(fake(0.0)):
    assert abs(x["res"]["mean"]) < 0.2
