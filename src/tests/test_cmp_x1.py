"""×1 이후 손실 추정 방법 비교의 상태별 지표 점검.
손으로 계산 가능한 작은 예: 완벽한 예측, 순서가 뒤집힌 예측, 값이 하나뿐인 예측.
"""
import numpy as np
from result.exp.cmp_x1 import per


# 정답을 그대로 맞히면 적중 1, regret 0, 제곱오차 0, 순위 상관 1
def test_perfect():
  Y = np.array([[0.0, 5.0, -3.0, 1.0], [0.0, -2.0, -4.0, -1.0]])
  p = per(Y.copy(), Y)
  assert np.allclose(p["hit"], 1) and np.allclose(p["regret"], 0) and np.allclose(p["se"], 0) and np.allclose(p["rho"], 1)


# 순서를 뒤집으면 순위 상관 −1, 가장 나쁜 후보를 골라 regret = 최대 − 최소
def test_reversed():
  Y = np.array([[0.0, 5.0, -3.0, 1.0]])
  p = per(-Y, Y)
  assert np.isclose(p["rho"][0], -1) and np.isclose(p["regret"][0], 8) and p["hit"][0] == 0


# 예측이 모두 같으면 순위 상관은 nan, 첫 후보(m = 1)를 고른다
def test_const():
  Y = np.array([[0.0, 5.0, -3.0, 1.0]])
  p = per(np.zeros((1, 4)), Y)
  assert np.isnan(p["rho"][0]) and p["pick"][0] == 0 and np.isclose(p["se"][0], 25 + 9 + 1)
