"""Q̂ 근사 테스트: ridge가 선형 함수를 되찾는지, MLP가 학습되는지, Q 객체가 pickle로 넘어가는지, 검증 점수의 정의.
자료는 난수로 만든 작은 행렬이다.
시뮬레이터는 쓰지 않는다.
"""
import pickle
from dataclasses import replace
import numpy as np
import pytest
from utils.params import Cfg
from value.fit import QT, scaler, ridge, mlp, score, target

CFG = replace(Cfg(), mlp_epochs=30)


# ridge(λ 작음)는 선형 함수를 정확히 되찾고, MLP는 평균 예측보다 오차가 작다, Q는 pickle 왕복 뒤 같은 값을 낸다
def test_fit():
  g = np.random.default_rng(0)
  X = g.normal(size=(500, 5)) * [1, 10, 100, 1, 0]
  Y = 3 + X @ [1.0, -0.2, 0.01, 2.0, 0.0]
  mu, sd = scaler(X)
  q = ridge(X, Y, mu, sd, Y.std(), 1e-9)
  assert q(X) == pytest.approx(Y, abs=1e-6)
  n = mlp(CFG, X, Y, mu, sd, Y.std(), 2, 16, 3e-3)
  assert ((n(X) - Y) ** 2).mean() < 0.1 * Y.var()
  assert np.array_equal(pickle.loads(pickle.dumps(n))(X), n(X))


# 검증 점수: 상태마다 점수 최대 후보(동점은 앞)의 실제 A 평균
def test_score():
  Y = np.array([[0.0, 5.0, -1.0], [0.0, -2.0, 3.0]])
  v = np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
  assert score(v, Y)["val"] == pytest.approx((5.0 + 0.0) / 2)


# TD 목표: 마지막 기간은 Π, 그 앞은 Π + 다음 상태 후보 중 최대 Q̂. QT는 첫 열 t/T로 기간의 Q̂를 고른다
def test_td():
  g = np.random.default_rng(1)
  P, Xn = g.normal(size=(4, 3)), g.normal(size=(4, 3, 5, 2))
  assert np.array_equal(target(None, P, Xn), P)
  assert target(lambda X: X[:, 1], P, Xn) == pytest.approx(P + Xn[..., 1].max(2))
  q = QT({1: lambda X: np.full(len(X), 1.0), 2: lambda X: np.full(len(X), 2.0)}, 30)
  assert q(np.array([[2 / 30, 0.0]]))[0] == 2.0 and q(np.array([[1 / 30, 0.0]]))[0] == 1.0


# MLP 조기 종료: hold를 주면 학습 자료와 같은 함수에서 오차가 작은 시점을 고른다
def test_mlp_hold():
  g = np.random.default_rng(2)
  X = g.normal(size=(300, 3))
  Y = X @ [1.0, -1.0, 0.5]
  mu, sd = scaler(X)
  n = mlp(replace(CFG, mlp_epochs=300), X[:200], Y[:200], mu, sd, Y.std(), 2, 16, 3e-3, hold=(X[200:], Y[200:]))
  assert ((n(X[200:]) - Y[200:]) ** 2).mean() < 0.1 * Y.var()
