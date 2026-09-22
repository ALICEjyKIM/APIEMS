"""선형 가치 근사 (기존 ADP).
시장 요약 지표를 표준화한 뒤 릿지 회귀로 V(s) = w · z + b를 맞춘다.
λ는 lin_lams 중 공통 검증 분할(vfa/train.val_mask)의 평균제곱오차가 가장 작은 값을 고른다 (튜닝 예산 = 설정 수).
"""
import numpy as np
from vfa.train import val_mask


class Linear:
  """릿지 회귀 가치 근사.
  fit은 학습 데이터 평균·표준편차로 지표를 표준화하고(표준편차 0인 지표는 그대로) [Z; √λ I] 최소제곱을 푼다.
  predict는 지표 행렬(행 = 상태)을 받아 가치 배열을 돌려준다.
  """

  def __init__(self, lam):
    self.lam = lam

  # 학습: 절편은 목표 평균, 가중치는 릿지 최소제곱 (λ = 0이면 최소노름 최소제곱)
  def fit(self, X, y):
    self.mu, self.sd = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1.0)
    Z = (X - self.mu) / self.sd
    self.b = y.mean()
    A = np.r_[Z, np.sqrt(self.lam) * np.eye(Z.shape[1])]
    self.w = np.linalg.lstsq(A, np.r_[y - self.b, np.zeros(Z.shape[1])], rcond=None)[0]
    return self

  # 가치 예측
  def predict(self, X):
    return (X - self.mu) / self.sd @ self.w + self.b


# λ 선택: lin_lams 각각을 검증 분할로 재고(검증 평균제곱오차), 가장 작은 λ로 전체 데이터를 다시 맞춘다
def select(cfg, X, y, g):
  va = val_mask(cfg, g)
  mse = {lam: float(np.mean((Linear(lam).fit(X[~va], y[~va]).predict(X[va]) - y[va]) ** 2)) for lam in cfg.lin_lams}
  lam = min(mse, key=mse.get)
  return Linear(lam).fit(X, y), mse
