"""이번 기간 이득 예측 점검 (ref/decisions.md "이번 기간 이득 예측 점검"): 250개 상태(check_state_x2.json)에서 입찰 전 정보로 실제 이번 기간 이득을 지금 예측보다 잘 맞히는지, 그 예측으로 고른 m이 더 버는지 본다.
m = 1.1 · 1.25 · 1.5마다 따로 반복 하나씩 빼는 교차 검증. 비교: 지금 예측(시나리오 · 평균 그대로), 시나리오 선형 보정, 피처 회귀, 실제 이득(상한).
결과 result/runs/check_now_pred_x2.json. 실행: src/에서 python -m result.check_now_pred
"""
import json
import numpy as np
from result.check_dev_ci import cci
from result.check_now_bound import cv_table
from result.check_state import COM, ls
from result.exp_long_x2 import BINS


# 반복 하나씩 빼고 X로 y를 맞춘 예측
def cv_fit(X, y, g):
  p = np.zeros(len(y))
  for k in np.unique(g):
    p[g == k] = X[g == k] @ ls(X[g != k], y[g != k])
  return p


# 예측 지표: 상관, 교차 검증 R²
def score(p, y):
  return dict(corr=float(np.corrcoef(p, y)[0, 1]), r2=float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()))


def summ(rows):
  g, t = np.array([r["rep"] for r in rows]), np.array([r["t"] for r in rows])
  b = np.searchsorted(BINS, t, side="right") - 1
  B = np.eye(len(BINS))[b]
  I, R = np.array([r["I"] for r in rows]), np.array([r["R"] for r in rows])
  I, R = I - I[:, :1], R - R[:, :, :1]
  D = R.mean(1)
  Lt = cv_table(D - I, b, g)
  Gs, Gm = (np.array([r[k] for r in rows]) for k in ("G_saa", "G_mean"))
  Gs, Gm = Gs - Gs[:, :1], Gm - Gm[:, :1]
  ex = np.array([r["ex"] for r in rows], float)
  ex = ex - ex[:, :1]
  C = np.array([[r[k] for k in COM] for r in rows], float)
  n, M = I.shape
  P = {k: np.zeros((n, M)) for k in ("cal", "reg")}
  pred = []
  for i in range(1, M):
    one = np.ones((n, 1))
    P["cal"][:, i] = cv_fit(np.c_[one, Gs[:, i]], I[:, i], g)
    P["reg"][:, i] = cv_fit(np.c_[B, Gs[:, i], Gm[:, i], C, ex[:, i]], I[:, i], g)
    e = lambda p: (I[:, i] - p) ** 2
    pred.append(dict(saa=score(Gs[:, i], I[:, i]), mean=score(Gm[:, i], I[:, i]), cal=score(P["cal"][:, i], I[:, i]), reg=score(P["reg"][:, i], I[:, i]),
                     gain=cci(e(P["cal"][:, i]) - e(P["reg"][:, i]), g)))
  a = np.arange(n)
  ch = {k: (v + Lt).argmax(1) for k, v in dict(saa=Gs, mean=Gm, cal=P["cal"], reg=P["reg"], now=I).items()}
  dec = {k: dict(D=cci(D[a, c], g), dev=float((c > 0).mean())) for k, c in ch.items()}
  dec["reg_vs_saa"], dec["reg_vs_cal"] = cci(D[a, ch["reg"]] - D[a, ch["saa"]], g), cci(D[a, ch["reg"]] - D[a, ch["cal"]], g)
  dec["closed"] = float((dec["reg"]["D"]["mean"] - dec["saa"]["D"]["mean"]) / (dec["now"]["D"]["mean"] - dec["saa"]["D"]["mean"]))
  return dict(n=n, pred=pred, dec=dec)


def main():
  out = summ(json.load(open("result/runs/check_state_x2.json"))["rows"])
  json.dump(out, open("result/runs/check_now_pred_x2.json", "w"), indent=1)
  print(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
