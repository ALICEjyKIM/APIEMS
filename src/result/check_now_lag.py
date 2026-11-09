"""지난 기간 이득 피처 점검 (ref/decisions.md "지난 기간 이득 피처 점검"): 이번 기간 이득 예측 점검에 지난 기간의 실제 이득(지난 기간 입찰로 m별 청산을 다시 푼 몫 차이)을 피처로 더한다.
지난 기간 값이 있는 t ≥ 6 상태에서 m마다 반복 하나씩 빼는 교차 검증. 예측: 시나리오 그대로, 선형 보정, 시나리오 + lag1, 시나리오 + 최근 3기간 평균(lag3), 피처 회귀 + lag1(참고).
결과 result/runs/check_now_lag_x2.json. 실행: src/에서 python -m result.check_now_lag
"""
import json
import numpy as np
from result.check_dev_ci import cci
from result.check_now_bound import cv_table
from result.check_now_pred import cv_fit, score
from result.check_state import COM
from result.exp_long_x2 import BINS


def main():
  rows = json.load(open("result/runs/check_state_x2.json"))["rows"]
  g, t = np.array([r["rep"] for r in rows]), np.array([r["t"] for r in rows])
  b = np.searchsorted(BINS, t, side="right") - 1
  I, R = np.array([r["I"] for r in rows]), np.array([r["R"] for r in rows])
  I, R = I - I[:, :1], R - R[:, :, :1]
  D = R.mean(1)
  Lt = cv_table(D - I, b, g)
  Gs = np.array([r["G_saa"] for r in rows])
  Gs = Gs - Gs[:, :1]
  ex = np.array([r["ex"] for r in rows], float)
  ex = ex - ex[:, :1]
  C = np.array([[r[k] for k in COM] for r in rows], float)
  at = {(r["rep"], r["t"]): i for i, r in enumerate(rows)}
  keep = np.array([(r["rep"], r["t"] - 1) in at for r in rows])
  lag1 = np.array([I[at[(r["rep"], r["t"] - 1)]] if k else I[i] * np.nan for i, (r, k) in enumerate(zip(rows, keep))])
  lag3 = np.array([np.mean([I[at[(r["rep"], s)]] for s in range(r["t"] - 3, r["t"]) if (r["rep"], s) in at], 0) if k else I[i] * np.nan
                   for i, (r, k) in enumerate(zip(rows, keep))])
  g, b, I, D, Lt, Gs, ex, C, lag1, lag3 = (x[keep] for x in (g, b, I, D, Lt, Gs, ex, C, lag1, lag3))
  B = np.eye(len(BINS))[b]
  n, M = I.shape
  one = np.ones((n, 1))
  P = {k: np.zeros((n, M)) for k in ("cal", "lag1", "lag3", "reg_lag")}
  pred = []
  for i in range(1, M):
    P["cal"][:, i] = cv_fit(np.c_[one, Gs[:, i]], I[:, i], g)
    P["lag1"][:, i] = cv_fit(np.c_[one, Gs[:, i], lag1[:, i]], I[:, i], g)
    P["lag3"][:, i] = cv_fit(np.c_[one, Gs[:, i], lag3[:, i]], I[:, i], g)
    P["reg_lag"][:, i] = cv_fit(np.c_[B, Gs[:, i], C, ex[:, i], lag1[:, i]], I[:, i], g)
    e = lambda p: (I[:, i] - p) ** 2
    pred.append(dict(saa=score(Gs[:, i], I[:, i]), **{k: score(P[k][:, i], I[:, i]) for k in P},
                     corr_lag1=float(np.corrcoef(lag1[:, i], I[:, i])[0, 1]), corr_lag3=float(np.corrcoef(lag3[:, i], I[:, i])[0, 1]),
                     gain_lag1=cci(e(P["cal"][:, i]) - e(P["lag1"][:, i]), g), gain_lag3=cci(e(P["cal"][:, i]) - e(P["lag3"][:, i]), g)))
  a = np.arange(n)
  ch = {k: (v + Lt).argmax(1) for k, v in dict(saa=Gs, now=I, **P).items()}
  dec = {k: dict(D=cci(D[a, c], g), dev=float((c > 0).mean())) for k, c in ch.items()}
  for k in ("lag1", "lag3", "reg_lag"):
    dec[k + "_vs_saa"] = cci(D[a, ch[k]] - D[a, ch["saa"]], g)
    dec[k + "_closed"] = float((dec[k]["D"]["mean"] - dec["saa"]["D"]["mean"]) / (dec["now"]["D"]["mean"] - dec["saa"]["D"]["mean"]))
  out = dict(n=n, pred=pred, dec=dec)
  json.dump(out, open("result/runs/check_now_lag_x2.json", "w"), indent=1)
  print(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
