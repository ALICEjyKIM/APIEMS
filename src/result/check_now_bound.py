"""이번 기간 이득 추가 이득 상한 (ref/decisions.md "이번 기간 이득 추가 이득 상한"): 한 번 올린 대가 점검 ×2의 250개 상태에서 기간별 평균 이후 손실 반영 정책의 예상 이번 기간 이득을 실제 이번 기간 이득으로 바꿔 고른 m의 끝까지 차이를 잰다.
이후 손실 표는 반복 하나씩 빼고 만든 기간 구간 평균(교차 검증)을 모든 비교에 똑같이 쓴다. 비교: 실제 이득 + 표, 시나리오 예측 + 표, 평균 예측 + 표, 실제 이득만(참고), 고정 padding 정책(0).
결과 result/runs/check_now_bound_x2.json. 실행: src/에서 python -m result.check_now_bound
"""
import json
import numpy as np
from result.check_dev_ci import cci
from result.exp_long_x2 import BINS


# 기간 구간별 평균 이후 손실 (반복 하나씩 빼고 계산해 그 반복 상태에 붙인다)
def cv_table(L, b, g):
  out = np.zeros_like(L)
  for k in np.unique(g):
    for j in np.unique(b):
      out[(g == k) & (b == j)] = L[(g != k) & (b == j)].mean(0)
  return out


def main():
  rows = json.load(open("result/runs/check_dev_x2.json"))["rows"]
  g, t = np.array([r["rep"] for r in rows]), np.array([r["t"] for r in rows])
  b = np.searchsorted(BINS, t, side="right") - 1
  I, R = np.array([r["I"] for r in rows]), np.array([r["R"] for r in rows])
  I, R = I - I[:, :1], R - R[:, :, :1]
  D = R.mean(1)
  Lt = cv_table(D - I, b, g)
  Gs, Gm = (np.array([r[k] for r in rows]) for k in ("G_saa", "G_mean"))
  Gs, Gm = Gs - Gs[:, :1], Gm - Gm[:, :1]
  a = np.arange(len(rows))
  ch = {k: v.argmax(1) for k, v in dict(now=I + Lt, saa=Gs + Lt, mean=Gm + Lt, now_only=I).items()}
  out = {k: dict(D=cci(D[a, c], g), dev=float((c > 0).mean()), m_count=np.bincount(c, minlength=D.shape[1]).tolist()) for k, c in ch.items()}
  out["now_vs_saa"], out["now_vs_mean"] = cci(D[a, ch["now"]] - D[a, ch["saa"]], g), cci(D[a, ch["now"]] - D[a, ch["mean"]], g)
  out["corr_saa"] = [float(np.corrcoef(Gs[:, i], I[:, i])[0, 1]) for i in range(1, D.shape[1])]
  out["corr_mean"] = [float(np.corrcoef(Gm[:, i], I[:, i])[0, 1]) for i in range(1, D.shape[1])]
  json.dump(dict(n=len(rows), bins=BINS, **out), open("result/runs/check_now_bound_x2.json", "w"), indent=1)
  print(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
