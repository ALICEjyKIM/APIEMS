"""상태 신호 점검 (ref/decisions.md "상태 신호 점검 (작은 시장)"): 제외 연쇄 점검 결과에서 상태별 이후 손실 L이 상태에 따라 정해지는지 반분 신뢰도로 본다.
m마다 미래 앞 절반 · 뒤 절반으로 잰 L의 상관(기간 구간 평균을 뺀 값 · 빼지 않은 값), 미래 전체 평균의 신뢰도(Spearman-Brown), 신뢰도 0.5에 필요한 미래 수, 기준 시장 대비 상관 차이.
결과 result/runs/check_rel_x1.json. 실행: src/에서 python -m result.check_rel base het1.0 ... (태그마다 check_chain_x1_{태그}.json, 첫 태그가 기준)
"""
import json
import sys
import numpy as np
from utils.params import Cfg
from result.check_state import bm, jack
from result.exp_long_x2 import BINS


# m별 반분 상관(raw · res), 미래 전체 평균의 신뢰도 sb, 신뢰도 0.5에 필요한 미래 수 need
def rel(rows):
  g, t = np.array([r["rep"] for r in rows]), np.array([r["t"] for r in rows])
  b = np.searchsorted(BINS, t, side="right") - 1
  I, R = np.array([r["I"] for r in rows]), np.array([r["R"] for r in rows])
  I = I.mean(1) if I.ndim == 3 else I  # 제외 연쇄 점검은 I를 미래마다 같은 값으로 저장
  I, R = I - I[:, :1], R - R[:, :, :1]
  h, out = R.shape[1] // 2, []
  La, Lb = R[:, :h].mean(1) - I, R[:, h:].mean(1) - I
  for i in range(1, R.shape[2]):
    rc = lambda k, res: np.corrcoef(*((x[k] - bm(x[k], b[k])) if res else x[k] for x in (La[:, i], Lb[:, i])))[0, 1]
    raw, res = jack(lambda k: rc(k, False), g), jack(lambda k: rc(k, True), g)
    sb = 2 * res["mean"] / (1 + res["mean"])
    out.append(dict(raw=raw, res=res, sb=float(sb), need=float(R.shape[1] * (1 - sb) / sb) if sb > 0 else None))
  return out


def main():
  tags = sys.argv[1:]
  R = {k: rel(json.load(open(f"result/runs/check_chain_x1_{k}.json"))["rows"]) for k in tags}
  for k in tags[1:]:
    for a, z in zip(R[k], R[tags[0]]):
      a["vs_base"] = dict(mean=a["res"]["mean"] - z["res"]["mean"], ci=float(np.hypot(a["res"]["ci"], z["res"]["ci"])))
  out = dict(base=tags[0], ms=list(Cfg().m_grid[1:]), rel=R)
  json.dump(out, open("result/runs/check_rel_x1.json", "w"), indent=1)
  print(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
