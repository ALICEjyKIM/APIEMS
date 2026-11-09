"""공급자 교체 난수 점검 (ref/decisions.md "공급자 교체 난수 점검"): 같은 상태 · 같은 미래에서 m = 1과 M_HI 두 세계를 끝까지 돌려, 대체 후보 난수 키(time, slot)별로 갈라지는 정도를 잰다.
준비 기간은 기본 키(time)로 돌리고 미래만 키를 바꾼다. 측정: 미래별 끝까지 차이의 표준편차, 두 세계 누적 순이윤의 상관, 마지막에 한쪽에만 있는 공급자 · 주문자 비율.
결과 result/runs/check_crn_x2.json. 실행: src/에서 python -m result.check_crn [Cfg 키=값 ...] (예: n_proc=4)
"""
from copy import deepcopy
from dataclasses import replace
import json
from multiprocessing import Pool
import sys
import numpy as np
from utils.params import Cfg, SCALE
from env.sim import Sim
from result.headroom import fut

REPS, T1, KF, M_HI, KEYS = (1000, 1001, 1002, 1003), 12, 6, 1.5, ("time", "slot")


# 상태 하나(반복 rep, 기간 T1) × 키: 미래 k마다 두 세계의 남은 누적 순이윤과 마지막 참여자 번호
def one(args):
  cfg, rep, key = args
  s = Sim(cfg, rep)
  while s.t < T1:
    s.step(1.0, 0.0)
  out = []
  for k in range(KF):
    w = []
    for m in (1.0, M_HI):
      b = deepcopy(s)
      b.cfg, b.rep = replace(cfg, rep_key=key), fut(rep, k)
      R = b.step(m, 0.0)["Pi"]
      while b.t < cfg.T:
        R += b.step(1.0, 0.0)["Pi"]
      w.append((float(R), set(int(i) for i in b.B), set(int(p["id"]) for p in b.S.values())))
    out.append(dict(R1=w[0][0], R2=w[1][0], sup=len(w[0][2] ^ w[1][2]) / len(w[0][2] | w[1][2]), buy=len(w[0][1] ^ w[1][1]) / len(w[0][1] | w[1][1])))
  return dict(rep=rep, key=key, fut=out)


# 키별 요약: 상태마다 미래 사이 D 표준편차 · 상관 · 갈라진 비율의 평균
def summ(rows):
  out = {}
  for key in KEYS:
    st = []
    for r in (x for x in rows if x["key"] == key):
      R1, R2 = np.array([f["R1"] for f in r["fut"]]), np.array([f["R2"] for f in r["fut"]])
      st.append(dict(rep=r["rep"], sd_D=float((R2 - R1).std(ddof=1)), sd_R1=float(R1.std(ddof=1)), corr=float(np.corrcoef(R1, R2)[0, 1]),
                     D=float((R2 - R1).mean()), sup=float(np.mean([f["sup"] for f in r["fut"]])), buy=float(np.mean([f["buy"] for f in r["fut"]]))))
    out[key] = dict(states=st, **{k: float(np.mean([x[k] for x in st])) for k in ("sd_D", "sd_R1", "corr", "D", "sup", "buy")})
  out["ratio_sd_D"] = out["slot"]["sd_D"] / out["time"]["sd_D"]
  return out


def main():
  ov = dict(a.split("=") for a in sys.argv[1:])
  cfg = replace(Cfg(), **SCALE["x2"])
  cfg = replace(cfg, **{k: type(getattr(cfg, k))(v) for k, v in ov.items()})
  with Pool(cfg.n_proc) as p:
    rows = p.map(one, [(cfg, r, key) for r in REPS for key in KEYS])
  out = summ(rows)
  json.dump(dict(cfg=repr(cfg), reps=REPS, t=T1, K=KF, m_hi=M_HI, **out, rows=rows), open("result/runs/check_crn_x2.json", "w"), indent=1)
  print(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
