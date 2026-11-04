"""한 번 올린 대가 점검 (ref/decisions.md "한 번 올린 대가 점검"): 예측 점검과 같은 250개 상태에서 후보 m마다 이번 기간만 m(θ = 0), 이후 고정 padding으로 T까지 돌려 누적 이윤을 미래 난수 8개로 잰다.
이번 기간 입찰은 공통이라 이번 기간 이득 I_m은 확정값이고, 끝까지 차이 D_m에서 I_m을 뺀 L_m이 이후 영향이다.
규칙 평가: 시나리오 예측의 m*를 예상 이득 Δ̂ ≥ τ일 때만 쓰는 규칙, 평균 예측의 m*, 실제 입찰의 m*(상한), D 최대 후보(반분 검증 포함). 결과 result/runs/check_dev[_x2].json. 실행: src/에서 python -m result.check_dev [x2|x4]
"""
from copy import deepcopy
from dataclasses import replace
import json
from multiprocessing import Pool
import sys
import numpy as np
from utils.params import Cfg, SCALE
from env.sim import Sim
from mech.chu import clear
from policy.base import est, scen
from result.check_pred import REPS, T0
from result.headroom import fut, K

TAUS = (0, 5, 10, 15, 20, 30, 40, 50)


# 상태 하나: 후보 m별 예측(평균·시나리오), 이번 기간 이득, 미래 k별 남은 누적 이윤
def one(args):
  cfg, rep, t = args
  s = Sim(cfg, rep)
  while s.t < t:
    s.step(1.0, 0.0)
  o, ms = s.pre(), list(cfg.m_grid)
  We, ge = est(o)
  Gm = [clear(cfg, o.q, We, o.K, ge, m).G for m in ms]
  Gs = np.array([[clear(cfg, o.q, W, o.K, g, m).G for m in ms] for W, g in scen(cfg, o, cfg.saa_k)]).mean(0)
  R, I = np.zeros((K, len(ms))), np.zeros(len(ms))
  for k in range(K):
    for i, m in enumerate(ms):
      b = deepcopy(s)
      b.rep = fut(rep, k)
      rec = b.step(m, 0.0)
      I[i], r = rec["G"], rec["Pi"]
      while b.t < cfg.T:
        r += b.step(1.0, 0.0)["Pi"]
      R[k, i] = r
  return dict(rep=rep, t=t, G_mean=Gm, G_saa=Gs.tolist(), I=I.tolist(), R=R.tolist())


# 규칙이 고른 후보의 끝까지 차이 평균과 95% 신뢰구간, 올린 비율
def rule(D, i):
  v = D[np.arange(len(D)), i]
  return dict(net=float(v.mean()), ci=float(1.96 * v.std(ddof=1) / np.sqrt(len(v))), dev=float((i > 0).mean()))


# 요약: m별 I·D·L(전체와 이번 기간 이득이 양수인 상태만), 규칙별 끝까지 차이. D 최대 후보는 미래 반으로 고르고 나머지 반으로 평가한 값(split)도 같이
def summ(rows):
  I, Gs, Gm = (np.array([x[k] for x in rows]) for k in ("I", "G_saa", "G_mean"))
  R = np.array([x["R"] for x in rows])
  I, R = I - I[:, :1], R - R[:, :, :1]
  D = R.mean(1)
  L = D - I
  n, h = len(rows), R.shape[1] // 2
  ci = lambda a: (1.96 * a.std(0, ddof=1) / np.sqrt(len(a))).tolist()
  by_m = dict(I=I.mean(0).tolist(), I_ci=ci(I), D=D.mean(0).tolist(), D_ci=ci(D), L=L.mean(0).tolist(), L_ci=ci(L))
  pos = []
  for i in range(1, I.shape[1]):
    k = I[:, i] > 0
    pos.append(dict(share=float(k.mean()), I=float(I[k, i].mean()), L=float(L[k, i].mean()), D=float(D[k, i].mean()), D_ci=ci(D[k, i:i + 1])[0]))
  i_saa, dh = Gs.argmax(1), Gs.max(1) - Gs[:, 0]
  rules = {f"saa_tau{t}": rule(D, np.where(dh >= t, i_saa, 0)) for t in TAUS}
  rules.update(mean=rule(D, Gm.argmax(1)), oracle_now=rule(D, I.argmax(1)), best_D=rule(D, D.argmax(1)), fixed=rule(D, np.zeros(n, int)))
  Da, Db = R[:, :h].mean(1), R[:, h:].mean(1)
  v = (Db[np.arange(n), Da.argmax(1)] + Da[np.arange(n), Db.argmax(1)]) / 2
  rules["best_D_split"] = dict(net=float(v.mean()), ci=float(1.96 * v.std(ddof=1) / np.sqrt(n)), dev=float(((Da.argmax(1) > 0).mean() + (Db.argmax(1) > 0).mean()) / 2))
  return dict(n=n, by_m=by_m, pos=pos, rules=rules)


def main():
  sc = sys.argv[1] if sys.argv[1:] else "x1"
  cfg = replace(Cfg(), **SCALE[sc])
  with Pool(cfg.n_proc) as p:
    rows = p.map(one, [(cfg, r, t) for r in REPS for t in range(T0, cfg.T)], chunksize=1)
  out = summ(rows)
  json.dump(dict(cfg=repr(cfg), scale=sc, K=K, **out, rows=rows), open(f"result/runs/check_dev{'' if sc == 'x1' else '_' + sc}.json", "w"), indent=1)
  print(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
