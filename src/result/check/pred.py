"""예측 점검 (한 기간, ref/decisions.md "예측 점검"): 고정 padding 경로(반복 1000~1009, t ≥ 5)의 각 기간에서 예측기별로 고른 m의 실제 플랫폼 몫을 비교한다.
예측기: 평균(est), 시나리오 saa_k개와 앞 절반(scen), 재참여자 지난 신고가 + 신규 시나리오(참고, scen_ind), 실제 입찰(상한). 후보는 m_grid, θ는 이번 기간 몫에 영향이 없어 뺀다.
결과 result/runs/check_pred.json. 실행: src/에서 python -m result.check.pred [작업자 수]
"""
import json
import sys
import time
from multiprocessing import Pool
import numpy as np
from utils.params import Cfg
from env.sim import Sim
from mech.chu import clear
from policy.base import est, scen, scen_ind
from result.common import REPS, T0


# 고정 padding 경로의 기간별 (반복, t, 입찰 전 관측, 실제 입찰)
def path(args):
  cfg, rep = args
  s, out = Sim(cfg, rep), []
  for _ in range(cfg.T):
    if s.t >= T0:
      out.append((rep, s.t, s.pre(), s.W.copy(), s.g.copy()))
    s.step(1.0, 0.0)
  return out


# 후보 m별 플랫폼 몫
def gs(cfg, o, W, g):
  return [clear(cfg, o.q, W, o.K, g, m).G for m in cfg.m_grid]


# 한 기간: 예측기별로 고른 m (m_grid 번호)과 결정 시간, 실제 입찰의 후보별 몫
def one(args):
  cfg, rep, t, o, W, g = args
  K, pick, sec = cfg.saa_k, {}, {}
  G = gs(cfg, o, W, g)
  for name, S in (("mean", [est(o)]), ("saa", scen(cfg, o, K)), ("ind", scen_ind(cfg, o, K))):
    t0 = time.time()
    M = np.array([gs(cfg, o, Ws, gk) for Ws, gk in S])
    sec[name], pick[name] = time.time() - t0, int(M.mean(0).argmax())
    if name == "saa":
      pick["saa_half"], sec["saa_half"] = int(M[:K // 2].mean(0).argmax()), sec[name] / 2
  pick["oracle"] = int(np.argmax(G))
  return dict(rep=rep, t=t, nb=len(W), ns=len(g), new=float(np.isnan(o.W0).mean()), G=G, pick=pick, sec=sec)


def main():
  cfg, n = Cfg(), int(sys.argv[1]) if len(sys.argv) > 1 else Cfg().n_proc
  with Pool(len(REPS)) as p:
    P = [x for ps in p.map(path, [(cfg, r) for r in REPS]) for x in ps]
  with Pool(n) as p:
    rows = p.map(one, [(cfg, *x) for x in P], chunksize=1)
  G = np.array([r["G"] for r in rows])
  best, summ = G.max(1), {}
  for name in ("mean", "saa", "saa_half", "ind", "oracle"):
    i = np.array([r["pick"][name] for r in rows])
    v = G[np.arange(len(G)), i]
    summ[name] = dict(G=float(v.mean()), gain=float((v - G[:, 0]).mean()), ci=float(1.96 * (v - G[:, 0]).std(ddof=1) / np.sqrt(len(v))),
                      recover=float((v - G[:, 0]).sum() / (best - G[:, 0]).sum()), agree=float((i == G.argmax(1)).mean()),
                      m_gt1=float((i > 0).mean()), regret=float((best - v).mean()), sec=float(np.mean([r["sec"].get(name, 0.0) for r in rows])))
  summ["fixed"] = dict(G=float(G[:, 0].mean()), gain=0.0, ci=0.0, recover=0.0, agree=float((G.argmax(1) == 0).mean()), m_gt1=0.0,
                       regret=float((best - G[:, 0]).mean()), sec=0.0)
  out = dict(cfg=repr(cfg), n=len(rows), summary=summ, rows=rows)
  json.dump(out, open("result/runs/check_pred.json", "w"), indent=1)
  print(json.dumps(summ, indent=1))


if __name__ == "__main__":
  main()
