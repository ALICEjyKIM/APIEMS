"""Chu IPB 청산의 해 시간 측정 (N1 완료 기준).
기준 시장에서 주문자 약 20명(신규 두 기간 분량, 정상 상태 활동 주문자 수 lam_buy / (1 − ret_ss))과 기간 0 공급자로 m 후보마다 청산한다.
결과는 result/runs/check_time_chu.json. 실행: src/에서 python -m result.time_chu
"""
import json
import time
import numpy as np
from utils.params import Cfg
from utils import market
from mech.chu import clear


# 반복 rep의 기준 시장 입찰 (q, W, K, g)
def bids(cfg, rep):
  mk = market.make(cfg, rep)
  bs = market.new_buyers(cfg, mk, rep, 0) + market.new_buyers(cfg, mk, rep, 1)
  q, W = np.array([b.qty for b in bs]), np.array([(b.qty * b.price).sum() for b in bs])
  return q, W, np.array([s.qty for s in mk.sups]), np.array([s.price for s in mk.sups])


def main():
  cfg, rows = Cfg(), []
  for rep in range(5):
    q, W, K, g = bids(cfg, rep)
    for m in cfg.m_grid:
      t = time.perf_counter()
      o = clear(cfg, q, W, K, g, m)
      rows.append(dict(rep=rep, m=m, nb=len(W), sec=time.perf_counter() - t, milp=o.n, win=int(o.win.sum()), pairs=int((o.y > 0).sum()), G=o.G, sw=o.sw, Ub=int(o.Ub.sum()), Uj=int(o.Uj.sum())))
      print(rows[-1])
  sec = [r["sec"] for r in rows]
  out = dict(cfg=repr(cfg), rows=rows, mean_sec=float(np.mean(sec)), max_sec=float(np.max(sec)))
  json.dump(out, open("result/runs/check_time_chu.json", "w"), indent=1)
  print("mean", out["mean_sec"], "max", out["max_sec"])


if __name__ == "__main__":
  main()
