"""진단 s7: SCALE15 기준 칸에서 Myopic이 Rule보다 누적 이윤이 큰 이유 (실험 7 해석용, 모형·파라미터는 바꾸지 않는다).
Rule (0.3, 0.3), 완전 Myopic, Myopic-sup (주문자 몫 0.3, 공급자 c = 0)을 평가 seed 반복 0~4에서 공통 난수로 돌려
기간별 재참여확률·유지 수·신규 공급자 수·지급 잉여·플랫폼 이윤을 result/runs/diag_s7_myopic.json에 저장한다.
"""
import json
import time
import numpy as np
from env.platform import Env
from match.milp_solve import Policy
from bench.rule_split import rule
from result.exp5_fair import cell_cfg
from result import diag

OUT = diag.RUNS / "diag_s7_myopic.json"
REPS = range(5)
POLS = {"Rule": lambda c: rule(c), "Myopic": lambda c: Policy(c), "Myopic-sup": lambda c: Policy(c, split=(c.sh_buy, None))}
KEYS = ("profit", "margin", "sb", "ss", "n_ok", "n_buy", "stay_buy", "new_buy", "n_sup", "stay_sup", "new_sup", "pb", "ps")


# 정책 하나의 반복 rep 기간별 원자료 (new_* = 다음 기간에 새로 들어온 참여자 수, pb·ps = 평균 재참여확률)
def episode(cfg, mk, rep):
  env, pol = Env(cfg, rep), mk(cfg)
  env.reset()
  out = {k: [] for k in KEYS}
  for _ in range(cfg.T):
    o = env.o
    r = env.step(pol.act(o))
    vals = (r["profit"], r["rev"] + r["sb"].sum() + r["ss"].sum(), r["sb"].sum(), r["ss"].sum(), r["n_ok"],
            r["n_buy"], r["stay_buy"], len(set(env.o.bid) - set(o.bid)), r["n_sup"], r["stay_sup"],
            len(set(env.o.sid) - set(o.sid)), r["pb"].mean() if len(r["pb"]) else np.nan, r["ps"].mean())
    for k, v in zip(KEYS, vals):
      out[k].append(float(v))
  return out


if __name__ == "__main__":
  t0 = time.time()
  cfg = cell_cfg()
  res = dict(note="실험 7 해석용 진단. 모형·파라미터 조정에 쓰지 않는다.", cell="alt4_conc0.5_k2", reps=list(REPS),
             runs={n: [episode(cfg, mk, rep) for rep in REPS] for n, mk in POLS.items()})
  res["sec"] = time.time() - t0
  OUT.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
  print(f"saved {OUT} ({res['sec']:.0f}s)")
