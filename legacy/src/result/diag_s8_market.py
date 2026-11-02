"""진단 s8: 조달 시장 v2 환경 점검 (정책 비교 아님, 파라미터 조정에 쓰지 않는다).
PROC15 preset에서 Myopic·Rule·Myopic-sup을 평가 seed 반복 0~4로 돌려 공급자 교체, 품목별 활동 공급자 수·용량, 주문자 묶음 반복,
경험별 재참여확률, 정책 간 공통 난수를 기간별로 기록해 result/runs/diag_s8_market.json에 저장한다. het 수준별 연결 구조 분포도 함께 낸다.
"""
import json
import time
from dataclasses import replace
import numpy as np
from env.platform import Env
from match.milp_solve import Policy
from bench.rule_split import rule
from utils.params import Cfg, PROC15
from utils.market2 import make2
from result import diag

OUT = diag.RUNS / "diag_s8_market.json"
CFG = replace(Cfg(), **PROC15)
REPS = range(5)
POLS = {"Myopic": lambda c: Policy(c), "Rule": lambda c: rule(c), "Myopic-sup": lambda c: Policy(c, split=(c.sh_buy, None))}
HETS = (0.0, 0.25, 0.5, 0.75, 1.0)


# 정책 하나의 반복 rep 기간별 원자료: 품목별 활동 공급자 수·용량, 교체 사건(떠난 공급자와 그 자리에 들어온 공급자의 품목·용량),
# 경험별 재참여확률, 남은 주문자의 묶음 반복 여부, 다음 기간 신규 주문자(ID·품목) (공급자 ID % n_sup = 자리)
def episode(cfg, mk, rep):
  env, pol = Env(cfg, rep), mk(cfg)
  env.reset()
  first, left, rows, events = {}, {}, [], []
  for t in range(cfg.T):
    o, nb = env.o, env.nb
    same = [bool(np.array_equal(q > 0, first.setdefault(i, q > 0))) for i, q in zip(o.bid, o.q)]
    r = env.step(pol.act(o))
    for k, i in enumerate(o.sid):
      if not r["kept_s"][k]:
        left[i % cfg.n_sup] = (t, o.cap[k])
    for j in r["joined"]:
      if j in left:
        t_left, cap = left.pop(j)
        got = env.sups[j][1].qty
        events.append(dict(seat=j, t_left=t_left, t_join=env.t, same_items=bool(np.array_equal(cap > 0, got > 0)),
                           same_cap=bool(np.array_equal(cap, got)), lost=cap.tolist(), got=got.tolist()))
    rows.append(dict(t=t, n_i=(o.cap > 0).sum(0).tolist(), cap_i=o.cap.sum(0).tolist(), n_sup=len(o.sid), n_pend=len(o.wait),
                     profit=r["profit"], pb_ok=r["pb"][r["ok"]].tolist(), pb_no=r["pb"][~r["ok"]].tolist(),
                     ps_al=r["ps"][r["alloc"]].tolist(), ps_no=r["ps"][~r["alloc"]].tolist(), same_bundle=same,
                     arr=[[int(i), (q > 0).astype(int).tolist()] for i, q in zip(env.o.bid, env.o.q) if i >= nb]))
  return dict(rows=rows, events=events, n_tgt=env.inst.n_tgt.tolist(), K=env.inst.K.tolist())


# het 수준별 품목 목표 대체 공급자 수 (반복 0~49)
def network(cfg):
  return {str(h): [make2(replace(cfg, het=h), rep).n_tgt.tolist() for rep in range(50)] for h in HETS}


if __name__ == "__main__":
  t0 = time.time()
  res = dict(note="조달 시장 v2 환경 점검. 정책 비교·파라미터 조정에 쓰지 않는다.", preset=PROC15, reps=list(REPS),
             runs={n: [episode(CFG, mk, rep) for rep in REPS] for n, mk in POLS.items()}, network=network(CFG))
  res["sec"] = time.time() - t0
  OUT.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
  print(f"saved {OUT} ({res['sec']:.0f}s)")
