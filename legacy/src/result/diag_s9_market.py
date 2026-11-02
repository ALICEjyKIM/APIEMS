"""진단 s9a: 조달 시장 v2 기준 환경 sanity check (정책 비교 아님, 판정에 정책 이윤을 쓰지 않는다).
PROC15(잠정), PROC15B(기준, onboard 2)와 기준 주변 한 축씩 바꾼 조건에서 Myopic·Rule을 평가 seed 반복 0~4로 돌려
활동 공급자·주문자, 품목별 활동 공급자 수 분포, 0명 비율, 활동 용량, 주문 가능률·충족률, 이탈률, 대기열을 result/runs/diag_s9_market.json에 저장한다.
"""
import json
import time
from dataclasses import replace
import numpy as np
from env.platform import Env
from match.milp_solve import Policy
from bench.rule_split import rule
from utils.params import Cfg, PROC15, PROC15B
from result import diag

OUT = diag.RUNS / "diag_s9_market.json"
BASE = replace(Cfg(), **PROC15B)
CONDS = {"PROC15": replace(Cfg(), **PROC15), "base": BASE,
         "L1": replace(BASE, onboard=1), "L4": replace(BASE, onboard=4),
         "het0": replace(BASE, het=0.0), "het1": replace(BASE, het=1.0),
         "cov1.0": replace(BASE, cover=1.0), "cov1.6": replace(BASE, cover=1.6),
         "k1": replace(BASE, items_per_order=1), "k3": replace(BASE, items_per_order=3)}
POLS = {"Myopic": lambda c: Policy(c), "Rule": lambda c: rule(c)}
REPS = range(5)


# 한 반복의 시장 지표 (기간 합·평균). 주문 가능 = 주문의 모든 품목에서 활동 공급가능량 합 ≥ 요구수량
def episode(cfg, mk, rep):
  env, pol = Env(cfg, rep), mk(cfg)
  env.reset()
  tgt, K = env.inst.n_tgt, env.inst.K
  s = dict(n_sup=0, n_buy=0, stay_sup=0, stay_buy=0, n_ok=0, feas=0, pend=0, cap=0.0)
  hist, zero = np.zeros(5), dict(all=[0, 0], lo=[0, 0], hi=[0, 0])
  for _ in range(cfg.T):
    o = env.o
    deg, cap = (o.cap > 0).sum(0), o.cap.sum(0)
    hist += np.bincount(np.minimum(deg, 4), minlength=5)
    for k, m in (("all", np.ones(cfg.n_items, bool)), ("lo", tgt <= 2), ("hi", tgt >= 4)):
      zero[k][0] += (deg[m] == 0).sum()
      zero[k][1] += m.sum()
    s["feas"] += int(((o.q <= cap[None]) | (o.q == 0)).all(1).sum())
    s["pend"] += len(o.wait)
    s["cap"] += cap.sum() / K.sum()
    r = env.step(pol.act(o))
    for k in ("n_sup", "n_buy", "stay_sup", "stay_buy", "n_ok"):
      s[k] += r[k]
  T = cfg.T
  return dict(act_sup=s["n_sup"] / T, act_buy=s["n_buy"] / T, dep_sup=1 - s["stay_sup"] / s["n_sup"], dep_buy=1 - s["stay_buy"] / s["n_buy"],
              fill=s["n_ok"] / s["n_buy"], feas=s["feas"] / s["n_buy"], pend=s["pend"] / T, cap_ratio=s["cap"] / T,
              zero_all=zero["all"][0] / zero["all"][1], zero_lo=zero["lo"][0] / max(zero["lo"][1], 1), zero_hi=zero["hi"][0] / max(zero["hi"][1], 1),
              deg_hist=(hist / hist.sum()).tolist(), tgt_lo=float((tgt <= 2).mean()), tgt_hist=np.bincount(np.minimum(tgt, 4), minlength=5)[1:].tolist())


# 판정 (0절 사전 기준)
def judge(m):
  my, ru = m["Myopic"], m["Rule"]
  c1 = all(x["zero_all"] <= 0.10 and x["zero_hi"] <= 0.05 for x in (my, ru))
  c2 = my["dep_sup"] >= 0.10 and my["pend"] >= 1 and all(x["zero_lo"] > x["zero_hi"] for x in (my, ru))
  c3 = my["fill"] >= 0.5
  return dict(C1=c1, C2=c2, C3=c3)


if __name__ == "__main__":
  t0 = time.time()
  res = dict(note="v2 기준 환경 sanity check. 판정 기준은 ref/plan/interaction_diag.md 0절 (실행 전에 적음). 사전 기준안(onboard 4) 결과는 diag_s9_market_v0.json", conds={})
  for name, cfg in CONDS.items():
    m = {}
    for pn, mk in POLS.items():
      eps = [episode(cfg, mk, rep) for rep in REPS]
      m[pn] = {k: (np.mean([e[k] for e in eps], 0).tolist() if isinstance(eps[0][k], list) else float(np.mean([e[k] for e in eps]))) for k in eps[0]}
    res["conds"][name] = dict(metrics=m, judge=judge(m))
    j = res["conds"][name]["judge"]
    for pn, x in m.items():
      print(f"{name:7s} {pn:6s} 공급자 {x['act_sup']:5.1f} 주문자 {x['act_buy']:5.1f} 이탈 공급자 {x['dep_sup']:.2f} 주문자 {x['dep_buy']:.2f} "
            f"0명 전체 {x['zero_all']:.3f} 목표≤2 {x['zero_lo']:.3f} 목표≥4 {x['zero_hi']:.3f} 활동용량/명목 {x['cap_ratio']:.2f} "
            f"가능 {x['feas']:.2f} 충족 {x['fill']:.2f} 대기 {x['pend']:.1f} 활동차수 0/1/2/3/4+ {np.round(x['deg_hist'], 2).tolist()}")
    print(f"        판정 {j}", flush=True)
  res["sec"] = time.time() - t0
  OUT.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
  print(f"saved {OUT} ({res['sec']:.0f}s)")
