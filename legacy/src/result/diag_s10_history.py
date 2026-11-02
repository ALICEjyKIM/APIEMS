"""진단 s10: 거래 이력 환경(PROC15H) 점검 (정책 비교 아님, 이력 파라미터 조정에 쓰지 않는다).
(a) Myopic·Rule 평가 seed 반복 0~4에서 이력이 경험을 따라 움직이는지, 이탈·신규 처리, 같은 경험에서 이력에 따른 재참여확률 차이를 본다.
(b) 같은 상태에서 공급자(주문자) 한 명의 이력만 높게·낮게 바꾼 두 복사본을 같은 미래 키로 Rule로 돌려 재참여확률·잔류·continuation value 차이를 잰다.
"""
import copy
import json
import time
from dataclasses import replace
import numpy as np
from env.platform import Env
from match.milp_solve import Policy
from bench.rule_split import rule
from utils.params import Cfg, PROC15B, PROC15H
from result import diag

OUT = diag.RUNS / "diag_s10_history.json"
CFG = replace(Cfg(), **PROC15H)
POLS = {"Myopic": lambda c: Policy(c), "Rule": lambda c: rule(c)}
REPS = range(5)
M = 40
KEY0 = 4_000_000
HI, LO = 0.9, 0.1
WIN = 5


# (a) 한 반복의 기간별 기록: 참여자별 (이력, 이번 경험, 잉여, 재참여확률), 품목 (이력, 부족률), 최근 WIN기간 경험 수
def episode(cfg, mk, rep):
  env, pol = Env(cfg, rep), mk(cfg)
  env.reset()
  seenB, seenS, recent = dict.fromkeys(env.o.bid), dict.fromkeys(env.o.sid), {"buy": {}, "sup": {}, "item": []}
  rows = dict(sup=[], buy=[], item=[], new_h=[], gone_kept=0)
  for t in range(cfg.T):
    o = env.o
    r = env.step(pol.act(o))
    for kind, ids, h, e, s, p in (("sup", o.sid, o.hs, r["alloc"], r["ss"], r["ps"]), ("buy", o.bid, o.hb, r["ok"], r["sb"], r["pb"])):
      for i, hk, ek, sk, pk in zip(ids, h, e, s, p):
        hist = recent[kind].setdefault(int(i), [])
        rows[kind].append(dict(t=t, h=float(hk), e=int(ek), s=float(sk), p=float(pk), n_recent=sum(hist[-WIN:]), m_recent=len(hist[-WIN:])))
        hist.append(int(ek))
    recent["item"].append((r["short"] > 0).astype(int))
    rows["item"] += [dict(t=t, i=i, h=float(o.hi[i]), short=float(r["short"][i]), unmet=float(r["unmet"][i]),
                          n_recent=int(np.sum([x[i] for x in recent["item"][-WIN - 1:-1]]))) for i in range(cfg.n_items)]
    new = [h for i, h in zip(env.o.sid, env.o.hs) if i not in seenS] + [h for i, h in zip(env.o.bid, env.o.hb) if i not in seenB]
    rows["new_h"] += [float(x) for x in new]
    seenS.update(dict.fromkeys(env.o.sid)), seenB.update(dict.fromkeys(env.o.bid))
    rows["gone_kept"] += len((set(o.sid) - set(env.o.sid)) & set(env.hS)) + len((set(o.bid) - set(env.o.bid)) & set(env.hB))
  return rows


# (b) 상태 env에서 참여자 pid(kind)의 이력만 h로 바꾼 복사본
def with_h(env, kind, pid, h):
  e = copy.deepcopy(env)
  (e.hS if kind == "sup" else e.hB)[pid] = h
  e._observe()
  return e


# (b) 미래 M개: Rule로 vf_H기간 이윤 합, 첫 기간 그 참여자의 재참여확률, 1·3·5기간 뒤 잔류 여부
def futures(env, kind, pid):
  out = []
  for k in range(M):
    e = copy.deepcopy(env)
    e.rep = KEY0 + (env.rep * env.cfg.T + env.t) * 100 + k
    pol, prof, alive, p0 = rule(env.cfg), 0.0, [], None
    for s in range(env.cfg.vf_H):
      o = e.o
      r = e.step(pol.act(o))
      if s == 0:
        ids, ps = (o.sid, r["ps"]) if kind == "sup" else (o.bid, r["pb"])
        p0 = float(ps[ids.index(pid)])
      prof += r["profit"]
      alive.append(pid in (e.o.sid if kind == "sup" else e.o.bid))
    out.append(dict(profit=prof, p=p0, alive=alive))
  return out


# (b) 통제 사례: Rule로 걸은 반복 rep의 기간 t 상태에서 이번 기간 배정(성립)받는 공급자(주문자) 중 공급가능량(주문 금액)이 가장 큰 사람
def controlled(rep, t, kind):
  env, pol = Env(CFG, rep), rule(CFG)
  env.reset()
  for _ in range(t):
    env.step(pol.act(env.o))
  d = copy.deepcopy(env).step(rule(CFG).act(env.o))
  o = env.o
  if kind == "sup":
    cand = [k for k in range(len(o.sid)) if d["alloc"][k]]
    k = max(cand, key=lambda k: o.cap[k].sum())
    pid = o.sid[k]
  else:
    cand = [k for k in range(len(o.bid)) if d["ok"][k]]
    k = max(cand, key=lambda k: (o.p[k] * o.q[k]).sum())
    pid = o.bid[k]
  F = {lab: futures(with_h(env, kind, pid, h), kind, pid) for lab, h in (("hi", HI), ("lo", LO))}
  diff = np.array([a["profit"] - b["profit"] for a, b in zip(F["hi"], F["lo"])])
  return dict(rep=rep, t=t, kind=kind, pid=int(pid), h_hi=HI, h_lo=LO, cur_h=float((o.hs if kind == "sup" else o.hb)[k]),
              p_hi=F["hi"][0]["p"], p_lo=F["lo"][0]["p"],
              alive_hi=np.mean([f["alive"] for f in F["hi"]], 0).tolist(), alive_lo=np.mean([f["alive"] for f in F["lo"]], 0).tolist(),
              V_hi=float(np.mean([f["profit"] for f in F["hi"]])), V_lo=float(np.mean([f["profit"] for f in F["lo"]])),
              dV=float(diff.mean()), dV_se=float(diff.std(ddof=1) / np.sqrt(M)))


if __name__ == "__main__":
  t0 = time.time()
  res = dict(note="PROC15H 점검. 정책 비교·이력 파라미터 조정에 쓰지 않는다. (b)는 Rule-continuation 값.",
             runs={n: [episode(CFG, mk, rep) for rep in REPS] for n, mk in POLS.items()})
  res["controlled"] = [controlled(rep, 10, kind) for kind in ("sup", "buy") for rep in range(3)]
  res["sec"] = time.time() - t0
  OUT.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
  print(f"saved {OUT} ({res['sec']:.0f}s)")
