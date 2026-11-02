"""진단 s15: policy-consistent c (Myopic-sup continuation)로 한 기간 결정이 Myopic-sup보다 나아지는가 (PROC15H, s14와 같은 24상태).
c^M_i = V^Myopic-sup(S) − V^Myopic-sup(S∖i): 같은 미래 키(M = 40, vf_H = 10, s14 평가 키와 다른 영역)로 잰다.
Exact-MC Myopic-c 행동과 Myopic-sup 행동을 각각 적용하고 s14와 같은 평가 키로 Myopic-sup을 10기간 이어 간다 (Myopic-sup 행동의 결과는 s14 원자료를 재사용).
"""
import copy
import json
import sys
import time
from multiprocessing import Pool
import numpy as np
from env.platform import settle, worth
from result.diag_s14_onestep import CFG, REPS, TS, M, state, act, cont, msup, UNITS as U14
from result import diag

UNITS = diag.RUNS / "diag_s15_units"
OUT = diag.RUNS / "diag_s15_myopic_c.json"
KEY0 = 8_000_000
WORKERS = 12


# 공급자 ID 집합 drop을 뺀 상태에서 미래 M개 Myopic-sup vf_H기간 이윤 합 (같은 키라 drop만 다르다)
def value(env, drop):
  out = []
  for k in range(M):
    e = copy.deepcopy(env)
    e.rep = KEY0 + (env.rep * CFG.T + env.t) * 100 + k
    for s in drop:
      e.drop("sup", e.o.sid.index(s))
    pol = msup(CFG)
    out.append(sum(e.step(pol.act(e.o))["profit"] for _ in range(CFG.vf_H)))
  return out


# 작업: ("mc", 반복, 기간, 묶음, 제거 집합들) 또는 ("cont", 반복, 기간, 행동)
def unit(job):
  t0 = time.time()
  if job[0] == "mc":
    _, rep, t, c, drops = job
    out = UNITS / f"mc_r{rep}_t{t}_c{c}.json"
    if out.exists():
      return out.name, "skip"
    env = state(rep, t)
    res = dict(rep=rep, t=t, values={",".join(map(str, d)): value(env, d) for d in drops})
  else:
    _, rep, t, dec = job
    out = UNITS / f"cont_r{rep}_t{t}.json"
    if out.exists():
      return out.name, "skip"
    env = state(rep, t)
    res = dict(rep=rep, t=t, rows=cont(env, tuple(np.array(x) for x in dec), "Myopic-sup"))
  res["sec"] = time.time() - t0
  UNITS.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(res), encoding="utf-8")
  return out.name, f"{res['sec']:.0f}s"


# 상태의 c^M (평균, 표준오차)
def cM(env, rep, t):
  V = {}
  for f in UNITS.glob(f"mc_r{rep}_t{t}_c*.json"):
    V.update({k: np.array(v) for k, v in json.loads(f.read_text(encoding="utf-8"))["values"].items()})
  d = [V[""] - V[str(s)] for s in env.o.sid]
  return np.array([x.mean() for x in d]), np.array([x.std(ddof=1) / np.sqrt(len(x)) for x in d])


if __name__ == "__main__":
  t0 = time.time()
  keys = [(rep, t) for rep in REPS for t in TS]
  if "mc" in sys.argv:
    jobs = []
    for rep, t in keys:
      env = state(rep, t)
      drops = [[]] + [[int(s)] for s in env.o.sid]
      jobs += [("mc", rep, t, c, drops[i:i + 7]) for c, i in enumerate(range(0, len(drops), 7))]
    print(f"MC 작업 {len(jobs)}", flush=True)
    with Pool(WORKERS, maxtasksperchild=4) as p:
      for i, (name, msg) in enumerate(p.imap_unordered(unit, jobs), 1):
        print(f"[{i}/{len(jobs)}] {name} {msg} ({time.time() - t0:.0f}s)", flush=True)
  if "cont" in sys.argv:
    A, jobs = {}, []
    for rep, t in keys:
      env = state(rep, t)
      c, se = cM(env, rep, t)
      o = env.o
      rec = dict(rep=rep, t=t, c=c.tolist(), se=se.tolist(), act={})
      for name, cc in (("Myopic-sup", None), ("Exact-MC-M", c)):
        dec = act(env, cc)
        r = settle(CFG, o, dec)
        alloc = (dec[0] * r["ok"][:, None, None]).sum((0, 2)) > 0
        rec["act"][name] = dict(ok=r["ok"].astype(int).tolist(), alloc=alloc.astype(int).tolist(), ss=r["ss"].tolist(), profit=float(r["profit"]),
                                x=dec[0].sum(0).tolist())
        if cc is not None:
          jobs.append(("cont", rep, t, [dec[0].tolist(), dec[1].tolist(), dec[2].tolist()]))
      A[f"{rep}_{t}"] = rec
    (diag.RUNS / "diag_s15_actions.json").write_text(json.dumps(A), encoding="utf-8")
    with Pool(WORKERS) as p:
      for i, (name, msg) in enumerate(p.imap_unordered(unit, jobs), 1):
        print(f"[{i}/{len(jobs)}] {name} {msg}", flush=True)
  if "summary" in sys.argv:
    A = json.loads((diag.RUNS / "diag_s15_actions.json").read_text(encoding="utf-8"))
    c = np.concatenate([A[f"{r}_{t}"]["c"] for r, t in keys])
    se = np.concatenate([A[f"{r}_{t}"]["se"] for r, t in keys])
    rows = []
    for rep, t in keys:
      a = A[f"{rep}_{t}"]["act"]
      base = json.loads((U14 / f"cont_r{rep}_t{t}_Myopic-sup_Myopic-sup.json").read_text(encoding="utf-8"))["rows"]
      new = json.loads((UNITS / f"cont_r{rep}_t{t}.json").read_text(encoding="utf-8"))["rows"]
      assert np.isclose(np.mean([x["imm"] for x in base]), a["Myopic-sup"]["profit"])
      d = lambda k: float(np.mean([x[k] - y[k] for x, y in zip(new, base)]))
      m, b = a["Exact-MC-M"], a["Myopic-sup"]
      rows.append(dict(rep=rep, t=t, d_imm=d("imm"), d_fut=d("fut"), d_tot=d("imm") + d("fut"), d_fill=d("fill"), d_ret=d("ret_sup"), d_act=d("act_sup"),
                       d_ss=sum(m["ss"]) - sum(b["ss"]), paid=sum(x > 1e-6 for x in m["ss"]), alloc_diff=float(np.mean(np.array(m["alloc"]) != np.array(b["alloc"]))),
                       x_diff=float(np.mean(np.array(m["x"]) != np.array(b["x"]))), acc_diff=sum(m["ok"]) - sum(b["ok"]), same=bool(m["alloc"] == b["alloc"] and np.allclose(m["ss"], b["ss"]) and m["x"] == b["x"]),
                       pos_c=int(sum(x > 0 for x in A[f"{rep}_{t}"]["c"]))))
    reps = np.array([r["rep"] for r in rows])
    rng = np.random.default_rng(0)
    def ci(k):
      v = np.array([r[k] for r in rows])
      bs = [np.mean(np.concatenate([v[reps == q] for q in rng.choice(np.unique(reps), len(np.unique(reps)))])) for _ in range(2000)]
      return dict(mean=float(v.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
    res = dict(note="PROC15H, s14 평가 24상태. c^M = Myopic-sup continuation c (M = 40, vf_H = 10). 평가 후속 정책 Myopic-sup.",
               c=dict(mean=float(c.mean()), sd=float(c.std()), se_mean=float(se.mean()), reliability=float(1 - np.mean(se ** 2) / np.var(c)), pos=float((c > 0).mean()),
                      sig_pos=float((c / se > 1.96).mean())),
               diffs={k: ci(k) for k in ("d_imm", "d_fut", "d_tot", "d_fill", "d_ret", "d_act", "d_ss", "acc_diff")},
               better=float(np.mean([r["d_tot"] > 0 for r in rows])), same_action=float(np.mean([r["same"] for r in rows])),
               alloc_diff=float(np.mean([r["alloc_diff"] for r in rows])), x_diff=float(np.mean([r["x_diff"] for r in rows])), paid=float(np.mean([r["paid"] for r in rows])),
               rows=rows)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    x = res["c"]
    print(f"c^M: 평균 {x['mean']:.1f} sd {x['sd']:.1f} MC se {x['se_mean']:.1f} 신뢰도 {x['reliability']:.2f} 양수 {x['pos']:.2f} 유의 양수 {x['sig_pos']:.2f}")
    print(f"행동: 완전히 같은 상태 {res['same_action']:.2f}, 공급자 배정 다른 비율 {res['alloc_diff']:.3f}, 공급자×품목 배정량 다른 비율 {res['x_diff']:.3f}, 잉여 받은 공급자 {res['paid']:.2f}명")
    for k, v in res["diffs"].items():
      print(f"  {k:8s} {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
    print(f"  Exact-MC-M가 더 나은 상태 비율 {res['better']:.2f}")
  print(f"done ({time.time() - t0:.0f}s)")
