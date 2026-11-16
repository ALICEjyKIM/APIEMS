"""한 번 올린 대가 점검 (ref/decisions.md "한 번 올린 대가 점검"): 예측 점검과 같은 250개 상태에서 후보 m마다 이번 기간만 m(θ = 0), 이후 고정 padding으로 T까지 돌려 누적 이윤을 미래 난수 8개로 잰다. 이번 기간 이득 I_m은 확정값이고, 끝까지 차이 D_m에서 I_m을 뺀 L_m이 이후 영향이다.
규칙 평가: 시나리오 예측의 m*를 예상 이득 Δ̂ ≥ τ일 때만 쓰는 규칙, 평균 예측의 m*, 실제 입찰의 m*(상한), D 최대 후보(반분 검증 포함). ci: 250개 상태가 시장 반복 10개에서 나와 같은 반복 안의 상태끼리 상관이 있으므로 m별 I·L·D와 규칙별 끝까지 차이를 반복 군집 표준오차와 t(군집 수 − 1)로 다시 잰다.
결과 result/runs/check_dev[_x2][_바꾼Cfg키값][_K미래수].json, check_dev_cluster_ci.json. 실행: src/에서 python -m result.check.dev [x2|x4] [Cfg 키=값 ...] [K=미래 수] (m을 일부만 잴 때 예: m_grid=1.05,1.15 K=4, 요약은 첫 m이 1일 때만), python -m result.check.dev ci
"""
from copy import deepcopy
from dataclasses import replace
import json
import os
from multiprocessing import Pool
import sys
import numpy as np
from utils.params import Cfg, SCALE
from env.sim import Sim
from mech.chu import clear
from policy.base import est, scen
from result.common import REPS, T0, fut, K, ovr, save, part, cci

TAUS = (0, 5, 10, 15, 20, 30, 40, 50)


# 상태 하나: 후보 m별 예측(평균·시나리오), 이번 기간 이득, 미래 k별 남은 누적 이윤 (미래는 fut의 앞 nk개)
def one(args):
  cfg, rep, t, nk = args
  s = Sim(cfg, rep)
  while s.t < t:
    s.step(1.0, 0.0)
  o, ms = s.pre(), list(cfg.m_grid)
  full = cfg.m_grid[0] == 1.0  # 예상 이득은 m = 1 대비 요약에만 쓰므로 새 m만 잴 때는 건너뛴다 (청산은 상태를 바꾸지 않아 미래 경로는 같다)
  We, ge = est(o)
  Gm = [clear(cfg, o.q, We, o.K, ge, m).G for m in ms] if full else []
  Gs = np.array([[clear(cfg, o.q, W, o.K, g, m).G for m in ms] for W, g in scen(cfg, o, cfg.saa_k)]).mean(0) if full else np.zeros(0)
  R, I = np.zeros((nk, len(ms))), np.zeros(len(ms))
  for k in range(nk):
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
  sc = sys.argv[1] if sys.argv[1:2] and sys.argv[1] in SCALE else "x1"
  ov = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
  nk = int(ov.pop("K", K))
  cfg = ovr(replace(Cfg(), **SCALE[sc]), ov)
  tag = ("" if sc == "x1" else "_" + sc) + "".join(f"_{k}{v}" for k, v in ov.items()) + (f"_K{nk}" if nk != K else "")
  pth = f"result/runs/check_dev{tag}"
  # 상태 하나가 끝날 때마다 중간 저장, 같은 명령으로 다시 실행하면 저장된 상태는 건너뛴다 (미래 수는 파일 이름에 들어 있다)
  rows = part(pth + ".part.json", cfg)
  done = {(x["rep"], x["t"]) for x in rows}
  with Pool(cfg.n_proc) as p:
    for x in p.imap_unordered(one, [(cfg, r, t, nk) for r in REPS for t in range(T0, cfg.T) if (r, t) not in done]):
      rows.append(x)
      print(len(rows), x["rep"], x["t"], flush=True)
      save(pth + ".part.json", dict(cfg=repr(cfg), K=nk, rows=rows))
  rows.sort(key=lambda x: (x["rep"], x["t"]))
  out = summ(rows) if cfg.m_grid[0] == 1.0 else {}
  json.dump(dict(cfg=repr(cfg), scale=sc, K=nk, **out, rows=rows), open(pth + ".json", "w"), indent=1)
  os.remove(pth + ".part.json")
  print(json.dumps(out, indent=1))


# 파일 하나의 반복 군집 구간: m별 (전체·이득 양수 상태) I·L·D, 규칙별 끝까지 차이
def ci_one(name):
  rows = json.load(open(f"result/runs/{name}"))["rows"]
  g = np.array([r["rep"] for r in rows])
  I, R = np.array([r["I"] for r in rows]), np.array([r["R"] for r in rows])
  Gs, Gm = np.array([r["G_saa"] for r in rows]), np.array([r["G_mean"] for r in rows])
  I, D = I - I[:, :1], (R - R[:, :, :1]).mean(1)
  L, n = D - I, len(rows)
  out = dict(all={}, pos={}, rules={})
  for i in range(1, I.shape[1]):
    k = I[:, i] > 0
    out["all"][i] = {a: cci(x[:, i], g) for a, x in (("I", I), ("L", L), ("D", D))}
    out["pos"][i] = dict(share=float(k.mean()), **{a: cci(x[k, i], g[k]) for a, x in (("I", I), ("L", L), ("D", D))})
  pick = lambda j: D[np.arange(n), j]
  i_saa, dh = Gs.argmax(1), Gs.max(1) - Gs[:, 0]
  sel = dict(mean=Gm.argmax(1), saa_tau0=i_saa, saa_tau30=np.where(dh >= 30, i_saa, 0), oracle_now=I.argmax(1),
             oracle_small=np.where(I[:, 1] > 0, 1, 0))
  out["rules"] = {a: dict(dev=float((j > 0).mean()), **cci(pick(j), g)) for a, j in sel.items()}
  return out


# ci: check_dev.json(×1) · check_dev_x2.json(×2)의 반복 군집 구간
def ci():
  out = {s: ci_one(f) for s, f in (("x1", "check_dev.json"), ("x2", "check_dev_x2.json"))}
  json.dump(out, open("result/runs/check_dev_cluster_ci.json", "w"), indent=1)
  for s, o in out.items():
    print(s, "pos D:", [(round(o["pos"][i]["D"]["mean"]), round(o["pos"][i]["D"]["ci"]), round(o["pos"][i]["D"]["ci_iid"])) for i in (1, 2, 3)])
    print(s, "rules:", {a: (round(v["mean"], 1), round(v["ci"], 1)) for a, v in o["rules"].items()})


if __name__ == "__main__":
  if sys.argv[1:2] == ["ci"]:
    ci()
  else:
    main()
