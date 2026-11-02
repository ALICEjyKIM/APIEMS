"""슬라이스 1 회귀 확인: 구현 수정이 환경·Linear·MLP를 바꾸지 않았는지.
저장된 exp1_value·exp1_eval·exp3_mlp의 값과 다시 돌린 값을 같은 설정에서 비교한다 (규모·파라미터는 그대로).
구조가 바뀐 GNN은 과거 수치 재현 대상이 아니므로 학습·추론이 정상인지만 확인한다.
"""
import json
import time
from dataclasses import asdict, replace
import numpy as np
from utils.params import Cfg
from env.platform import Env
from env.graph import graph, phi_conc
from match.interface import rollout
from match.milp_solve import Policy
from bench.rule_split import rule
from utils.features import phi
from vfa.train import collect, collect_obs, val_mask, select as tselect
from vfa import linear, mlp
from vfa.gnn import GNN, hidden, n_params, mlp_params
from vfa.coef import raw, coefs, vfa_policy
from result import diag

OUT = diag.RUNS / "diag_s1_regress.json"
RTOL = 1e-9


def load(name):
  return json.loads((diag.RUNS / name).read_text(encoding="utf-8"))


# 두 값의 일치 기록 (상대 오차)
def cmp(old, new, rtol=RTOL):
  o, n = np.asarray(old, float), np.asarray(new, float)
  d = np.max(np.abs(n - o) / np.maximum(np.abs(o), 1e-12)) if o.size else 0.0
  return dict(old=old, new=new, max_rel=float(d), same=bool(d <= rtol))


# A. 환경 통계: 저장된 반복별 원자료와 같은 설정에서 다시 돌린 rollout 비교 (모델을 쓰지 않는 두 정책)
def env_stats():
  d = load("exp1_eval.json")
  cfg = replace(Cfg(), price_rank=False)
  assert cfg.reps == d["cfg"]["reps"] and cfg.T == d["cfg"]["T"] and cfg.seed == d["cfg"]["seed"]
  pols = {"근시안": lambda c: Policy(c), "규칙 기반 (0.3, 0.3)": lambda c: rule(c)}
  # 저장된 기간별 원자료에서 반복별 지표 (rollout이 내는 것과 같은 정의)
  per = lambda s: dict(profit=float(np.sum(s["profit"])), fill=float(np.sum(s["n_ok"]) / np.sum(s["n_buy"])),
                       ret_buy=float(np.sum(s["stay_buy"]) / np.sum(s["n_buy"])),
                       ret_sup=float(np.sum(s["stay_sup"]) / np.sum(s["n_sup"])))
  out = {}
  for name, mk in pols.items():
    saved = [per(s) for s in d["policies"][name]]
    got = [rollout(cfg, mk(cfg), rep) for rep in range(cfg.reps)]
    out[name] = {k: cmp([s[k] for s in saved], [g[k] for g in got]) for k in ("profit", "fill", "ret_buy", "ret_sup")}
  return out


# B. 실험 1 적합: 같은 설정에서 학습 데이터와 Linear·MLP 선택이 저장값과 같은지
def exp1_fit():
  d = load("exp1_value.json")
  cfg = replace(Cfg(), price_rank=False)
  X, y, g = collect(cfg)
  va = val_mask(cfg, g)
  f = d["fit"]
  out = dict(n=cmp(f["n"], len(y)), n_val=cmp(f["n_val"], int(va.sum())),
             y_mean=cmp(f["y_mean"], float(y.mean())), y_sd=cmp(f["y_sd"], float(y.std())))
  for name, sel, grid, key in (("선형", linear.select, cfg.lin_lams, "lam"), ("MLP", mlp.select, cfg.nn_wds, "wd")):
    m, mse = sel(cfg, X, y, g)
    a = getattr(m, key)
    r2 = float(1 - mse[a] / np.var(y[va]))
    out[name] = dict(val_mse=cmp(f[name]["val_mse"], [mse[k] for k in grid]),
                     pick=cmp(f[name]["pick"], a), val_r2=cmp(f[name]["val_r2"], r2))
  return out


# C. 실험 3 한 칸: MLP와 MLP + 편중도 요약 적합이 저장값과 같은지 (phi_conc가 바뀐 env/graph.py에 있으므로 확인)
def exp3_cell(tag="conc0_k1", conc=0.0, k=1):
  d = load("exp3_mlp.json")["cells"][tag]
  c = Cfg()
  cfg = replace(c, conc=conc, items_per_order=k, vf_reps=c.vf_reps3, price_rank=True)
  O, y, g = collect_obs(cfg)
  va = val_mask(cfg, g)
  out = dict(n=cmp(d["n"], len(y)), n_val=cmp(d["n_val"], int(va.sum())),
             y_mean=cmp(d["y_mean"], float(y.mean())), y_sd=cmp(d["y_sd"], float(y.std())),
             max_buyers=cmp(d["max_buyers"], max(len(o.bid) for o in O)))
  for name, enc in (("MLP", phi), ("MLP+편중도 요약", phi_conc)):
    X = np.array([enc(o) for o in O])
    mk = lambda wd: mlp.MLP(cfg, wd)
    m, mse = tselect(cfg, mk, cfg.nn_wds, X, y, g)
    r2 = float(1 - mse[m.wd] / np.var(y[va]))
    out[name] = dict(val_mse=cmp(d["fit"][name]["val_mse"], [mse[w] for w in cfg.nn_wds]),
                     pick=cmp(d["fit"][name]["pick"], m.wd), val_r2=cmp(d["fit"][name]["val_r2"], r2))
  return out


# D. GNN 정상성: 학습이 손실을 줄이는지, 예측·유지 가치가 유한한지, 자리 수·노드 순서에 불변인지, 정책이 돌아가는지
def gnn_sane():
  cfg = replace(Cfg(), price_rank=False)
  O, y, g = collect_obs(cfg)
  X = np.array([graph(cfg, o) for o in O])
  va = val_mask(cfg, g)
  mse = lambda m, s: float(np.mean((m.predict(X[s]) - y[s]) ** 2))
  m1 = GNN(replace(cfg, nn_epochs=1), 0.1).fit(X[~va], y[~va])
  m = GNN(cfg, 0.1).fit(X[~va], y[~va])
  tr0, tr1 = mse(m1, ~va), mse(m, ~va)
  out = dict(hidden=hidden(cfg), params=n_params(cfg, hidden(cfg)), mlp_params=mlp_params(cfg),
             train_mse_1epoch=tr0, train_mse_full=tr1, train_mse_ratio=tr1 / tr0, learned=bool(tr1 < tr0),
             train_r2=float(1 - tr1 / np.var(y[~va])), val_r2=float(1 - mse(m, va) / np.var(y[va])),
             pred_finite=bool(np.isfinite(m.predict(X)).all()),
             note=("val_r2는 weight decay 0.1 하나로 학습 분할에만 맞춘 값이다. "
                   "격자 선택을 한 Linear·MLP와 나란히 놓을 수 없으므로 공정 비교가 아니다 (슬라이스 5에서 같은 예산으로 비교한다)."))
  mf = GNN(cfg, 0.1).fit(X, y)
  o = O[-1]
  cb, cs, v = raw(mf, o)
  gb, gs = coefs(cfg, mf, o)
  out.update(coef_finite=bool(np.isfinite(cb).all() and np.isfinite(cs).all() and np.isfinite(v)),
             n_coef=cmp([len(o.bid), len(o.sid)], [len(cb), len(cs)]),
             guard_in_range=bool((gb >= 0).all() and (gs >= 0).all()
                                 and (gb <= cfg.coef_hi * max(v, 0) + 1e-9).all()
                                 and (gs <= cfg.coef_hi * max(v, 0) + 1e-9).all()),
             c_buy_mean=float(cb.mean()), c_sup_mean=float(cs.mean()), v=float(v))
  wide = replace(cfg, gnn_nb=cfg.gnn_nb + 17)
  m2 = GNN(wide, 0.1)
  for kk, vv in mf.__dict__.items():
    if kk not in ("cfg", "enc"):
      setattr(m2, kk, vv)
  perm = np.arange(len(o.bid))[::-1]
  o2 = replace(o, q=o.q[perm], p=o.p[perm], rb=o.rb[perm], bid=[o.bid[i] for i in perm])
  a = float(mf.predict(graph(cfg, o)[None])[0])
  out.update(slot_invariant=cmp(a, float(m2.predict(graph(wide, o)[None])[0]), 1e-5),
             order_invariant=cmp(a, float(mf.predict(graph(cfg, o2)[None])[0]), 1e-5))
  short = replace(cfg, T=6)
  r = rollout(short, vfa_policy(short, mf), 0)
  out["policy_runs"] = dict(profit=r["profit"], fill=r["fill"], ok=bool(r["profit"] >= 0 and 0 < r["fill"] <= 1))
  return out


NOTE = ("슬라이스 1 회귀 확인. A·B·C는 저장된 결과와 같은 설정에서 다시 돌려 비교한다 (환경·Linear·MLP는 수정하지 않았으므로 일치해야 한다). "
        "D는 구조가 바뀐 GNN의 정상성만 본다 (과거 수치 재현 대상이 아니다). 규모·파라미터는 바꾸지 않았다.")

if __name__ == "__main__":
  t0 = time.time()
  res = dict(note=NOTE, cfg=asdict(Cfg()))
  for name, fn in (("A_env", env_stats), ("B_exp1_fit", exp1_fit), ("C_exp3_cell", exp3_cell), ("D_gnn", gnn_sane)):
    t = time.time()
    res[name] = fn()
    print(f"{name} {time.time() - t:.0f}s", flush=True)
  res["sec"] = time.time() - t0
  OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
  print(f"saved {OUT} ({res['sec']:.0f}s)")
