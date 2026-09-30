"""실험 5: Linear·MLP·GNN 공정 비교 (확정 설계, 슬라이스 5).
튜닝 seed 반복을 학습 0~71(M = tgt_M) / 검증 72~89(M = val_M, 설정 선택) / 시험 90~107(M = val_M, 보고 전용)으로 나누고,
모델마다 설정 12개를 초기화 seed 하나로 골라 고른 설정을 seed 5개로 다시 학습해 시험 R²와 반복 블록 부트스트랩 CI를 낸다.
"""
import json
import sys
import time
from dataclasses import asdict, replace
import numpy as np
from utils.params import Cfg, SCALE15
from utils.features import phi
from env.platform import Env
from env.graph import graph
from bench.rule_split import rule
from match.mc_value import mc_value
from vfa.train import tkey, target
from vfa.linear import Linear
from vfa.mlp import MLP
from vfa.gnn import GNN, hidden, n_params
from result import diag

NAMES = ("선형", "MLP", "GNN")
PAIRS = (("MLP", "선형"), ("GNN", "선형"), ("GNN", "MLP"))


# 칸 설정: 확정 환경(SCALE15)에 시장 조건을 얹는다
def cell_cfg(n_alt=4, conc=0.5, k=2):
  c = Cfg()
  return replace(c, **{**SCALE15, "n_alt": n_alt, "conc": conc, "items_per_order": k}, vf_reps=c.vf_reps3)


# 칸 이름
def tag(cfg):
  return f"alt{cfg.n_alt}_conc{cfg.conc:g}_k{cfg.items_per_order}"


# 반복 범위: 학습 / 검증 / 시험 (시험은 vf_reps 뒤 test_reps개)
def splits(cfg):
  r0 = round(cfg.vf_reps * (1 - cfg.vf_val))
  return dict(train=range(r0), val=range(r0, cfg.vf_reps), test=range(cfg.vf_reps, cfg.vf_reps + cfg.test_reps))


# 튜닝 seed 궤적을 반복 reps로 걸으며 t ≤ T − vf_H 상태의 관측과 반복 번호를 모은다 (futures=k면 상태마다 미래 k개도)
def walk(cfg, reps, futures=0):
  tc = replace(cfg, seed=cfg.tune_seed)
  O, V, g = [], [], []
  for rep in reps:
    env, pol = Env(tc, rep), rule(tc)
    env.reset()
    for t in range(tc.T):
      if t <= tc.T - tc.vf_H:
        O.append(env.o)
        g.append(rep)
        if futures:
          V.append(mc_value(env, R=futures, key=tkey))
      env.step(pol.act(env.o))
  return O, (np.array(V) if futures else None), np.array(g)


# 칸 데이터: 분할마다 관측·미래 행렬·반복 번호. pre = {분할: 미래 행렬}이면 미래를 다시 재지 않고 관측만 복원한다
def data(cfg, pre=None):
  sp, k = splits(cfg), dict(train=cfg.tgt_M, val=cfg.val_M, test=cfg.val_M)
  out, sec = {}, {}
  for s, reps in sp.items():
    t0 = time.time()
    if pre and s in pre:
      O, _, g = walk(cfg, reps)
      V = np.asarray(pre[s])
      assert len(V) == len(O) and V.shape[1] >= k[s]
    else:
      O, V, g = walk(cfg, reps, k[s])
    out[s] = dict(O=O, V=V[:, :k[s]], g=g, y=target(V, k[s]))
    sec[s] = time.time() - t0
  return out, sec


# 모델별 설정 12개: (이름, 설정 → 초기화 seed s의 미학습 모델, 입력 변환)
def grid(cfg, name):
  base = replace(cfg, nn_epochs=cfg.es_epochs)
  if name == "선형":
    return [(f"lam={l:.4g}", lambda s, l=l: Linear(l), phi) for l in cfg.fair_lams]
  mk = MLP if name == "MLP" else GNN
  enc = phi if name == "MLP" else (lambda o, c=base: graph(c, o))
  return [(f"h={h} wd={w:g}", lambda s, h=h, w=w: mk(replace(base, nn_hidden=h, nn_seed=s), w), enc)
          for h in cfg.fair_hidden for w in cfg.fair_wds]


# 한 설정 학습: 신경망은 검증 데이터로 조기 종료, 선형은 닫힌 해
def fit(m, Xt, yt, Xv, yv):
  return m.fit(Xt, yt) if isinstance(m, Linear) else m.fit(Xt, yt, Xv, yv)


# 설정 선택: 학습·검증만 받는다 (시험 데이터는 이 함수에 들어올 수 없다)
def select(cfg, name, tr, va):
  rows, first = [], cfg.fair_seeds[0]
  for lab, make, enc in grid(cfg, name):
    Xt, Xv = np.array([enc(o) for o in tr["O"]]), np.array([enc(o) for o in va["O"]])
    t0 = time.time()
    m = fit(make(first), Xt, tr["y"], Xv, va["y"])
    sec = time.time() - t0
    mse = float(np.mean((m.predict(Xv) - va["y"]) ** 2))
    be = getattr(m, "best_epoch", None)
    if be is not None:
      assert abs(mse - min(m.val_curve)) <= 1e-6 * max(mse, 1.0)  # 되돌린 가중치가 곡선 최솟값과 같다
    rows.append(dict(config=lab, val_mse=mse, val_r2=1 - mse / float(np.var(va["y"])), best_epoch=be, sec=sec,
                     params=(n_params(m.cfg, m.h) if name == "GNN" else None)))
  k = int(np.argmin([r["val_mse"] for r in rows]))
  return k, rows


# 고른 설정을 초기화 seed 전부로 다시 학습해 시험 예측을 낸다 (시험은 여기서 처음 쓴다)
def final(cfg, name, k, tr, va, te):
  lab, make, enc = grid(cfg, name)[k]
  Xt, Xv, Xs = (np.array([enc(o) for o in d["O"]]) for d in (tr, va, te))
  preds, secs, eps = [], [], []
  for s in cfg.fair_seeds:
    t0 = time.time()
    m = fit(make(s), Xt, tr["y"], Xv, va["y"])
    secs.append(time.time() - t0)
    preds.append(m.predict(Xs))
    eps.append(getattr(m, "best_epoch", None))
  return dict(config=lab, pred=np.array(preds), sec=secs, best_epoch=eps)


# 시험 R²
def r2(p, y, idx=None):
  idx = np.arange(len(y)) if idx is None else idx
  return float(1 - np.mean((p[idx] - y[idx]) ** 2) / np.var(y[idx]))


# 시험 반복 블록 부트스트랩: 표본마다 seed 평균 R² 차이 (선형은 seed와 무관하지만 같은 모양으로 둔다)
def boot(cfg, P, y, g):
  reps, rng = np.unique(g), np.random.default_rng(cfg.seed)
  where = {r: np.where(g == r)[0] for r in reps}
  out = {}
  draws = [np.concatenate([where[r] for r in rng.choice(reps, len(reps))]) for _ in range(cfg.boot_B)]
  for a, b in PAIRS:
    ds = np.array([np.mean([r2(pa, y, i) - r2(pb, y, i) for pa, pb in zip(P[a], P[b])]) for i in draws])
    full = float(np.mean([r2(pa, y) - r2(pb, y) for pa, pb in zip(P[a], P[b])]))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    out[f"{a}-{b}"] = dict(diff=full, lo=float(lo), hi=float(hi), p_le0=float(np.mean(ds <= 0)))
  return out


# 달성 가능 R² 상한 (타깃 M개 평균의 잔여 잡음)
def ceiling(V):
  return float(1 - V.var(1, ddof=1).mean() / V.shape[1] / np.var(V.mean(1)))


# 칸 하나 전체: 데이터 → 모델별 선택 → 최종 5 seed → 부트스트랩
def run_cell(cfg, pre=None, out=None):
  t0 = time.time()
  D, sec_data = data(cfg, pre)
  tr, va, te = D["train"], D["val"], D["test"]
  assert not (set(tr["g"]) & set(va["g"]) | set(tr["g"]) & set(te["g"]) | set(va["g"]) & set(te["g"]))
  res = dict(cfg=asdict(cfg), tag=tag(cfg), n={s: len(D[s]["y"]) for s in D}, reps={s: [int(min(D[s]["g"])), int(max(D[s]["g"]))] for s in D},
             sec_data=sec_data, ceiling={s: ceiling(D[s]["V"]) for s in ("val", "test")}, models={})
  P = {}
  for name in NAMES:
    t1 = time.time()
    k, rows = select(cfg, name, tr, va)
    sec_sel = time.time() - t1
    f = final(cfg, name, k, tr, va, te)
    P[name] = f["pred"]
    tr2 = [r2(p, te["y"]) for p in f["pred"]]
    res["models"][name] = dict(pick=k, config=f["config"], edge=bool(k in (0, len(rows) - 1)) if name == "선형" else None,
                               select=rows, sec_select=sec_sel, sec_final=f["sec"], best_epoch_final=f["best_epoch"],
                               test_r2=tr2, test_r2_mean=float(np.mean(tr2)), test_r2_sd=float(np.std(tr2, ddof=1)))
    print(f"  {name}: 고른 설정 {f['config']} (검증 R² {rows[k]['val_r2']:.4f}) → 시험 R² {np.mean(tr2):.4f} ± {np.std(tr2, ddof=1):.4f} "
          f"| 선택 {sec_sel:.0f}s 최종 {sum(f['sec']):.0f}s", flush=True)
  res["boot"] = boot(cfg, P, te["y"], te["g"])
  res["test_pred"] = {n: P[n].tolist() for n in NAMES}
  res["test_y"], res["test_g"] = te["y"].tolist(), te["g"].tolist()
  res["futures"] = {s: D[s]["V"].tolist() for s in D}
  res["sec"] = time.time() - t0
  if out:
    out.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
  return res


if __name__ == "__main__":
  # 점검: baseline 칸 하나. 학습·검증 미래는 4b에서 잰 것을 쓰되 반복 하나씩 다시 재서 같은지 확인하고, 시험은 새로 잰다
  cfg = cell_cfg()
  if "check" in sys.argv:
    d4 = json.loads((diag.RUNS / "diag_s4b_train_m.json").read_text(encoding="utf-8"))
    sp = splits(cfg)
    Vtr, Vva = np.array(d4["Vtr"]), np.array(d4["Vva"])
    n_per = cfg.T - cfg.vf_H + 1
    _, v0, _ = walk(cfg, [sp["train"][0]], cfg.tgt_M)
    _, v1, _ = walk(cfg, [sp["val"][0]], cfg.val_M)
    same = bool(np.allclose(v0, Vtr[:n_per, :cfg.tgt_M]) and np.allclose(v1, Vva[:n_per, :cfg.val_M]))
    print(f"4b 미래 재현: {same}", flush=True)
    assert same
    res = run_cell(cfg, pre=dict(train=Vtr, val=Vva), out=diag.RUNS / f"exp5_check_{tag(cfg)}.json")
    print("부트스트랩:", {k: (round(v["diff"], 4), round(v["lo"], 4), round(v["hi"], 4)) for k, v in res["boot"].items()})
    print(f"total {res['sec']:.0f}s")
