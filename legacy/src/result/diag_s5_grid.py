"""슬라이스 5 준비: 넓힌 격자에서 최적 설정이 내부에 오는지 확인한다.
baseline 칸(SCALE15)에서 4b의 미래 표본을 재사용해 학습 타깃 M = tgt_M, 검증 타깃 M = val_M로 둔다.
선형 λ, MLP·GNN weight decay를 넓게 훑고, 신경망은 에폭별 검증 곡선을 남겨 조기 종료가 필요한지도 본다.
"""
import json
import time
from dataclasses import asdict, replace
import numpy as np
from torch.optim.optimizer import register_optimizer_step_post_hook
from utils.params import Cfg, SCALE15
from utils.features import phi
from env.platform import Env
from env.graph import graph
from bench.rule_split import rule
from vfa.train import target
from vfa import linear, mlp
from vfa.gnn import GNN
from result import diag

OUT = diag.RUNS / "diag_s5_grid.json"
LAMS = [float(x) for x in np.logspace(-2, 6, 17)]
WDS_MLP = [0.0, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0]
WDS_GNN = [0.0, 1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0]
EPOCHS, EVERY = 1000, 25


# baseline 칸
def cell_cfg():
  c = Cfg()
  return replace(c, **SCALE15, vf_reps=c.vf_reps3, nn_epochs=EPOCHS)


# 튜닝 seed 궤적을 다시 걸어 상태 관측을 복원한다 (궤적은 결정적이므로 4b와 같은 상태)
def states(cfg, reps):
  tc = replace(cfg, seed=cfg.tune_seed)
  O = []
  for rep in reps:
    env, pol = Env(tc, rep), rule(tc)
    env.reset()
    for t in range(tc.T):
      if t <= tc.T - tc.vf_H:
        O.append(env.o)
      env.step(pol.act(env.o))
  return O


# EVERY 에폭마다 학습·검증 MSE를 기록하며 학습한다
def curve(m, Xt, yt, Xv, yv):
  tr, va, n = [], [], [0]
  def hook(opt, args, kw):
    n[0] += 1
    if n[0] % EVERY == 0:
      tr.append(float(np.mean((m.predict(Xt) - yt) ** 2)))
      va.append(float(np.mean((m.predict(Xv) - yv) ** 2)))
  h = register_optimizer_step_post_hook(hook)
  t0 = time.time()
  m.fit(Xt, yt)
  h.remove()
  return tr, va, time.time() - t0


# 신경망 한 종류를 wd 격자로 훑는다: 마지막 에폭과 최적 에폭의 검증 R²
def sweep_nn(make, wds, Xt, yt, Xv, yv):
  vv, rows = float(np.var(yv)), []
  for wd in wds:
    tr, va, sec = curve(make(wd), Xt, yt, Xv, yv)
    k = int(np.argmin(va))
    rows.append(dict(wd=wd, sec=sec, train=tr, val=va, r2_last=1 - va[-1] / vv, r2_best=1 - va[k] / vv,
                     best_epoch=(k + 1) * EVERY, r2_train_last=1 - tr[-1] / float(np.var(yt))))
    print(f"    wd {wd:<7g} R² 마지막 {rows[-1]['r2_last']:.4f} 최적 {rows[-1]['r2_best']:.4f} @ {rows[-1]['best_epoch']} "
          f"(train {rows[-1]['r2_train_last']:.3f}) {sec:.0f}s", flush=True)
  return rows


# 격자 끝 판정: 최적이 양 끝이면 범위가 모자라다
def edge(vals, scores):
  k = int(np.argmax(scores))
  return dict(pick=vals[k], idx=k, edge=bool(k in (0, len(vals) - 1)))


NOTE = ("슬라이스 5 준비. baseline 칸(SCALE15)에서 4b의 미래 표본을 재사용한다 (학습 타깃 M = tgt_M, 검증 타깃 M = val_M). "
        "선형 λ는 1e-2~1e6 로그 17점, MLP wd 10점, GNN wd 8점, 신경망은 1000에폭까지 25에폭마다 검증 MSE를 기록한다. "
        "이 탐색은 설계용이며 근사 방법 비교가 아니다.")

if __name__ == "__main__":
  t0, cfg = time.time(), cell_cfg()
  d4 = json.loads((diag.RUNS / "diag_s4b_train_m.json").read_text(encoding="utf-8"))
  r0 = round(cfg.vf_reps * (1 - cfg.vf_val))
  Otr, Ova = states(cfg, range(r0)), states(cfg, range(r0, cfg.vf_reps))
  ytr, yva = target(np.array(d4["Vtr"]), cfg.tgt_M), target(np.array(d4["Vva"]), cfg.val_M)
  assert len(Otr) == len(ytr) and len(Ova) == len(yva)
  print(f"학습 {len(ytr)} (M={cfg.tgt_M}), 검증 {len(yva)} (M={cfg.val_M}), 상태 복원 {time.time() - t0:.0f}s", flush=True)
  Xt, Xv = np.array([phi(o) for o in Otr]), np.array([phi(o) for o in Ova])
  vv = float(np.var(yva))
  res = dict(note=NOTE, cfg=asdict(cfg), n_train=len(ytr), n_val=len(yva), lams=LAMS, wds_mlp=WDS_MLP, wds_gnn=WDS_GNN,
             epochs=EPOCHS, every=EVERY)
  print("선형", flush=True)
  lin = [1 - float(np.mean((linear.Linear(l).fit(Xt, ytr).predict(Xv) - yva) ** 2)) / vv for l in LAMS]
  res["linear"] = dict(r2=lin, **edge(LAMS, lin))
  print(f"    최적 λ {res['linear']['pick']:g} R² {max(lin):.4f} 끝 {res['linear']['edge']}", flush=True)
  print("MLP", flush=True)
  res["mlp"] = sweep_nn(lambda wd: mlp.MLP(cfg, wd), WDS_MLP, Xt, ytr, Xv, yva)
  print("GNN", flush=True)
  Gt, Gv = np.array([graph(cfg, o) for o in Otr]), np.array([graph(cfg, o) for o in Ova])
  res["gnn"] = sweep_nn(lambda wd: GNN(cfg, wd), WDS_GNN, Gt, ytr, Gv, yva)
  for k, wds in (("mlp", WDS_MLP), ("gnn", WDS_GNN)):
    res[f"{k}_edge_last"] = edge(wds, [r["r2_last"] for r in res[k]])
    res[f"{k}_edge_best"] = edge(wds, [r["r2_best"] for r in res[k]])
  res["sec"] = time.time() - t0
  OUT.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
  print(f"saved {OUT} ({res['sec']:.0f}s)")
