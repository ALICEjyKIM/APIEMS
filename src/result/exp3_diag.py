"""실험 3 v1 진단: 학습 곡선, 정답(vf_H기간 이윤 합)의 잡음 크기, GNN 입력 크기, 정답 평균화 시 수집 시간 추정.
기존 학습 코드(MLP.fit, GNN.fit)를 그대로 쓰고 옵티마이저 스텝 뒤 훅으로 에폭마다 학습·검증 평균제곱오차를 기록한다 (튜닝 seed만).
원자료 result/runs/exp3_diag.json, 그림 result/figures/exp3_diag_*.png, 학습한 모델 result/models/exp3_v1_*.pt.
"""
import json
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook
from env.platform import Env
from bench.rule_split import rule
from match.mc_value import mc_value
from vfa.train import collect_obs, val_mask
from vfa.mlp import MLP
from vfa.gnn import GNN
from env.graph import phi_conc, graph
from utils.features import phi
from result import diag
from result.exp3_graph import CELLS, NAMES, cell_cfg, tag

ROOT = Path(__file__).resolve().parent
NOISE_N = 20  # 잡음 크기를 재는 학습 상태 수 (학습 반복 0 ~ NOISE_N − 1에서 하나씩)
NOISE_R = 10  # 상태당 미래 수
AVG_R = 5  # 수집 시간 추정: 상태당 미래 AVG_R번 평균을 정답으로 할 때


# 모델 이름 → (만들기, 입력 변환)
def makers(cfg):
  enc = {"MLP": phi, "MLP+편중도 요약": phi_conc, "GNN": lambda o: graph(cfg, o)}
  mk = {"MLP": lambda wd: MLP(cfg, wd), "MLP+편중도 요약": lambda wd: MLP(cfg, wd), "GNN": lambda wd: GNN(cfg, wd)}
  return {n: (mk[n], enc[n]) for n in NAMES}


# 학습 곡선: 스텝 뒤 훅으로 매 에폭 학습·검증 평균제곱오차를 기록하며 학습 분할에 맞춘다
def curve(m, Xt, yt, Xv, yv):
  tr, va = [], []
  def hook(opt, args, kw):
    tr.append(float(np.mean((m.predict(Xt) - yt) ** 2)))
    va.append(float(np.mean((m.predict(Xv) - yv) ** 2)))
  h = register_optimizer_step_post_hook(hook)
  m.fit(Xt, yt)
  h.remove()
  return tr, va


# 판정 (roadmap 결정 기록의 기준): 조기 종료 필요 / 외움 / 덜 배움 / 정상
def judge(tr, va, vt, vv, ceil):
  E, k = len(va), int(np.argmin(va))
  r2t, r2v = 1 - tr[-1] / vt, 1 - va[-1] / vv
  lab = []
  if k < 0.9 * E and va[-1] >= 1.05 * va[k]:
    lab.append("조기 종료 필요")
  if r2t - r2v >= 0.3:
    lab.append("외움")
  if r2t < 0.5 * ceil:
    lab.append("덜 배움")
  return dict(labels=lab or ["정상"], r2_train=r2t, r2_val=r2v, best_ep=k + 1, best_r2_val=1 - va[k] / vv, r2_val_last_over_best=va[-1] / va[k])


# 모델 저장: 신경망 가중치와 표준화 값 (입력 변환은 이름으로 복원)
def save(m, name, cell, split):
  d = ROOT / "models"
  d.mkdir(exist_ok=True)
  p = d / f"exp3_v1_{tag(*cell)}_{'GNN' if name == 'GNN' else ('MLPconc' if name != 'MLP' else 'MLP')}_{split}.pt"
  torch.save(dict(name=name, cell=cell, split=split, wd=m.wd, state=m.net.state_dict(), mu=m.mu, sd=m.sd, ym=m.ym, ys=m.ys), p)
  return p.name


# 저장한 모델 불러오기 → fit 없이 predict·vfa/coef.raw에 쓸 수 있는 모델
def load(path):
  s = torch.load(path, weights_only=False)
  cfg = cell_cfg(*s["cell"])
  mk, enc = makers(cfg)[s["name"]]
  m = mk(s["wd"])
  m.enc, m.mu, m.sd, m.ym, m.ys = enc, s["mu"], s["sd"], s["ym"], s["ys"]
  if s["name"] == "GNN":
    from vfa.gnn import Net
    m.net = Net(2 * cfg.n_items + 2, cfg.gnn_hidden, cfg.gnn_layers)
  else:
    dims = [len(s["mu"])] + [cfg.nn_hidden] * cfg.nn_layers
    m.net = torch.nn.Sequential(*[x for a, b in zip(dims[:-1], dims[1:]) for x in (torch.nn.Linear(a, b), torch.nn.ReLU())], torch.nn.Linear(dims[-1], 1))
  m.net.load_state_dict(s["state"])
  return m


# 잡음 크기: 학습 반복 r의 기간 5 + 5 × (r mod 4) 상태에서 미래 NOISE_R번의 vf_H기간 이윤 합
def noise(cfg):
  tc, rows, sec = replace(cfg, seed=cfg.tune_seed), [], 0.0
  for r in range(NOISE_N):
    t0, env, pol = 5 + 5 * (r % 4), Env(tc, r), rule(tc)
    env.reset()
    for _ in range(t0):
      env.step(pol.act(env.o))
    s = time.time()
    rows.append(dict(rep=r, t=t0, v=mc_value(env, R=NOISE_R).tolist()))
    sec += time.time() - s
  return rows, sec / (NOISE_N * NOISE_R)


# GNN 입력 크기: 활성 노드 표준화 특징의 종류별 평균 절댓값, 주문자 수 범위
def gnn_scale(cfg, Xg, m):
  n, d = cfg.gnn_nb + cfg.n_sup, 2 * cfg.n_items + 2
  x, mask = Xg[:, :n * d].reshape(-1, n, d), Xg[:, n * d:n * d + n].astype(bool)
  z = (x - m.mu) / m.sd
  nb = cfg.gnn_nb
  zb, zs = z[:, :nb][mask[:, :nb]], z[:, nb:][mask[:, nb:]]
  I = cfg.n_items
  grp = lambda a: dict(qty=float(np.abs(a[:, :I]).mean()), price=float(np.abs(a[:, I:2 * I]).mean()), ret=float(np.abs(a[:, 2 * I]).mean()), all=float(np.abs(a).mean()), max=float(np.abs(a).max()))
  raw = lambda a: dict(qty=float(a[:, :I].mean()), price=float(a[:, I:2 * I].mean()))
  B = mask[:, :nb].sum(1)
  return dict(z_buy=grp(zb), z_sup=grp(zs), raw_buy=raw(x[:, :nb][mask[:, :nb]]), raw_sup=raw(x[:, nb:][mask[:, nb:]]),
              n_buy=[int(B.min()), float(B.mean()), int(B.max())], n_sup=[int(mask[:, nb:].sum(1).min()), float(mask[:, nb:].sum(1).mean()), int(mask[:, nb:].sum(1).max())])


# 빈 자리 점검: (출력 차이) 비활성 자리의 특징과 비활성끼리의 연결선에 난수를 넣어도 GNN 출력이 같은지, (연결선 수) 데이터에서 비활성 자리에 닿는 연결선 수
def mask_check(cfg, Xg, m):
  n, d, nb = cfg.gnn_nb + cfg.n_sup, 2 * cfg.n_items + 2, cfg.gnn_nb
  mk = Xg[:, n * d:n * d + n]
  touch = float(((1 - mk[:, :nb, None] * mk[:, None, nb:]) * Xg[:, n * d + n:].reshape(-1, nb, cfg.n_sup)).sum())
  X = Xg[:50].copy()
  x, mask = X[:, :n * d].reshape(-1, n, d), X[:, n * d:n * d + n]
  A = X[:, n * d + n:].reshape(-1, nb, cfg.n_sup)
  g = np.random.default_rng(0)
  x += g.normal(size=x.shape) * 100 * (1 - mask[..., None])
  A += (g.random(A.shape) < 0.5) * (1 - mask[:, :nb, None]) * (1 - mask[:, None, nb:])
  X2 = np.c_[x.reshape(len(X), -1), mask, A.reshape(len(X), -1)]
  return dict(out_diff=float(np.abs(m.predict(X2) - m.predict(Xg[:50])).max()), edges_touching_empty=touch)


# 칸 하나: 수집 → 모델 × weight decay {0, 0.1} 학습 곡선 → v1 선택값 전체 재학습·저장 → 잡음 → GNN 점검
def run(cell):
  t0, cfg = time.time(), cell_cfg(*cell)
  O, y, g = collect_obs(cfg)
  tcol = time.time() - t0
  va = val_mask(cfg, g)
  rows, sec_roll = noise(cfg)
  V = np.array([r["v"] for r in rows])
  ceil = float(1 - V.var(1, ddof=1).mean() / y.var())
  out = dict(n=len(y), y_var=float(y.var()), y_var_tr=float(y[~va].var()), y_var_va=float(y[va].var()), collect_sec=tcol, sec_per_rollout=sec_roll,
             noise=rows, ceil=ceil, models={})
  pick = json.loads((diag.RUNS / ("exp3_gnn.json")).read_text(encoding="utf-8"))["cells"][tag(*cell)]["fit"]
  pick.update(json.loads((diag.RUNS / "exp3_mlp.json").read_text(encoding="utf-8"))["cells"][tag(*cell)]["fit"])
  for n, (mk, enc) in makers(cfg).items():
    X = np.array([enc(o) for o in O])
    rec = {}
    for wd in (0.0, 0.1):
      s = time.time()
      m = mk(wd)
      tr, vl = curve(m, X[~va], y[~va], X[va], y[va])
      rec[str(wd)] = dict(train=tr, val=vl, sec=time.time() - s, file=save(m, n, cell, f"tr_wd{wd:g}"), **judge(tr, vl, y[~va].var(), y[va].var(), ceil))
    s = time.time()
    m = mk(pick[n]["pick"]).fit(X, y)
    rec["full"] = dict(wd=m.wd, sec=time.time() - s, file=save(m, n, cell, "full"))
    if n == "GNN":
      rec["scale"] = gnn_scale(cfg, X, m)
      rec["mask_diff"] = mask_check(cfg, X, m)
    out["models"][n] = rec
  out["sec"] = time.time() - t0
  return tag(*cell), out


# 칸별 학습 곡선 그림 (행 = 모델, 열 = weight decay 0 / 0.1, 세로축 = 목표 분산 대비 MSE)
def figures(D):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  (ROOT / "figures").mkdir(exist_ok=True)
  for tg, c in D["cells"].items():
    f, ax = plt.subplots(3, 2, figsize=(10, 10), sharey=True)
    for i, n in enumerate(NAMES):
      for j, wd in enumerate(("0.0", "0.1")):
        r, a = c["models"][n][wd], ax[i, j]
        a.plot(np.array(r["train"]) / c["y_var_tr"], label="train")
        a.plot(np.array(r["val"]) / c["y_var_va"], label="val")
        a.axhline(1 - c["ceil"], ls=":", c="gray", label="noise floor")
        a.set_title(f"{n if n != 'MLP+편중도 요약' else 'MLP+conc'} wd={wd}  [{', '.join(r['labels'])}]".replace("조기 종료 필요", "early-stop").replace("외움", "memorize").replace("덜 배움", "underfit").replace("정상", "ok"), fontsize=9)
        a.set_ylim(0, 1.6)
        a.set_xlabel("epoch")
    ax[0, 0].set_ylabel("MSE / var(y)")
    ax[0, 0].legend(fontsize=8)
    f.suptitle(f"exp3 v1 diag {tg} (source: exp3_diag.json)")
    f.tight_layout()
    f.savefig(ROOT / "figures" / f"exp3_diag_{tg}.png", dpi=110)
    plt.close(f)


# 표: 칸 × 모델 판정, 잡음 크기, GNN 입력 크기, 수집 시간 추정
def tables(D):
  out = ["출처: exp3_diag.json", "", "| 칸 | 모델 | wd | 판정 | 학습 R² | 검증 R² (마지막) | 검증 R² 최고 (에폭) | 마지막/최저 검증 MSE |", "|---|---|---|---|---|---|---|---|"]
  for tg, c in D["cells"].items():
    for n in NAMES:
      for wd in ("0.0", "0.1"):
        r = c["models"][n][wd]
        out.append(f"| {tg} | {n} | {wd} | {', '.join(r['labels'])} | {r['r2_train']:.3f} | {r['r2_val']:.3f} | {r['best_r2_val']:.3f} ({r['best_ep']}) | {r['r2_val_last_over_best']:.2f} |")
  out += ["", "| 칸 | 한 번 돌린 값 표준편차 (상태 안 평균) | 상태 간 표준편차 (미래 10번 평균의) | 학습 목표 전체 표준편차 | 잡음 상한 R² (1회) | 잡음 상한 R² (5회 평균) | 롤아웃당 초 |", "|---|---|---|---|---|---|---|"]
  for tg, c in D["cells"].items():
    V = np.array([r["v"] for r in c["noise"]])
    w = V.var(1, ddof=1).mean()
    out.append(f"| {tg} | {np.sqrt(w):.0f} | {V.mean(1).std(ddof=1):.0f} | {np.sqrt(c['y_var']):.0f} | {c['ceil']:.3f} | {1 - w / AVG_R / (c['y_var'] - w + w / AVG_R):.3f} | {c['sec_per_rollout']:.3f} |")
  out += ["", "| 칸 | 주문자 수 (최소/평균/최대) | 공급자 수 | 표준화 특징 평균 |z| 주문자 (수량/가격/재참여) | 공급자 (수량/가격/재참여) | 최대 |z| 주문자 / 공급자 | 원래 평균 수량 주문자 / 공급자 | 빈 자리 교란 시 출력 차이 |", "|---|---|---|---|---|---|---|---|"]
  for tg, c in D["cells"].items():
    s, zb, zs = c["models"]["GNN"]["scale"], c["models"]["GNN"]["scale"]["z_buy"], c["models"]["GNN"]["scale"]["z_sup"]
    out.append(f"| {tg} | {s['n_buy'][0]}/{s['n_buy'][1]:.1f}/{s['n_buy'][2]} | {s['n_sup'][0]}/{s['n_sup'][1]:.1f}/{s['n_sup'][2]} | {zb['qty']:.2f}/{zb['price']:.2f}/{zb['ret']:.2f} | "
               f"{zs['qty']:.2f}/{zs['price']:.2f}/{zs['ret']:.2f} | {zb['max']:.1f} / {zs['max']:.1f} | {s['raw_buy']['qty']:.1f} / {s['raw_sup']['qty']:.1f} | {c['models']['GNN']['mask_diff']['out_diff']:.2e} (빈 자리 연결선 {c['models']['GNN']['mask_diff']['edges_touching_empty']:.0f}) |")
  cfg = cell_cfg(*CELLS[0])
  n_st = cfg.vf_reps * (cfg.T - cfg.vf_H + 1)
  out += ["", "| 칸 | 기존 수집 초 | 미래 5번 평균 정답 추가 초 (1프로세스) | 합계 분 (1프로세스) | 합계 분 (칸당 6프로세스) |", "|---|---|---|---|---|"]
  for tg, c in D["cells"].items():
    add = n_st * AVG_R * c["sec_per_rollout"]
    out.append(f"| {tg} | {c['collect_sec']:.0f} | {add:.0f} | {(c['collect_sec'] + add) / 60:.1f} | {(c['collect_sec'] + add / 6) / 60:.1f} |")
  out += ["", "| 칸 | 모델 | 학습 곡선 1회 초 (wd 0 / 0.1) | 전체 재학습 초 |", "|---|---|---|---|"]
  for tg, c in D["cells"].items():
    for n in NAMES:
      r = c["models"][n]
      out.append(f"| {tg} | {n} | {r['0.0']['sec']:.0f} / {r['0.1']['sec']:.0f} | {r['full']['sec']:.0f} |")
  return "\n".join(out)


if __name__ == "__main__":
  t0 = time.time()
  path = diag.RUNS / "exp3_diag.json"
  if "run" in sys.argv:
    with Pool(len(CELLS)) as p:
      cells = dict(p.map(run, CELLS))
    note = f"튜닝 seed, v1 설정(학습 {cell_cfg(*CELLS[0]).vf_reps}반복), 잡음 상태 {NOISE_N}개 × 미래 {NOISE_R}번"
    path.write_text(json.dumps(dict(note=note, cells=cells)), encoding="utf-8")
    print(path, f"{time.time() - t0:.0f}s", flush=True)
  D = json.loads(path.read_text(encoding="utf-8"))
  if "fig" in sys.argv:
    figures(D)
  if "table" in sys.argv:
    print(tables(D))
