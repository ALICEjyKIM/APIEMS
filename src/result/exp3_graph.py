"""실험 3: MLP vs MLP + 편중도 요약 vs GNN (축소 4칸: 공급 편중도 {0, 1} × 주문당 품목 수 {1, 2}, n_alt 2, 새 가격 생성 방식).
칸마다 같은 관측 데이터(튜닝 seed vf_reps3반복)로 세 모델을 학습하고, 검증 반복 상태에서 시뮬레이션 기준치 대비 예측 오차와 유지 가치 변환 오차를 잰다.
mlp 단계 → result/runs/exp3_mlp.json (기준치 포함), gnn 단계 → result/runs/exp3_gnn.json (같은 상태의 GNN 예측). 칸들은 프로세스 병렬로 돈다.
"""
import json
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
import numpy as np
from utils.params import Cfg
from env.platform import Env
from bench.rule_split import rule
from match.mc_value import mc_value
from vfa.train import collect_obs, val_mask, select
from vfa.mlp import MLP
from vfa.gnn import GNN
from vfa.coef import raw, mc_coefs, conv_err
from env.graph import phi_conc, graph
from utils.features import phi
from result import diag

CELLS = [(0.0, 1), (0.0, 2), (1.0, 1), (1.0, 2)]  # (공급 편중도, 주문당 품목 수)
PRED_T = (5, 10, 15, 20)  # 예측 오차 상태의 기간 (실험 1과 같음)
CONV_T = (10, 20)  # 변환 오차 상태의 기간 (실험 1과 같음, 10기간 간격이라 conv_R개 미래 키가 겹치지 않음)
NAMES = ("MLP", "MLP+편중도 요약", "GNN")


# 칸의 설정: 새 가격 생성 방식, 학습 반복 vf_reps3
def cell_cfg(conc, k):
  c = Cfg()
  return replace(c, conc=conc, items_per_order=k, vf_reps=c.vf_reps3, price_rank=True)


# 칸 이름
def tag(conc, k):
  return f"conc{conc:g}_k{k}"


# 튜닝 seed 검증 반복 중 앞쪽 e3_state_reps개를 규칙 기반으로 돌리며 기간 ts의 (반복, 기간, 환경)을 낸다
def states(cfg, ts):
  tc, r0 = replace(cfg, seed=cfg.tune_seed), round(cfg.vf_reps * (1 - cfg.vf_val))
  for rep in range(r0, r0 + cfg.e3_state_reps):
    env, pol = Env(tc, rep), rule(tc)
    env.reset()
    for t in range(max(ts) + 1):
      if t in ts:
        yield rep, t, env
      env.step(pol.act(env.o))


# 모델 하나 학습 → (모델, 적합 기록). enc가 있으면 모델 입력 변환으로 둔다
def fit(cfg, make, O, y, g, enc):
  X = np.array([enc(o) for o in O])
  m, mse = select(cfg, make, cfg.nn_wds, X, y, g)
  va, a = val_mask(cfg, g), m.wd
  return m, dict(grid=list(cfg.nn_wds), val_mse=[mse[w] for w in cfg.nn_wds], pick=a, edge=a in (cfg.nn_wds[0], cfg.nn_wds[-1]), val_r2=float(1 - mse[a] / np.var(y[va])))


# MLP·MLP + 편중도 요약 모델 만들기 (입력 변환 enc를 붙인다)
def mlp_make(cfg, enc):
  def mk(wd):
    m = MLP(cfg, wd)
    m.enc = enc
    return m
  return mk


# mlp 단계 (칸 하나): 학습 데이터 수집, 두 MLP 학습, 기준치와 예측·가드 전 유지 가치
def run_mlp(cell):
  t0, cfg = time.time(), cell_cfg(*cell)
  O, y, g = collect_obs(cfg)
  ms, fits = {}, {}
  for n, enc in (("MLP", phi), ("MLP+편중도 요약", phi_conc)):
    ms[n], fits[n] = fit(cfg, mlp_make(cfg, enc), O, y, g, enc)
  pred = [dict(rep=r, t=t, mc=mc_value(env).tolist(), **{n: float(raw(ms[n], env.o)[2]) for n in ms}) for r, t, env in states(cfg, PRED_T)]
  conv = []
  for r, t, env in states(cfg, CONV_T):
    cb, cs, _, sb, ss = mc_coefs(env, cfg.conv_R)
    row = dict(rep=r, t=t, mc_buy=cb.tolist(), mc_sup=cs.tolist(), se_buy=sb.tolist(), se_sup=ss.tolist())
    for n in ms:
      c = raw(ms[n], env.o)
      row[n] = dict(buy=c[0].tolist(), sup=c[1].tolist())
    conv.append(row)
  return tag(*cell), dict(cfg=asdict(cfg), n=len(y), n_val=int(val_mask(cfg, g).sum()), y_mean=float(y.mean()), y_sd=float(y.std()),
                          max_buyers=max(len(o.bid) for o in O), fit=fits, pred=pred, conv=conv, sec=time.time() - t0)


# gnn 단계 (칸 하나): 같은 학습 데이터로 GNN 학습, mlp 단계와 같은 상태에서 예측·가드 전 유지 가치
def run_gnn(cell):
  t0, cfg = time.time(), cell_cfg(*cell)
  O, y, g = collect_obs(cfg)
  m, f = fit(cfg, lambda wd: GNN(cfg, wd), O, y, g, lambda o: graph(cfg, o))
  pred = [dict(rep=r, t=t, GNN=float(raw(m, env.o)[2])) for r, t, env in states(cfg, PRED_T)]
  conv = []
  for r, t, env in states(cfg, CONV_T):
    c = raw(m, env.o)
    conv.append(dict(rep=r, t=t, GNN=dict(buy=c[0].tolist(), sup=c[1].tolist())))
  params = sum(p.numel() for p in m.net.parameters())
  return tag(*cell), dict(cfg=asdict(cfg), n=len(y), fit={"GNN": f}, params=params, pred=pred, conv=conv, sec=time.time() - t0)


# 단계 실행: 칸 4개를 프로세스 4개로 돌리고 json 하나로 저장
def stage(name, fn, note):
  with Pool(len(CELLS)) as p:
    out = dict(p.map(fn, CELLS))
  path = diag.RUNS / f"exp3_{name}.json"
  path.write_text(json.dumps(dict(note=note, cells=out)), encoding="utf-8")
  return path


# 두 json에서 칸별 표: 적합, 예측 오차(편향·평균 절대 오차·모델 간 짝지은 차이), 변환 오차(모델 간 짝지은 차이, 잡음 바닥)
def tables():
  A = json.loads((diag.RUNS / "exp3_mlp.json").read_text(encoding="utf-8"))
  G = json.loads((diag.RUNS / "exp3_gnn.json").read_text(encoding="utf-8"))
  ci = lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))
  pm = lambda x: f"{np.mean(x):+.1f} ± {ci(x):.1f}"
  out = [f"원자료: exp3_mlp.json ({A['note']}), exp3_gnn.json ({G['note']})", ""]
  pairs = [("GNN", "MLP+편중도 요약"), ("GNN", "MLP"), ("MLP+편중도 요약", "MLP")]
  for tg, a in A["cells"].items():
    b = G["cells"][tg]
    assert [(r["rep"], r["t"]) for r in a["pred"]] == [(r["rep"], r["t"]) for r in b["pred"]]
    fits = {**a["fit"], **b["fit"]}
    out += [f"### {tg} (학습 {a['n']}개, 검증 {a['n_val']}개, 목표 평균 {a['y_mean']:.0f}, 최대 주문자 수 {a['max_buyers']}, GNN 파라미터 {b['params']})", "",
            "| 모델 | 고른 weight decay (격자 끝) | 검증 R² | 예측 편향 (95% CI) | 예측 평균 절대 오차 | 변환 주문자 평균 절대 차이 | 변환 공급자 평균 절대 차이 | 모델 c 평균 주문자 / 공급자 |",
            "|---|---|---|---|---|---|---|---|"]
    mc = np.array([np.mean(r["mc"]) for r in a["pred"]])
    e = {n: np.array([r[n] for r in (b if n == "GNN" else a)["pred"]]) - mc for n in NAMES}
    cc = {n: {k: np.concatenate([np.asarray(r[n][k], float) for r in (b if n == "GNN" else a)["conv"]]) for k in ("buy", "sup")} for n in NAMES}
    mk = {k: np.concatenate([np.asarray(r[f"mc_{k}"], float) for r in a["conv"]]) for k in ("buy", "sup")}
    sk = {k: np.concatenate([np.asarray(r[f"se_{k}"], float) for r in a["conv"]]) for k in ("buy", "sup")}
    for n in NAMES:
      f = fits[n]
      out.append(f"| {n} | {f['pick']} ({'예' if f['edge'] else '아니오'}) | {f['val_r2']:.3f} | {pm(e[n])} | {np.abs(e[n]).mean():.1f} | "
                 f"{conv_err(cc[n]['buy'], mk['buy']):.1f} | {conv_err(cc[n]['sup'], mk['sup']):.1f} | {cc[n]['buy'].mean():.1f} / {cc[n]['sup'].mean():.1f} |")
    out += ["", f"기준치: 예측 상태 {len(mc)}개 평균 {mc.mean():.0f} (표준오차 평균 {np.mean([np.std(r['mc'], ddof=1) / np.sqrt(len(r['mc'])) for r in a['pred']]):.0f}), "
                f"변환 상태 {len(a['conv'])}개 기준치 c 평균 주문자 {mk['buy'].mean():.1f} / 공급자 {mk['sup'].mean():.1f}, 잡음 바닥(표준오차 평균) {sk['buy'].mean():.1f} / {sk['sup'].mean():.1f}", "",
            "| 짝지은 차이 (앞 − 뒤, 음수면 앞 모델이 더 정확) | 예측 절대 오차 (상태 단위) | 변환 절대 차이 주문자 (참여자 단위) | 변환 절대 차이 공급자 (참여자 단위) |", "|---|---|---|---|"]
    for x, z in pairs:
      out.append(f"| {x} − {z} | {pm(np.abs(e[x]) - np.abs(e[z]))} | {pm(np.abs(cc[x]['buy'] - mk['buy']) - np.abs(cc[z]['buy'] - mk['buy']))} | "
                 f"{pm(np.abs(cc[x]['sup'] - mk['sup']) - np.abs(cc[z]['sup'] - mk['sup']))} |")
    out.append("")
  return "\n".join(out)


if __name__ == "__main__":
  t0 = time.time()
  c = Cfg()
  note = f"튜닝 seed {c.tune_seed}, 학습 {c.vf_reps3}반복, 예측 상태 = 검증 반복 {c.e3_state_reps}개 × 기간 {PRED_T} (rollout {c.mc_R}), 변환 상태 = 검증 반복 {c.e3_state_reps}개 × 기간 {CONV_T} (rollout {c.conv_R}), H {c.vf_H}, price_rank True"
  if "mlp" in sys.argv:
    print(stage("mlp", run_mlp, note), f"{time.time() - t0:.0f}s", flush=True)
  if "gnn" in sys.argv:
    print(stage("gnn", run_gnn, note), f"{time.time() - t0:.0f}s", flush=True)
  if "table" in sys.argv:
    print(tables())
