"""슬라이스 5b: 용량 축 끝값 점검 (baseline 칸).
exp5 점검에서 고른 weight decay(MLP 0.03, GNN 0.01)를 고정하고 은닉 64와 128을 초기화 seed 5개로 학습해 같은 시험 표본에서 짝지어 비교한다.
판정(결과 전에 고정): seed 평균 시험 R² 차이(128 − 64)의 반복 블록 부트스트랩 95% CI 하한 > 0이면 의미 있는 개선. 검증 R²도 같은 방식으로 보고한다.
"""
import json
import time
from dataclasses import replace
import numpy as np
from utils.features import phi
from env.graph import graph
from vfa.mlp import MLP
from vfa.gnn import GNN, hidden, n_params, mlp_params
from result.exp5_fair import cell_cfg, splits, walk, r2
from result import diag

SRC = diag.RUNS / "exp5_check_alt4_conc0.5_k2.json"
OUT = diag.RUNS / "diag_s5b_width.json"
WD = {"MLP": 0.03, "GNN": 0.01}
WIDTHS = (64, 128)


# 반복 블록 부트스트랩: seed 평균 R² 차이 (b − a)
def boot(cfg, Pa, Pb, y, g):
  reps, rng = np.unique(g), np.random.default_rng(cfg.seed)
  where = {r: np.where(g == r)[0] for r in reps}
  ds = []
  for _ in range(cfg.boot_B):
    i = np.concatenate([where[r] for r in rng.choice(reps, len(reps))])
    ds.append(np.mean([r2(pb, y, i) - r2(pa, y, i) for pa, pb in zip(Pa, Pb)]))
  full = float(np.mean([r2(pb, y) - r2(pa, y) for pa, pb in zip(Pa, Pb)]))
  lo, hi = np.percentile(ds, [2.5, 97.5])
  return dict(diff=full, lo=float(lo), hi=float(hi), meaningful=bool(lo > 0))


NOTE = ("슬라이스 5b: baseline 칸에서 고른 wd를 고정하고 은닉 64·128을 seed 5개로 학습해 같은 시험·검증 표본에서 짝지어 비교. "
        "판정 기준(결과 전 고정): seed 평균 R² 차이(128 − 64)의 반복 블록 부트스트랩 95% CI 하한 > 0. "
        "64는 같은 seed로 다시 학습해 exp5 점검의 저장 예측과 일치하는지도 확인한다.")

if __name__ == "__main__":
  t0 = time.time()
  cfg = cell_cfg()
  src = json.loads(SRC.read_text(encoding="utf-8"))
  sp, D = splits(cfg), {}
  for s in ("train", "val", "test"):
    O, _, g, _ = walk(cfg, sp[s])
    V = np.array(src["futures"][s])
    D[s] = dict(O=O, g=g, y=V.mean(1))
  assert np.allclose(D["test"]["y"], src["test_y"])
  base = replace(cfg, nn_epochs=cfg.es_epochs)
  res = dict(note=NOTE, wd=WD, widths=list(WIDTHS), models={})
  for name, mk, enc in (("MLP", MLP, phi), ("GNN", GNN, lambda o: graph(base, o))):
    X = {s: np.array([enc(o) for o in D[s]["O"]]) for s in D}
    out = {}
    for w in WIDTHS:
      pv, pt, secs, eps = [], [], [], []
      for sd in cfg.fair_seeds:
        c = replace(base, nn_hidden=w, nn_seed=sd)
        t1 = time.time()
        m = mk(c, WD[name]).fit(X["train"], D["train"]["y"], X["val"], D["val"]["y"])
        secs.append(time.time() - t1)
        pv.append(m.predict(X["val"]))
        pt.append(m.predict(X["test"]))
        eps.append(m.best_epoch)
      c = replace(base, nn_hidden=w)
      out[w] = dict(val_r2=[r2(p, D["val"]["y"]) for p in pv], test_r2=[r2(p, D["test"]["y"]) for p in pt],
                    sec=secs, best_epoch=eps, pv=np.array(pv), pt=np.array(pt),
                    params=(n_params(c, hidden(c)) if name == "GNN" else mlp_params(c)),
                    gnn_hidden=(hidden(c) if name == "GNN" else None))
      print(f"  {name} 은닉 {w}: 검증 R² {np.mean(out[w]['val_r2']):.4f} 시험 R² {np.mean(out[w]['test_r2']):.4f} ± {np.std(out[w]['test_r2'], ddof=1):.4f} "
            f"| 학습 1회 평균 {np.mean(secs):.0f}s, 파라미터 {out[w]['params']}", flush=True)
    stored = np.array(src["test_pred"][name])
    repro = bool(np.allclose(out[64]["pt"], stored, rtol=1e-5, atol=1e-3))
    res["models"][name] = dict(
      repro_64=repro,
      **{f"w{w}": dict(val_r2=out[w]["val_r2"], test_r2=out[w]["test_r2"], val_r2_mean=float(np.mean(out[w]["val_r2"])),
                       test_r2_mean=float(np.mean(out[w]["test_r2"])), test_r2_sd=float(np.std(out[w]["test_r2"], ddof=1)),
                       sec=out[w]["sec"], sec_mean=float(np.mean(out[w]["sec"])), best_epoch=out[w]["best_epoch"],
                       params=out[w]["params"], gnn_hidden=out[w]["gnn_hidden"]) for w in WIDTHS},
      boot_test=boot(cfg, out[64]["pt"], out[128]["pt"], D["test"]["y"], D["test"]["g"]),
      boot_val=boot(cfg, out[64]["pv"], out[128]["pv"], D["val"]["y"], D["val"]["g"]))
    b = res["models"][name]
    print(f"  {name} 128 − 64: 시험 {b['boot_test']['diff']:+.4f} [{b['boot_test']['lo']:+.4f}, {b['boot_test']['hi']:+.4f}] "
          f"검증 {b['boot_val']['diff']:+.4f} [{b['boot_val']['lo']:+.4f}, {b['boot_val']['hi']:+.4f}] | 64 재현 {repro}", flush=True)
  res["sec"] = time.time() - t0
  OUT.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
  print(f"saved {OUT} ({res['sec']:.0f}s)")
