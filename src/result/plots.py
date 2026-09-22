"""진행 문서(tex)용 그림: result/runs/의 json 원자료에서만 만든다.
정답 잡음과 R² 상한(exp3_diag.json), 공급자 유지 가치 과소평가(exp1_value.json, exp3_mlp.json, exp3_gnn.json), GNN 입력 척도(exp3_diag_scale.json).
결과는 result/figures/에 저장한다. 주색 크림슨 #B30B00, 보조색 초록 #70AD47.
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from result import diag

FIG = Path(__file__).resolve().parent / "figures"
RED, GREEN = "#B30B00", "#70AD47"
AVG_R = 5  # 정답을 미래 AVG_R번 평균으로 할 때의 R² 상한 (exp3_diag.AVG_R와 같음)


# json 읽기
def load(name):
  return json.loads((diag.RUNS / name).read_text(encoding="utf-8"))


# 그림 저장
def save(f, name):
  FIG.mkdir(exist_ok=True)
  f.tight_layout()
  f.savefig(FIG / name, dpi=150)
  plt.close(f)


# 정답 잡음: 칸별 상태 안 표준편차(한 번 돌린 값) vs 상태 간 표준편차, R² 상한(1회, 5회 평균) 주석
def noise():
  D = load("exp3_diag.json")["cells"]
  tg = list(D)
  f, ax = plt.subplots(figsize=(7, 3.8))
  x = np.arange(len(tg))
  for i, c in enumerate(tg):
    V = np.array([r["v"] for r in D[c]["noise"]])
    w, yv = V.var(1, ddof=1).mean(), D[c]["y_var"]
    ws, bs = np.sqrt(w), V.mean(1).std(ddof=1)
    ax.bar(i - 0.2, ws, 0.4, color=RED, label="within-state SD (single rollout)" if i == 0 else None)
    ax.bar(i + 0.2, bs, 0.4, color=GREEN, label="between-state SD (mean of 10)" if i == 0 else None)
    c1, c5 = 1 - w / yv, 1 - w / AVG_R / (yv - w + w / AVG_R)
    ax.text(i, max(ws, bs) + 80, f"R² ceiling\n1 run {c1:.2f} / 5-avg {c5:.2f}", ha="center", fontsize=8)
  ax.set_xticks(x, tg)
  ax.set_ylabel("10-period profit SD")
  ax.set_ylim(0, 3000)
  ax.legend(fontsize=8, loc="lower right")
  save(f, "label_noise.png")


# 공급자 유지 가치: 모델 평균(가드 전) vs 시뮬레이션 기준치 평균, 실험 1(기존 가격 생성)과 실험 3 칸별
def supplier_c():
  E1, A, G = load("exp1_value.json"), load("exp3_mlp.json")["cells"], load("exp3_gnn.json")["cells"]
  sup = lambda rows, k: np.concatenate([np.asarray(r[k]["sup"] if isinstance(r[k], dict) else r[k], float) for r in rows])
  groups = [("exp1\n(old prices)", sup(E1["conv"], "mc_sup"), [("linear", sup(E1["conv"], "선형")), ("MLP", sup(E1["conv"], "MLP"))])]
  for tg in A:
    groups.append((f"exp3\n{tg}", sup(A[tg]["conv"], "mc_sup"),
                   [("MLP", sup(A[tg]["conv"], "MLP")), ("MLP+conc", sup(A[tg]["conv"], "MLP+편중도 요약")), ("GNN", sup(G[tg]["conv"], "GNN"))]))
  f, ax = plt.subplots(figsize=(9, 4))
  sty = {"linear": dict(color=GREEN, alpha=0.45), "MLP": dict(color=GREEN), "MLP+conc": dict(color=GREEN, hatch="//", edgecolor="white"), "GNN": dict(color=GREEN, hatch="..", edgecolor="white")}
  seen, pos, ticks = set(), 0.0, []
  ci = lambda a: 1.96 * a.std(ddof=1) / np.sqrt(len(a))
  for name, mc, ms in groups:
    bars = [("simulation baseline", mc, dict(color=RED))] + [(n, a, sty[n]) for n, a in ms]
    xs = pos + np.arange(len(bars)) * 0.8
    for xx, (n, a, s) in zip(xs, bars):
      ax.bar(xx, a.mean(), 0.8, yerr=ci(a), capsize=2, label=None if n in seen else n, **s)
      seen.add(n)
    ticks.append((xs.mean(), name))
    pos = xs[-1] + 1.6
  ax.set_xticks([t for t, _ in ticks], [n for _, n in ticks], fontsize=8)
  ax.set_ylabel("mean supplier retention value (pre-guard)")
  ax.legend(fontsize=8, ncol=5, loc="upper left")
  save(f, "supplier_c.png")


# GNN 입력 척도: 칸별 표준화 뒤 수량 특징의 주문자·공급자 상자 그림 (저장된 통계, 수염 1.5 IQR)
def scale():
  D = load("exp3_diag_scale.json")["cells"]
  f, ax = plt.subplots(figsize=(8, 3.8))
  stats, cols, ticks = [], [], []
  for i, (tg, c) in enumerate(D.items()):
    for j, (k, lab, col) in enumerate((("z_buy", "buyer", RED), ("z_sup", "supplier", GREEN))):
      s = dict(c[k], label=lab, fliers=[c[k]["max"]])
      stats.append(s)
      cols.append(col)
    ticks.append((3 * i + 1.5, tg))
  pos = [3 * i + 1 + j for i in range(len(D)) for j in range(2)]
  b = ax.bxp(stats, positions=pos, widths=0.7, patch_artist=True, showfliers=True)
  for p, col in zip(b["boxes"], cols):
    p.set_facecolor(col)
    p.set_alpha(0.8)
  for m in b["medians"]:
    m.set_color("black")
  ax.set_xticks([t for t, _ in ticks], [n for _, n in ticks])
  ax.set_ylabel("standardized quantity feature (z)")
  ax.axhline(0, color="gray", lw=0.6)
  ax.legend([b["boxes"][0], b["boxes"][1]], ["buyer nodes", "supplier nodes"], fontsize=8)
  save(f, "gnn_scale.png")


if __name__ == "__main__":
  for fn in (noise, supplier_c, scale):
    if fn.__name__ in sys.argv or "all" in sys.argv:
      fn()
      print(fn.__name__, "saved in", FIG)
