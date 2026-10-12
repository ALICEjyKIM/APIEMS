"""진단 s9b 요약: diag_s9_plan.json과 diag_s9_units/에서만 표를 만든다 (Rule-continuation interaction value).
쌍별 I_ij와 표준오차, 유형(A 공통 품목 / B 묶음 보완 / C 약한 관계)·꼬리표(D bottleneck / E 중복)별 분포와 잡음 보정 크기, c_i 대비 비율,
공통 품목의 활동 공급자 수별 크기, 조건(k, het, L)별 크기, 무작위 집합 제거의 additive·pairwise 예측 오차, c_i 순위와 interaction 반영 순위를 비교한다.
"""
import csv
import json
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from result.diag_s9_interaction import PLAN, UNITS, OUT, CSV, CONDS

TOP = 3


# 상태별 미래 값 {제거 집합 문자열: 미래 M개 배열}
def load():
  P = json.loads(PLAN.read_text(encoding="utf-8"))
  vals = defaultdict(dict)
  for f in UNITS.glob("*.json"):
    u = json.loads(f.read_text(encoding="utf-8"))
    vals[(u["cond"], u["rep"], u["t"])].update({k: np.array(v) for k, v in u["values"].items()})
  return P, vals


# 평균과 표준오차
def ms(x):
  return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


# 잡음 보정 RMS: sqrt(max(평균 제곱 − 평균 표준오차², 0))
def rms(m, se):
  m, se = np.asarray(m), np.asarray(se)
  return float(np.sqrt(max(np.mean(m ** 2) - np.mean(se ** 2), 0.0))) if len(m) else float("nan")


# 상태 하나: 단일 c (미래별), 쌍 기록, 무작위 집합 오차, 순위 비교
def per_state(s, V):
  full = V[""]
  d = {i: full - V[str(i)] for i in s["sid"]}
  pairs = []
  Iarr = {}
  for p in s["pairs"]:
    I = full - V[str(p["i"])] - V[str(p["j"])] + V[f"{p['i']},{p['j']}"]
    Iarr[(p["i"], p["j"])] = Iarr[(p["j"], p["i"])] = I
    (im, ise), (ci, cise), (cj, cjse) = ms(I), ms(d[p["i"]]), ms(d[p["j"]])
    pairs.append(dict(cond=s["cond"], rep=s["rep"], t=s["t"], **p, I=im, I_se=ise, ci=ci, ci_se=cise, cj=cj, cj_se=cjse,
                      norm=abs(im) / (abs(ci) + abs(cj) + 1.0)))
  sets = []
  for D in s["sets"]:
    delta = full - V[",".join(map(str, D))]
    add = sum(d[i] for i in D)
    pw = add - sum(Iarr[(a, b)] for a in D for b in D if a < b)
    sets.append(dict(size=len(D), delta=ms(delta), err_add=ms(delta - add), err_pair=ms(delta - pw)))
  rank = None
  if len(Iarr) == len(s["sid"]) * (len(s["sid"]) - 1):
    q = dict(zip(s["sid"], 1 - np.array(s["rs"])))
    ci = {i: d[i] - sum(q[j] * Iarr[(i, j)] for j in s["sid"] if j != i) for i in s["sid"]}
    a = np.array([d[i].mean() for i in s["sid"]])
    b = np.array([ci[i].mean() for i in s["sid"]])
    ta, tb = set(np.argsort(-a)[:TOP]), set(np.argsort(-b)[:TOP])
    flips = []
    for x in range(len(s["sid"])):
      for y in range(len(s["sid"])):
        if a[x] > a[y] and b[x] < b[y]:
          diff = ci[s["sid"][y]] - ci[s["sid"][x]]
          m, se = ms(diff)
          dc, dse = ms(d[s["sid"][x]] - d[s["sid"][y]])
          flips.append(dict(hi_c=s["sid"][x], lo_c=s["sid"][y], c_gap=dc, c_gap_se=dse, int_gap=m, int_gap_se=se))
    rank = dict(spearman=float(spearmanr(a, b)[0]), top3_overlap=len(ta & tb) / TOP, n=len(a), flips=len(flips),
                sig_flips=sum(f["int_gap"] / f["int_gap_se"] > 2 and f["c_gap"] / f["c_gap_se"] > 2 for f in flips),
                adj=float(np.mean(np.abs(a - b))), c_abs=float(np.mean(np.abs(a))), flip_list=sorted(flips, key=lambda f: -f["int_gap"] / f["int_gap_se"])[:3],
                c=dict(zip(map(str, s["sid"]), a.tolist())), c_int=dict(zip(map(str, s["sid"]), b.tolist())))
  return pairs, sets, rank, {i: ms(d[i]) for i in s["sid"]}


# 쌍 묶음의 요약: 개수, 평균 I (상태 블록 부트스트랩 CI), 음수·유의 음수·유의 양수 비율, 잡음 보정 RMS(I)·RMS(c)와 비
def agg(ps, seed=0):
  if not ps:
    return dict(n=0)
  I = np.array([p["I"] for p in ps])
  se = np.array([p["I_se"] for p in ps])
  cm = np.r_[[p["ci"] for p in ps], [p["cj"] for p in ps]]
  cse = np.r_[[p["ci_se"] for p in ps], [p["cj_se"] for p in ps]]
  st = sorted({(p["cond"], p["rep"], p["t"]) for p in ps})
  by = {k: [x["I"] for x in ps if (x["cond"], x["rep"], x["t"]) == k] for k in st}
  g = np.random.default_rng(seed)
  bs = [np.mean(np.concatenate([by[st[i]] for i in g.choice(len(st), len(st))])) for _ in range(2000)]
  ri, rc = rms(I, se), rms(cm, cse)
  return dict(n=len(ps), states=len(st), mean=float(I.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
              neg=float((I < 0).mean()), sig_neg=float((I / se < -1.96).mean()), sig_pos=float((I / se > 1.96).mean()),
              rms_I=ri, rms_c=rc, ratio=ri / rc if rc > 0 else float("nan"), mean_abs_I=float(np.abs(I).mean()), mean_se=float(se.mean()),
              norm_median=float(np.median([p["norm"] for p in ps])))


if __name__ == "__main__":
  P, vals = load()
  allp, allsets, ranks, singles = [], [], [], []
  for s in P:
    V = vals.get((s["cond"], s["rep"], s["t"]), {})
    if not all(",".join(map(str, d)) in V for d in s["drops"]):
      print("미완료", s["cond"], s["rep"], s["t"])
      continue
    ps, sets, rank, cs = per_state(s, V)
    allp += ps
    allsets += [dict(cond=s["cond"], rep=s["rep"], t=s["t"], **x) for x in sets]
    singles += [dict(cond=s["cond"], rep=s["rep"], t=s["t"], c=m, se=e) for m, e in cs.values()]
    if rank:
      ranks.append(dict(rep=s["rep"], t=s["t"], **rank))
  res = dict(note="Rule-continuation interaction value (PROC15B, M = 20, vf_H = 10). optimal·oracle 가치가 아니다.", conds={}, by_deg={}, tags={}, sets={}, ranks=ranks)
  for cond in CONDS:
    ps = [p for p in allp if p["cond"] == cond]
    sg = [x for x in singles if x["cond"] == cond]
    res["conds"][cond] = dict(all=agg(ps), **{t: agg([p for p in ps if p["type"] == t]) for t in "ABC"},
                              c_rms=rms([x["c"] for x in sg], [x["se"] for x in sg]), c_mean=float(np.mean([x["c"] for x in sg])) if sg else None)
  base = [p for p in allp if p["cond"] == "base"]
  for k in (2, 3, 4, 5):
    sel = [p for p in base if p["type"] == "A" and (p["min_deg_shared"] == k if k < 5 else p["min_deg_shared"] >= 5)]
    res["by_deg"][f"A_min_deg_{k}{'+' if k == 5 else ''}"] = agg(sel)
  for k in (1, 2, 3, 4):
    sel = [p for p in base if (p["min_deg"] == k if k < 4 else p["min_deg"] >= 4)]
    res["by_deg"][f"all_min_deg_{k}{'+' if k == 4 else ''}"] = agg(sel)
  res["tags"] = {"D": agg([p for p in base if p["D"]]), "E": agg([p for p in base if p["E"]]),
                 "B_coneed>=3": agg([p for p in base if p["type"] == "B" and p["coneed"] >= 3]), "B_coneed1-2": agg([p for p in base if p["type"] == "B" and p["coneed"] < 3])}
  for n in (3, 6):
    S = [x for x in allsets if x["size"] == n]
    if S:
      res["sets"][str(n)] = dict(n=len(S), delta_mean=float(np.mean([x["delta"][0] for x in S])),
                                 add_rms=rms([x["err_add"][0] for x in S], [x["err_add"][1] for x in S]),
                                 pair_rms=rms([x["err_pair"][0] for x in S], [x["err_pair"][1] for x in S]),
                                 add_mean_abs=float(np.mean([abs(x["err_add"][0]) for x in S])), pair_mean_abs=float(np.mean([abs(x["err_pair"][0]) for x in S])),
                                 add_se=float(np.mean([x["err_add"][1] for x in S])), pair_se=float(np.mean([x["err_pair"][1] for x in S])),
                                 rows=S)
  ex = {}
  for t in "ABC":
    ps = sorted([p for p in base if p["type"] == t], key=lambda p: -abs(p["I"]) / p["I_se"])[:3]
    ex[t] = ps
  res["examples"] = ex
  OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
  keys = ["cond", "rep", "t", "i", "j", "type", "D", "E", "shared", "min_deg_shared", "min_deg", "coneed", "I", "I_se", "ci", "ci_se", "cj", "cj_se", "norm"]
  with CSV.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    for p in allp:
      w.writerow({k: p[k] for k in keys})
  f2 = lambda x: f"n {x['n']} 평균 {x['mean']:+.1f} [{x['lo']:+.1f}, {x['hi']:+.1f}] 음수 {x['neg']:.2f} 유의음 {x['sig_neg']:.2f} 유의양 {x['sig_pos']:.2f} RMS(I) {x['rms_I']:.1f} RMS(c) {x['rms_c']:.1f} 비 {x['ratio']:.2f} 평균 se {x['mean_se']:.1f}" if x["n"] else "n 0"
  for cond, r in res["conds"].items():
    print(f"\n[{cond}] c 평균 {r['c_mean']:.1f} RMS(c) {r['c_rms']:.1f}")
    for t in ("all", "A", "B", "C"):
      print(f"  {t}: {f2(r[t])}")
  print("\n[base 차수별]")
  for k, x in res["by_deg"].items():
    print(f"  {k}: {f2(x)}")
  print("\n[base 꼬리표]")
  for k, x in res["tags"].items():
    print(f"  {k}: {f2(x)}")
  print("\n[무작위 집합 제거: 실제 손실 대비 예측 오차]")
  for n, x in res["sets"].items():
    print(f"  {n}명 n {x['n']}: 손실 평균 {x['delta_mean']:.0f} | additive 오차 RMS {x['add_rms']:.1f} (평균|오차| {x['add_mean_abs']:.1f}, se {x['add_se']:.1f}) | pairwise 오차 RMS {x['pair_rms']:.1f} (평균|오차| {x['pair_mean_abs']:.1f}, se {x['pair_se']:.1f})")
  print("\n[순위: c_i vs c_i − Σ(1 − p_j) I_ij]")
  for r in ranks:
    print(f"  r{r['rep']} t{r['t']}: n {r['n']} Spearman {r['spearman']:.2f} top3 겹침 {r['top3_overlap']:.2f} 뒤집힌 쌍 {r['flips']} (둘 다 유의 {r['sig_flips']}) 평균|조정| {r['adj']:.1f} vs 평균|c| {r['c_abs']:.1f}")
  print(f"\nsaved {OUT}, {CSV}")
