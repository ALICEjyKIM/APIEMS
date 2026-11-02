"""실험 5 요약: 18칸 원자료(result/runs/exp5_<칸>.json)에서만 표를 만든다.
칸별 seed 평균 시험 R²·표준편차, 짝지은 부트스트랩 95% CI, 고른 설정, 계산시간을 정리하고,
n_alt·주문당 품목 수·공급 편중도별로 GNN의 상대 성능(GNN − MLP, GNN − 선형)이 어떻게 변하는지 모은다.
"""
import json
import numpy as np
from result import diag
from result.exp5_fair import CELLS, NAMES, PAIRS, cell_cfg, tag

OUT = diag.RUNS / "exp5_summary.json"


# 18칸 원자료 (없는 칸은 건너뛴다)
def load():
  out = {}
  for cell in CELLS:
    p = diag.RUNS / f"exp5_{tag(cell_cfg(*cell))}.json"
    if p.exists():
      out[cell] = json.loads(p.read_text(encoding="utf-8"))
  return out


# 칸별 행
def rows(R):
  out = []
  for (a, c, k), d in R.items():
    m = d["models"]
    r = dict(n_alt=a, conc=c, k=k, tag=d["tag"], ceiling=d["ceiling"]["test"],
             sec_data=float(sum(d["sec_data"].values())))
    for n in NAMES:
      r[n] = dict(mean=m[n]["test_r2_mean"], sd=m[n]["test_r2_sd"], config=m[n]["config"],
                  val_r2=m[n]["select"][m[n]["pick"]]["val_r2"], edge=m[n]["edge"],
                  sec_select=m[n]["sec_select"], sec_final=float(sum(m[n]["sec_final"])))
    for a_, b_ in PAIRS:
      r[f"{a_}-{b_}"] = d["boot"][f"{a_}-{b_}"]
    out.append(r)
  return out


# 요인 수준별 평균: 다른 두 요인에 대해 칸 평균 (칸 수도 함께)
def margins(rs, key):
  out = {}
  for fac in ("n_alt", "k", "conc"):
    lv = sorted({r[fac] for r in rs})
    out[fac] = {str(v): dict(mean=float(np.mean([r[key]["diff"] for r in rs if r[fac] == v])),
                             n=len([r for r in rs if r[fac] == v]),
                             pos=sum(r[key]["lo"] > 0 for r in rs if r[fac] == v),
                             neg=sum(r[key]["hi"] < 0 for r in rs if r[fac] == v)) for v in lv}
  return out


# 표 문자열
def table(rs):
  f = lambda x: f"{x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]"
  s = ["| n_alt | 편중도 | k | 선형 | MLP | GNN | MLP − 선형 | GNN − 선형 | GNN − MLP | 상한 |", "|---|---|---|---|---|---|---|---|---|---|"]
  for r in sorted(rs, key=lambda r: (r["n_alt"], r["conc"], r["k"])):
    s.append(f"| {r['n_alt']} | {r['conc']:g} | {r['k']} | {r['선형']['mean']:.3f} | {r['MLP']['mean']:.3f} ± {r['MLP']['sd']:.3f} | "
             f"{r['GNN']['mean']:.3f} ± {r['GNN']['sd']:.3f} | {f(r['MLP-선형'])} | {f(r['GNN-선형'])} | {f(r['GNN-MLP'])} | {r['ceiling']:.3f} |")
  return "\n".join(s)


if __name__ == "__main__":
  R = load()
  rs = rows(R)
  res = dict(n_cells=len(rs), rows=rs, margins={k: margins(rs, k) for k in ("GNN-MLP", "GNN-선형", "MLP-선형")})
  OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
  print(f"칸 {len(rs)}/{len(CELLS)} (원자료 result/runs/exp5_<칸>.json)\n")
  print(table(rs))
  for key in ("GNN-MLP", "GNN-선형"):
    print(f"\n{key} 요인별 칸 평균 (유의하게 양수 / 음수 칸 수)")
    for fac, lv in res["margins"][key].items():
      print(f"  {fac}: " + ", ".join(f"{v}: {x['mean']:+.3f} (+{x['pos']}/−{x['neg']} of {x['n']})" for v, x in lv.items()))
