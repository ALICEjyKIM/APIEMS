"""진단·비교 실행과 원자료 저장.
정책 집합을 같은 반복(공통 난수)으로 돌려 반복별·기간별 원자료를 result/runs/diag_<tag>.json에 저장한다.
표(정책별 평균과 95% 신뢰구간, 기준 정책 대비 짝지은 차이, 기간 구간별 차이)는 그 json에서만 만든다.
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
from env.platform import Env, Obs

RUNS = Path(__file__).resolve().parent / "runs"
KEYS = ("profit", "n_ok", "n_buy", "stay_buy", "n_sup", "stay_sup", "zero", "capr", "eco", "sb", "ss")


# 정책 하나를 반복 rep로 돌린 기간별 원자료: 이윤, 성립·전체 주문 수, 남은·전체 주문자·공급자 수, 공급자 없는 품목 수,
# 거절 원인(용량 경쟁 = 혼자 넣으면 성립 / 단독 불성립 = 혼자 넣어도 불성립), 배분 잉여 합(주문자·공급자)
def episode(cfg, mk, rep):
  env, pol = Env(cfg, rep), mk(cfg)
  env.reset()
  pol.reset(rep)
  out = {k: [] for k in KEYS}
  for _ in range(cfg.T):
    o = env.o
    d = pol.act(o)
    ok = (d[0].sum(1) == o.q).all(1)
    capr = eco = 0
    for b in np.where(~ok)[0]:
      solo = Obs(o.t, o.q[b:b + 1], o.p[b:b + 1], o.cap, o.c, [0], o.sid)
      eco += (mk(cfg).act(solo)[0] == 0).all()
      capr += not (mk(cfg).act(solo)[0] == 0).all()
    r = env.step(d)
    vals = (r["profit"], r["n_ok"], r["n_buy"], r["stay_buy"], r["n_sup"], r["stay_sup"],
            ((o.cap > 0).sum(0) == 0).sum(), capr, eco, r["sb"].sum(), r["ss"].sum())
    for k, v in zip(KEYS, vals):
      out[k].append(float(v))
  return out


# 정책들(이름 → cfg를 받아 정책을 만드는 함수)을 cfg.reps 반복으로 돌려 원자료를 저장하고 경로를 돌려준다
def run(cfg, policies, tag, note=""):
  data = dict(tag=tag, note=note, cfg=asdict(cfg),
              policies={n: [episode(cfg, mk, rep) for rep in range(cfg.reps)] for n, mk in policies.items()})
  path = RUNS / f"diag_{tag}.json"
  path.write_text(json.dumps(data), encoding="utf-8")
  return path


# 한 반복의 누적 지표: 누적 이윤, 충족률, 재참여율, 평균 활동 참여자 수, 공급자 자리 점유율, 공급자 없는 품목 수, 거절 원인 비율
def per_rep(ep, cfg):
  s = {k: float(np.sum(v)) for k, v in ep.items()}
  return dict(profit=s["profit"], fill=s["n_ok"] / s["n_buy"], ret_buy=s["stay_buy"] / s["n_buy"], ret_sup=s["stay_sup"] / s["n_sup"],
              act_buy=s["n_buy"] / cfg["T"], act_sup=s["n_sup"] / cfg["T"], occ=s["n_sup"] / cfg["T"] / cfg["n_sup"],
              zero=s["zero"] / cfg["T"], capr=s["capr"] / s["n_buy"], eco=s["eco"] / s["n_buy"])


# 95% 신뢰구간 반폭
def ci(x):
  x = np.asarray(x, float)
  return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


# json 하나에서 표를 만든다: 정책별 평균(누적 이윤은 95% CI 포함). ref·other를 주면 other − ref 짝지은 차이와 기간 구간별 차이를 덧붙인다
def table(path, ref=None, other=None, width=5):
  d = json.loads(Path(path).read_text(encoding="utf-8"))
  cfg, P = d["cfg"], d["policies"]
  out = [f"원자료: {Path(path).name}. {d['note']}".rstrip(),
         f"cfg: T={cfg['T']} reps={cfg['reps']} seed={cfg['seed']} p_sup_new={cfg['p_sup_new']} b_buy={cfg['b_buy']:.4f} b_sup={cfg['b_sup']:.4f}",
         "", "| 정책 | 재참여율 주문자 / 공급자 | 활동 주문자 / 공급자 | 공급자 자리 점유율 | 공급자 없는 품목 | 충족률 | 용량 경쟁 거절 | 단독 불성립 거절 | 누적 이윤 (95% CI) |",
         "|---|---|---|---|---|---|---|---|---|"]
  for n, eps in P.items():
    R = [per_rep(e, cfg) for e in eps]
    m = {k: np.mean([r[k] for r in R]) for k in R[0]}
    out.append(f"| {n} | {m['ret_buy']:.3f} / {m['ret_sup']:.3f} | {m['act_buy']:.1f} / {m['act_sup']:.2f} | {m['occ']:.3f} | {m['zero']:.2f} "
               f"| {m['fill']:.3f} | {m['capr']:.3f} | {m['eco']:.3f} | {m['profit']:.0f} ± {ci([r['profit'] for r in R]):.0f} |")
  if ref and other:
    A, B = np.array([e["profit"] for e in P[other]]), np.array([e["profit"] for e in P[ref]])
    diff = A.sum(1) - B.sum(1)
    out += ["", f"짝지은 차이 ({other} − {ref}) 누적 이윤: {diff.mean():+.0f} ± {ci(diff):.0f}, 양수 {int((diff > 0).sum())}/{len(diff)}, "
            f"반복별 {np.round(diff).astype(int).tolist()}"]
    bins = []
    for a in range(0, cfg["T"], width):
      dd = A[:, a:a + width].mean(1) - B[:, a:a + width].mean(1)
      bins.append(f"t{a}–{min(a + width, cfg['T']) - 1}: {A[:, a:a + width].mean():.0f} | {B[:, a:a + width].mean():.0f} ({dd.mean():+.0f} ± {ci(dd):.0f})")
    out.append(f"기간 구간별 기간당 이윤 ({other} | {ref}, 짝지은 차이): " + ", ".join(bins))
    cum = np.cumsum(A - B, axis=1).mean(0)
    first = next((t for t in range(cfg["T"]) if (cum[t:] > 0).all()), None)
    out.append(f"평균 누적 차이가 그 뒤로 계속 양수인 첫 기간: {first}")
  return "\n".join(out)


if __name__ == "__main__":
  print(table(sys.argv[1], *sys.argv[2:4]))
