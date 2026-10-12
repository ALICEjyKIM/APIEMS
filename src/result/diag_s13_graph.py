"""진단 s13 그래프: PROC15H 상태 → Buyer–Item–Supplier 그래프 (vfa/hgnn 입력).
주문자 노드 [주문 품목 수, 총 요구수량, 주문 금액, 평균 마크업, hB, 직전 재참여확률], 품목 노드 [기준가, hI, 대기 후보 공급가능량·수, 최소 대기],
공급자 노드는 Own MLP와 같은 own feature. 연결선: 주문자–품목 [요구수량, 마크업], 공급자–품목 [공급가능량, 공급가격 수준]. deg=True면 품목 노드에 활동 공급자 수·총 공급가능량·총 요구량을 더한다 (공정성 점검용).
"""
import numpy as np
from result.diag_s11_context import feats
from result.diag_s11_summary import OWN


# 상태 env의 그래프 (공급자 순서 = o.sid)
def graph(env, deg=False):
  o, cfg, base = env.o, env.cfg, env.inst.base
  qb = o.q > 0
  mk = np.where(qb, o.p / base[None], 0.0)
  xb = np.c_[qb.sum(1), o.q.sum(1), (o.p * o.q).sum(1), mk.sum(1) / np.maximum(qb.sum(1), 1), o.hb, o.rb].astype(float)
  pend = np.zeros(cfg.n_items)
  npend = np.zeros(cfg.n_items)
  wait = np.full(cfg.n_items, cfg.onboard + 1.0)
  for j, (ready, prof) in env.pipe.items():
    m = prof.qty > 0
    pend[m] += prof.qty[m]
    npend[m] += 1
    wait[m] = np.minimum(wait[m], ready - o.t)
  xi = np.c_[base, o.hi, pend, npend, wait]
  if deg:
    xi = np.c_[xi, (o.cap > 0).sum(0), o.cap.sum(0), o.q.sum(0)]
  xs = np.array([[feats(env, k)[0][f] for f in OWN] for k in range(len(o.sid))], float).reshape(-1, len(OWN))
  bb, bi = np.nonzero(qb)
  ss, si = np.nonzero(o.cap > 0)
  ebi = np.c_[o.q[bb, bi], o.p[bb, bi] / base[bi]].astype(float)
  esi = np.c_[o.cap[ss, si], o.c[ss, si] / base[si]].astype(float)
  return dict(xb=xb, xi=xi.astype(float), xs=xs, ebi=ebi, esi=esi, bi_b=bb, bi_i=bi, si_s=ss, si_i=si, sid=[int(s) for s in o.sid])
