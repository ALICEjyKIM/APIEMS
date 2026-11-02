"""반복 조달 시장의 기간 엔진: 입찰 전 관측(PreObs) → 1단계 결정 (m, θ) → Chu IPB 청산 → 쿠폰 → 재참여 → 다음 기간 시장.
1단계 정책은 pre()만 받는다. 이번 기간 입찰가 W, g는 엔진 안에만 있고 청산에만 쓴다 (ref/design.md 2장).
도착·변동·재참여 난수는 (seed, 반복, 스트림, 키)로 정해 정책이 달라도 같다 (utils/rng.py).
"""
from dataclasses import dataclass
import numpy as np
from utils import market
from utils.rng import ret_u
from mech.chu import clear, pad
from env.ret import Ret


@dataclass
class PreObs:
  """입찰 전 관측: 공개 정보와 지난 기간까지의 기록 (이번 기간 입찰가는 없다).
  주문자 행: bid 번호, q 수량, hb 이력, rb 기준점(nan = 거래 기록 없음), tb 재임, eb 지난 기간 padding 제외, W0 지난 신고 가치(nan = 신규).
  공급자 행(활동 자리 순): slot, sid 번호, K 용량, hs, rs, ts, es, g0 지난 신고 원가. pend·wait onboard 대기 용량·남은 기간, hi 품목 부족 이력, vbar·cbar 과거 품목 평균 단위 가치·원가, owe 미지급 쿠폰.
  """
  t: int
  bid: np.ndarray
  q: np.ndarray
  hb: np.ndarray
  rb: np.ndarray
  tb: np.ndarray
  eb: np.ndarray
  W0: np.ndarray
  slot: np.ndarray
  sid: np.ndarray
  K: np.ndarray
  hs: np.ndarray
  rs: np.ndarray
  ts: np.ndarray
  es: np.ndarray
  g0: np.ndarray
  pend: np.ndarray
  wait: np.ndarray
  hi: np.ndarray
  vbar: np.ndarray
  cbar: np.ndarray
  owe: float


# 참여자 하나의 상태: 원래 프로필, 이번 기간 주문·공급, 이력, 기준점, 재임, 지난 기간 padding 제외, 지난 신고가
def _new(pid, prof, cfg, last):
  return dict(id=pid, prof=prof, cur=prof, h=cfg.h0, r=np.nan, ten=0, ex=False, last=last)


# 1단계 후보: m_grid × theta_grid 중 Q_c가 품목 c의 활동 공급 용량을 넘지 않는 것
def cands(cfg, o):
  ok = [m for m in cfg.m_grid if (pad(cfg, m, o.q, o.K) <= o.K.sum(0)).all()]
  return [(m, th) for m in ok for th in cfg.theta_grid]


class Sim:
  """한 반복(rep)의 시장. 생성하면 기간 0 시장이 준비되고, step(m, θ)마다 한 기간을 진행해 기록을 돌려준다.
  주문자는 도착 순 번호, 공급자는 자리 j에 번호(기간 0은 j, 기간 t에 들어온 후보는 t·n_sup + j)를 둔다.
  떠난 공급자 자리는 onboard 기간 뒤 새 후보가 채우고, 쿠폰은 발행 다음 기간에 돌아온 사람에게만 지급한다.
  """

  def __init__(self, cfg, rep):
    nc = cfg.n_items
    self.cfg, self.rep, self.t, self.nb = cfg, rep, 0, 0
    self.mk, self.ret = market.make(cfg, rep), Ret(cfg)
    self.B, self.pipe, self.owe = {}, {}, {}
    self.S = {j: _new(j, s, cfg, np.full(nc, np.nan)) for j, s in enumerate(self.mk.sups)}
    self.hi, self.vs, self.vn, self.cs, self.cn = (np.zeros(nc) for _ in range(5))
    self._arrive()

  # 기간 t 신규 주문자를 받고 이번 기간 행렬을 만든다 (주문자: 남은 주문자 다음 신규, 공급자: 자리 순)
  def _arrive(self):
    for b in market.new_buyers(self.cfg, self.mk, self.rep, self.t):
      self.B[self.nb] = _new(self.nb, b, self.cfg, np.nan)
      self.nb += 1
    nc, bs, ss = self.cfg.n_items, list(self.B.values()), list(self.S.values())
    self.bid, self.slot = np.array(list(self.B), int), np.array(list(self.S), int)
    self.q = np.array([b["cur"].qty for b in bs], int).reshape(-1, nc)
    self.W = np.array([(b["cur"].qty * b["cur"].price).sum() for b in bs])
    self.K = np.array([s["cur"].qty for s in ss], int).reshape(-1, nc)
    self.g = np.array([s["cur"].price for s in ss], float).reshape(-1, nc)

  # 입찰 전 관측 (이번 기간 W, g를 읽지 않는다)
  def pre(self):
    nc, bs, ss, pj = self.cfg.n_items, list(self.B.values()), list(self.S.values()), sorted(self.pipe)
    a = lambda v, k: np.array([x[k] for x in v], float)
    with np.errstate(invalid="ignore"):
      vbar, cbar = self.vs / self.vn, self.cs / self.cn
    return PreObs(self.t, self.bid.copy(), self.q.copy(), a(bs, "h"), a(bs, "r"), a(bs, "ten"), a(bs, "ex"),
                  a(bs, "last"),
                  self.slot.copy(), np.array([s["id"] for s in ss], int), self.K.copy(), a(ss, "h"), a(ss, "r"), a(ss, "ten"), a(ss, "ex"),
                  np.array([s["last"] for s in ss], float).reshape(-1, nc),
                  np.array([self.pipe[j][1].qty for j in pj], int).reshape(-1, nc), np.array([self.pipe[j][0] - self.t for j in pj], int),
                  self.hi.copy(), vbar, cbar, float(sum(self.owe.values())))

  # 이번 기간 입찰 (진실 입찰: 주문자 묶음 가치 W, 공급자 단위 원가 g)
  def bids(self):
    return self.W.copy(), self.g.copy()

  # 1단계 결정 (m, θ)로 한 기간을 진행하고 기록을 돌려준다
  def step(self, m, th):
    cfg, rep, t, mk = self.cfg, self.rep, self.t, self.mk
    q, W, K, g = self.q, self.W, self.K, self.g
    o = clear(cfg, q, W, K, g, m)
    paid = float(sum(self.owe.values()))
    sb, ss = np.where(o.win, W - o.p, 0.0), o.R.sum(1) - (g * o.y).sum(1)
    eb, es = o.win, o.y.sum(1) > 0
    wb, ws = q @ mk.base, K @ mk.base
    tot = wb[o.Ub].sum() + ws[o.Uj].sum()
    f = th * o.G / tot if tot > 0 else 0.0
    kb = np.where(o.Ub, np.minimum(f * wb, cfg.cbar * wb), 0.0)
    ks = np.where(o.Uj, np.minimum(f * ws, cfg.cbar * ws), 0.0)
    xb, xs = (sb + cfg.cpn_w * kb) / W, (ss + cfg.cpn_w * ks) / (g * K).sum(1)
    bs, sp = [self.B[i] for i in self.bid], [self.S[j] for j in self.slot]
    h = lambda v, k: np.array([x[k] for x in v], float)
    pb = self.ret.prob("buy", eb, xb, h(bs, "h"), h(bs, "r"), o.Ub)
    ps = self.ret.prob("sup", es, xs, h(sp, "h"), h(sp, "r"), o.Uj)
    zb = np.array([ret_u(cfg, rep, "buy", b["id"], t) < p for b, p in zip(bs, pb)], bool)
    zs = np.array([ret_u(cfg, rep, "sup", s["id"], t) < p for s, p in zip(sp, ps)], bool)
    rec = dict(t=t, m=m, th=th, Q=o.Q, G=o.G, paid=paid, Pi=o.G - paid, sw=o.sw, milp=o.n, nb=len(W), win=o.win.sum(), ns=len(sp),
               empty=cfg.n_sup - len(sp), cpn=kb.sum() + ks.sum(), bid=self.bid.copy(), sid=h(sp, "id").astype(int), slot=self.slot.copy(),
               eb=eb, es=es, sb=sb, ss=ss, xb=xb, xs=xs, pb=pb, ps=ps, zb=zb, zs=zs, Ub=o.Ub, Uj=o.Uj, kb=kb, ks=ks, W=W, y=o.y)
    self._update(bs, eb, xb, o.Ub, list(W))
    self._update(sp, es, xs, o.Uj, list(g))
    self._stats(q, W, K, g)
    self.owe = {**{("buy", b["id"]): k for b, k, z in zip(bs, kb, zb) if z and k > 0},
                **{("sup", s["id"]): k for s, k, z in zip(sp, ks, zs) if z and k > 0}}
    self._next(zb, zs)
    return rec

  # 이력 h ← h_rho h + (1 − h_rho) e, 거래하면 기준점 r ← r_rho r + (1 − r_rho) x (첫 거래 r = x), 재임·제외·신고가 기록
  def _update(self, v, e, x, u, last):
    c = self.cfg
    for p, ek, xk, uk, lk in zip(v, e, x, u, last):
      p["h"] = c.h_rho * p["h"] + (1 - c.h_rho) * float(ek)
      if ek:
        p["r"] = xk if np.isnan(p["r"]) else c.r_rho * p["r"] + (1 - c.r_rho) * xk
      p["ten"], p["ex"], p["last"] = p["ten"] + 1, bool(uk), np.asarray(lk, float)

  # 품목 통계: 과거 평균 단위 가치(W / Σq, 묶음에 든 품목마다), 과거 평균 원가, 공급 부족 이력
  def _stats(self, q, W, K, g):
    has, sup = q > 0, K > 0
    self.vs += (has * (W / np.maximum(q.sum(1), 1))[:, None]).sum(0)
    self.vn += has.sum(0)
    self.cs += (g * sup).sum(0)
    self.cn += sup.sum(0)
    dem, cap = q.sum(0), K.sum(0)
    short = np.where(dem > 0, np.maximum(dem - cap, 0) / np.maximum(dem, 1), 0.0)
    self.hi = self.cfg.h_rho * self.hi + (1 - self.cfg.h_rho) * short

  # 다음 기간: 남은 주문자·공급자는 프로필에 변동을 더하고, 떠난 자리에 후보를 예약하고, 대기가 끝난 후보가 들어오고, 신규 주문자가 도착한다
  def _next(self, zb, zs):
    cfg, rep, t = self.cfg, self.rep, self.t
    self.B = {i: self.B[i] for i, z in zip(self.bid, zb) if z}
    for j, z in zip(self.slot, zs):
      if not z:
        del self.S[j]
        self.pipe[j] = (t + cfg.onboard, market.candidate(cfg, self.mk, rep, t, j))
    self.t = t + 1
    for p in self.B.values():
      p["cur"] = market.perturb(cfg, rep, "buy", p["id"], self.t, p["prof"])
    for p in self.S.values():
      p["cur"] = market.perturb(cfg, rep, "sup", p["id"], self.t, p["prof"])
    for j in [j for j in sorted(self.pipe) if self.pipe[j][0] <= self.t]:
      self.S[j] = _new(self.t * cfg.n_sup + j, self.pipe.pop(j)[1], cfg, np.full(cfg.n_items, np.nan))
    self.S = dict(sorted(self.S.items()))
    self._arrive()


# 정책 pol(PreObs) → (m, θ)으로 T기간을 돌려 기간별 기록
def run(cfg, rep, pol):
  s = Sim(cfg, rep)
  return [s.step(*pol(s.pre())) for _ in range(cfg.T)]
