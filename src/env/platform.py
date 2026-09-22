"""플랫폼 시장 환경.
매 기간 신규 주문자가 도착하고 공급자가 공급을 제안하며, 정책의 결정을 검사해 이윤을 정산한다.
정산 후 참여자는 배분 잉여에 따른 재참여 확률로 남거나 떠나고, 빈 공급자 자리는 새 공급자가 채운다.
"""
from dataclasses import dataclass
import numpy as np
from utils.instance import make
from utils.arrivals import new_buyers, new_sups, perturb
from utils.common import ret_u
from env.response import Logistic


@dataclass
class Obs:
  """기간 t에 정책이 보는 시장.
  q·p: 주문자별 품목 요구수량·제안가격 (B×I), cap·c: 공급자별 품목 공급가능량·공급가격 (J×I).
  bid·sid: 행 순서대로의 주문자·공급자 식별자, rb·rs: 참여자별 직전 기간 재참여확률 (신규 참여자는 잉여 0의 값).
  """
  t: int
  q: np.ndarray
  p: np.ndarray
  cap: np.ndarray
  c: np.ndarray
  bid: list
  sid: list
  rb: np.ndarray = None
  rs: np.ndarray = None


# 참여자 프로필 목록 → (수량, 가격) 행렬 (참여자가 없으면 0행)
def _mat(profs, n):
  return np.array([p.qty for p in profs], int).reshape(-1, n), np.array([p.price for p in profs], float).reshape(-1, n)


# 결정 (배정 x[b,j,i], 주문자 잉여 sb, 공급자 잉여 ss)을 검사하고 정산한다
# 모든 품목 요구수량이 채워진 주문만 성립(한 품목을 여러 공급자가 나눠 채워도 됨), 잉여는 자기 거래 마진 이내, 플랫폼 이윤 ≥ 0
def settle(cfg, o, dec):
  x, sb, ss = dec
  assert np.issubdtype(x.dtype, np.integer) and x.shape == (len(o.q), len(o.cap), o.q.shape[1])
  assert (x >= 0).all() and (x.sum(0) <= o.cap).all()
  ok = (x.sum(1) == o.q).all(1)
  m = (o.p[:, None] - o.c[None]) * x * ok[:, None, None]
  g, mj = m.sum((1, 2)), m.sum((0, 2))
  assert sb.shape == g.shape and (sb >= -cfg.tol).all() and (sb <= g + cfg.tol).all()
  assert ss.shape == mj.shape and (ss >= -cfg.tol).all() and (ss <= mj + cfg.tol).all()
  rev, ship = m.sum() - sb.sum() - ss.sum(), cfg.f_ship * ok.sum()
  assert rev - ship >= -cfg.tol
  cmin = np.where(o.cap > 0, o.c, np.inf).min(0, initial=np.inf)
  pot = (o.q * np.where(o.q > 0, o.p - cmin, 0)).sum(1) - cfg.f_ship
  return dict(ok=ok, n_ok=ok.sum(), rev=rev, ship=ship, profit=rev - ship, opp=np.maximum(pot, 0)[~ok].sum(), sb=sb, ss=ss)


# 참여자 제안 금액: 주문자 Σ 제안가격×수량, 공급자 Σ 공급가격×공급가능량 (잉여율의 분모)
def worth(o):
  return (o.p * o.q).sum(1), (o.c * o.cap).sum(1)


class Env:
  """한 반복의 시장 환경.
  reset으로 기간 0 시장을 만들고, step(결정)으로 정산·재참여 판정 후 다음 기간 시장을 o에 둔다.
  도착·변동·재참여 난수가 기간·참여자로 키가 잡혀 있어 결정이 달라도 같은 참여자에게 같은 난수가 쓰인다 (공통 난수).
  """

  def __init__(self, cfg, rep, resp=None):
    self.cfg, self.rep = cfg, rep
    self.resp = resp or Logistic(cfg)
    self.inst = make(cfg, rep)

  # 기간 0 시장: 기간 0 공급자(번호 = 자리)와 신규 주문자
  def reset(self):
    self.t, self.nb = 0, 0
    self.buys, self.ords, self.lb, self.ls = {}, {}, {}, {}
    self.sups = {j: (j, s) for j, s in enumerate(self.inst.sups)}
    self.offs = dict(enumerate(self.inst.sups))
    self._arrive()
    return self.o

  # 기간 t 신규 주문자를 받고 관측을 만든다 (주문자는 남은 주문자 다음에 신규, 공급자는 자리 순)
  def _arrive(self):
    bs = new_buyers(self.cfg, self.inst, self.rep, self.t)
    for m, b in enumerate(bs):
      self.buys[self.nb + m] = self.ords[self.nb + m] = b
    self.nb += len(bs)
    q, p = _mat(self.ords.values(), self.cfg.n_items)
    cap, c = _mat(self.offs.values(), self.cfg.n_items)
    sid = [self.sups[j][0] for j in self.offs]
    rb = np.array([self.lb.get(i, self.cfg.ret_p0) for i in self.ords])
    rs = np.array([self.ls.get(i, self.cfg.ret_p0) for i in sid])
    self.o = Obs(self.t, q, p, cap, c, list(self.ords), sid, rb, rs)

  # 결정을 정산하고 재참여 판정(ret_u < p) 후 다음 기간으로: 남은 참여자는 프로필에 변동을 더해 주문·공급한다
  def step(self, dec):
    cfg, rep, t, o = self.cfg, self.rep, self.t, self.o
    r = settle(cfg, o, dec)
    wb, ws = worth(o)
    r["pb"], r["ps"] = self.resp.prob("buy", r["sb"], wb), self.resp.prob("sup", r["ss"], ws)
    self.lb, self.ls = dict(zip(o.bid, r["pb"])), dict(zip(o.sid, r["ps"]))
    self.t = t + 1
    self.buys = {i: self.buys[i] for i, p in zip(o.bid, r["pb"]) if ret_u(cfg, rep, "buy", i, t) < p}
    self.sups = {j: s for (j, s), p in zip(self.sups.items(), r["ps"]) if ret_u(cfg, rep, "sup", s[0], t) < p}
    new = new_sups(cfg, self.inst, rep, self.t, [j for j in range(len(self.inst.sups)) if j not in self.sups])
    self.sups = dict(sorted({**self.sups, **{j: (self.t * cfg.n_sup + j, s) for j, s in new.items()}}.items()))
    self.offs = {j: new[j] if j in new else perturb(cfg, rep, "sup", i, self.t, s) for j, (i, s) in self.sups.items()}
    self.ords = {i: perturb(cfg, rep, "buy", i, self.t, b) for i, b in self.buys.items()}
    self._arrive()
    r.update(n_buy=len(o.bid), stay_buy=len(set(o.bid) & set(self.o.bid)),
             n_sup=len(o.sid), stay_sup=len(set(o.sid) & set(self.o.sid)))
    return r
