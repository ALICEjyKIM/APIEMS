"""플랫폼 시장 환경.
매 기간 신규 주문자가 도착하고 공급자가 공급을 제안하며, 정책의 배정 결정을 검사해 이윤을 정산한다.
슬라이스 1: 주문자는 한 기간만 참여하고 공급자는 모두 남는다 (재참여 반응은 슬라이스 2).
"""
from dataclasses import dataclass
import numpy as np
from utils.instance import make
from utils.arrivals import new_buyers, perturb


@dataclass
class Obs:
  """기간 t에 정책이 보는 시장.
  q·p: 주문자별 품목 요구수량·제안가격 (B×I), cap·c: 공급자별 품목 공급가능량·공급가격 (J×I).
  bid·sid: 행 순서대로의 주문자·공급자 식별자.
  """
  t: int
  q: np.ndarray
  p: np.ndarray
  cap: np.ndarray
  c: np.ndarray
  bid: list
  sid: list


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
  cmin = np.where(o.cap > 0, o.c, np.inf).min(0)
  pot = (o.q * np.where(o.q > 0, o.p - cmin, 0)).sum(1) - cfg.f_ship
  return dict(ok=ok, n_ok=ok.sum(), rev=rev, ship=ship, profit=rev - ship, opp=np.maximum(pot, 0)[~ok].sum(), sb=sb, ss=ss)


class Env:
  """한 반복의 시장 환경.
  reset으로 기간 0 시장을 만들고, step(x)로 배정을 정산한 뒤 다음 기간 시장을 o에 둔다.
  도착·변동 난수가 기간·참여자로 키가 잡혀 있어 결정이 달라도 같은 시장이 온다 (공통 난수).
  """

  def __init__(self, cfg, rep):
    self.cfg, self.rep = cfg, rep
    self.inst = make(cfg, rep)

  # 기간 0 시장: 기간 0 공급자와 신규 주문자
  def reset(self):
    self.t, self.nb = 0, 0
    self.sups = dict(enumerate(self.inst.sups))
    self.off = dict(self.sups)
    self._arrive()
    return self.o

  # 기간 t 신규 주문자를 받고 관측을 만든다
  def _arrive(self):
    bs = new_buyers(self.cfg, self.inst, self.rep, self.t)
    self.ords = dict(zip(range(self.nb, self.nb + len(bs)), bs))
    self.nb += len(bs)
    q, p = _mat(self.ords.values(), self.cfg.n_items)
    cap, c = _mat(self.off.values(), self.cfg.n_items)
    self.o = Obs(self.t, q, p, cap, c, list(self.ords), list(self.off))

  # 결정을 정산하고 다음 기간으로: 주문자는 떠나고, 공급자는 남아 프로필에 변동을 더해 공급한다
  def step(self, dec):
    o = self.o
    r = settle(self.cfg, o, dec)
    self.t += 1
    self.off = {j: perturb(self.cfg, self.rep, "sup", j, self.t, s) for j, s in self.sups.items()}
    self._arrive()
    r.update(n_buy=len(o.bid), stay_buy=len(set(o.bid) & set(self.o.bid)),
             n_sup=len(o.sid), stay_sup=len(set(o.sid) & set(self.o.sid)))
    return r
