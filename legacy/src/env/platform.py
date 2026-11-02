"""플랫폼 시장 환경.
매 기간 신규 주문자가 도착하고 공급자가 공급을 제안하며, 정책의 결정을 검사해 이윤을 정산한다.
정산 후 참여자는 배분 잉여에 따른 재참여 확률로 남거나 떠나고, 빈 공급자 자리는 새 공급자가 채운다.
cfg.market == "v2"면 Env(cfg, rep)가 Env2(조달 시장 v2)를 만든다: 새 후보 공급자가 onboarding 지연 뒤 자리를 채우고 재참여에 경험이 들어간다.
cfg.hist까지 켜면 EnvH: 주문자·공급자·품목 이력이 거래 결과로 누적되고 재참여에 들어간다.
"""
from dataclasses import dataclass
import numpy as np
from utils.instance import make
from utils.arrivals import new_buyers, new_sups, perturb
from utils.common import ret_u
from env.response import response
from utils.market2 import make2, candidate


@dataclass
class Obs:
  """기간 t에 정책이 보는 시장.
  q·p: 주문자별 품목 요구수량·제안가격 (B×I), cap·c: 공급자별 품목 공급가능량·공급가격 (J×I).
  bid·sid: 행 순서대로의 주문자·공급자 식별자, rb·rs: 참여자별 직전 기간 재참여확률 (신규 참여자는 잉여 0의 값).
  pend·wait: v2에서 onboarding 중인 후보의 품목별 공급가능량과 활동까지 남은 기간 (legacy는 None).
  hb·hs·hi: 이력 환경(EnvH)의 주문자 성립 이력, 공급자 배정 이력, 품목 공급 부족 이력 (그 밖에는 None).
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
  pend: np.ndarray = None
  wait: np.ndarray = None
  hb: np.ndarray = None
  hs: np.ndarray = None
  hi: np.ndarray = None


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

  # cfg.market == "v2"면 Env2를 만든다 (deepcopy는 인자 없이 부른다)
  def __new__(cls, cfg=None, rep=None, resp=None):
    if cls is Env and cfg is not None and cfg.market == "v2":
      cls = EnvH if cfg.hist else Env2
    return super().__new__(cls)

  def __init__(self, cfg, rep, resp=None):
    self.cfg, self.rep = cfg, rep
    self.resp = resp or response(cfg)
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
    self._observe()

  # 현재 참여자로 관측을 만든다
  def _observe(self):
    q, p = _mat(self.ords.values(), self.cfg.n_items)
    cap, c = _mat(self.offs.values(), self.cfg.n_items)
    sid = [self.sups[j][0] for j in self.offs]
    rb = np.array([self.lb.get(i, self.cfg.ret_p0) for i in self.ords])
    rs = np.array([self.ls.get(i, self.cfg.ret_p0) for i in sid])
    self.o = Obs(self.t, q, p, cap, c, list(self.ords), sid, rb, rs)

  # 현재 기간 관측의 종류 kind k번째 참여자를 상태에서 뺀다 (주문자는 주문·프로필 삭제, 공급자는 자리를 비움)
  def drop(self, kind, k):
    if kind == "buy":
      i = self.o.bid[k]
      del self.buys[i], self.ords[i]
    else:
      j = list(self.offs)[k]
      del self.sups[j], self.offs[j]
    self._observe()

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


class Env2(Env):
  """조달 시장 v2의 한 반복 (Env(cfg, rep)가 cfg.market == "v2"일 때 만든다).
  기간 t에 떠난 공급자 자리는 새로 생성한 후보(품목·용량·가격이 새것)가 t + onboard부터 채우고, 그동안 후보는 pend·wait로만 보인다.
  재참여는 잉여율과 경험(주문자 성립, 공급자 거래 배정)에 의존하고, step은 배정·경험·잔류 결과를 원자료로 함께 돌려준다.
  """

  def __init__(self, cfg, rep, resp=None):
    self.cfg, self.rep = cfg, rep
    self.resp = resp or response(cfg)
    self.inst = make2(cfg, rep)

  # 기간 0 시장 (onboarding 대기열은 비어 있다)
  def reset(self):
    self.pipe = {}
    return super().reset()

  # 현재 참여자와 onboarding 후보로 관측을 만든다 (신규 참여자의 직전 재참여확률은 경험 없음·잉여 0의 값)
  def _observe(self):
    q, p = _mat(self.ords.values(), self.cfg.n_items)
    cap, c = _mat(self.offs.values(), self.cfg.n_items)
    sid = [self.sups[j][0] for j in self.offs]
    p0 = float(self.resp.prob("buy", 0.0, 1.0, 0.0))
    rb = np.array([self.lb.get(i, p0) for i in self.ords])
    rs = np.array([self.ls.get(i, p0) for i in sid])
    pend = np.array([self.pipe[j][1].qty for j in sorted(self.pipe)], int).reshape(-1, self.cfg.n_items)
    wait = np.array([self.pipe[j][0] - self.t for j in sorted(self.pipe)], int)
    self.o = Obs(self.t, q, p, cap, c, list(self.ords), sid, rb, rs, pend, wait)

  # 자리 j가 기간 t에 비었다: 후보를 만들어 t + onboard에 활동하도록 대기열에 넣는다
  def _leave(self, j, t):
    self.pipe[j] = (t + self.cfg.onboard, candidate(self.cfg, self.inst, self.rep, t, j))

  # 참여자를 뺀다 (공급자는 이번 기간에 떠난 것으로 보고 자리에 후보를 예약한다)
  def drop(self, kind, k):
    if kind == "sup":
      self._leave(list(self.offs)[k], self.t)
    super().drop(kind, k)

  # 재참여확률 (주문자: 성립 경험, 공급자: 배정 경험)
  def _probs(self, o, r, alloc):
    wb, ws = worth(o)
    return self.resp.prob("buy", r["sb"], wb, r["ok"]), self.resp.prob("sup", r["ss"], ws, alloc)

  # 정산 결과를 상태에 남긴다 (Env2는 남기지 않는다. EnvH는 이력을 갱신)
  def _record(self, o, r, alloc):
    pass

  # 정산 → 경험 의존 재참여 판정 → 떠난 자리에 후보 예약 → 대기 끝난 후보 활동 → 다음 기간 시장
  def step(self, dec):
    cfg, rep, t, o = self.cfg, self.rep, self.t, self.o
    r = settle(cfg, o, dec)
    x = dec[0]
    alloc = (x * r["ok"][:, None, None]).sum((0, 2)) > 0
    r["pb"], r["ps"] = self._probs(o, r, alloc)
    self.lb, self.ls = dict(zip(o.bid, r["pb"])), dict(zip(o.sid, r["ps"]))
    self._record(o, r, alloc)
    self.t = t + 1
    self.buys = {i: self.buys[i] for i, p in zip(o.bid, r["pb"]) if ret_u(cfg, rep, "buy", i, t) < p}
    stay = {j: s for (j, s), p in zip(self.sups.items(), r["ps"]) if ret_u(cfg, rep, "sup", s[0], t) < p}
    for j in self.sups:
      if j not in stay:
        self._leave(j, t)
    new = {j: self.pipe.pop(j)[1] for j in sorted(self.pipe) if self.pipe[j][0] <= self.t}
    self.sups = dict(sorted({**stay, **{j: (self.t * cfg.n_sup + j, s) for j, s in new.items()}}.items()))
    self.offs = {j: new[j] if j in new else perturb(cfg, rep, "sup", i, self.t, s) for j, (i, s) in self.sups.items()}
    self.ords = {i: perturb(cfg, rep, "buy", i, self.t, b) for i, b in self.buys.items()}
    self._arrive()
    r.update(n_buy=len(o.bid), stay_buy=len(set(o.bid) & set(self.o.bid)),
             n_sup=len(o.sid), stay_sup=len(set(o.sid) & set(self.o.sid)),
             x=x, alloc=alloc, bid=list(o.bid), sid=list(o.sid),
             kept_b=np.isin(o.bid, self.o.bid), kept_s=np.isin(o.sid, self.o.sid), joined=sorted(new))
    return r


class EnvH(Env2):
  """조달 시장 v2 + 거래 이력 (cfg.hist).
  참여자 ID마다 주문자 성립 이력 hB, 공급자 배정 이력 hS를, 품목마다 공급 부족 이력 hI를 지수 평활 h ← ρh + (1 − ρ)x로 누적한다.
  재참여는 정산 전 이력 h_t로 정하고 그 뒤에 갱신한다. 떠난 참여자의 이력은 지우고, 신규 참여자(새 ID)는 h0에서 시작한다.
  """

  def reset(self):
    self.hB, self.hS, self.hI = {}, {}, np.zeros(self.cfg.n_items)
    return super().reset()

  # 관측에 이력을 붙인다 (활동 참여자의 이력만 남기고, 처음 보는 ID는 h0)
  def _observe(self):
    super()._observe()
    o, h0 = self.o, self.cfg.h0
    self.hB = {i: self.hB.get(i, h0) for i in o.bid}
    self.hS = {i: self.hS.get(i, h0) for i in o.sid}
    o.hb = np.array([self.hB[i] for i in o.bid])
    o.hs = np.array([self.hS[i] for i in o.sid])
    o.hi = self.hI.copy()

  # 재참여확률: 이번 기간 경험·잉여율과 정산 전 이력
  def _probs(self, o, r, alloc):
    wb, ws = worth(o)
    return (self.resp.prob("buy", r["sb"], wb, r["ok"], o.hb), self.resp.prob("sup", r["ss"], ws, alloc, o.hs))

  # 이력 갱신: 주문자 ← 성립, 공급자 ← 배정, 품목 ← 공급 부족률 max(0, 요구량 − 활동 공급가능량) / 요구량 (요구량 0이면 0)
  def _record(self, o, r, alloc):
    rho = self.cfg.h_rho
    for i, h, y in zip(o.bid, o.hb, r["ok"]):
      self.hB[i] = rho * h + (1 - rho) * float(y)
    for i, h, e in zip(o.sid, o.hs, alloc):
      self.hS[i] = rho * h + (1 - rho) * float(e)
    dem, cap = o.q.sum(0), o.cap.sum(0)
    short = np.where(dem > 0, np.maximum(dem - cap, 0) / np.maximum(dem, 1), 0.0)
    unmet = np.where(dem > 0, (o.q * ~r["ok"][:, None]).sum(0) / np.maximum(dem, 1), 0.0)
    self.hI = rho * self.hI + (1 - rho) * short
    r.update(short=short, unmet=unmet)
