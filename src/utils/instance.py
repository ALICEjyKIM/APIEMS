"""주문자·공급자 프로필 생성.
공급자 수는 고정이고, 자리 구조는 시장 조건(공급 편중도, 대체 공급자 수)으로 정해져 반복 내내 유지된다.
주문자 프로필은 주문당 품목 수만큼 품목을 고르고 품목별 수량과 제안가격 수준을 가진다.
"""
from dataclasses import dataclass, field
import numpy as np
from utils.common import rng


@dataclass
class Prof:
  """참여자 프로필 (주문자·공급자 공통).
  items: 품목 포함 여부, qty: 품목별 수량(공급자는 공급가능량), price: 품목별 가격.
  포함하지 않는 품목의 qty·price는 0이다.
  """
  items: np.ndarray
  qty: np.ndarray
  price: np.ndarray


@dataclass
class Inst:
  """한 반복의 시장 인스턴스.
  base: 품목 기준가, sup_items·sup_cap: 공급자 자리별 공급 품목과 공급가능량.
  sups: 기간 0의 공급자 프로필 (자리 j의 공급자).
  """
  base: np.ndarray
  sup_items: np.ndarray
  sup_cap: np.ndarray
  sups: list = field(default_factory=list)


# 품목당 주문 수량 범위: 주문당 품목 수 k에 k_ref/k배를 곱해 주문 총량과 품목별 기대 수요를 k와 무관하게 맞춘다
def qty_range(cfg):
  s = cfg.qty_unit * cfg.k_ref / cfg.items_per_order
  return round(cfg.qty_lo * s), round(cfg.qty_hi * s)


# 공급자별 품목 수:균등 수열과 최대 편중 수열을 공급 편중도로 섞고 합이 연결 수가 되게 반올림
def _degrees(cfg):
  L, S, I = cfg.n_items * cfg.n_alt, cfg.n_sup, cfg.n_items
  assert cfg.n_alt <= S <= L
  eq = np.full(S, L // S)
  eq[:L % S] += 1
  mx, r = np.ones(S, int), L - S
  for j in range(S):
    k = min(I - 1, r)
    mx[j] += k
    r -= k
  x = np.round((1 - cfg.conc) * eq + cfg.conc * mx, 9)
  d = np.floor(x).astype(int)
  d[np.argsort(-(x - d), kind="stable")[:L - d.sum()]] += 1
  return d


# 공급자 자리 구조: 품목 수가 많은 공급자부터 남은 필요 공급자 수가 큰 품목을 고른다 (Ryser 탐욕법)
def _slots(cfg, g):
  d = _degrees(cfg)
  need = np.full(cfg.n_items, cfg.n_alt)
  items = np.zeros((cfg.n_sup, cfg.n_items), bool)
  for j in np.argsort(-d, kind="stable"):
    pick = np.lexsort((g.random(cfg.n_items), -need))[:d[j]]
    items[j, pick] = True
    need[pick] -= 1
  assert (need == 0).all()
  return items


# 반복 rep의 인스턴스: 기준가, 공급자 자리 구조와 용량(안정 상태 활동 주문자 수요 × cover), 기간 0 공급자
def make(cfg, rep):
  g = rng(cfg, rep, "inst")
  base = g.uniform(cfg.base_lo, cfg.base_hi, cfg.n_items)
  items = _slots(cfg, g)
  act = cfg.lam_buy / (1 - cfg.ret_ss)
  lo, hi = qty_range(cfg)
  dem = act * cfg.items_per_order / cfg.n_items * (lo + hi) / 2
  tot = round(cfg.cover * dem)
  cap = np.zeros(items.shape, int)
  for i in range(cfg.n_items):
    cap[items[:, i], i] = 1 + g.multinomial(tot - cfg.n_alt, g.dirichlet(np.ones(cfg.n_alt)))
  inst = Inst(base, items, cap)
  inst.sups = [new_sup(cfg, inst, g, j) for j in range(len(items))]
  return inst


# 자리 j의 새 공급자: 자리의 품목·공급가능량, 새 가격 수준
def new_sup(cfg, inst, g, j):
  it = inst.sup_items[j]
  return Prof(it.copy(), inst.sup_cap[j].copy(), inst.base * g.uniform(cfg.cost_lo, cfg.cost_hi) * it)


# 주문자 프로필 n개: 품목 구성, 품목별 수량, 제안가격 수준
def new_buy(cfg, inst, g, n):
  out = []
  for _ in range(n):
    it = np.zeros(cfg.n_items, bool)
    it[g.choice(cfg.n_items, cfg.items_per_order, replace=False)] = True
    lo, hi = qty_range(cfg)
    qty = g.integers(lo, hi + 1, cfg.n_items) * it
    out.append(Prof(it, qty, inst.base * g.uniform(cfg.markup_lo, cfg.markup_hi) * it))
  return out
