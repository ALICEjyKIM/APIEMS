"""주문자·공급자 프로필 생성.
공급자 자리 구조는 시장 조건(다품목 공급자 비율, 대체 공급자 수)으로 정해지고 반복 내내 고정된다.
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


# 공급자 자리 구조: 품목마다 n_alt 자리, 서로 다른 품목 자리 둘을 합쳐 다품목 공급자를 만든다
def _slots(cfg, g):
  cnt = np.full(cfg.n_items, cfg.n_alt)
  m = round(cfg.multi_ratio * cnt.sum() / (1 + cfg.multi_ratio))
  sets = []
  for _ in range(m):
    a, b = np.lexsort((g.random(cfg.n_items), -cnt))[:2]
    sets.append((a, b))
    cnt[a] -= 1
    cnt[b] -= 1
  sets += [(i,) for i in range(cfg.n_items) for _ in range(cnt[i])]
  items = np.zeros((len(sets), cfg.n_items), bool)
  for j, s in enumerate(sets):
    items[j, list(s)] = True
  return items


# 반복 rep의 인스턴스: 기준가, 공급자 자리 구조와 용량(안정 상태 활동 주문자 수요 × cover), 기간 0 공급자
def make(cfg, rep):
  g = rng(cfg, rep, "inst")
  base = g.uniform(cfg.base_lo, cfg.base_hi, cfg.n_items)
  items = _slots(cfg, g)
  act = cfg.lam_buy / (1 - cfg.ret_ss)
  dem = act * cfg.items_per_order / cfg.n_items * (cfg.qty_lo + cfg.qty_hi) / 2
  cap = np.zeros(items.shape)
  for i in range(cfg.n_items):
    js = np.flatnonzero(items[:, i])
    cap[js, i] = np.maximum(1, np.round(cfg.cover * dem * g.dirichlet(np.ones(len(js)))))
  inst = Inst(base, items, cap.astype(int))
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
    qty = g.integers(cfg.qty_lo, cfg.qty_hi + 1, cfg.n_items) * it
    out.append(Prof(it, qty, inst.base * g.uniform(cfg.markup_lo, cfg.markup_hi) * it))
  return out
