"""조달 시장 v2의 시장 생성 (cfg.market == "v2"에서만 쓴다. legacy 생성은 utils/instance.py 그대로).
품목마다 qualification 가중치를 뽑아 품목별 대체 공급자 수가 달라지게 하고(het), 공급자 자리의 품목 수는 conc로 정한다.
품목 명목 용량은 정책과 무관하게 명목 수요 × cover로 두고, 떠난 공급자 자리는 새로 생성한 후보가 onboarding 지연 뒤 채운다.
"""
from dataclasses import dataclass, field
import numpy as np
from utils.common import rng
from utils.instance import Prof, qty_range, _degrees


@dataclass
class Market:
  """한 반복의 v2 시장 인스턴스.
  base: 품목 기준가, w: 품목 qualification 가중치(합 1), n_tgt: 품목별 목표 대체 공급자 수, K: 품목 명목 용량.
  deg: 공급자 자리별 품목 수, sups: 기간 0 공급자 프로필 (자리 j).
  """
  base: np.ndarray
  w: np.ndarray
  n_tgt: np.ndarray
  K: np.ndarray
  deg: np.ndarray
  sups: list = field(default_factory=list)


# 품목 명목 수요: 명목 활동 주문자 수(lam_buy / (1 − ret_ss)) × 주문당 품목 수 / 품목 수 × 평균 품목 수량 (정책과 무관)
def demand(cfg):
  lo, hi = qty_range(cfg)
  return cfg.lam_buy / (1 - cfg.ret_ss) * cfg.items_per_order / cfg.n_items * (lo + hi) / 2


# 품목 qualification 가중치: het = 0이면 균등, 아니면 Gamma(1/het², 1) 정규화 (가중치의 변동계수 ≈ het)
def weights(cfg, g):
  if cfg.het == 0:
    return np.full(cfg.n_items, 1 / cfg.n_items)
  x = g.gamma(1 / cfg.het ** 2, 1.0, cfg.n_items)
  return x / x.sum()


# 자리 품목 수 deg와 품목별 공급자 수 n으로 연결 구조를 만들 수 있는지 (Gale–Ryser 조건)
def feasible(deg, n):
  n = np.sort(n)[::-1]
  return all(n[:k].sum() <= np.minimum(deg, k).sum() for k in range(1, len(n) + 1))


# 품목별 목표 대체 공급자 수: 품목마다 공통 하한 f를 먼저 주고 나머지 연결을 가중치로 최대 나머지 반올림 (상한 n_sup)
# f는 1부터 올려 연결 구조를 만들 수 있는 가장 작은 값 (conc가 크면 품목을 많이 가진 자리가 모든 품목의 하한을 올린다)
def targets(cfg, w, deg):
  L, I = cfg.n_items * cfg.n_alt, cfg.n_items
  for f in range(1, cfg.n_alt + 1):
    x = (L - f * I) * w
    n = np.floor(x).astype(int)
    n[np.argsort(-(x - n), kind="stable")[:L - f * I - n.sum()]] += 1
    n += f
    if n.max() <= cfg.n_sup and feasible(deg, n):
      return n
  raise AssertionError("연결 구조를 만들 수 없음")


# 기간 0 자리 구조: 품목 수가 많은 자리부터 남은 필요 공급자 수가 큰 품목을 고른다 (legacy _slots와 같은 Ryser 탐욕법, need = 목표 수)
def slots(cfg, g, deg, n_tgt):
  need = n_tgt.copy()
  items = np.zeros((cfg.n_sup, cfg.n_items), bool)
  for j in np.argsort(-deg, kind="stable"):
    pick = np.lexsort((g.random(cfg.n_items), -need))[:deg[j]]
    assert (need[pick] > 0).all()
    items[j, pick] = True
    need[pick] -= 1
  assert (need == 0).all()
  return items


# 공급자 하나의 프로필: 품목 it의 연결 용량 = 품목 명목 용량 / 목표 대체 공급자 수 × Gamma(dir_alpha, 1/dir_alpha) (최소 1),
# 가격 수준은 연결마다 U(cost_lo, cost_hi)
def profile(cfg, mk, g, it):
  cap = np.maximum(1, np.round(mk.K / mk.n_tgt * g.gamma(cfg.dir_alpha, 1 / cfg.dir_alpha, cfg.n_items))).astype(int) * it
  lvl = g.uniform(cfg.cost_lo, cfg.cost_hi, cfg.n_items)
  return Prof(it.copy(), cap, mk.base * lvl * it)


# 반복 rep의 v2 시장: 기준가(legacy와 같은 첫 난수), qualification 가중치, 목표 수, 명목 용량, 자리 품목 수, 기간 0 공급자
def make2(cfg, rep):
  g = rng(cfg, rep, "inst")
  base = g.uniform(cfg.base_lo, cfg.base_hi, cfg.n_items)
  w = weights(cfg, g)
  deg = _degrees(cfg)
  n_tgt = targets(cfg, w, deg)
  mk = Market(base, w, n_tgt, np.full(cfg.n_items, cfg.cover * demand(cfg)), deg)
  items = slots(cfg, g, deg, n_tgt)
  mk.sups = [profile(cfg, mk, g, items[j]) for j in range(cfg.n_sup)]
  return mk


# 자리 j가 기간 t에 비면 생기는 대체 후보: 품목 deg[j]개를 qualification 가중치로 비복원 추출, 용량·가격은 새로 생성
# 난수 키 (반복, arr 스트림, 2, t, j): 정책이 달라도 같은 자리가 같은 기간에 비면 같은 후보가 온다
def candidate(cfg, mk, rep, t, j):
  g = rng(cfg, rep, "arr", 2, t, j)
  it = np.zeros(cfg.n_items, bool)
  it[g.choice(cfg.n_items, mk.deg[j], replace=False, p=mk.w)] = True
  return profile(cfg, mk, g, it)
