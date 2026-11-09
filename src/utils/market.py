"""시장 생성: 품목, 공급자 자리 구조·용량·원가, 신규 주문자, 재참여 변동, 빈자리 후보.
이전 구조의 조달 시장 v2(PROC15H) 생성 규칙을 그대로 옮겼고, 같은 seed에서 같은 값이 나온다 (tests/test_market.py).
가격은 참여자만 아는 값이다: 주문자 price = 품목별 단위 가치, 공급자 price = 품목별 단위 원가.
"""
from dataclasses import dataclass, field
import numpy as np
from utils.rng import rng, KINDS


@dataclass
class Prof:
  """참여자 프로필 (주문자·공급자 공통).
  items: 품목 포함 여부, qty: 품목별 수량(공급자는 공급가능량), price: 품목별 단위 가치(주문자) 또는 원가(공급자).
  포함하지 않는 품목의 qty·price는 0이다.
  """
  items: np.ndarray
  qty: np.ndarray
  price: np.ndarray


@dataclass
class Market:
  """한 반복의 시장 인스턴스.
  base: 품목 기준가, w: 품목 자격 가중치(합 1), n_tgt: 품목별 목표 대체 공급자 수, K: 품목 명목 용량.
  deg: 공급자 자리별 품목 수, sups: 기간 0 공급자 프로필 (자리 j).
  """
  base: np.ndarray
  w: np.ndarray
  n_tgt: np.ndarray
  K: np.ndarray
  deg: np.ndarray
  sups: list = field(default_factory=list)


# 품목당 주문 수량 범위: 주문당 품목 수 k에 k_ref/k배를 곱해 품목별 기대 수요를 k와 무관하게 맞춘다
def qty_range(cfg):
  s = cfg.qty_unit * cfg.k_ref / cfg.items_per_order
  return round(cfg.qty_lo * s), round(cfg.qty_hi * s)


# 품목 명목 수요: 명목 활동 주문자 수 × 주문당 품목 수 / 품목 수 × 평균 품목 수량 (정책과 무관)
def demand(cfg):
  lo, hi = qty_range(cfg)
  return cfg.lam_buy / (1 - cfg.ret_ss) * cfg.items_per_order / cfg.n_items * (lo + hi) / 2


# 기준 자리 점유율: 경험 있음·잉여 0으로 머무는 공급자의 평균 재임 1 / (1 − r2_p1)과 빈 기간 onboard − 1의 비 (해석값)
def occ_ref(cfg):
  ten = 1 / (1 - cfg.r2_p1)
  return ten / (ten + cfg.onboard - 1)


# 품목 명목 용량: cover를 실효 공급 배수로 보고 기준 자리 점유율로 나눈다
def capacity(cfg):
  return cfg.cover * demand(cfg) / occ_ref(cfg)


# 공급자 자리별 품목 수: 균등 수열과 최대 편중 수열을 conc로 섞고 합이 연결 수가 되게 반올림
def degrees(cfg):
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


# 품목 자격 가중치: het = 0이면 균등, 아니면 Gamma(1/het², 1) 정규화 (변동계수 ≈ het)
def weights(cfg, g):
  if cfg.het == 0:
    return np.full(cfg.n_items, 1 / cfg.n_items)
  x = g.gamma(1 / cfg.het ** 2, 1.0, cfg.n_items)
  return x / x.sum()


# 자리 품목 수 deg와 품목별 공급자 수 n으로 연결 구조를 만들 수 있는지 (Gale–Ryser 조건)
def feasible(deg, n):
  n = np.sort(n)[::-1]
  return all(n[:k].sum() <= np.minimum(deg, k).sum() for k in range(1, len(n) + 1))


# 품목별 목표 대체 공급자 수: 공통 하한 f를 주고 나머지 연결을 가중치로 최대 나머지 반올림 (f는 가능한 가장 작은 값)
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


# 기간 0 자리 구조: 품목 수가 많은 자리부터 남은 필요 공급자 수가 큰 품목을 고른다 (Ryser 탐욕법)
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


# 공급자 하나의 프로필: 품목 용량 = 명목 용량 / 목표 대체 공급자 수 × Gamma(dir_alpha, 1/dir_alpha) (최소 1), 원가 수준은 품목마다 U(cost_lo, cost_hi)
def profile(cfg, mk, g, it):
  cap = np.maximum(1, np.round(mk.K / mk.n_tgt * g.gamma(cfg.dir_alpha, 1 / cfg.dir_alpha, cfg.n_items))).astype(int) * it
  lvl = g.uniform(cfg.cost_lo, cfg.cost_hi, cfg.n_items)
  return Prof(it.copy(), cap, mk.base * lvl * it)


# 반복 rep의 시장: 기준가, 자격 가중치, 목표 수, 명목 용량, 자리 품목 수, 기간 0 공급자
def make(cfg, rep):
  g = rng(cfg, rep, "inst")
  base = g.uniform(cfg.base_lo, cfg.base_hi, cfg.n_items)
  w = weights(cfg, g)
  deg = degrees(cfg)
  n_tgt = targets(cfg, w, deg)
  mk = Market(base, w, n_tgt, np.full(cfg.n_items, capacity(cfg)), deg)
  items = slots(cfg, g, deg, n_tgt)
  mk.sups = [profile(cfg, mk, g, items[j]) for j in range(cfg.n_sup)]
  return mk


# 자리 j가 기간 t에 비면 생기는 대체 후보 (sub = 3이면 t는 자리의 교체 순번): 품목 deg[j]개를 자격 가중치로 비복원 추출, 용량·원가는 새로 생성
def candidate(cfg, mk, rep, t, j, sub=2):
  g = rng(cfg, rep, "arr", sub, t, j)
  it = np.zeros(cfg.n_items, bool)
  it[g.choice(cfg.n_items, mk.deg[j], replace=False, p=mk.w)] = True
  return profile(cfg, mk, g, it)


# 기간 t의 신규 주문자 (Poisson 도착): 품목 구성, 품목별 수량, 단위 가치 = 기준가 × U(markup_lo, markup_hi)
def new_buyers(cfg, mk, rep, t):
  g = rng(cfg, rep, "arr", 0, t)
  out, lo, hi = [], *qty_range(cfg)
  for _ in range(g.poisson(cfg.lam_buy)):
    it = np.zeros(cfg.n_items, bool)
    it[g.choice(cfg.n_items, cfg.items_per_order, replace=False)] = True
    qty = g.integers(lo, hi + 1, cfg.n_items) * it
    out.append(Prof(it, qty, mk.base * g.uniform(cfg.markup_lo, cfg.markup_hi) * it))
  return out


# 재참여 참여자의 기간 t 주문·공급: 프로필 수량·가격에 작은 상대 변동을 더한다
def perturb(cfg, rep, kind, pid, t, prof):
  g = rng(cfg, rep, "noise", KINDS[kind], pid, t)
  n = len(prof.items)
  qty = np.maximum(1, np.round(prof.qty * (1 + cfg.noise_qty * g.standard_normal(n)))).astype(int)
  price = prof.price * (1 + cfg.noise_price * g.standard_normal(n))
  return Prof(prof.items.copy(), qty * prof.items, price * prof.items)
