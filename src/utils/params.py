"""실험 설정을 한곳에 모은 frozen Cfg.
소규모 예비 실험의 규모, 시장 조건, 주문자·공급자 생성 범위를 담는다.
시장 조건을 바꿀 때는 dataclasses.replace(cfg, multi_ratio=...)를 쓴다.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
  """모든 수치 설정.
  규모, 시장 조건, 주문자·공급자 생성 범위, 재참여 변동.
  값을 바꾸려면 dataclasses.replace로 새 Cfg를 만든다.
  """
  # 규모
  seed: int = 0
  n_items: int = 5
  T: int = 25
  reps: int = 10
  # 시장 조건
  multi_ratio: float = 0.3
  n_alt: int = 2
  items_per_order: int = 2
  # 주문자
  lam_buy: float = 4.0
  qty_lo: int = 2
  qty_hi: int = 5
  markup_lo: float = 1.1
  markup_hi: float = 1.5
  # 공급자
  cost_lo: float = 0.8
  cost_hi: float = 1.1
  cover: float = 1.3
  p_sup_new: float = 0.5
  # 품목
  base_lo: float = 10.0
  base_hi: float = 20.0
  # 재참여 변동
  noise_qty: float = 0.1
  noise_price: float = 0.05
