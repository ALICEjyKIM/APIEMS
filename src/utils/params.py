"""실험 설정을 한곳에 모은 frozen Cfg.
소규모 예비 실험의 규모, 시장 조건, 주문자·공급자 생성 범위를 담는다.
시장 조건을 바꿀 때는 dataclasses.replace(cfg, conc=...)를 쓴다.
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
  n_items: int = 6
  n_sup: int = 6
  T: int = 30
  reps: int = 30  # 정책 비교(평가 seed) 반복 수. 10반복은 몫 간 차이를 가려낼 검정력이 부족해 30으로 늘림
  # 시장 조건
  conc: float = 0.5
  n_alt: int = 2
  items_per_order: int = 2
  # 주문자
  lam_buy: float = 10.0  # 기간당 신규 주문자 수 평균. 품목별 총 공급용량도 같은 비율로 커져 k = 1 주문 한 건(평균 21)이 용량(151)의 1/4 이하
  qty_lo: int = 2
  qty_hi: int = 5
  qty_unit: int = 3  # 수량 단위 배수 (공급용량도 같은 비율로 커짐)
  k_ref: int = 2  # 품목당 수량 범위 [qty_lo, qty_hi] × qty_unit이 적용되는 기준 주문당 품목 수
  markup_lo: float = 1.1
  markup_hi: float = 1.5
  # 공급자
  cost_lo: float = 0.8
  cost_hi: float = 1.1
  # 공급자 가격 수준: True면 품목별 공급용량처럼 (품목, 그 품목의 k번째 공급자)마다 뽑아 품목별 평균 공급가격이 편중도와 무관하다 (실험 3부터).
  # False면 공급자마다 하나를 뽑는 기존 방식이다 (슬라이스 2~4의 diag_*·exp1_* 결과를 재현할 때 고정)
  price_rank: bool = True
  cover: float = 1.3  # 기대 실효 공급(품목별 총 공급용량 × 공급자 자리 점유율) / 안정 상태 수요
  # 품목 용량을 대체 공급자들에게 나누는 Dirichlet 집중 파라미터. 총 공급량은 값과 무관하게 같고, 클수록 고르게 나뉜다.
  # 1.0이면 기존 동작(단체 위 균등)이라 옛 결과가 그대로 재현된다
  dir_alpha: float = 1.0
  occ: float = 0.6022222222222221  # 공급자 자리 점유율: 기준 몫 sh_ref로 운영한 튜닝용 반복의 실측값 (bench/tune.measure로 한 번 재서 고정)
  ret_ss: float = 0.5
  p_sup_new: float = 0.25  # 빈 공급자 자리 충원 확률 (빈자리 평균 4기간). 새 공급자 검증·계약·등록에 시간이 걸리고, 충원이 빠르면 공급자를 잃는 손해가 없어 유지 가치가 의미를 잃는다
  # 품목
  base_lo: float = 10.0
  base_hi: float = 20.0
  # 재참여 변동
  noise_qty: float = 0.1
  noise_price: float = 0.05
  # 재참여 반응: 잉여율(배분 잉여 / 제안 금액)의 로지스틱을 두 점으로 정한다 (주문자·공급자 동일):
  # 잉여 0이면 ret_p0, 기준 잉여율 r_ref(기준 몫 sh_ref로 마진의 30%를 받을 때의 평균 잉여율, 튜닝용 반복 실측)이면 ret_ref
  # 기울기 b = (logit(ret_ref) − logit(ret_p0)) / r_ref (bench/tune.measure)
  ret_p0: float = 0.5
  ret_ref: float = 0.85
  r_ref_buy: float = 0.10478922733208504
  r_ref_sup: float = 0.10198023148034958
  b_buy: float = 16.55323833900429
  b_sup: float = 17.00918923411489
  # 재참여확률 구간선형 근사: 잉여율 0 ~ 확률 pwl_hi 지점 등간격 pwl_n점 + 잉여율 pwl_rmax 끝점
  pwl_n: int = 5
  pwl_hi: float = 0.97
  pwl_rmax: float = 1.0
  # 고정 잉여 비율 (거래 마진 기준, 나머지는 플랫폼 몫). sh_ref = 사전 기준 몫(주문자, 공급자): 반응 보정의 기준 잉여율과 규칙 기반 튜닝의 비교 기준.
  # sh_buy·sh_sup = 규칙 기반이 쓰는 몫 (기본 시장 튜닝 결과)
  sh_ref: tuple = (0.3, 0.3)
  sh_buy: float = 0.3
  sh_sup: float = 0.3
  sh_buy_grid: tuple = (0.0, 0.1, 0.2, 0.3)
  sh_sup_grid: tuple = (0.1, 0.2, 0.3, 0.4)
  # 가치 근사: 가치 = 후속 정책(규칙 기반)으로 vf_H기간 이윤 합, 학습 데이터는 튜닝 seed vf_reps반복
  # 시뮬레이션 기준치는 미래 난수 키만 바꾼 mc_R개 rollout 평균 (키 = mc_rep0 + (rep × T + t) × mc_R + k)
  vf_H: int = 10
  vf_reps: int = 30
  mc_R: int = 10
  mc_rep0: int = 1000000
  vf_val: float = 0.2  # 검증 분할: 학습 반복의 뒤 비율 (모든 근사 방법 공통)
  # 미래가치 타깃은 같은 상태에서 미래 키만 바꾼 rollout의 Monte Carlo 평균이다 (슬라이스 4).
  # 키 = tgt_rep0 + (rep × T + t) × tgt_stride + k. mc_value의 기준치 키(mc_rep0 계열)와 분리된 스트림이며,
  # stride가 tgt_M·val_M보다 크므로 옆 상태와 겹치지 않는다. 학습·검증은 반복이 달라 서로 겹칠 수 없다
  tgt_rep0: int = 2000000
  tgt_stride: int = 16
  val_M: int = 10  # 검증 타깃의 미래 rollout 수 (슬라이스 4에서 고정)
  tgt_M: int = 5  # 학습 타깃의 미래 rollout 수 (슬라이스 4b에서 결정: M=5가 M=1·2보다 유의하게 낫다, runs/diag_s4b_train_m.json)
  # 슬라이스 5 공정 비교 설계 (확정): 학습 반복 0~71(M = tgt_M) / 검증 72~89(M = val_M, 설정 선택) / 시험 90~107(M = val_M, 보고 전용)
  # 시험 반복은 vf_reps3 뒤 test_reps개이며 설정 선택에 전혀 쓰지 않는다
  test_reps: int = 18
  # 최종 보고용 시험 반복의 시작: 90~107은 baseline 칸에서 용량 격자 판단(슬라이스 5b)에 한 번 써서 은퇴시켰다.
  # 모든 칸이 같은 분할을 쓰도록 시험은 108~125로 둔다 (학습 0~71, 검증 72~89는 그대로)
  test_rep0: int = 108
  # 튜닝 예산: 모델마다 설정 12개. 선형 λ 12점 (1e0~1e4 로그), MLP·GNN은 wd 6개 × 용량 2단계
  # GNN의 용량 단계는 각 nn_hidden의 MLP와 파라미터 수가 가장 가까운 은닉 크기다 (vfa/gnn.hidden)
  fair_lams: tuple = tuple(float(10 ** (4 * i / 11)) for i in range(12))
  fair_wds: tuple = (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1)
  fair_hidden: tuple = (32, 64)
  # 조기 종료 (MLP·GNN 공통): es_epochs까지 es_every에폭마다 검증 MSE를 재고 최소였던 에폭의 가중치로 되돌린다
  es_epochs: int = 1000
  es_every: int = 25
  # 초기화 seed: 선택은 첫 seed 하나로, 고른 설정을 전부로 다시 학습한다. 시험 반복 블록 부트스트랩 횟수
  fair_seeds: tuple = (0, 1, 2, 3, 4)
  boot_B: int = 2000
  lin_lams: tuple = (0.0, 0.1, 1.0, 10.0, 100.0)  # 선형 근사 릿지 λ 후보 (튜닝 예산 5개, MLP와 같게)
  # MLP: 은닉 nn_layers층 × nn_hidden, Adam nn_lr, 전체 배치 nn_epochs회, weight decay 후보 nn_wds (튜닝 예산 5개, 선형과 같게)
  nn_hidden: int = 64
  nn_layers: int = 2
  nn_lr: float = 1e-3
  nn_epochs: int = 500
  nn_wds: tuple = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
  nn_seed: int = 0
  # GNN: 주문자 자리 수(활성 마스크), 은닉 크기, 메시지 전달 층 수
  # gnn_hidden = 0이면 MLP 파라미터 수에 가장 가까운 값을 자동으로 쓴다 (vfa/gnn.hidden). 고정하려면 값을 넣는다
  gnn_nb: int = 80
  gnn_hidden: int = 0
  gnn_layers: int = 2
  # 실험 3: 학습 반복 수(실험 1의 3배), 예측·변환 오차 상태를 뽑는 검증 반복 수
  vf_reps3: int = 90
  e3_state_reps: int = 6
  # 유지 가치 변환 오차의 시뮬레이션 기준치 rollout 수 (예측 오차는 mc_R)
  conv_R: int = 30
  # 유지 가치 가드 (모든 근사 방법 공통): 종류 평균 쪽 shrink 비율, 상한 = coef_hi × 상태 가치
  shrink: float = 0.5
  coef_hi: float = 0.2
  # 튜닝·보정용 반복 (평가 seed와 분리)
  tune_seed: int = 1
  n_tune: int = 10
  # 조달 시장 v2 (market = "v2"일 때만 쓴다. 기본값 "legacy"는 기존 환경 그대로이며 아래 값을 읽지 않는다)
  # het: 품목 qualification 가중치의 변동계수 (0 = 모든 품목 대체 공급자 n_alt개, 클수록 bottleneck 품목이 생김)
  # onboard: 떠난 공급자 자리를 새 후보가 채우기까지의 기간 L (기간 t에 떠나면 t + L부터 활동, 빈 기간 L − 1)
  # v2의 품목 명목 용량은 cover × 명목 수요이며 occ(정책 실측값)를 쓰지 않는다
  # 재참여: p = σ(β0 + β2 · 경험 + β1 · 잉여율). 경험 = 주문자 주문 성립, 공급자 거래 배정 여부 (주문자·공급자 같은 식)
  # 세 점으로 정한다: 경험 없음·잉여 0이면 r2_p0, 경험 있음·잉여 0이면 r2_p1, 경험 있음·잉여율 r2_rate면 r2_p2
  market: str = "legacy"
  het: float = 0.0
  onboard: int = 4
  r2_p0: float = 0.3
  r2_p1: float = 0.6
  r2_p2: float = 0.85
  r2_rate: float = 0.1
  # 허브 통합 배송: 성립 주문당 허브→주문자 고정비만 (개당 비용·분할 공급 추가 비용 없음)
  f_ship: float = 25.0
  # MILP (gurobipy)
  tol: float = 1e-4  # 결정 검사 허용 오차 (잉여 범위, 플랫폼 이윤 ≥ 0). 솔버 정수 허용오차 1e-5가 마진을 거쳐 전파되는 크기보다 크게
  grb_seed: int = 0
  grb_threads: int = 1
  mip_gap: float = 1e-6

# 새 규모 후보 (슬라이스 3에서 정함): 품목 15, 공급자 자리 24, n_alt 4, 묶음 2, 편중도 0.5.
# Cfg 기본값(품목 6, 공급자 6)은 옛 결과 재현을 위해 그대로 두고, 이 preset을 dataclasses.replace로 적용한다.
# occ와 기준 잉여율은 이 규모에서 고정점까지 다시 잰 값이다 (result/diag_s3_scaleup.py → runs/diag_s3_scaleup.json).
# b_buy·b_sup은 두 점 기울기 (logit(ret_ref) − logit(ret_p0)) / r_ref로 r_ref에서 유도된다.
# dir_alpha 5는 용량 꼬리만 자르고 공급자 간 규모 차이는 남기는 값이다 (result/diag_s3b_alpha.py → runs/diag_s3b_alpha.json).
SCALE15 = dict(
  n_items=15,
  n_sup=24,
  n_alt=4,
  items_per_order=2,
  conc=0.5,
  dir_alpha=5.0,
  occ=0.6530555555555556,
  r_ref_buy=0.10308345926154251,
  r_ref_sup=0.11525964887763726,
  b_buy=16.827152171786267,
  b_sup=15.049508412346505,
)

# 조달 시장 v2 preset (2026-10-08): SCALE15와 같은 규모·가격·수요 생성에 공급자 교체·품목별 대체 공급자 수 이질성·경험 의존 재참여를 더한다.
# 정책 실측값(occ, r_ref, b_*)을 쓰지 않는다. het·onboard·cover·r2_*는 결과를 보기 전에 정한 잠정값이며 실험 전에 확정한다.
PROC15 = dict(
  n_items=15,
  n_sup=24,
  n_alt=4,
  items_per_order=2,
  conc=0.5,
  dir_alpha=5.0,
  market="v2",
  het=0.5,
  onboard=4,
  cover=1.3,
)
