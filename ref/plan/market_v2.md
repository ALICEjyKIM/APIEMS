# 조달 시장 v2 (PROC15) 설계

작성 2026-10-08. legacy 환경(Cfg 기본값, SCALE15)과 실험 1~7 결과는 그대로 둔다. v2는 `market = "v2"`인 설정에서만 동작한다.

## 1. 기존 코드 진단

| 요구 | 상태 | 근거 (legacy 코드) |
|---|---|---|
| finite horizon, AON 주문, split sourcing, 가격·수량 이질성 | 이미 만족 | `env/platform.settle`, `match/milp_build` |
| ID 유지, 남은 참여자 프로필 지속 + 작은 변동 | 이미 만족 | `Env.step`, `utils/arrivals.perturb` |
| CRN (도착·변동·재참여 난수를 정책과 무관하게 키로 고정) | 이미 만족 | `utils/common.rng` |
| 공급자 이탈 뒤 동일 network 복원 안 됨 | 수정 필요 | 빈 자리를 같은 품목·같은 용량의 새 공급자가 확률 0.25로 채움 (`new_sup`) |
| 품목별 대체 공급자 수 이질성 | 수정 필요 | 모든 품목이 정확히 n_alt개 (`_slots`의 need = n_alt) |
| 주문자 반복 구매 프로필 | 이미 만족 (더 강함) | 남은 주문자는 처음 묶음을 그대로 반복, 수량·가격만 변동. 품목 변동은 없음 |
| 재참여가 매칭 경험에도 의존 | 수정 필요 | p = σ(b · 잉여 / 제안 금액), 잉여 0이면 항상 0.5 |
| 정책 독립 용량 보정 | 부분 만족 | 수요는 명목값이지만 용량에 Rule 실측 점유율 occ = 0.653을 나눔 |
| 정책 독립 반응 보정 | 수정 필요 | 기준 잉여율 r_ref를 Rule로 실측 |
| post-decision value | 수정 필요 (설계만) | 타깃이 V^Rule(S_t), Rule trajectory 상태만 |
| Buyer–Item–Supplier 그래프용 원자료 | 부분 만족 | Obs에 q·p·cap·c·ID는 있으나 배정 x·잔류 결과는 step에서 버려짐 |

## 2. 설계

### 2.1 공급자 교체와 onboarding
- 현재: 자리 j가 비면 매 기간 확률 0.25로 같은 품목·용량, 새 가격의 공급자가 채운다.
- v2: 기간 t에 자리 j의 공급자가 떠나면 그 자리에 후보가 생기고 t + L부터 활동한다 (빈 기간 L − 1). 후보의 품목은 자리의 품목 수 deg[j]개를 품목 qualification 가중치 w로 비복원 추출하고, 용량과 가격은 새로 뽑는다. 대기 중 후보는 `Obs.pend`(품목별 공급가능량)와 `Obs.wait`(남은 기간)로만 보인다.
- 이유: 떠난 공급자의 품목 묶음이 다시 같은 자리에 오지 않는다 (sanity check에서 같은 품목 묶음 2~3%, 떠난 품목을 새 공급자가 다시 가진 비율 0.17~0.18). 자리 수 24는 유지해 시장 규모와 그래프 패딩이 고정된다. 자리 품목 수 deg는 유지해 공급자 portfolio 편중(conc) 축이 시간이 지나도 보존된다.
- L은 결정론적 지연이다. 외생 시장 특성으로 해석한다 (fast / medium / slow). 잠정값 L = 4는 legacy의 평균 빈 기간 3과 같다.
- 후보 난수 키 (반복, arr, 2, t, j): 같은 자리가 같은 기간에 비면 정책과 무관하게 같은 후보가 온다.
- 코드: `utils/market2.candidate`, `env/platform.Env2._leave`, `Env2.step`, `Env2.drop`.

### 2.2 품목별 대체 공급자 수 (het)
- 현재: 모든 품목 n_alt개.
- v2: 품목 가중치 w ~ Gamma(1/het², 1) 정규화 (het = 0이면 균등). 목표 수 n_i = f + 최대 나머지 반올림((L − f·I) · w), 총 연결 L = n_items × n_alt는 legacy와 같다. 공통 하한 f는 1부터 올려 Gale–Ryser 조건을 만족하는 가장 작은 값이다. 기간 0 연결은 legacy와 같은 Ryser 탐욕법으로 n_i와 deg를 정확히 맞춘다.
- conc와의 관계: conc는 공급자 쪽(자리별 품목 수) 분포, het는 품목 쪽(품목별 공급자 수) 분포다. 두 축은 따로 움직인다. 단, conc가 크면 전 품목 공급자가 생겨 모든 품목의 하한이 올라가므로 het의 실제 폭이 줄어든다 (n_alt 4, conc 1, het 1에서 범위 2~16, conc 0.5에서 1~22).
- 분리: 가격분포, 품목 수요, 품목 명목 용량은 het와 무관하다. bottleneck 품목은 공급자 수가 적을 뿐 명목 용량은 같다 (연결 용량 = K_i / n_i × Gamma 잡음). 공급자 하나에 용량이 몰려 있어 그 공급자가 떠나면 손실이 크다.
- 수준 후보 (n_alt 4, conc 0.5, 50반복): het 0 → 4~4, 0.25 → 2~7, 0.5 → 1~11 (1~2개 품목 13%), 1.0 → 1~22 (37%).
- 코드: `utils/market2.weights`, `targets`, `feasible`, `slots`.

### 2.3 주문자 반복 구매 프로필
- 현재 구현이 이미 만족한다. 신규 주문자는 품목 묶음·품목별 수량·마크업을 프로필로 받고, 남아 있는 동안 같은 묶음을 반복한다 (sanity check 100%). 주문자가 떠나면 그 묶음 수요가 실제로 사라진다.
- 변경 없음. 묶음 안의 품목 변동(예: 확률 ε로 한 품목 교체)은 필요해지면 `perturb`에 선택 파라미터로 넣는다.

### 2.4 재참여 함수
- 현재: p = σ(b · s / w), s = 배분 잉여, w = 제안 금액.
- v2: p = σ(β0 + β2 · e + β1 · s / w). e는 주문자면 주문 성립 여부, 공급자면 성립 주문에서 배정을 받았는지. β는 세 점으로 정한다: (e = 0, s = 0) → r2_p0 = 0.3, (e = 1, s = 0) → r2_p1 = 0.6, (e = 1, s / w = 0.1) → r2_p2 = 0.85. 주문자·공급자 같은 식.
- 이유: 매칭(e)과 잉여(s / w)가 따로 재참여를 움직인다. 정책 실측값을 쓰지 않는다.
- 공급자 분모: 제안 금액(Σ 공급가격 × 공급가능량)을 유지했다. 실현 거래액을 분모로 쓰면 MILP의 PWL 꺾인 점이 결정변수에 의존해 선형이 깨진다. 거래 기회 효과는 e가 따로 맡는다. 유휴 용량이 많은 공급자는 같은 잉여에도 잉여율이 낮다는 legacy 성질은 남는다.
- MILP: 재참여확률 = PWL_{e=1}(잉여) − gap · (1 − e), gap = r2_p1 − r2_p0. 주문자 e = y[b], 공급자 e = z[j] (z ≤ 배정 합, 공급가능량 합 · z ≥ 배정 합). 잉여는 자기 마진 이내라 e = 0이면 잉여 0이므로 이 식은 정확하다 (PWL 근사 오차 제외). legacy 반응 함수에는 gap이 없어 MILP가 바뀌지 않는다.
- 코드: `env/response.Retention`, `response`, `match/milp_build.exp_sup`.

### 2.5 정책 독립 용량
- 현재: 품목 용량 = cover × 명목 수요 / occ (occ = Rule 실측 점유율).
- v2: 품목 명목 용량 K_i = cover × 명목 수요. 명목 수요 = lam_buy / (1 − ret_ss) × k / n_items × 평균 품목 수량 = 28 (legacy와 같은 정의, ret_ss = 0.5는 명목 가정). cover = 1.3이면 K = 36.4로 legacy의 실효 목표(56 × 0.653 = 36.6)와 같다.
- 실제 활동 용량은 정책과 재참여 파라미터에 따라 달라지고 결과로 보고한다. sanity check에서 명목 대비 Myopic 0.44, Rule 0.62였다.
- 코드: `utils/market2.demand`, `make2`.

### 2.6 설정 구조
- Cfg에 `market = "legacy"`(기본), `het`, `onboard`, `r2_p0`, `r2_p1`, `r2_p2`, `r2_rate`를 더했다. legacy는 이 값을 읽지 않는다.
- preset `PROC15` = SCALE15의 규모·가격·수요 생성 + market v2 (het 0.5, onboard 4, cover 1.3). occ, r_ref, b_*는 쓰지 않는다.
- `Env(cfg, rep)`는 `cfg.market == "v2"`면 `Env2`를 만든다. 호출부(rollout, mc_value, 실험 스크립트)는 바꾸지 않는다. `Policy`와 `Env`의 기본 반응 함수는 `response(cfg)`가 시장 버전에 맞춰 고른다.

## 3. VFA와 post-decision value (설계만)

불일치
- 타깃 V^Rule(S_t)는 후속 정책이 Rule로 고정된다. 실험 7에서 이 정의의 c는 Rule보다는 나았지만 Myopic-sup을 넘지 못했다.
- 상태는 Rule trajectory에서만 모았다. 다른 정책이 방문하는 상태(공급자가 적은 상태 등)는 학습 분포 밖이다.
- v2의 상태에는 onboarding 대기열(pend, wait)이 있다. 기존 시장 요약 지표와 그래프는 이를 쓰지 않는다.
- v2의 재참여는 e에 의존한다. c_i · p_i의 p_i가 매칭 결정에도 의존한다 (MILP에는 반영했다).

post-decision 정식화
- S_t → (매칭 x, 잉여 u, v) → S_t^+ = (참여자, 경험 e, 잉여, 재참여확률 p_i, 대기열) → 재참여 판정·도착·후보 활동 → S_{t+1}.
- `Env2.step`이 정산과 재참여 판정 사이에 이미 S_t^+의 원자료(x, ok, alloc, sb, ss, pb, ps, pend, wait)를 갖고 있다. 이를 `post(o, r)` 하나로 묶어 저장하면 된다.
- 지금의 목적함수 profit + Σ c_i p_i는 V(S_t^+)를 p에 대해 1차 근사한 것이다 (c_i = ∂V / ∂p_i, multilinear extension의 기울기). 이 관점에서 c_i는 leave-one-out 차이가 아니라 V(S^+)의 p_i 방향 기울기로 정의할 수 있다.
- 타깃: S_t^+에서 재참여 난수만 공통 키로 뽑고 후속 정책 π로 H기간 돌린 이윤 합.
- 후속 정책 π의 선택이 실험 7의 핵심 쟁점이다. 후보: (a) policy iteration (π_k의 V로 π_{k+1}을 만들고 그 trajectory에서 다시 수집), (b) 여러 정책(Myopic, Rule, 잡음 섞인 탐색 정책)의 상태를 섞어 수집하고 π는 고정, (c) 1회 rollout improvement를 여러 번 반복.
- 신경망 구조 변경은 보류한다.

## 4. 그래프와 c_ij용 원자료

- Buyer–Item 연결: `Obs.q > 0`. Supplier–Item 연결과 용량: `Obs.cap`. 가격: `Obs.p`, `Obs.c`. ID: `Obs.bid`, `Obs.sid` (공급자 자리 = ID % n_sup).
- onboarding 후보의 품목·용량: `Obs.pend`, 남은 기간 `Obs.wait`.
- 거래·배정 결과: `Env2.step` 반환의 `x` (주문자 × 공급자 × 품목), `ok`, `alloc`, `sb`, `ss`.
- 재참여 결과: `pb`, `ps` (확률), `kept_b`, `kept_s` (실제 잔류), `joined` (이번에 활동을 시작한 자리).
- 손실 없이 보존된다. 다만 rollout(`match/interface`)은 합계만 남기므로, c_ij 분석 때는 기간별 원자료를 저장하는 실행부가 필요하다 (`result/diag_s8_market.py`가 예시).
- c_ij는 계산하지 않았다.

## 5. 실험 전에 정할 값

| 파라미터 | 잠정값 | 정할 것 |
|---|---|---|
| onboard L | 4 | fast / medium / slow 수준 (예: 1, 4, 8) |
| het | 0.5 | 수준 (예: 0, 0.5, 1.0) |
| cover | 1.3 | loose / medium / tight 수준. 명목 기준인지, 활동 용량 결과를 함께 보고할지 |
| r2_p0, r2_p1, r2_p2, r2_rate | 0.3, 0.6, 0.85, 0.1 | 근거 (문헌·현실 가정). 주문자·공급자를 다르게 둘지 |
| 공급자 경험 정의 | 배정 여부 (1개라도) | 배정 비율·최소 물량 기준이 필요한지 |
| 묶음 안 품목 변동 | 없음 | 넣을지, 넣으면 ε |
| 명목 활동 주문자 ret_ss | 0.5 | v2 재참여와 맞출지 (지금은 legacy 정의 유지) |
| 평가 T, 반복 수 | 30, 미정 | 사전 등록 |
| 후속 정책 / 상태 수집 정책 | 미정 | 3장의 (a)~(c) |
