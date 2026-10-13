# 모델 크기 통제와 결정 유용성 (진단 s14)

작성 2026-10-14. PROC15H 고정, c_i = V^Rule(S) − V^Rule(S∖i) (Rule-continuation). 코드 `result/diag_s14_size.py`, `result/diag_s14_onestep.py`.
원자료 `result/runs/diag_s14_size.json`, `diag_s14_pred.npz`, `diag_s14_onestep.json`, `diag_s14_actions.json`, `diag_s14_units/`, 모형 `result/models/s14/`.

## Part I. 모델 크기 통제 (96상태, c_i 1,849개, s13 보조와 같은 분할)

Large Context MLP: SUM-GNN의 (은닉, 층) 6개마다 같은 층 수에서 파라미터 수가 가장 가까운 폭 (93, 79, 194, 160, 396, 323) × wd 3개 = 18설정 × seed 3. 입력은 Context MLP와 같다.

| 모형 | 파라미터 수 (후보) | R² | MAE | RMSE | 상태 내 Spearman | 상위 3 | 상위 20% |
|---|---|---|---|---|---|---|---|
| Context MLP | 2,017 ~ 36,865 | 0.545 | 64.4 | 95.6 | 0.638 | 0.608 | 0.651 |
| Large Context MLP | 11,533 ~ 218,995 | 0.537 | 65.4 | 96.5 | 0.639 | 0.615 | 0.651 |
| SUM-GNN | 11,553 ~ 219,137 | 0.735 | 51.9 | 73.0 | 0.740 | 0.719 | 0.756 |

- SUM-GNN − Large MLP: R² +0.198 [+0.170, +0.228], MAE −13.5 [−16.5, −10.8], Spearman +0.101 [+0.074, +0.128], 상위 3 +0.104 [+0.035, +0.174].
- Large MLP − Context MLP: R² −0.008 [−0.023, +0.004]. 파라미터를 늘려도 MLP는 좋아지지 않는다.
- 묶음 8개 모두에서 SUM-GNN이 낫다 (R² 0.64~0.82 vs Large MLP 0.40~0.59).
- thin market: SUM-GNN − Large MLP R² +0.250 [+0.224, +0.285], Spearman +0.150 [+0.093, +0.198].
- 결론: SUM-GNN의 우위는 파라미터 수 때문이 아니다.

## Part II. 결정 유용성 Step A (한 기간 결정)

- 평가 상태: 학습에 쓰지 않은 평가 seed 반복 16~23을 Myopic-sup로 걸은 기간 6·12·18 (24상태).
- 정책 (모두 같은 MILP, 같은 재참여 PWL, 주문자 잉여 몫 0.3 고정, 공급자 c 출처만 다름):
  - Myopic-sup: 공급자 c 없음.
  - Exact-MC Rule-c: 그 상태에서 MC + CRN으로 잰 Rule-continuation c (M = 40, vf_H = 10). optimal·oracle 가치가 아니다.
  - Context-MLP-c / SUM-GNN-c: s14 final 모형 (반복 0~13 학습, 14~15 검증으로 18설정 중 선택: MLP (32, 2, 0.01), GNN (128, 3, 0)), seed 3개 평균.
  - c는 가드 없이 그대로 쓴다 (실험 7과 같음). γ = 1.
- 이 상태들에서 Exact-MC c 대비 예측: SUM-GNN R² 0.72, 상태 내 Spearman 0.76 / Context MLP 0.56, 0.70.
- 평가: 각 행동을 적용하고 같은 미래 키 40개로 후속 정책을 10기간 이어 간다. 가치 = 이번 기간 이윤 + 이후 10기간 이윤 (할인 없음). 후속 정책은 Rule과 Myopic-sup 두 가지.

행동 (24상태 평균)

| 정책 | 수락 주문 | 배정 공급자 | 잉여 받은 공급자 | 공급자 잉여 | 이번 기간 이윤 |
|---|---|---|---|---|---|
| Myopic-sup | 15.5 | 16.5 | 0.0 | 0 | 1,103 |
| Exact-MC Rule-c | 15.3 | 15.0 | 1.4 | 42 | 1,052 |
| Context-MLP-c | 15.6 | 16.1 | 1.0 | 21 | 1,081 |
| SUM-GNN-c | 15.5 | 15.4 | 0.7 | 18 | 1,081 |

공급자 배정 일치율은 정책 쌍마다 87~94%다. c를 쓰는 정책도 기간당 공급자 1명 안팎에게만 작은 잉여를 준다.

이어 간 가치 차이 (상태 블록 부트스트랩 95% CI)

| 비교 | 후속 Rule | 후속 Myopic-sup |
|---|---|---|
| Exact-MC − Myopic-sup | +2.9 [−16.9, +22.0] | +10.2 [−10.7, +30.2] |
| SUM-GNN − Exact-MC | +3.7 [−8.2, +16.3] | +14.2 [−8.0, +35.0] |
| SUM-GNN − Context-MLP | +5.9 [−9.2, +20.2] | +9.8 [−7.3, +34.3] |
| SUM-GNN − Myopic-sup | +6.6 [−8.6, +23.7] | +24.5 [+4.0, +44.8] |
| Context-MLP − Myopic-sup | +0.7 [−8.7, +9.9] | +14.7 [+2.3, +28.0] |

- 총가치는 후속 Rule에서 약 7,200, 후속 Myopic-sup에서 약 12,800이다. 차이는 모두 0.2% 이하다.
- 충족률(0.51~0.53), 공급자 잔류(0.69~0.71), 활동 공급자(18.5 / 20.4), 0명 품목 비율(0.05~0.06)은 정책 사이에 거의 같다.
- c를 쓰는 정책은 이번 기간 이윤을 22~51 포기하고 이후 이윤을 비슷한 만큼 더 얻는다.

## 판정

- Exact-MC Rule-c는 Myopic-sup보다 낫다고 할 수 없다 (두 후속 정책 모두 CI가 0 포함). 판정 기준의 Case C다.
- 이것은 GNN 문제가 아니다. 정확한 Rule-continuation c도 결정을 의미 있게 바꾸지 못한다. 실험 7(PROC15 이전 legacy 환경)과 같은 방향이다.
- 가능한 이유: (1) target이 Rule 후속을 가정해 c를 쓰는 정책 자체와 맞지 않는다 (policy mismatch). (2) 이 환경에서 c의 크기가 재참여를 올리는 비용에 비해 작아 MILP가 거의 쓰지 않는다 (공급자 잉여 기간당 18~42, 이번 기간 이윤 1,100). (3) Exact-MC c에도 MC 잡음(표준오차 약 40)이 있다.
- 학습된 c가 후속 Myopic-sup에서 Myopic-sup보다 조금 낫지만 (+15~25, 0.2%) 후속 Rule에서는 구분되지 않는다. 결정 차이로 의미를 두기 어렵다.
- Step B(30기간 closed-loop)는 Step A에서 결정 신호가 없어 실행하지 않았다. Exact-MC 정책의 closed-loop은 반복당 수 시간이 들고, 0.2% 크기의 차이는 반복 수십 개로도 가려내기 어렵다.

## 다음 단계

- target 재설계가 먼저다: policy-consistent c (c를 쓰는 정책 자체를 후속 정책으로), approximate policy iteration, post-decision value, action-conditioned continuation value 중 하나.
- 함께 볼 것: 이 환경에서 공급자 유지가 결정에 쓰일 만큼 가치가 있는지 (c의 크기 대비 재참여를 올리는 비용). 정확한 c로도 결정이 거의 바뀌지 않으면 표현 모형보다 이 경제 구조가 먼저 문제다.
- SUM-GNN은 표현력 우위가 확인됐으므로, 재설계한 target에서 다시 비교할 후보로 남긴다.
