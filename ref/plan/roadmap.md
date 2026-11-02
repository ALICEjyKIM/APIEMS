# 로드맵

작성 2026-11-02. 설계는 `ref/design.md`, 결정은 `ref/decisions.md`. 슬라이스마다 pytest 통과 + MILP 최적해 확인 후 커밋한다.

| 슬라이스 | 내용 | 완료 기준 | 시기 |
|---|---|---|---|
| N0 정리 | legacy 동결, 새 src 골격, 시장 생성 이식 | 시장 생성이 이전 PROC15H와 같은 값 (tests/test_market.py) | 11/2 완료 |
| N1 Chu IPB 청산 | `mech/chu.py`: padding 낙찰, 주문자 가격, 공급자 임계 가격·대금, 배정, Q = 0 청산과 padding 제외자 | Chu 5.3 예제(주문자 1이 A 2개를 6에, 공급자 1이 임계 가격 3에서 5를 받음. VCG 보상 6에서 1을 뗌)와 보고서 6.1 예제(플랫폼 몫 6) 재현, 전수 열거와 일치, G ≥ 0, 해 시간 측정 | 11/2 완료 (tests/test_chu.py, result/runs/check_time_chu.json) |
| N2 시뮬레이터 | `env/sim.py`: PreObs/Bids, 기간 엔진, 재참여(기준점·제외 항), 이력·기준점 갱신, 쿠폰 장부, 빈자리 onboard | 공통 난수, 입찰가 불변 테스트, 재참여 확률표 재현 | 11/2 완료 (tests/test_sim.py) |
| N3 출발 점검 | padding 배수별 빠지는 거래·플랫폼 몫, 잉여율 분포(기준점 측정), 다품목 반례 탐색 | result/runs/check_start_*.json, decisions.md 갱신 | 11/3~11/4 |
| KSCM 슬라이드 | 질문·가설·설계, 출발 점검 결과 | 11/5 마감 | 11/5 |
| N4 비교 정책 | 고정, 최선 고정값, 한 기간 최적, 예상 청산 | 공통 난수 rollout | 11/6~11/7 |
| N5 Q̂ | 선형, MLP, MLP + 구조 요약, GNN, 같은 예산 | 검증 오차 기록 | 11/8~11/15 |
| N6 본 실험 | 메인, 질문 1~3 | result/runs/exp_*.json | 11/15~11/30 |
| N7 근사 점검 | 상태 432 정확해 | 오차, 최적성 격차 | 11월 |
| N8 가정·조작 점검 | 재참여 함수 교차, 여러 기간 조작 | 시간이 되면 | 12월 |

KSCM 발표는 11/13, APIEMS 원고는 12/1부터.
