# 반복 B2B 조달 플랫폼의 진실 이중경매와 동적 padding 결정

여러 품목을 묶어 주문하는 주문자와 품목별 용량을 가진 공급자를 매 기간 연결하는 반복 조달 플랫폼을 다룬다.
청산은 진실성을 보장하는 Chu(2009) IPB padding 이중경매(MILP)로 하고, 플랫폼은 입찰 전에 padding 배수와 쿠폰 비율을 정한다.
padding으로 빠진 참여자는 다음 기간에 덜 돌아오므로, 이 선택을 학습한 장기 가치로 한다.

**연구 질문**: 진실 청산 규칙 위에서 사전 결정 변수를 장기 이윤 기준으로 정하면, 한 기간 최적 정책보다 누적 이윤이 높은가? 그 차이는 어디서 나오고(padding·쿠폰), 어떤 시장에서 커지며, 거래 연결 구조(GNN)가 필요한가?

> 소규모 예비 실험이다. 설계와 진행 상황은 [연구 보고](https://claude.ai/artifact/57P5nTY8XGntb32KbzBLUg)와 `ref/plan/roadmap.md`에 있다.

## 현재 상태 (2026-11-02)

| 단계 | 상태 |
|---|---|
| 메커니즘·실험 설계 | 완료 (ref/design.md) |
| 새 src 골격, 시장 생성 이식 | 완료 |
| Chu IPB 청산 모듈 | 다음 |
| 출발 점검, 본 실험 | 예정 |

## 폴더

- `src/`: 새 구조 코드 (공유 모듈 utils·env와 새 모듈)
- `legacy/`: 이전 구조(MILP 잉여배분 + 유지 가치 c_i)의 코드·결과·문서. `legacy/README.md` 참고
- `ref/`: design.md(설계), decisions.md(결정 기록), plan/(로드맵, 시장 환경 정의)
- `paper/`: 선행연구 PDF

## 실행

conda 환경 apiems (gurobipy 포함). 테스트는 `src/`에서 `python -m pytest tests`.
