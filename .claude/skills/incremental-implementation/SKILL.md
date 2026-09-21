---
name: incremental-implementation
description:
 Implement a feature or module in small vertical slices, each verified by pytest and an optimal MILP solve before committing.
 Use when the user asks to implement, build, or extend code under src/ (env, match, vfa, bench, result),
 or to continue the next slice. 구현, 슬라이스 진행, 다음 단계 구현 요청 시 사용.
argument-hint: "[slice number or feature]"
---

# Incremental Implementation

CLAUDE.md의 파일 구조·코딩 컨벤션·주의사항과 `ref/plan/roadmap.md`의 슬라이스 정의를 먼저 읽고,
요청된 구현($ARGUMENTS)을 진행합니다.

## 원칙

- **수직 슬라이싱**: 환경 → 정책 → 평가를 관통하는 최소 경로를 먼저 만들고, 기능은 그 위에 하나씩 추가합니다.
  각 슬라이스가 끝나면 시스템 전체가 항상 실행·테스트 가능한 상태여야 합니다.
- **Rule 0 단순함 최우선**: 가장 적은 줄로 작성합니다. 요청되지 않은 예외처리, 안전장치, 일반화를 넣지 않습니다.
- **Rule 1 한 번에 하나씩**: 한 commit에는 하나의 논리적 변경만 담습니다.
- **Rule 2 항상 실행 가능**: 매 commit 후 pytest 전체 통과, MILP를 쓰는 경로는 Optimal 유지.

## Workflow

1. **범위 파악** — 요청이 슬라이스 번호면 `ref/plan/roadmap.md`의 해당 정의를 따릅니다.
   번호가 아닌 기능 요청이면 관련 모듈과 기존 테스트를 읽고 하위 단계로 나눕니다.
   한 단계는 한 번의 commit으로 설명 가능한 크기로 잡습니다.
2. **테스트 먼저** — 기대 동작을 확인하는 테스트를 `src/tests/test_*.py`에 추가합니다.
   작은 인스턴스에서 손으로 계산 가능한 정답(주문 성립 조건, 잉여 합계 보존, 계수 0 = 근시안 정책 등)을 우선 검사합니다.
3. **구현** — 테스트를 통과하는 최소한의 코드를 CLAUDE.md의 코딩 컨벤션대로 작성합니다.
4. **검증** — `src/`에서 `pytest tests`로 전체 테스트를 실행합니다.
   MILP를 건드린 단계는 소규모 인스턴스에서 솔버 상태가 Optimal인지 확인합니다.
   실패하면 원인을 고친 뒤 다시 검증하고, 테스트를 약화시켜 통과시키지 않습니다.
5. **commit** — 검증을 통과한 단계만 commit합니다. 메시지는 `feat(<모듈>): <내용>` 형식입니다.
6. **정직성 점검** — 요청된 슬라이스를 마치면 다음을 스스로 점검합니다.
   비교군(근시안, 규칙 기반, 선형 근사)을 제안 방법보다 약하게 설정하지 않았는가,
   근사 방법 간 입력 정보·튜닝 예산이 같은가, 공통 난수를 쓰는가.
7. **멈추고 보고** — 요청된 슬라이스가 끝나면 다음 슬라이스로 넘어가지 않고 멈춥니다.
   완료한 단계, 테스트 결과, 정직성 점검 결과, 다음 슬라이스에서 할 일을 요약합니다.

## 주의

- 모든 정책은 같은 MILP를 공유하고 계수만 다르게 합니다.
- 실험 결과는 `result/runs/*.json`에만 저장하고, 그림·요약은 그 json에서만 생성합니다.