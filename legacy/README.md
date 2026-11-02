# legacy: 이전 구조 (MILP 잉여배분 + 유지 가치 c_i)

2026-11-02에 새 구조(Chu IPB 청산 + 사전 결정 변수의 동적 선택)로 바꾸면서 이전 연구를 이 폴더로 옮기고 동결했다. 앞으로 고치지 않는다.
이전 구조에서는 매 기간 MILP(당기 이윤 + Σ c_i × 구간선형 재참여확률)가 주문 수락, 배정, 잉여배분을 한 번에 정했고, c_i = V(s) − V(s에서 i 제외)를 선형·MLP·GNN으로 근사했다.
연구 보고 6.2절이 인용하는 진단 s11~s15와 실험 1~7의 원자료·코드가 여기 있다.

## 구성

- `src/`: 이전 구조 코드 전체. 공유 모듈(utils, env) 사본을 포함해 새 `src/`와 독립이다.
- `src/result/runs/`: 원자료 json과 실행 로그(*.log, *.err). 로그는 지우지 않는다.
- `src/result/models/`, `src/result/figures/`: 학습된 모델과 그림
- `ref/`: 이전 방향의 제안서·리뷰·문헌 조사, `ref/plan/`: 이전 로드맵과 진단 s11~s15 문서
- `docs/tex/`: 이전 진행 문서, `outputs/`: 2026-10-21 데이터 내보내기
- `CLAUDE_legacy.md`, `README_legacy.md`: 이전 프로젝트 지침과 README

## 실행

- 테스트: `legacy/src/`에서 `python -m pytest tests` (2026-11-02 동결 시점 210개 통과)
- 스크립트: `legacy/src/`에서 `python -m result.<스크립트>`
