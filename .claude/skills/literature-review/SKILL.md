---
name: literature-review
description:
 Run a literature review using paper search and primary-source synthesis.
 Use when the user asks for a lit review, paper survey, state of the art, related work,
 선행연구, 문헌 조사, 관련 연구, or academic landscape summary on a research topic.
argument-hint: "[topic]"
---

# Literature Review

같은 폴더의 [details.md](details.md)에 있는 workflow를 실행하여 주제에 대한 문헌 검토를 수행합니다.
주제는 사용자가 넘긴 인자($ARGUMENTS)를 사용하고, 인자가 없으면 `ref/literature-review.md`의 조사 범위를,
그 파일도 없으면 CLAUDE.md의 연구 질문을 주제로 삼습니다.

사용되는 에이전트: `researcher`, `verifier`, `reviewer`

출력:
- 중간 산출물: `ref/_review/<slug>_*.md`
- 최종 리뷰: `ref/<slug>.md`
- 설계 점검: `ref/plan/<slug>_spec_audit.md`