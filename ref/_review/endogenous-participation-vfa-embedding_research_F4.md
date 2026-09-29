# 주제 F4 증거 수집: 확인된 선례들의 상호의존이 대체형뿐인가 보완형을 포함하는가

작성일: 2026-09-28
조사자 지침: 우리에게 유리한 판정을 만들지 않는다. **보완형 선례를 찾는 것이 목적**이었다.
사용 도구: arXiv PDF 전문 다운로드 + pypdf 텍스트 추출 + 용어 전수 검색, 저자(Rotman) 호스팅 워킹페이퍼 PDF 전문 추출,
OpenAlex `abstract_inverted_index` 복원, Crossref 초록, Unpaywall OA 탐색, Semantic Scholar 피인용 + 인용 문맥(contexts), WebSearch.

**질문 진행 상태**: F4a done(워킹페이퍼 전문) / F4b **부분 — (i) 판정불가, (ii) done(초록·인용문맥)** / F4c done(초록 + 피인용 37건 전수) /
F4d done(초록 전문) / F4e done(전문)

읽은 수준 표기: `전문`=논문 전체 텍스트를 직접 추출해 읽음, `워킹페이퍼전문`=출판본이 아닌 저자 호스팅 원고 전문,
`초록전문`=초록 전문을 직접 복원해 읽음, `인용문맥`=타 논문이 인용한 문장만 직접 확인, `판정불가`=근거 없음.

---

## 1. 판정 표 (맨 앞)

| # | 논문 | 상호의존 종류 | 증분 가치 부호 | `c_i`류 양을 최적화 계수로? | 근거 문장 / 근거 없음 | 읽은 수준 |
|---|------|---------------|----------------|------------------------------|------------------------|-----------|
| 1 | Afèche, Araghi, Baron (2017) M&SOM 19(4):674–691 [1] | **대체형 지배 + 보완 방향 외부성 1개(§5.2 부정적 WOM). 묶음·보완형 상호의존은 없음** | **명시적 부호 진술 없음.** 새 고객 가치 `V_0^N`은 서비스되는 유형들의 Vμ 지수의 **볼록결합**(식 28) → 최대 지수로 상계, 초가법적 확대는 구조적으로 불가 | **그렇다(지수형).** "This metric and the Vμ indices **drive the optimal policy** under fixed advertising in §§4.2.1-4.2.2" [1]. 단, MILP 계수가 아니라 유체모형에서 해석적으로 도출한 우선순위 지수 | "the value of serving a new customer, `V_0^N`, depends on the service policy, that is, not only on her own Vμ index, but also on the Vμ indices of base customers that are served ... These key implications derive from two distinctive features of our model: the presence of **customer base transitions** and their dependence on the service quality" (Remark 1) [1] / 39쪽 원고에서 bundle·complement·supermodular·all-or-nothing **출현 0회** | 워킹페이퍼전문 (2016-12 Rotman 원고) |
| 2 | Klein & Kolb (2015) Omega 55:111–125 [2] | **대체형(용량 경쟁)만 확인.** 보완 언급 없음 | **미언급**(초록 수준) | **(i) 판정불가.** 초록은 "Finally, we investigate the impact of limited capacity on the CLV by **introducing an opportunity cost-based approach** that understands customer profitability as a customer's contribution to customer equity"로 끝난다 [2]. 최적화 계수인지 사후 지표인지 초록만으로 구분 불가 | "when demand exceeds capacity, because the **isolated determination and optimization of a single customer's lifetime value is no longer feasible**" [2]; 단일 제품군임을 타 논문이 확인: "prior research on CLV is primarily focused on analysing transaction patterns associated with **only one product category** (e.g., Klein and Kolb, 2015)" [5] | 초록전문 + 인용문맥 |
| 3 | Ovchinnikov, Boulu-Reshef, Pfeifer (2014) MS 60(8):2002–2019 [3] | **대체형(용량 경쟁)만** | **증분 < 독립.** "when capacity is unlimited, VIC equals CLV, but when capacity is limited, **VIC is much smaller**" [3] | 아니다(초록 수준). VIC는 DP에서 도출되는 한계 가치이고, 결정변수는 획득·유지 **지출액**이다 [3] | 위 인용 + "changes dynamically depending on the number of customers and their mix" [3]. 피인용 37건 전수 확인 결과 **VIC > CLV / 묶음 / 보완형을 다룬 후속 없음** [6] | 초록전문 + 피인용 전수 |
| 4 | Pfeifer & Ovchinnikov (2011) JIM 25(3):178–189 [4] | **혼재를 명시적으로 언급.** 자기 논문의 기여는 대체형(공급측 용량)이지만, **보완 방향의 수요측 의존(추천/referral)을 "잘 알려진 것"으로 명시적으로 인정하고 배제**한다 | **양방향 모두 시사.** 독립성 정의가 부호 중립적이고("the acquisition (retention) of Jane Doe has no effect on the cash flows of any other current or future customers"), 용량 제약 하에서는 "the firm can prefer a **lower-CLV** customer" [4] | 아니다(초록 수준). WTS는 지출 상한 해석량이다 | **결정적 문장**: "In contrast to **well-understood demand-side dependencies among customer relationships (such as referrals)**, this paper highlights a particular kind of **supply-side dependency**—that created when the firm is limited in the number of customers it can serve" [4] | 초록전문 |
| 5 | Kishimoto 외 (2026) ICLR / arXiv:2602.15752 [7] | **둘 다 아님.** 사용자별 유지곡선의 **자기 수확체감(오목)** 만 있고 참여자 간 상호의존이 목적함수에 없다 | **미언급.** 오목성은 같은 사용자의 매칭 수에 대한 체감이며 참여자 집합 부호 문제와 무관 | 아니다. 계수는 개인별 유지곡선의 1기간 이득이고 정렬 점수로 쓴다(식 12) | Assumption 1: "For every user x∈X, the reward function f(x,·) is **concave** in the number of matches m≥0" [7]; 목적 식 (9)는 receiver 이득 + 추천된 사용자들 이득의 **단순 합**. 11쪽 전문에서 bundle 0회, complement 0회, substitut 0회, capacit **0회**, supermodul 0회, submodul 0회, all-or 0회, quota 0회 [7] | 전문 |
| 6 | **Subramanian, Raju, Zhang (2014) MS 60(2):494–507** [8] — 확인된 5편 밖에서 발견한 **반례** | **보완 방향(경쟁 외부성)** | **증분 > 독립.** "even though a low-cost customer is more profitable when **viewed in isolation**, a high-cost customer may be **strategically more valuable** by discouraging poaching" [8] | 아니다(해석적 2사 게임모형) | "traditional customer lifetime value metrics may lead to poor retention decisions because they do not account for the **competitive externality that actions toward some customers impose on the cash flows from other customers**"; "evolve from a segmentation mindset, which views each customer **in isolation**, to a **customer portfolio mindset**, which recognizes that the value of different customers is **interlinked**" [8] | 초록전문(Crossref) |
| 7 | Schmitt, Skiera, Van den Bulte (2011) JM 75(1):46–59 [9] | 보완 방향(추천 = 수요측 양의 의존)의 실증 | 해당 없음(설정 자체가 집합함수 차분이 아님) | 아니다 | "The average value of a referred customer is at least **16% higher** than that of a nonreferred customer with similar demographics and time of acquisition" [9] | 초록전문(Wharton 호스팅 PDF 직접 추출) |

---

## 2. F4a~F4e 직답

### F4a. Afèche, Araghi, Baron (2017)의 "policy-dependent value": 대체형뿐인가, 보완형을 포함하는가

**답: 참여자 간(cross-customer) 상호의존은 대체형(병목 용량 경쟁)이다. 보완 방향의 외부성이 §5.2 확장에 하나 있으나,
우리가 말하는 묶음 보완형(한 참여자가 다른 참여자의 거래 성립을 가능하게 함)은 전혀 없다.**
근거는 저자(Rotman) 호스팅 2016-12 워킹페이퍼 전문이다. **출판본이 아니므로 초록이 출판 초록과 일치함만 확인했고,
출판본에서 절 번호·식 번호가 달라질 수 있다.**

세 갈래로 분해해 판정했다.

**(1) 병목 용량 = 대체형.** 논문의 핵심 처방은 용량 배급(rationing)과 서비스 거부다:
"we find that under mild conditions it is optimal to **ration capacity**, whereby the firm serves only new and lucrative
base customers, but **denies service to unprofitable base customers**. We derive optimality conditions for capacity
rationing" [1]. 한 유형에 용량을 주면 다른 유형이 받지 못하는 순수 경쟁 구조다.

**(2) "policy-dependent value"의 실제 메커니즘은 보완이 아니라 같은 고객의 유형 전이(customer base transition)다.**
이것이 F4a의 핵심이다. 식 (27)의 분자를 논문은 이렇게 설명한다:
"The numerator sums the immediate profit of serving a new customer plus **her expected future profit as a base customer**;
with probability ... **she switches to type j** and her CLV is ..., depending on whether type j is served or not" [1].
그리고 Remark 1:

> "Importantly, the value of serving a new customer, `V_0^N`, depends on the service policy, that is, not only on her own
> Vμ index, but also on the Vμ indices of base customers that are served. As a result, new customers' optimal priority
> ranking does not correspond to the ranking of their Vμ index, in contrast to standard index policies in the literature.
> These key implications derive from two distinctive features of our model: **the presence of customer base transitions
> and their dependence on the service quality**." [1]

즉 "다른 유형의 지수에 의존한다"는 말은 **그 고객이 장래에 그 유형이 되기 때문**이며, 서로 다른 두 참여자가 함께 있어야
가치가 생기는 보완이 아니다. 더 결정적으로 식 (28)은
"`V_0^N` equals the **load-weighted convex combination** of the Vμ indices of types that are served" [1]
로, `V_0^N`은 개별 지수들의 **볼록결합**이다. 볼록결합은 최대 지수를 넘을 수 없으므로 **초가법(초모듈) 방향의 확대가 구조적으로 불가능**하다.
(이 마지막 문장은 식 (28)의 형태에서 우리가 도출한 **추론**이며, 논문이 그렇게 진술한 것은 아니다.)

**(3) §5.2 부정적 WOM = 보완 방향의 외부성이 실제로 존재한다.** 이 부분은 우리에게 불리하므로 그대로 적는다.
실효 신규 도착률을 `λ_0^e := λ_0 − η·N_1(1−ρ_1)`로 두고 "The parameter η ≥ 0 captures the **WOM intensity**" [1].
즉 기존 고객에게 서비스하면 신규 고객 획득이 늘어나는 **양의 외부성**이다. 다만 논문의 결론은 이 외부성이 대체형 논리를
**약화시키되 뒤집지 않는다**는 것이다: "WOM **reduces, but does not eliminate**, the capacity cost range in which
rationing capacity is optimal" (Proposition 4 도입부) [1].

**용어 전수 검색(39쪽 원고 텍스트)**: `complement` 0회, `bundle` 0회, `supermodular` 0회, `submodular` 0회,
`all-or` 0회, `externalit` 0회. `substitut` 13회는 전부 대수적 "substituting (식 대입)"이고,
`cross-sell` 3회는 문헌리뷰·참고문헌, `word of mouth` 4회는 §5.2다.

**`c_i`류 양의 용도**: 최적화의 **결정 기준**으로 쓴다. "This metric and the Vμ indices **drive the optimal policy**
under fixed advertising in §§4.2.1-4.2.2" [1], Proposition 1은 Vμ 지수 순 우선순위를 처방한다. 단, 이는 유체모형에서
해석적으로 도출한 **지수 정책**이고, 학습된 V의 노드 삭제 차분을 솔버 계수로 넣는 것이 아니다.

### F4b. Klein & Kolb (2015): (i) 최적화 계수인가 사후 지표인가, (ii) 대체형뿐인가

**(i) 판정불가. 확인하지 못했다.** 이것을 추측으로 메우지 않는다.
Unpaywall 조회 결과 `is_oa: false`, OA 사본 0건이다(§4 참조). OpenAlex에 초록 색인이 없고(`abstract_inverted_index: None`),
**Crossref에도 Elsevier가 초록을 등록하지 않았다**(`abstract` 필드 부재, §7 재검증 로그 참조).
Augsburg OPUS 리포지터리(`.../docId/40463`)는 이 환경에서 네트워크 접속이 차단되었으며(ECONNREFUSED / http=000),
ScienceDirect·SSRN은 403이다. 초록 전문은 **RePEc 두 미러(EconPapers·IDEAS)에서 직접 추출해 확보했다**(§7).
따라서 확보한 최상위 근거는 출판사 초록뿐이고, 그 마지막 문장은
"**Finally**, we investigate the impact of limited capacity on the customer lifetime value by **introducing an
opportunity cost-based approach** that understands customer profitability as a customer's contribution to customer equity" [2]이다.

여기서 두 해석이 모두 가능하고 초록으로는 구분되지 않는다.

- (A) 사후 지표 해석: "Finally"가 기여 목록의 마지막이고 "investigate the impact ... on the CLV"는 분석·평가 서술이다.
  MDP를 가치반복으로 풀면 수락/거절은 Bellman 식에서 바로 나오므로 기여분을 계수로 꽂을 필요가 없다.
- (B) 최적화 계수(bid-price) 해석: 수익관리에서 기회비용 기반 수락 규칙("수익 ≥ 기회비용이면 수락")은 **그 자체가 통제 규칙**이다.
  "opportunity cost-based approach"라는 표현은 이쪽을 강하게 시사한다.

**(B)가 맞으면 우리 `c_i`의 신규성은 E6 판정보다 더 약해진다.** 도서관 확인 질문은 §4에 정확히 적었다.
(위 (A)·(B) 비교는 초록 문장에 대한 우리 **해석**이며, 어느 쪽도 원문 확인이 아니다.)

**(ii) 상호의존은 대체형만 확인했다.** 초록의 상호의존 서술은 전부 용량 경쟁이다:
"The allocation process becomes complex, when **demand exceeds capacity**, because the isolated determination and
optimization of a single customer's lifetime value is **no longer feasible**" [2].
보완·교차판매·묶음 언급은 초록에 없다. **EconPapers가 싣은 출판사 키워드 목록도 상호의존 쪽 단어가 전혀 없다**:
"Markov decision process ; Customer relationship management ; Revenue management ; Customer equity ;
Capacity allocation ; Repurchase behavior" — bundling·cross-selling·complementarity·multi-item 없음 [2](§7 참조).
초록의 나머지 상호의존 서술도 시간축 하나뿐이다: "Furthermore, we analyze when and how **intertemporal customer
behavior** influences capacity allocation" [2]. 보강 근거로, 이 논문을 인용한 IMDS 2017 논문이
"prior research on CLV is primarily focused on analysing transaction patterns associated with **only one product
category** (e.g., Klein and Kolb, 2015)"라고 적었다 [5]. 단일 제품군이면 묶음 보완이 발생할 여지가 없다.
(이는 2차 출처의 특성화이며, 원문 확인은 아니다.)

### F4c. Ovchinnikov, Boulu-Reshef, Pfeifer (2014): VIC > CLV가 되는 경우를 본 논문이나 후속이 다루는가

**답: 본 논문은 다루지 않는다. 피인용 37건을 전수 확인한 결과 후속도 다루지 않는다.**

본 논문의 부호는 단방향이다:

> "we introduce a concept of the value of an incremental customer (VIC), and show that when capacity is unlimited,
> **VIC equals customer lifetime value (CLV)**, but when capacity is limited, **VIC is much smaller** and changes
> dynamically depending on the number of customers and their mix." [3]

즉 그들의 모형에서 VIC ≤ CLV이고, 등호는 용량 무제한일 때만이다. VIC > CLV는 나오지 않는다.

**후속 문헌 확인 절차와 결과**: Semantic Scholar 인용 API로 피인용 37건의 제목·연도·게재지·초록을 받아
`cross-sell`, `bundl`, `complement`, `referral`, `network value`, `word of mouth`, `synerg`, `viral`, `spillover`,
`supermodul` 열 개 키워드로 초록 전수 검색했다 [6]. 키워드가 걸린 것은 두 건뿐이고 둘 다 VIC > CLV를 다루지 않는다.

- Social Promotion (POM 2020, doi 10.1111/poms.13247): "reallocate the promotional rewards based on consumers'
  **social network value** rather than their personal value to the firm"이라는 실증·이론 논문으로, 보완 방향의
  가치 개념(사회적 네트워크 가치 > 개인 가치)을 담고 있으나 VIC/CLV 부호 비교 틀이 아니다.
- 에이전트 기반 스타트업 유지 모형(ESWA 2021)에 WOM 언급.

나머지 35건은 전부 획득-유지 지출, 용량·수익관리 결합, 행동실험, 리뷰 계열이었다 [6].
**결론: VIC > CLV(보완형)를 다룬 후속은 찾지 못했다.**
**이 결론은 두 번째 인용 데이터베이스로 교차검증했다**: OpenCitations에서 피인용 DOI 34건을 받아 Semantic Scholar 집합과
대조하니 SS가 놓친 6건이 있었고, 그 6건을 Crossref 초록으로 전수 선별한 결과 보완 관련 키워드는 **0건**이었다(§7).
(부재 증명은 아니다. 두 창구 모두 **초록** 검색이고 본문 검색이 아니다. OpenAlex `filter=cites:` 순회는
일일 예산 소진으로 못 했다 — 정확한 오류 본문과 시도한 쿼리는 §7에 적었다.)

### F4d. Pfeifer & Ovchinnikov (2011): "independent"이 깨지는 방식으로 보완을 언급하는가

**답: 언급한다. 그리고 이것이 F4 전체에서 가장 우리에게 불리한 발견이다.**
초록 전문(OpenAlex `abstract_inverted_index` 복원)에 다음 문장이 있다:

> "**In contrast to well-understood demand-side dependencies among customer relationships (such as referrals)**,
> this paper highlights a particular kind of **supply-side dependency**—that created when the firm is limited in the
> number of customers it can serve." [4]

세 가지가 따라 나온다.

1. 이 논문은 **보완형 의존(추천)을 알고 있고, 그것을 "잘 알려진(well-understood)" 것으로 분류하며, 의도적으로 논외로 둔다.**
   즉 "경쟁·잠식만 다룬다"는 우리 쪽 서술은 틀리다. 보완을 **명시적으로 언급**한다.
2. 이 논문의 독립성 정의는 **부호 중립적**이다: "By independent we mean that the acquisition (retention) of Jane Doe has
   **no effect on the cash flows of any other current or future customers**" [4]. 즉 CLV ≠ WTS의 원인으로
   양의 의존(추천)과 음의 의존(용량) 둘 다를 포괄하는 틀을 이미 제공한다.
3. 논문 자신의 모형(Blattberg–Deighton 확장)은 대체형이며 부호는 아래로 간다:
   "for a firm at capacity (in this model), CLV is no longer relevant to marketing spending decisions and the firm can
   **prefer a lower-CLV customer**" [4].

**추천 방향에서 WTS > CLV라고 이 논문이 명시적으로 쓰는지는 확인하지 못했다**(본문 미확보, §4 참조).
다만 추천이 양의 의존이라는 것과 그 가치 프리미엄은 독립 문헌에서 직접 확인된다: 독일 대형 은행 약 1만 명 3년 추적에서
"The average value of a referred customer is at least **16% higher** than that of a nonreferred customer" [9],
그리고 CRV(customer referral value)를 CLV와 별도 축으로 두고 고객을 4분면으로 나누는 처방이 JM 2010에 있다 [10]
([10]은 서지·검색요약 수준이며 본문은 확인하지 않았다).

### F4e. Kishimoto 외 (2026) ICLR: 묶음·보완형 상호의존이 전혀 없음을 확인

**답: 확인했다. 전혀 없다. 11쪽 전문을 읽고 용어를 전수 검색했다.**

**용어 출현 횟수(전문 텍스트, 대소문자 무시)**: `bundl` 0, `complement` 0, `substitut` 0, `all-or` 0,
`supermodul` 0, `submodul` 0, `capacit` **0**, `package` 0, `multi-item` 0, `quota` 0, `congestion` 0 [7].
용량 제약이라는 단어조차 없다.

**유일한 비선형성은 사용자 자신의 수확체감이다.**

> Assumption 1 (Concavity of the Reward Function). "For every user x ∈ X, the reward function f(x,·) is **concave in the
> number of matches** m ≥ 0." [7]
>
> 해설: "Intuitively, this assumption implies that as users receive more matches, their retention gain is **decelerated**.
> ... a user who gets their first match gains more incremental retention probability than when they get the 10th match." [7]

**목적함수에 참여자 간 결합항이 없다.** 식 (9)는 receiver x의 이득과 추천된 사용자들의 이득의 **단순 합**이다
(직접 읽은 식 (9)의 두 중괄호 항: "Gain for receiver x" + "Gain for recommended users") [7].
그리고 이 오목성 때문에 Jensen 하계(Lemma 1)와 선형 하계(Lemma 2)를 써서 **NP-hard 문제를 위치별 점수 합의 정렬 문제(식 12)로
환원**한다 [7]. 즉 이 논문의 해법 자체가 **가법 분해 가능성에 의존**하므로, 보완형(비분해) 상호의존을 담을 수 없는 구조다.
매칭 기회가 한정되어 있다는 경쟁 요소는 정식화에 용량 제약이 아니라 상위 K개 슬롯(위치 가중치 α_k)으로만 들어간다 [7].

**요약**: 오목 = 수확체감 = 열모듈(대체) 방향이며, 보완형이 아니다. 지시받은 판정과 일치한다.

---

## 3. G1 현상 문장에 대한 최종 판정

검토 대상 문장:

> "다품목 묶음 주문이 만드는 보완형 상호의존 때문에, 참여자의 증분 가치가 그의 독립 가치를 넘어선다.
> 용량 경쟁이 만드는 대체형 상호의존에서는 증분 가치가 독립 가치보다 작아지므로(VIC < CLV),
> **기존 선례의 부호와 반대다.** 이 부호 반전이 잉여배분 결정을 질적으로 다르게 만든다."

**판정: 절반 성립, 절반 좁혀야 한다. 세 번째 문장("기존 선례의 부호와 반대다")은 현재 형태로 쓸 수 없다.**

### 성립하는 부분

**확인된 다섯 선례의 상호의존은 모두 묶음 보완형이 아니고, 증분 > 독립의 부호 반전을 다루지 않는다.** 근거:

- Ovchinnikov 외 2014 [3]: 부호가 명시적으로 단방향(VIC ≤ CLV, 등호는 용량 무제한). 피인용 37건 전수 확인에서도 반대 부호 없음 [6].
- Klein & Kolb 2015 [2]: 초록의 상호의존이 전부 용량 경쟁. 단일 제품군(2차 근거 [5]).
- Afèche 외 2017 [1]: 39쪽 원고에서 complement/bundle/supermodular 0회. 새 고객 가치가 지수들의 **볼록결합**(식 28)이라
  초가법 확대가 구조적으로 불가능. cross-customer 결합은 용량 배급뿐.
- Pfeifer & Ovchinnikov 2011 [4]: 자기 모형은 공급측(용량) 의존이고 "prefer a lower-CLV customer".
- Kishimoto 외 2026 [7]: 참여자 간 결합항 자체가 없고 오목(수확체감)뿐. capacit 포함 관련 용어 전부 0회.

따라서 "**다품목 all-or-nothing 묶음이 만드는 참여자 간 보완형 상호의존을 다룬 선례는 이 다섯 편 중 없다**"는
안전하게 쓸 수 있다.

### 좁혀야 하는 부분 — 세 가지

**(1) "기존 선례의 부호와 반대다"는 과잉 일반화다. 증분 > 독립은 이미 Management Science에 있다.**
Subramanian, Raju, Zhang (2014) MS 60(2) [8]:
"even though a low-cost customer is more profitable when **viewed in isolation**, a high-cost customer may be
**strategically more valuable** by discouraging poaching"; "traditional customer lifetime value metrics may lead to poor
retention decisions because they do not account for the competitive externality"; "evolve from a segmentation mindset,
which views each customer in isolation, to a **customer portfolio mindset**, which recognizes that the value of
different customers is **interlinked**" [8].
→ 부호(증분 > 독립)와 정책 함의(독립 가치가 낮은 참여자를 오히려 붙잡아라)가 **우리 주장과 같다.**
메커니즘만 다르다(경쟁사 포칭 억제 vs 묶음 성립). 이 논문을 인용하지 않고 "부호 반전"을 신규로 내세우면
마케팅·OM 교차 심사자에게 걸린다.

**(2) Pfeifer & Ovchinnikov 2011 자신이 보완형 의존을 "잘 알려진 것"으로 명시한다** [4].
따라서 "선례는 대체형만 알았다"는 서술은 그 논문의 초록 한 문장으로 즉시 반박된다.
CLV 문헌에는 CRV(추천 가치) 계보가 있고 거기서 참여자 가치는 자기 CLV를 넘는다 [9][10].

**(3) 우리 자신의 모형에서 부호가 아직 미결이다(D 조사 §3 결론과 동일).**
n_alt ≥ 2에서 같은 품목의 두 공급자는 대체(열모듈 방향), 다른 품목의 공급자는 보완(초모듈 방향)이라 **혼재**이고,
순 부호는 측정 대상이다. G1이 "보완형이라서 증분 > 독립"을 **전제**로 깔면, 측정 결과가 반대로 나올 때 주장 전체가 무너진다.

### 권고하는 재서술 (요소 나열로 돌아가지 않으면서 방어 가능한 형태)

세 문장으로 분리하고, 각각의 검증·인용 책임을 명시한다.

1. **선행 정리(인용)**: 참여자 간 상호의존이 있으면 개별 생애가치와 한계(증분) 가치가 갈라진다는 것은 확립된 결과이며,
   그 부호는 의존의 방향에 달려 있다 — 용량 경쟁에서는 증분이 작아지고([3][4][2]), 경쟁 외부성·추천에서는 증분이 커진다([8][9][10]).
2. **우리 기여(신규)**: 그 갈라짐을 **다품목 all-or-nothing 묶음 성립**이라는 **새로운 발생 원인**에서 만들고,
   그것을 **양면(주문자·공급자) 참여자 집합 위의 학습된 V의 노드 삭제 차분**으로 계산해,
   **개별 참여자에게 갈 금전 잉여 몫과 매칭을 하나의 MILP에서 동시에** 정하는 계수로 쓴다.
   이 조합을 한 모형에 담은 선례는 위 어디에도 없다(F4 + E6/E7 교차 확인).
3. **측정으로 답할 것(주장 아님)**: 이 시장에서 `V(S) − V(S∖{i})` 대 `V({i}) − V(∅)`의 순 부호와 그 크기가
   공급 편중도·n_alt·주문당 품목 수에 따라 어떻게 움직이는지. 대체·보완이 혼재하므로 부호는 가정하지 않고 보고한다.

이 형태는 "그 차이가 왜 중요한가"에 답하면서(기존 방법이 놓치는 것 = 묶음발 보완 성분), 부호를 **선점된 주장**이 아니라
**측정 결과**로 돌려 리뷰어 반박 면적을 줄인다.

---

## 4. 미해결 · 도서관에서 확인할 정확한 질문

### (가) Klein & Kolb (2015) Omega 55:111–125 — **최우선. `c_i` 신규성 판정에 직결**

접근 실패 기록: Unpaywall `is_oa=false`, OA 사본 0건 / OpenAlex 초록 색인 없음 / Augsburg OPUS(docId 40463) 네트워크 차단 /
ScienceDirect·SSRN 403.

확인할 것:

1. **"customer's contribution to customer equity"를 정의한 식 번호를 찾아라.** 그 식이 (a) 최적 가치함수 `V`의 차분
   `V(상태) − V(그 고객을 뺀 상태)` 형태인지, (b) 그것이 수락/거절 결정 규칙에 **직접** 들어가는지
   ("accept if revenue ≥ opportunity cost" 형태의 명제·알고리즘이 있는지), (c) 아니면 수치실험 절에서만 계산·해석되는
   사후 지표인지 판정하라. → **(b)이면 우리 `c_i` 서술을 "기회비용 기반 수락 규칙의 학습·양면·묶음 확장"으로 더 내려야 한다.**
2. **모형 절의 상태·행동·제약**: 고객 세그먼트 간 결합이 용량 제약 하나뿐인지, 아니면 세그먼트 간 수요 상호작용
   (교차판매·묶음·다품목)이 있는지. 특히 요청(request)이 **여러 자원/제품을 동시에** 소비하는지.
3. **수치실험 절의 비교표**: "CLV 기반 배분"과 "기여분(기회비용) 기반 배분"의 성과 차이를 보고하는 표가 있는지,
   그리고 기여분이 CLV보다 **작다**고 명시하는지(부호).

### (나) Pfeifer & Ovchinnikov (2011) JIM 25(3):178–189

접근 실패 기록: Unpaywall `is_oa=false` / SSRN(abstract_id=1335471) 403.

확인할 것:

1. **서론에서 referral(추천) 의존을 논하는 단락**을 찾아, 그 경우 **WTS > CLV**라고 부호를 명시하는지 확인하라.
   명시하면 우리 "부호 반전" 주장의 선행이 되고, 명시하지 않으면 "방향만 언급, 부호는 미진술"로 정확히 인용할 수 있다.
2. **Blattberg–Deighton 확장 모형의 식**에서 용량 제약이 어떻게 들어가고, WTS와 CLV의 차이가 어느 항으로 나타나는지
   (그 항이 우리 `c_i − CLV_i`에 대응하는지).

### (다) Ovchinnikov, Boulu-Reshef, Pfeifer (2014) MS 60(8):2002–2019

접근 실패 기록: Unpaywall `is_oa=false` / HAL(hal-01290217)에 초록만, PDF 없음.

확인할 것:

1. **VIC의 정의식과 그 부호 명제(Proposition)를 특정하라.** VIC ≤ CLV가 정리로 증명되는지, 수치예시로만 보이는지.
2. **DP의 상태가 "고객 수와 구성(mix)"이라면, 구성이 가치함수에 들어가는 방식이 가법적인지** 확인하라.
   가법적이면 그 모형에서는 보완형이 원리적으로 나올 수 없다(우리 갭 논거 보강).

### (라) Afèche, Araghi, Baron (2017) M&SOM — 출판본 대조

현재 근거는 2016-12 Rotman 워킹페이퍼 전문이다. 출판본에서 확인할 것:

1. Remark 1과 식 (28)(볼록결합 진술)이 **출판본에 그대로 있는지**, 절·식 번호를 출판본 기준으로 다시 적어라.
2. **온라인 부록**에 §5.1(유형 전이)·§5.2(WOM)의 일반화가 있고, 거기서 여러 유형을 **동시에** 서비스해야 가치가 생기는
   형태의 항이 나타나는지(= 보완형). 현재까지 근거로는 없다.

### (마) 확인하지 못한 주변 항목

- Kumar, Aksoy, Donkers 외 (2010) JSR "Undervalued or Overvalued Customers: Capturing Total Customer Engagement Value"
  (doi 10.1177/1094670510375602): Unpaywall이 Groningen 리포지터리 OA PDF를 보고하지만 이 환경에서 403이었다.
  → CLV만 세면 고객이 "undervalued"된다는 진술의 원문 문장을 확인하면 §3의 (2)를 더 강하게 쓸 수 있다.
- Kishimoto 외 (2026)의 ICLR 2026 게재는 arXiv 주석의 **저자 신고 정보**이며 OpenReview로 직접 확인하지 않았다(E 문서와 동일 한계).
- Subramanian, Raju, Zhang (2014) [8]은 **초록만** 읽었다. 본문의 어느 명제가 "증분 > 독립"을 증명하는지,
  그리고 그 모형이 집합함수 차분 형태인지(아마 아닐 것)는 확인하지 않았다. **내용 추정 금지.**
  이 한 편의 본문 확인이 §3의 (1) 판정 강도를 좌우한다. 우선순위는 (가) 다음이다.

---

## 5. 출처

1. Afèche, P., M. Araghi, O. Baron. "Customer Acquisition, Retention, and Service Access Quality: Optimal Advertising,
   Capacity Level, and Capacity Allocation." *Manufacturing & Service Operations Management* 19(4):674–691, 2017.
   출판본: https://doi.org/10.1287/msom.2017.0635 — 읽은 것은 저자 호스팅 2016-12 워킹페이퍼 전문:
   https://www.rotman.utoronto.ca/media/rotman/content-assets/images/areas/news-hub/2016/AfecheAraghiBaron2016AcquisitionRetentionServiceQuality.pdf
2. Klein, R., J. Kolb. "Maximizing customer equity subject to capacity constraints." *Omega* 55:111–125, 2015.
   https://doi.org/10.1016/j.omega.2015.02.008 — 초록: https://ideas.repec.org/a/eee/jomega/v55y2015icp111-125.html
3. Ovchinnikov, A., B. Boulu-Reshef, P. E. Pfeifer. "Balancing Acquisition and Retention Spending for Firms with Limited
   Capacity." *Management Science* 60(8):2002–2019, 2014. https://doi.org/10.1287/mnsc.2013.1842 —
   초록(복원 출처): https://api.openalex.org/works/doi:10.1287/mnsc.2013.1842 · https://hal.science/hal-01290217
4. Pfeifer, P. E., A. Ovchinnikov. "A Note on Willingness to Spend and Customer Lifetime Value for Firms with Limited
   Capacity." *Journal of Interactive Marketing* 25(3):178–189, 2011. https://doi.org/10.1016/j.intmar.2011.02.003 —
   초록(복원 출처): https://api.openalex.org/works/doi:10.1016/j.intmar.2011.02.003 ·
   SSRN 초기판 기록: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1335471
5. "A model to improve management of banking customers." *Industrial Management & Data Systems*, 2017.
   https://doi.org/10.1108/IMDS-03-2016-0107 — Klein & Kolb를 단일 제품군 연구로 특성화한 인용 문맥의 출처.
   인용 문맥 확인 API: https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.omega.2015.02.008/citations?fields=title,year,contexts
6. Ovchinnikov 외 2014 피인용 37건 전수 목록·초록(전수 키워드 검색에 사용):
   https://api.semanticscholar.org/graph/v1/paper/DOI:10.1287/mnsc.2013.1842/citations?fields=title,year,venue,externalIds,abstract&limit=100
   그중 §2 F4c에서 언급한 건: Social Promotion (POM 2020) https://doi.org/10.1111/poms.13247 ·
   에이전트 기반 스타트업 유지 (ESWA 2021) https://doi.org/10.1016/j.eswa.2021.114861
7. Kishimoto, Takehi, Tanaka, Nomura, Togashi, Tomita, Saito. "Beyond Match Maximization and Fairness:
   Retention-Optimized Two-Sided Matching." arXiv:2602.15752 (주석: Published as a conference paper at ICLR 2026).
   https://arxiv.org/abs/2602.15752 · 전문 PDF: https://arxiv.org/pdf/2602.15752
8. Subramanian, U., J. S. Raju, Z. J. Zhang. "The Strategic Value of High-Cost Customers."
   *Management Science* 60(2):494–507, 2014. https://doi.org/10.1287/mnsc.2013.1771 —
   초록(직접 확보 출처): https://api.crossref.org/works/10.1287/mnsc.2013.1771
9. Schmitt, P., B. Skiera, C. Van den Bulte. "Referral Programs and Customer Value." *Journal of Marketing* 75(1):46–59, 2011.
   https://doi.org/10.1509/jm.75.1.46 — 전문 PDF(저자 기관 호스팅):
   https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Schmitt-Skiera-vandenBulte-2011-Referral-Programs-Customer-Value.pdf
10. Kumar, V., J. A. Petersen, R. P. Leone. "Driving Profitability by Encouraging Customer Referrals: Who, When, and How."
    *Journal of Marketing* 74(5):1–17, 2010. https://doi.org/10.1509/jmkg.74.5.001 (서지·검색요약 수준. 본문 미확보)
11. Kumar, V. 외. "Undervalued or Overvalued Customers: Capturing Total Customer Engagement Value."
    *Journal of Service Research*, 2010. https://doi.org/10.1177/1094670510375602 (서지만. 본문·초록 미확보 — §4(마) 참조)
12. OA 부재 확인에 쓴 Unpaywall 조회(네 편 모두 `is_oa: false`):
    https://api.unpaywall.org/v2/10.1016/j.omega.2015.02.008 ·
    https://api.unpaywall.org/v2/10.1016/j.intmar.2011.02.003 ·
    https://api.unpaywall.org/v2/10.1287/mnsc.2013.1842 ·
    https://api.unpaywall.org/v2/10.1287/msom.2017.0635 (각각 `?email=` 파라미터 필요)

---

## 6. Coverage Status

**직접 확인한 것**

- Kishimoto 외 2026: **전문**(11쪽 PDF 텍스트 추출). Assumption 1, 식 (9)(12), Lemma 1·2 직접 읽음.
  bundle/complement/substitut/capacit/supermodul/submodul/all-or/quota/congestion **전부 0회**를 기계적으로 확인.
- Afèche 외 2017: 저자 호스팅 **워킹페이퍼 전문**(39쪽). Remark 1, 식 (26)(27)(28), Lemma 1, Proposition 1, §5.2 직접 읽음.
  용어 전수 검색 완료. **출판본 대조는 미완.**
- Ovchinnikov 외 2014 / Pfeifer & Ovchinnikov 2011: **초록 전문**(OpenAlex 색인 복원). 부호 진술 문장 직접 확인.
- Ovchinnikov 외 2014 **피인용 37건 전수** 제목·게재지·초록 키워드 검색(10개 키워드).
- Subramanian 외 2014: **초록 전문**(Crossref). 반례 문장 직접 확인.
- Schmitt 외 2011: **초록 전문**(Wharton 호스팅 PDF 직접 추출).
- Klein & Kolb 2015: **초록 + 인용 문맥**. 본문 미확보.

**여전히 불확실한 것**

- **F4b(i) 최적화 계수 여부 — 판정불가.** 이것이 남은 최대 위험이다. 도서관 확인 질문 §4(가) 참조.
- Pfeifer & Ovchinnikov가 추천 방향에서 WTS > CLV를 **명시**하는지.
- Ovchinnikov 외의 VIC ≤ CLV가 **정리**인지 수치예시인지.
- Afèche 외 출판본의 절·식 번호와 온라인 부록 내용.
- Subramanian 외 2014의 본문 명제 형태(집합함수 차분인지 여부).

**완료하지 못한 작업**

- OpenAlex `filter=cites:` 순회: **일일 예산 소진(HTTP 429)으로 끝내 실패했다.** 35초 간격 3회 재시도 포함.
  **Semantic Scholar + OpenCitations 두 창구로 대체 교차검증했다.** 정확한 오류 본문·시도 쿼리는 §7.
  OpenAlex 쪽 전수 확인은 UTC 자정 이후 재시도 과제로 남는다.
- Semantic Scholar 키워드 검색("value of an incremental customer" 등)도 429로 실패했다. WebSearch로 대체했으나
  학술 결과가 거의 없었다(상위 결과 대부분 마케팅 SEO 블로그). **이 경로는 재시도 가치가 있다.**
- Augsburg OPUS 리포지터리 접근(네트워크 차단). 다른 네트워크에서 재시도 필요.
- Kumar 외 2010 JSR OA PDF(Groningen 403).
- **Ovchinnikov 외 2014 피인용 중 1건 선별 불가**: "Retention or Acquisition? Behavior-Based Quality Disclosure"
  (*Management Science* 2025, https://doi.org/10.1287/mnsc.2022.01081 / 프리프린트 https://doi.org/10.2139/ssrn.5106550).
  Crossref·Semantic Scholar 어디에도 초록이 등록되어 있지 않다. **제목으로 내용을 단정하지 않는다.**
  확인할 질문: 이 논문의 모형에 고객 간 **보완형** 의존(교차판매·묶음·추천)이 있는지, 그리고 VIC/CLV 부호 비교를 하는지.
- 진행 기록·우회 시도 전문은 §7 재검증 로그에 있다.

---

## 7. 재검증 로그 (코디네이터 지시에 따른 우회 전술 결과)

작성 시점: 2026-09-28, §1~§6을 쓴 뒤 추가 수행. **판정은 하나도 바뀌지 않았고, 근거는 두 곳이 강해졌다.**

### 7.1 Crossref 등록 초록으로 3편 독립 교차검증 — 성공

`https://api.crossref.org/works/<DOI>` 를 네 편에 대해 호출했다(모두 http=200).
OpenAlex `abstract_inverted_index` 복원본과 **별개 경로**에서 같은 문장을 얻었으므로, §2의 인용은 단일 출처 의존이 아니다.

| DOI | Crossref `abstract` | 판정에 쓴 문장 재확인 |
|---|---|---|
| 10.1016/j.intmar.2011.02.003 [4] | 있음(1,276자) | "In contrast to well-understood demand-side dependencies among customer relationships (such as referrals), this paper highlights a particular kind of supply-side dependency" **일치 확인** |
| 10.1287/mnsc.2013.1842 [3] | 있음(1,933자) | "when capacity is unlimited, VIC equals customer lifetime value (CLV), but when capacity is limited, VIC is much smaller" **일치 확인** |
| 10.1287/msom.2017.0635 [1] | 있음(2,693자) | "her policy-dependent value, which reflects the Vμ indices of other served types" **일치 확인** (워킹페이퍼 초록과도 일치) |
| 10.1016/j.omega.2015.02.008 [2] | **없음**(Elsevier 미등록) | — |

### 7.2 Klein & Kolb 초록을 RePEc 두 미러에서 직접 추출 — 성공 (E 문서 재인용이 아니라 자체 확보)

- EconPapers: https://econpapers.repec.org/article/eeejomega/v_3a55_3ay_3a2015_3ai_3ac_3ap_3a111-125.htm (http=200)
- IDEAS: https://ideas.repec.org/a/eee/jomega/v55y2015icp111-125.html (http=200)

두 미러의 초록 본문이 **글자 단위로 동일**하다. §2 F4b에서 쓴 두 문장 외에 새로 확보한 문장이 하나 있다:

> "Furthermore, we analyze when and how **intertemporal customer behavior** influences capacity allocation." [2]

즉 초록이 명시하는 상호의존 축은 **(a) 용량 경쟁, (b) 시간축(재구매 행동)** 둘뿐이고, 참여자 간 보완은 없다.

**추가로 확보한 출판사 키워드 목록**(EconPapers 게재):
"Markov decision process ; Customer relationship management ; Revenue management ; Customer equity ;
Capacity allocation ; Repurchase behavior"
→ bundling·cross-selling·complementarity·multi-item·assortment **어느 것도 없다.** F4b(ii) 판정(대체형만)을 보강한다.

**F4b(i)은 여전히 판정불가다.** Crossref·RePEc·Unpaywall·OpenAlex 네 경로 모두 본문을 주지 않으므로,
"기여분을 최적화 계수로 쓰는가"는 §4(가)의 도서관 질문으로 남는다. 추측으로 메우지 않았다.

### 7.3 OpenAlex 429의 정확한 성격 — 코디네이터 진단과 **다르다**. 그대로 보고한다.

코디네이터가 부모 세션에서 본 본문은 "Anonymous search is temporarily rate-limited ... retry in 31s"(초 단위 스로틀)였다.
**내 환경의 429 본문은 다른 오류다.** 전문을 그대로 붙인다:

```
{"error":"Rate limit exceeded","message":"Insufficient budget. This request has no API key, so it counts against
the free daily budget shared by everyone on your network's IP address, and that budget is used up ($0 remaining;
resets at midnight UTC). Use your own key instead ...","retryAfter":52496,"costUsd":0.0001,
"dailyRemainingUsd":0,"prepaidRemainingUsd":0,"creditsRequired":1,"creditsRemaining":0}
```

`retryAfter`가 **52,496초(약 14.6시간, UTC 자정까지)** 다. 즉 내 쪽은 초 단위 스로틀이 아니라 **일일 예산 소진**이다.
지시대로 35초 간격으로 **3회 재시도**했고 세 번 모두 같은 429였다.

**다만 코디네이터의 경로 관찰은 맞았다.** 같은 시점에 두 경로의 결과가 갈렸다:

| 시도한 쿼리 | 결과 |
|---|---|
| `https://api.openalex.org/works?filter=cites:W1970322222&per-page=200&select=...` (3회, 35초 간격) | **http=429** (위 본문) |
| `https://api.openalex.org/works/doi:10.1287/mnsc.2013.1771?select=id,title` | **http=200** (정상 응답 수신) |

즉 `works/doi:` 단일 조회는 예산 소진 상태에서도 통과하고, `works?filter=` 목록 조회는 막힌다.
**F4의 핵심 수단(DOI 직접 조회 + `abstract_inverted_index` 복원)은 이미 §1~§2 작성 시 성공적으로 끝냈다**
(Ovchinnikov 외·Pfeifer & Ovchinnikov·Afèche 외 초록 복원 완료). 실패한 것은 **피인용 목록 순회 하나**다.

### 7.4 피인용 순회를 OpenCitations로 대체 — 성공, F4c 판정 보강

OpenAlex 목록 경로가 막혔으므로 **제3의 무료 인용 데이터베이스**를 썼다.
`https://opencitations.net/index/api/v2/citations/doi:10.1287/mnsc.2013.1842` (http=200, 리다이렉트 추적 필요)

- OpenCitations 피인용 DOI: **34건**
- Semantic Scholar 피인용 DOI: **32건**(전체 37건 중 DOI 보유분)
- **OpenCitations에만 있는 6건**(= Semantic Scholar가 놓친 것)을 Crossref 초록으로 전수 선별:

| DOI | 연도·제목 | 보완 키워드 14개 검사 |
|---|---|---|
| 10.1287/mnsc.2018.3139 | 2019 Strategic Consumers, Revenue Management, and the Design of Loyalty Programs | **0건** |
| 10.2139/ssrn.2903548 | 2017 In Pursuit of Enhanced Customer Retention Management: Review, Key Issues, and Future Directions | **0건** |
| 10.2139/ssrn.3260826 | 2018 Impact of Workforce Flexibility on Customer Satisfaction | **0건** |
| 10.2139/ssrn.3994561 | 2021 A Theory of Predictive Sales Analytics Adoption | **0건** |
| 10.2139/ssrn.5106550 | 2025 Retention or Acquisition? Behavior-Based Quality Disclosure | 초록 미등록(0자) — **판정불가 1건** |
| 10.2139/ssrn.5336880 | 2025 Cross-Channel Advertising Allocation and Consumer Dynamics | **0건** |

검사 키워드: cross-sell, cross sell, bundl, complement, referral, network value, word of mouth, synerg, viral,
spillover, supermodul, all-or-nothing, multi-item, portfolio.

**결론: 두 인용 데이터베이스 합집합에서 VIC > CLV 또는 묶음·보완형을 다룬 후속은 없다.**
단 위 표의 마지막에서 두 번째 항목(SSRN 5106550, MS 2025 "Retention or Acquisition? Behavior-Based Quality Disclosure")은
**초록이 어디에도 등록되어 있지 않아 선별하지 못했다.** 제목상 행동 기반 품질 공시 연구이고 묶음과 무관해 보이지만,
**제목으로 내용을 단정하지 않는다.** 미확인 1건으로 §4에 남긴다(출판본 DOI: 10.1287/mnsc.2022.01081).

### 7.5 절차상 보고 — 사용자 이메일 사용

§1~§6 작성 중 OpenAlex·Crossref·Unpaywall 호출 일부에 `mailto=` / `email=` 파라미터로 사용자 이메일을 넣었다
(API 예절(polite pool) 관행을 따른 것이지만, **사전 허락을 받지 않았다**).
코디네이터 지시를 받은 시점부터 중단했고, **§7의 모든 호출에는 이메일을 넣지 않았다**(Crossref·RePEc·OpenCitations·OpenAlex 모두
이메일 없이 http=200으로 동작함을 확인했다). 산출물 §5에 적은 URL에도 이메일을 남기지 않았다.
해당 세 서비스는 학술 서지 API이며 이메일 외 다른 정보는 보내지 않았다.

### 7.6 재검증 후 판정 변화 요약

| 항목 | 변화 |
|---|---|
| F4a | 변화 없음. Crossref 초록이 워킹페이퍼 초록과 일치해 **근거 강화** |
| F4b(i) | **여전히 판정불가.** Crossref에도 초록 미등록 확인으로 접근 실패 경로가 하나 더 문서화됨 |
| F4b(ii) | **강화.** RePEc 자체 확보 초록 + 출판사 키워드 목록에 보완 관련 단어 0건 |
| F4c | **강화.** 인용 DB 2개 교차검증(SS 37 + OpenCitations 34), 추가 6건 선별 완료, 미확인 1건 명시 |
| F4d | 변화 없음. Crossref 초록으로 결정적 문장 **독립 재확인** |
| F4e | 변화 없음(애초에 OpenAlex 불필요, arXiv 전문으로 완결) |
| §3 G1 판정 | **변화 없음.** "다섯 선례 모두 묶음 보완형 아님"은 유지되고, "기존 선례의 부호와 반대다"는 여전히 Subramanian 외 2014 [8] 때문에 쓸 수 없다 |
