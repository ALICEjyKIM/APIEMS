# 주제 E 증거 수집: 배분 몫의 특성화 · CLV 기반 배분 · 오지정 강건성

작성일: 2026-09-28
수집 도구: WebSearch(각도별 병렬), OpenAlex API(검색·피인용 순회·초록 색인 프로브), Crossref API(서지 검증),
Semantic Scholar API(초록·OA PDF 탐색), RePEc/IDEAS(출판사 제공 초록), arXiv API, pypdf로 저자 호스팅 PDF 초록 추출
중복 회피: `..._research_A.md` §2·§3·§4.A3을 먼저 읽고, 이미 조사된 Balseiro 2017, Chen 2020 TR-B, Luy 2024 POM,
Bhargava 2022, Cachon·Lariviere 2005, Musalem 2023은 **재조사하지 않았다.** 필요한 곳에서만 A 문서 번호를 `[A##]`로 참조한다.

**질문 진행 상태**: E6 done / E7 done / E8 done / E1 done / E9 done / E4 done(부분) / E3 done(부분) / E5 done
미해결은 §7에 열거한다.

읽은 수준 표기: `초록전문`=초록 전문을 직접 확보해 읽음, `본문일부`=본문 해당 절을 직접 추출, `서지`=서지정보만 확인(내용 추정 금지),
`검색요약`=검색엔진 요약만 확보(원문·초록 미확보, medium 이하 신뢰도), `색인프로브`=OpenAlex 초록 색인에 특정 용어가 있는지만 확인.

---

## 0. 최우선 직답 — E6 · E7

### E6. "개체별 생애가치를 추정해 최적화 모형의 계수로 넣는" 선례가 있는가?

**있다. 단정적으로 그렇다. 그리고 우리 `c_i`의 정의 자체(전체 가치 − 그 개체를 뺀 가치)도 이미 있는 개념이다.**
우리에게 불리한 방향으로 답한다.

가장 직접적인 세 편:

**(a) Klein & Kolb (2015), *Omega* 55:111–125 [5]** — 우리 `c_i`와 개념적으로 같은 것을 명시적으로 도입했다.
초록 전문(RePEc 제공, 직접 확보) 인용: "This paper considers a firm that wants to optimally allocate limited capacity to
heterogeneous customer segments in order to maximize its customer equity. The decision whether to accept or to reject
a customer's request in a current period influences his repurchase behavior in later periods. The allocation process
becomes complex, when demand exceeds capacity, **because the isolated determination and optimization of a single
customer's lifetime value is no longer feasible.** Using a Markov decision process formulation, we study how to trade
off short-term attainable revenues and long-term customer relationships. ... Finally, we investigate the impact of
limited capacity on the customer lifetime value by **introducing an opportunity cost-based approach that understands
customer profitability as a customer's contribution to customer equity**." [5]
→ "고객의 customer equity에 대한 기여분"은 `V(s) − V(s에서 i 제외)`와 같은 종류의 양이다.
또 "개별 고객의 CLV를 따로 떼어 최적화하는 것이 더는 불가능하다"는 진술은 **우리가 D 주제에서 주장하려던 상호의존성 문제를
이미 언어로 표현한 것이다.**

**(b) Ovchinnikov, Boulu-Reshef, Pfeifer (2014), *Management Science* 60(8):2002–2019 [3]** — 한계 개체 가치를 이름 붙여 도입했다.
초록 전문(RePEc, 직접 확보) 인용: "we introduce a concept of the **value of an incremental customer (VIC)**, and show that
when capacity is unlimited, VIC equals customer lifetime value (CLV), but when capacity is limited, **VIC is much
smaller and changes dynamically depending on the number of customers and their mix**. As a result, the optimal spending
is constant and depends on CLV for the firms with unlimited capacity, but changes dynamically and is generally
unrelated to CLV when capacity is limited." [3]
→ VIC는 "고객 1명이 추가로 있을 때의 가치 차분"이며 동적계획 안에서 계산된다. 우리 `c_i`와 기준점만 반대인 같은 차분이다.
그리고 **용량 제약이 있으면 한계 가치가 표준 CLV와 크게 달라진다**는 결과는, 우리가 "용량·묶음 결합 때문에 단순 CLV가 아니라
차분이 필요하다"고 말하려던 논거를 이미 선점한다.

**(c) Pfeifer & Ovchinnikov (2011), *Journal of Interactive Marketing* 25(3):178–189 [4]** — 그 차이의 조건을 명제로 진술했다.
출판사 제공 초록(RePEc, 직접 확보)의 인용: CLV와 willingness to spend(WTS, "the maximum amount the firm should be willing
to spend to acquire (retain) the customer")를 구분하고, "**CLV will equal WTS if (and, for the most part, only if) the
firm's relationships with customers are independent**"라고 진술한다. 또 "for a firm at capacity ... **CLV is no longer
relevant to marketing spending decisions and the firm can prefer a lower-CLV customer**"라고 한다 [4].
→ "참여자 간 상호의존이 없을 때만 개별 가치 = 한계 지불용의"라는 명제는, 우리가 "묶음·용량 결합 때문에 모듈러 근사가 깨진다"고
논증할 때 **인용해야 하는 선행 문헌**이지, 우리가 처음 발견한 것이 아니다.

추가 배경 계보:
- Pfeifer & Carraway (2000) [2]가 고객 관계를 마르코프 연쇄로 모형화하는 표준을 세웠다(초록 앞부분만 확보).
- Lewis (2005), *Management Science* 51(6):986–994 [1]은 **잠재계층 로짓으로 구매행동을 추정 + 동적계획으로 가격(할인) 경로 최적화**를
  하고, 초록에서 "the DP framework allows the calculation of CV to be an explicit function of marketing policies and
  customer status"라고 말한다 [1]. 즉 "생애가치를 정책의 함수로 계산해 현재 결정에 되먹임"은 CRM/CLV 동적계획의 표준 정식화다.
- Montoya, Netzer, Jedidi (2010), *Marketing Science* [29]는 (1) 계층 베이즈 비동질 은닉마르코프로 개체별 반응을 추정하고
  (2) **POMDP로 의사(physician)별 디테일링·샘플링을 동적 배분**한다 [29]. "개체별 반응 추정 → 개체별 동적 배분" 2단계 구조가 우리와 같다.

**결론(E6)**: "개체별 생애가치(또는 그 한계 차분)를 추정해 최적화의 계수·목적함수로 쓴다"는 것은 **CRM/CLV·수익관리 문헌의
확립된 표준**이다. 우리 `c_i`는 **새 개념이 아니다.** 논문에서 `c_i`를 신규 기여로 제시하면 리뷰어가 [3][4][5]로 즉시 반박할 수 있다.
`c_i`는 "기존 개념(VIC / customer's contribution to customer equity)을 학습된 비선형 V로 계산하고 매칭 MILP 계수로 쓴다"는
**구현·결합 수준의 기여**로 내려서 서술해야 한다.

### E7. 그 문헌이 조합(정수) 결정과 결합된 사례가 있는가, 아니면 연속량(마케팅 예산·할인·쿠폰)뿐인가?

**"연속량뿐"이라는 방어는 성립하지 않는다. 조합·이산 결정과 결합된 사례가 있다.** 이것도 우리에게 불리하게 답한다.

확인한 결합 사례:
1. **수락/거절 용량배분(이산)**: Klein & Kolb (2015) [5]은 "accept or reject a customer's request"를 결정하는 용량배분 MDP다 [5].
   Buhl, Klein, Kolb, Landherr (2011) [6]은 세그먼트별 제품에 희소 용량을 배분하는 CR2M을 제안한다(서지·검색요약 수준).
2. **0-1 대상 선택 수리계획**: Bhaskar, Sundararajan, Krishnan (2009), *JORS* 60:717–727 [25]은
   크로스셀 캠페인의 **대상 고객 리스트 선택**을 퍼지 수리계획으로 푼다. 초록 전문: "deciding on the list of customers to whom
   the offer should be sent such that a certain set of business objectives are met/optimized"이고, 계수는 "response propensity,
   expected volume, expected profit from a customer" 추정치다 [25]. 개체별 추정 가치가 이산 선택 문제의 계수로 들어간다.
3. **실무 규모의 수리계획 + 마르코프**: Sundararajan 외 7인 (2011), *Interfaces* 41:485–505 [26]은
   "making optimal product offers to customers of a retail bank by using techniques including **Markov chains, genetic
   algorithms, mathematical programming**, and design of experiments"이며 약 2천만 달러 효과를 보고한다 [26].
4. **용량 배분 + 타 참여자에 의존하는 개체 가치**: Afèche, Araghi, Baron (2017), *M&SOM* 19:674–691 [7]은
   "how to allocate capacity and tailor service access quality levels to different customer types"를 풀고,
   "a customer's lifetime value; her Vμ index ...; and **her policy-dependent value, which reflects the Vμ indices of
   other served types**"라는 가치지표를 도출한다 [7]. "다른 참여자에 의존하는 개체 가치"라는 점에서 우리 `c_i`와 문제의식이 같다.
5. **배분 규칙 + 참여자 집합의 조합 선택**: Alaei, Makhdoumi, Malekian, Pekeč (2022), *Management Science* 68(12):8699–8721 [20]은
   pro-rata / user-centric 수익배분 규칙이 **어떤 아티스트 집합을 플랫폼에 유지할 수 있는지**를 특성화하고,
   "the platform's problem of selecting an optimal portfolio of artists is **NP-complete**. However, by establishing
   connections to the **Knapsack problem**, we develop a Polynomial Time Approximation Scheme (PTAS)"라고 한다
   (저자 원고 PDF에서 직접 추출) [20]. 수익배분 + 참여 유지 + 조합 최적화가 한 편에 다 있다.
6. **학습된 가치함수 차분을 매칭의 변 가중치로**: Xu 외 (2018), KDD [8]. 초록 전문:
   "**each driver-order-pair is valued in consideration of both immediate rewards and future gains, and then dispatch
   is solved using a combinatorial optimizing algorithm**" [8]. Qin 외 (2020), *INFORMS Journal on Applied Analytics*
   50:272–286 [9]은 같은 시스템이 "a combinatorial optimization approach to one that encompasses a semi-Markov
   decision-process model and deep reinforcement learning"으로 진화했다고 기술한다 [9].
   → **"학습된 V(s)의 차분을 개체별 계수로 만들어 조합최적화에 넣는다"는 구조 자체가 이미 실무·학술 표준이다.**
     우리 파이프라인의 골격(학습 V → 개체별 차분 → 매칭 최적화)은 여기서 새롭지 않다.

**남는 진짜 차별점(우리에게 유리한 부분만 정확히 좁혀서)**
- (i) **배분 대상이 "금전 잉여 몫"이고 그 몫이 연속 결정변수로 같은 MILP 안에 있다.** [5][6][7]은 용량·서비스 접근성(비금전)을,
  [25][26]은 오퍼 대상 선택을, [8][9]는 매칭 자체를 결정한다. Alaei 외 [20]은 배분규칙을 고르지만 **정태**이고 개별 참여자별
  맞춤 몫이 아니라 규칙 클래스 선택이다. 위 문헌 중 "개별 참여자에게 갈 금전 몫 + 매칭"을 **하나의 수학적 모형에서 동시에** 정하는 것은 없었다.
- (ii) **양면(주문자·공급자) 모두에 `c_i`를 둔다.** [1][3][4][5][25][26][27][28][29]는 모두 고객 한 면만 다룬다. [8][9]는 기사 상태값만 쓴다.
- (iii) **all-or-nothing 다품종 묶음** 때문에 참여자 간 가치가 결합된다. 위 문헌 중 묶음 완전충족 제약을 둔 것은 없었다.
- (iv) **유지 가치를 학습된 비선형 V(그래프 포함)의 노드 삭제 차분으로 계산**한다. [3][5]는 작은 상태공간 MDP를 정확히 풀고,
  [8][9]는 격자/신경망 V를 쓰지만 유지·이탈 반응이 없다.

**리뷰어 통과 여부에 대한 솔직한 판정**
- (i)+(iii)의 조합은 "왜 중요한가"를 통과할 가능성이 있다. 금전 몫은 참여자별로 **연속적으로** 조절 가능한 유일한 레버이고,
  묶음 제약은 `c_i`의 가법성을 깨므로 [3][4]가 진술한 "상호의존이면 CLV ≠ 한계가치"의 **새로운 발생 원인**을 준다.
- (ii)만으로는 통과하기 어렵다. "양면으로 확장했다"는 증분 기여로 읽힌다.
- (iv)만으로는 통과하기 어렵다. [8][9] 계보가 이미 학습된 V의 차분을 조합최적화 계수로 쓴다. 우리 차이는 "차분 대상이
  참여자 노드이고 가중치가 구간선형 재참여확률"이라는 정식화 세부이며, 이는 B·C 주제(Neur2RO 계보)와 묶어야 한다.
- **가장 위험한 반박**: "당신들의 `c_i`는 Klein & Kolb의 customer's contribution to customer equity이고,
  Ovchinnikov 외의 VIC이며, Pfeifer & Ovchinnikov가 CLV ≠ WTS 조건으로 이미 정리했다. 새로운 것은 무엇인가?"
  → 이 질문에 대한 답을 **"금전 몫 결정 + 묶음 제약 + 양면 + 하나의 MILP"의 조합으로만** 준비해야 하고,
    `c_i` 개념 자체를 기여로 주장하면 안 된다.

---

## 1. 증거 표

사분위는 **하나도 직접 확인하지 않았다.** 저널명과 본 프로젝트 기준(OR/OM 주요 저널 여부)만 적는다.

| # | 제목 | 저자·연도 | 게재지(사분위 미확인) | 배분 대상(결정변수) | 반응 함수 | 최적화 형태 | 우리와의 차이 한 문장 | URL | 읽은 수준 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Research Note: A Dynamic Programming Approach to Customer Relationship Pricing | Lewis 2005 | Management Science 51(6):986-994 | 고객별 가격(할인) | **잠재계층 로짓**(구매확률) | 동적계획(가격 경로) | 한 면(고객)만. 매칭·용량·묶음 없음 | https://doi.org/10.1287/mnsc.1050.0373 | 초록전문 |
| 2 | Modeling customer relationships as Markov chains | Pfeifer & Carraway 2000 | Journal of Interactive Marketing 14(2):43-55 | (모형화) | 마르코프 전이 | 마르코프 연쇄 CLV | 최적화·매칭 없음. CLV 마르코프 표준의 원전 | https://doi.org/10.1002/(sici)1520-6653(200021)14:2<43::aid-dir4>3.0.co;2-h | 초록 앞부분만 |
| 3 | Balancing Acquisition and Retention Spending for Firms with Limited Capacity | Ovchinnikov, Boulu-Reshef, Pfeifer 2014 | Management Science 60(8):2002-2019 | 획득·유지 **지출액**(연속) | 초록에 함수형 명시 없음(확인 필요) | 동적계획 + 행동실험 | **VIC(한계 고객 가치)가 우리 c_i와 동형.** 금전 몫 배분·매칭·묶음 없음 | https://doi.org/10.1287/mnsc.2013.1842 | 초록전문 |
| 4 | A Note on Willingness to Spend and Customer Lifetime Value for Firms with Limited Capacity | Pfeifer & Ovchinnikov 2011 | Journal of Interactive Marketing 25(3):178-189 | 획득·유지 지출액 | - | 고객자산 확장 모형(해석적) | **"관계가 독립일 때만 CLV = WTS"를 진술.** 우리 모듈러 근사 논증의 선행 문헌 | https://doi.org/10.1016/j.intmar.2011.02.003 | 초록전문(RePEc) |
| 5 | Maximizing customer equity subject to capacity constraints | Klein & Kolb 2015 | Omega 55:111-125 | **용량 수락/거절**(이산) | 수락·거절이 재구매 행동에 영향(함수형 미확인) | **MDP** | **"고객의 customer equity 기여분"을 기회비용으로 도입.** 금전 몫·양면·묶음·학습 VFA 없음 | https://doi.org/10.1016/j.omega.2015.02.008 | 초록전문(RePEc) |
| 6 | CR2M - an approach for capacity control considering long-term effects on the value of a customer | Buhl, Klein, Kolb, Landherr 2011 | Journal of Management Control 22:187-204 | 세그먼트별 제품에 희소 용량 배분 | 확인 필요 | 확인 필요(RM+CRM 결합) | 초록 미확보. 내용 단정 금지 | https://doi.org/10.1007/s00187-011-0133-8 | 서지 + 검색요약 |
| 7 | Customer Acquisition, Retention, and Service Access Quality: Optimal Advertising, Capacity Level, and Capacity Allocation | Afeche, Araghi, Baron 2017 | M&SOM 19:674-691 | 광고지출·용량수준·**용량 배분** | 획득·유지·행동이 서비스 접근품질에 의존 | **유체 모형** + 대규모 큐 시뮬레이션 | **"policy-dependent value"(타 유형에 의존하는 개체 가치)를 이미 도출.** 금전 몫·묶음·학습 VFA 없음 | https://doi.org/10.1287/msom.2017.0635 | 초록전문(끝부분 절단) |
| 8 | Large-Scale Order Dispatch in On-Demand Ride-Hailing Platforms: A Learning and Planning Approach | Xu 외 2018 | KDD 2018 (학회) | 주문-기사 **배정** | 없음(유지·이탈 반응 없음) | 학습된 시공간 상태가치 + **조합최적화** | **학습 V 차분을 매칭 계수로 쓰는 구조의 선례.** 잉여배분·재참여·묶음 없음 | https://doi.org/10.1145/3219819.3219824 | 초록전문 |
| 9 | Ride-Hailing Order Dispatching at DiDi via Reinforcement Learning | Qin, Tang, Jiao, Zhang, Xu, Zhu, Ye 2020 | INFORMS Journal on Applied Analytics 50:272-286 | 주문-기사 배정 | 없음 | 조합최적화에서 semi-MDP + 심층강화학습으로 진화 | 동일. OR 계열 심사지 판본 | https://doi.org/10.1287/inte.2020.1047 | 초록전문 |
| 10 | Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching (MRet) | Kishimoto, Takehi, Tanaka, Nomura, Togashi, Tomita, Saito 2026 | arXiv(주석: Published as a conference paper at ICLR 2026) | **매칭 기회(추천 슬롯)** | **학습된 개인별 유지 곡선, 매칭 수에 대해 오목**(Assumption 1) | NP-hard를 **정렬 기반 O(N log N)** LTR로 환원 | 배분 대상이 노출이고 **금전 아님.** 단일기간 LTR이며 V(s)·MILP·잉여배분 없음 | https://arxiv.org/abs/2602.15752 | 본문일부(Eq.9·14·15, Assumption 1, 4.1절) |
| 11 | Not All Matches Are Equally Valuable: Retention-Focused Recommendation in a Job-Matching Platform | Ute, Ichimura, Saito 2026 | arXiv(RecSys in HR 2026 워크숍) | 추천 랭킹 점수 보정 | **실증**: 최근 매칭 적은 사용자의 이탈위험 급증, 성공 사용자의 추가 매칭은 한계효과 작음 | 사후처리 점수 부스트 + 온라인 실험 | 최적화 모형 아님. 효과가 통상 수준에서 유의하지 않았다고 스스로 보고 | https://arxiv.org/abs/2609.01652 | 초록전문(arXiv API) |
| 12 | Dynamic Learning and Pricing with Model Misspecification | Nambiar, Simchi-Levi, Wang 2019 | Management Science 65(11):4980-5000 | 가격 | **오지정된 수요모형** | 온라인 학습 + **random price shock(RPS)**, 이론 보증 | 우리는 추정 루프가 없고 반응을 알려진 것으로 둔다. 결정-예측오차 상관(내생성)을 다루지 않는다 | https://doi.org/10.1287/mnsc.2018.3194 | 초록전문 |
| 13 | Testing the Validity of a Demand Model: An Operations Perspective | Besbes, Phillips, Zeevi 2010 | M&SOM 12(1) | (검정 방법론) | "the ubiquitous **logit** model" 등 모수 소비자반응 모형 | **성과 기반 오지정 가설검정**(점근 특성화) | 우리는 오지정의 성과 영향을 두 점(재학습/불일치)으로만 본다. 통계적 검정 절차가 없다 | https://doi.org/10.1287/msom.1090.0264 | 초록전문(저자 호스팅 PDF 직접 추출) |
| 14 | Models of the Spiral-Down Effect in Revenue Management | Cooper, Homem-de-Mello, Kleywegt 2006 | Operations Research 54(5):968-987 | 보호수준(booking control) | 잘못된 고객행동 가정 | 예측-최적화 반복의 동태 분석, 나선형 하강 발생 조건 | **오지정 + 최적화 되먹임이 자기강화 편향을 낳는다**는 표준 경고. 우리 실험 2에 이 축이 없다 | https://doi.org/10.1287/opre.1060.0304 | 검색요약(초록 미확보, medium) |
| 15 | Robust Control of Markov Decision Processes with Uncertain Transition Matrices | Nilim & El Ghaoui 2005 | Operations Research | (방법론) | 전이확률 불확실성 집합 | robust MDP | 우리는 불확실성 집합을 정의하지 않는다 | https://doi.org/10.1287/opre.1050.0216 | 서지 |
| 16 | Robust Dynamic Programming | Iyengar 2005 | Mathematics of Operations Research | (방법론) | 전이확률 불확실성 집합 | robust DP | 동일 | https://doi.org/10.1287/moor.1040.0129 | 서지 |
| 17 | Robust Markov Decision Processes | Wiesemann, Kuhn, Rustem 2013 | Mathematics of Operations Research | (방법론) | 불확실성 집합(일반화) | robust MDP | 동일 | https://doi.org/10.1287/moor.1120.0566 | 서지 |
| 18 | Distributionally Robust Markov Decision Processes | Xu & Mannor 2010 | NeurIPS 2010 (학회) | (방법론) | 전이분포의 모호집합 | DR-MDP | 동일. 분포적 강건 계열의 씨앗 | https://papers.nips.cc/paper/3927-distributionally-robust-markov-decision-processes | 서지 |
| 19 | Treatment Allocation with Strategic Agents | Munro 2023 | Management Science (mnsc.2022.01629) | 개체별 처치(할당) | **전략적 개체가 공변량을 바꾼다** | 베이즈 최적화 기반 순차실험, 최적 규칙에 **무작위화** 포함 | 우리는 참여자를 비전략적으로 둔다. E3 한계 서술의 직접 근거 | https://doi.org/10.1287/mnsc.2022.01629 | 초록전문 |
| 20 | Revenue-Sharing Allocation Strategies for Two-Sided Media Platforms: Pro-Rata vs. User-Centric | Alaei, Makhdoumi, Malekian, Pekec 2022 | Management Science 68(12):8699-8721 | **수익배분 규칙**(pro-rata / user-centric / 임의 규칙) | 아티스트 참여 지속 조건(정태) | 규칙 비교 + **Knapsack 환원, NP-complete, PTAS**, 쌍대성 기반 다항시간 알고리즘 | **정태.** 확률적 재참여·동적 상태전이·학습 VFA 없음 | https://doi.org/10.1287/mnsc.2022.4307 | 초록전문(저자 원고 PDF 직접 추출) |
| 21 | Dynamic Resource Allocation on Multi-Category Two-Sided Platforms | Li, Shen, Bart 2021 | Management Science | 양면·다카테고리·다기간 **투자 배분** | 네트워크 효과(직접·간접·교차) | 2기간 이론모형 + 실증 + 시뮬레이션 | 배분 대상이 집계 투자액. 개별 참여자 잉여 몫·매칭 없음 | https://doi.org/10.1287/mnsc.2020.3586 | 초록전문 |
| 22 | Balancing Acquisition and Retention Resources to Maximize Customer Profitability | Reinartz, Thomas, Kumar 2005 | Journal of Marketing 69(1):63-79 | 획득·유지 마케팅 지출(채널별) | 확인 필요(초록 미확보) | 수익성 최대화 배분 프레임워크 | 초록 미확보. 1차 조건의 정확한 형태 확인 필요 | https://doi.org/10.1509/jmkg.69.1.63.55511 | 서지 |
| 23 | Managing Retention in Service Relationships | Aflaki & Popescu 2014 | Management Science 60(2):415-433 | 기간별 **서비스 수준**(연속) | 과거 경험에서 기대·만족을 거쳐 갱신으로(행동이론 기반) | 동적 모형, 최적정책 구조 | 금전 몫 아님. "장기적으로 서비스를 변동시키는 것이 최적이 아니다"가 우리 차등배분과 대비된다 | https://doi.org/10.1287/mnsc.2013.1775 | 초록전문 |
| 24 | Managing Churn to Maximize Profits | Lemmens & Gupta 2020 | Marketing Science 39(5):956-973 | 사전 유지 캠페인 대상 | 인과추론(기계학습) 기반 처치효과 | 이익 최적화(캠페인) | 초록이 2문장으로 짧다. 최적화 형태 확인 필요 | https://doi.org/10.1287/mksc.2020.1229 | 초록전문(짧음) |
| 25 | A fuzzy mathematical programming approach for cross-sell optimization in retail banking | Bhaskar, Sundararajan, Krishnan 2009 | Journal of the Operational Research Society 60:717-727 | **대상 고객 리스트 선택**(0-1) | 반응성향 추정치(삼각퍼지수) | **퍼지 수리계획**, 그룹 수준 정식화 | 개체 가치가 이산 선택 계수로 들어가는 선례. 동적 상태전이·매칭·묶음 없음 | https://doi.org/10.1057/palgrave.jors.2602609 | 초록전문 |
| 26 | Marketing Optimization in Retail Banking | Sundararajan 외 7인 2011 | Interfaces 41:485-505 | 고객별 제품 오퍼 | 반응 추정 불확실성 언급 | **마르코프 연쇄 + 수리계획 + 유전알고리즘** | 실무 사례(약 2천만 달러 효과). 동적 참여자 집합 전이·양면 없음 | https://doi.org/10.1287/inte.1110.0597 | 초록전문 |
| 27 | Customer lifetime value: stochastic optimization approach | Ching, Ng, Wong, Altman 2004 | Journal of the Operational Research Society 55:860-868 | 광고·판촉 예산 | 마르코프 연쇄 | **확률적 동적계획**(무한·유한 지평) | 연속 예산. 조합 결정·양면·묶음 없음 | https://doi.org/10.1057/palgrave.jors.2601755 | 초록전문 |
| 28 | Cross-Selling the Right Product to the Right Customer at the Right Time | Li, Sun, Montgomery 2011 | Journal of Marketing Research 48(4):683-700 | 고객별 크로스셀 권유(제품·시점·채널) | 고객반응 모형(수요 진화, 판촉·광고·교육의 다면 역할) | **확률적 동적계획** | 한 면. 매칭·용량·묶음 없음 | https://doi.org/10.1509/jmkr.48.4.683 | 초록전문 |
| 29 | Dynamic Allocation of Pharmaceutical Detailing and Sampling for Long-Term Profitability | Montoya, Netzer, Jedidi 2010 | Marketing Science 29(5):909-924 | 의사별 디테일링·샘플링(언제·어떻게) | **계층 베이즈 비동질 은닉마르코프**(추정) | **POMDP**(사후분포 적분) | "개체별 반응 추정에서 개체별 동적 배분으로"의 2단계가 우리와 같다. 금전 몫·양면·매칭 없음 | https://doi.org/10.1287/mksc.1100.0570 | 초록전문 |

A 문서에 이미 있는 항목(Chen·Xu 2026 [A17], Bhargava 외 2022 [A14], Balseiro 외 2017 [A13], Cachon·Lariviere 2005 [A15],
Chen 외 2020 TR-B [A12], Luy 외 2024 [A10], Masorgo 외 2026 [A26])은 재수록하지 않고 필요할 때 [A##]로 참조한다.

---

## 2. E1 — 배분 몫의 최적 수준을 참여 반응의 1차 조건으로 특성화한 결과

**결론: 우리 1차 조건 `b·c_i·p(1−p)/w = 1`과 "같은 구조"의 결과가 CRM 문헌에 이미 있다.
단, "개별 참여자의 잉여 몫"에 대해 그 조건을 쓴 것은 찾지 못했다.**

- **같은 구조의 FOC가 있는 계보**: CLV 기반 유지 지출 최적화의 표준 1차 조건은
  "유지지출 한 단위의 한계 유지확률 상승 × 생애가치 = 1"이다. Ovchinnikov 외 (2014)의 초록이 이 구조를 직접 진술한다:
  "**the optimal spending is constant and depends on CLV** for the firms with unlimited capacity, but changes
  dynamically and is generally unrelated to CLV when capacity is limited" [3].
  우리 조건은 이 구조에서 (a) 생애가치를 `c_i`로, (b) 한계 유지확률을 로지스틱 도함수 `b·p(1−p)/w`로 바꾼 것이다.
  즉 **1차 조건의 구조 자체는 신규가 아니고, 로지스틱을 대입하면 `p(1−p)` 인자가 나오는 것은 대수적 귀결이다.**
- **참여 유지 조건을 배분 규칙의 함수로 특성화한 정태 결과**: Alaei 외 (2022) [20]이
  "we characterize when these two allocation rules can sustain a set of artists on the platform"으로
  수익배분 규칙과 참여 지속을 직접 연결한다 [20]. Bhargava 외 (2022) [A14]는 생산자 규모별 차등 수익배분율의
  최적 설계를 정태로 푼다.
- **플랫폼 수수료율의 FOC**: 검색 수준에서 Stackelberg 계열(예: *Journal of Revenue and Pricing Management*의
  "Optimal revenue sharing in platform markets: a Stackelberg model", https://doi.org/10.1057/s41272-018-00180-4)이
  존재하지만 **초록·원문을 확보하지 못해 내용을 단정하지 않는다.** 검색 요약이 제시한 닫힌 해 형태는 출처를
  검증하지 못했으므로 인용하지 않는다.
- **찾지 못한 것**: "개별 참여자 i에게 가는 잉여 몫"의 최적성을 `그 참여자의 한계 유지가치 × 반응 도함수 = 1`
  형태로 진술한 결과. 아래 쿼리에서 해당 결과가 나오지 않았다.
  - WebSearch `optimal revenue sharing rate platform first order condition retention elasticity commission`
    → 학술 결과는 정태 Stackelberg 계열뿐이고 상위 결과 다수가 마켓플레이스 SEO 블로그였다.
  - WebSearch `optimal retention spending first order condition "marginal" retention rate times customer lifetime value equals one logistic response function`
    → 정확한 수식 형태를 담은 학술 출처를 얻지 못했다(Blattberg·Deighton 조건에 대한 2차 언급만 나왔다. 7절 참조).
- **판정**: 우리 1차 조건은 **논문의 독립 기여로 내세울 수 없다.** "우리 MILP가 암묵적으로 만족하는 조건을
  CRM의 표준 FOC와 연결해 해석"하는 용도로만 쓰고, 이때 [3]을 인용하는 것이 정직하다.

---

## 3. E8 — 유지 최적화를 매칭과 결합한 최근 연구

**확인했다. 단서로 지적된 arXiv 2602.15752는 실재하고, 우리 문제 설정과 가장 가깝다.**

**Kishimoto, Takehi, Tanaka, Nomura, Togashi, Tomita, Saito (2026), arXiv:2602.15752v3,
"Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching" [10]** —
arXiv 주석에 "Published as a conference paper at ICLR 2026"으로 적혀 있다(저자 신고 정보이며 OpenReview로 직접 확인하지 않았다).
초록과 본문에서 직접 확인한 것:
- 문제 정의: "we formally define the new problem setting of **maximizing user retention in two-sided matching
  platforms**" [10]. 즉 "양면 매칭 플랫폼에서 유지 최대화"라는 문제 설정을 **새로운 것으로 선언한 논문이 이미 있다.**
- 반응 함수: "For every user x in X, the reward function f(x,.) is **concave** in the number of matches m >= 0"
  (Assumption 1). 합성실험은 이차-지수형(만족 매칭 수 b_x까지 이차 증가 후 포화, Eq. 15), 실데이터는
  로그인 데이터의 k-means 군집별 유지 곡선을 쓴다 [10].
- 최적화: 원 문제(Eq. 9)는 NP-hard이고, **오목성을 이용해 정렬 문제로 환원**한다(Eq. 14, O(N log N)) [10].
- 양면 반영: "the retention probability gain for both the receiving user x in X and the recommended users" [10].
- 참여자 집합: 사용자가 확률적으로 이탈하며 이탈·잔존 집합으로 구분된다(4.1절). 다만 정식화는 **단일기간 LTR 틀**이며
  명시적 다기간 동적계획이 아니다 [10].
- 비교군: Max Match(Eq. 3), FairCo, Uniform, 그리고 참 유지함수를 아는 oracle(MRet-best) [10].
- 금전: 구독 모형을 동기로 언급하지만 금전 이전·가격·경제적 잉여 배분은 모형화하지 않는다(해당 언급을 본문에서 찾지 못했다) [10].

**유사 연구 추가 확인**
- Ute, Ichimura, Saito (2026), arXiv:2609.01652(RecSys in HR 2026 워크숍) [11]: 실제 구직 매칭 플랫폼에서
  "users with very few recent matches are indeed much more likely to leave the platform, while **additional matches
  for already successful users provide limited marginal value for retention**"를 실증하고, 사후처리 점수 부스트를
  온라인 실험했다. 처리군의 이탈이 방향적으로 낮았으나 **통상 유의수준에서 유의하지 않았다**고 스스로 보고한다 [11].
  (A 문서 [A34]의 저자 미확인 항목이 이 논문이다. arXiv API로 저자를 확정했다: Tatsuya Ute, Chiaki Ichimura, Yuta Saito.)
- Xu 외 2018 [8] / Qin 외 2020 [9]: 유지·이탈 반응은 없지만 **학습 V의 차분을 매칭 계수로** 쓰는 구조의 선례다.
- Afeche, Araghi, Baron 2017 [7]: 획득·유지를 **용량 배분**과 결합한 M&SOM 논문. 매칭은 아니지만
  "누구에게 용량을 주면 그가 남는다"는 구조는 같다.
- 검색에서 반복 등장했으나 **초록을 확보하지 못한 후보**(내용 단정하지 않는다):
  "Optimizing Long-Term Efficiency and Fairness in Ride-Hailing via Joint Order Dispatching and Driver Repositioning"
  (KDD 2022, https://dl.acm.org/doi/10.1145/3534678.3539060),
  "A vehicle value based ride-hailing order matching and dispatching algorithm"
  (https://www.sciencedirect.com/science/article/abs/pii/S095219762400112X).

**우리 설정과 [10]의 차이(방어선)**: 배분 대상이 노출(매칭 기회)이며 **금전 잉여가 아니다**; 반응이 매칭 수의 오목함수이고
잉여율의 로지스틱이 아니다; **V(s)를 학습해 미래 가치를 근사하지 않고** 1기간 유지 이득만 본다; 오목성으로 정렬 환원하므로
**MILP·품목별 용량 제약·묶음 완전충족이 없다**; 공급 용량과 all-or-nothing 다품목이 없다.
**그러나 "양면 매칭 플랫폼의 유지 최적화"라는 문제 선언을 먼저 한 논문이 존재하므로, 우리 논문에서 문제 설정의
신규성을 주장할 때 반드시 [10]을 인용하고 차이를 명시해야 한다.** 인용하지 않으면 ICLR·KDD 계열을 읽은 리뷰어에게 걸린다.

---

## 4. E9 — 반응 모형 오지정 강건성의 표준 접근, 그리고 우리 실험 2에 빠진 것

### 표준 접근 네 갈래 (확인한 것만)

1. **성과 기반 오지정 검정** — Besbes, Phillips, Zeevi (2010) [13].
   초록 전문(저자 호스팅 PDF에서 pypdf로 직접 추출):
   "managers are typically more interested in the performance of a decision rather than the statistical validity of the
   underlying model. We propose a framework and a statistical test that **incorporates decision performance into a
   measure of statistical validity**. ... We show that traditional model-based goodness-of-fit tests may consistently
   reject simple parametric models of consumer response (e.g., **the ubiquitous logit model**), while at the same time
   these models may 'pass' the proposed performance-based test. Such situations arise when decisions derived from a
   postulated (and possibly incorrect) model, generate results that cannot be distinguished statistically from the best
   achievable performance -- i.e., when demand relationships are fully known." [13]
2. **오지정 하 학습·최적화(내생성 교정 + 후회 보증)** — Nambiar, Simchi-Levi, Wang (2019) [12]. 초록 전문:
   "model misspecification leads to a **correlation between price and prediction error of demand** per period, which,
   in turn, leads to inconsistent price elasticity estimates and hence suboptimal pricing decisions. We propose a
   'random price shock' (RPS) algorithm ... We show that the RPS algorithm has **strong theoretical performance
   guarantees**, that it is robust to model misspecification" [12].
3. **예측·최적화 되먹임의 자기강화 편향** — Cooper, Homem-de-Mello, Kleywegt (2006) [14].
   검색요약 수준에서 확인: 잘못된 고객행동 가정이 판매·추정·보호수준의 하향 나선을 만들고, 논문이 그 발생 조건을 제시한다 [14].
   **초록 미확보이므로 medium 신뢰도.**
4. **강건·분포적 강건 동적계획** — Nilim & El Ghaoui 2005 [15], Iyengar 2005 [16], Wiesemann, Kuhn, Rustem 2013 [17],
   Xu & Mannor 2010 [18]. **이 네 편은 서지만 확인했다.** 제목 수준을 넘어 내용을 단정하지 않는다.
   (전이확률 또는 그 분포의 불확실성 집합 위에서 최악의 경우를 최대화한다는 일반적 설명은 검색 요약 근거이며 medium이다.)

### 우리 실험 2 설계(재학습 / 로지스틱으로 학습하고 다른 반응에서 평가)에 빠진 것

우리 설계는 "모형 오지정의 **성과 영향을 두 점으로 측정**"하는 민감도 분석이다. 위 표준에 비해 다음이 없다.

1. **불확실성 집합이 없다.** [15][16][17][18] 계열은 반응·전이를 하나의 대안으로 바꾸는 것이 아니라
   **집합 위에서 최악을 보장**한다. 우리는 강건 정책을 설계하지 않고 강건성을 측정만 한다.
   → 논문에서 "robust"라는 단어를 쓰면 안 되고 "sensitivity to response misspecification"으로 써야 한다.
2. **추정 루프가 없다.** [12]의 핵심 문제는 "결정이 예측오차와 상관되어 탄력성 추정이 비일관"이 되는 것이다 [12].
   우리는 반응 함수를 **알려진 것으로** 두고 평가 반응만 바꾼다. 실제 플랫폼은 배분 몫을 정하면서 반응을 추정해야 하므로
   이 내생성이 그대로 발생한다. **우리 설정은 오히려 [12]보다 나쁜 조건이다**: 배분 결정이 잉여율을 정하고,
   잉여율이 재참여를 정하며, 남은 참여자만 다음 기간 데이터에 나타난다(생존 선택).
3. **되먹임 편향 분석이 없다.** [14]의 나선형 하강과 구조가 같은 위험이 우리에게 있다: 유지 가치 `c_i`를 과소추정하면
   참여자에게 잉여를 덜 주고, 참여자가 떠나고, 학습 데이터의 참여자 구성이 좁아지고, `c_i` 추정이 더 나빠진다.
   현재 실험 2는 이 경로를 측정하지 않는다. **이것이 실험 2의 가장 큰 공백이다.**
4. **성과 기반 검정이 없다.** [13]은 "통계적으로 기각되는 모형도 의사결정 성능 기준으로는 통과할 수 있다"는 검정을 준다 [13].
   결과를 "예측 오차가 커졌다 / 누적 이윤이 떨어졌다"로만 보고하면 그 차이가 통계적으로 구분 가능한지에 대한 절차가 없다.
   최소한 반복 30회 차이에 대한 신뢰구간·검정을 붙여야 한다.
5. **보증이 없다.** [12]는 후회 보증을, [14]는 발생 조건을 준다. 우리는 소규모 예비 실험이므로 보증을 제시할 수 없고,
   이를 한계로 명시해야 한다.
6. **[13]이 주는 유리한 근거 하나**: 로짓·로지스틱을 "the ubiquitous logit model"로 부르므로, 우리 로지스틱 가정 자체는
   OM 문헌에서 표준적 선택이라고 인용할 수 있다 [13].

---

## 5. E4 · E3 · E5 직답

### E4. 이탈·유지가 금전 배분에 로지스틱으로 반응한다는 가정의 실증 근거 또는 반증

**부분적 근거는 있다. 그러나 "금전 배분에서 유지확률로"의 관계를 로지스틱으로 추정한 운영 문헌은 이번에도 찾지 못했다.**
1차 조사(A3)의 결론을 뒤집을 증거는 없었고 인접 근거만 보강되었다.

근거 쪽
- **로짓 자체가 관행**: Besbes, Phillips, Zeevi (2010)가 "the ubiquitous logit model"이라고 부른다 [13].
  소비자 반응을 로짓으로 두는 것은 OM 가격결정·수익관리에서 표준적 선택이다.
- **금전(할인)에 대한 로짓 반응을 실데이터로 추정한 사례**: Lewis (2005) [1]. 초록 전문:
  "The first component is a **latent class logit model** that is used to model customer buying behavior"이며
  대도시 신문 구독자 데이터로 추정하고, 신규 고객 할인 관행을 지지하되 "**a series of decreasing discounts based on the
  length of customer tenure** rather than a single steep discount for first-time purchasers"를 제안한다 [1].
  → 단 이것은 **구매(갱신) 확률의 로짓**이며 설명변수가 가격·할인이다. 우리 "배분 잉여율의 로지스틱"과 완전히 같지 않다.
- **포화·오목성의 실증과 가정**: Ute, Ichimura, Saito (2026) [11]이 구직 플랫폼에서 "매칭이 적은 사용자는 이탈위험이 크고,
  이미 성공한 사용자에게 추가 매칭은 한계 유지가치가 작다"를 실증한다 [11].
  Kishimoto 외 [10]은 유지 곡선의 **오목성**을 가정으로 놓는다 [10].
  → 이는 **로지스틱의 상반부(오목 구간)와 부합하지만, 하반부의 볼록 구간(작은 금액에서 반응이 거의 없다가 급증)을
    지지하지 않는다.** 우리 로지스틱은 잉여율 0에서 p = 0.5로 시작하는 S자이므로 위 실증은 부분적 지지일 뿐이다.
- **문헌의 실제 표준은 함수형이 아니라 성질**: A 문서에서 확인한 대안 형태는 유보값·임계형, 선형 혼합(Luy 외 [A10]),
  `1 − exp(−ηx)`(Chen·Xu [A17])이다. 공통점은 **단조 증가 + 포화**이며 로지스틱은 그중 한 선택일 뿐이다.
- **확보하지 못한 근거**: 이탈 예측에서 로지스틱 회귀가 관행이라는 점은 웹 검색으로 보이지만 상위 결과가 대부분
  블로그·학위논문이어서 SCIE 수준 출처를 특정하지 못했다. **인용 가능한 출처가 없으므로 근거로 쓰지 않는다.**

반증·주의 쪽
- **차등 대우의 최적성 자체에 반대되는 결과**: Aflaki & Popescu (2014), *Management Science* [23] 초록 전문:
  "we find that **varying service in the long run is not optimal**. ... **Loyal or high-margin customers need not
  warrant better service**; those who anchor less on past service experiences do -- provided that retention is improved
  by better past experiences" [23].
  → "생애가치가 높은 참여자에게 더 많이 준다"는 우리 `c_i` 비례 논리에 대한 직접적 반례가 심사지 논문으로 존재한다.
    반응 함수가 기준점 의존(참조 효과)을 가지면 최적 정책이 **일정 수준 유지**가 될 수 있다.
    실험 2의 대안 반응 후보로 검토할 가치가 있다.
- **행동 편향 증거**: Ovchinnikov 외 (2014) [3]은 "providing CLV information **exacerbates** these biases and leads to a
  loss of net revenue when capacity is limited, but providing information about the **marginal costs** of acquisition and
  retention eliminated these biases and increases net revenue"라고 보고한다 [3].
  → 우리가 CLV 대신 `c_i`(한계 개념)를 쓰는 것은 이 결과와 **같은 방향**이므로 유리한 인용이다.
- A 문서 [A26](Masorgo 외 2026, JOM)의 "금전만의 함수가 아니다"는 반박은 그대로 유효하다(서지만 확인).

**서술 권고**: "재참여확률 = 로지스틱(잉여율)"은 (a) 로짓이 OM의 관행적 선택이라는 점 [13],
(b) 단조·포화 성질이 관련 실증과 부합한다는 점 [10][11], (c) 두 점으로 고정한 방식이 Luy 외 [A10]의 두 점 보간과
같은 수준의 단순화라는 점으로 정당화하고, **로지스틱 자체의 직접적 실증 근거는 없다고 명시**해야 한다.
실험 2가 바로 이 때문에 필요하다는 논리로 연결한다.

### E3. 개별 참여자에게 차등 배분하는 것의 공정성·전략적 문제

**전략적 문제에 대한 직접 근거를 확보했다. 공정성 실증 근거는 확보하지 못했다.**

- **Munro (2023), *Management Science* [19]** 초록 전문: "**Treatment personalization introduces incentives for
  individuals to modify their behavior to obtain a better treatment. Strategic behavior shifts the joint distribution of
  covariates and potential outcomes.** The optimal rule without strategic behavior allocates treatments only to those
  with a positive conditional average treatment effect. **With strategic behavior, we show that the optimal rule can
  involve randomization**, allocating treatments with less than 100% probability even to those who respond positively on
  average to the treatment." [19]
  → 우리 모형에서 참여자는 프로필(품목 구성·요구수량·제안가격·공급가능량)을 비전략적으로 낸다.
    참여자가 `c_i`를 높이도록 프로필을 조작할 수 있으면 최적 배분이 달라지고
    **무작위화가 최적에 포함될 수 있다**는 것이 [19]의 결과다. 한계 서술에 이 논문을 인용하는 것이 정확하다.
  - 우리 모형의 특수한 취약점(**이것은 우리 추론이며 [19]의 주장이 아니다**): `c_i = V(s) − V(s에서 i 제외)`는
    "대체 공급자가 적은 참여자"에게 큰 값을 준다. 따라서 공급자가 **의도적으로 품목 다양성을 줄이거나 공급가능량을 조절해**
    희소성을 만들 유인이 생긴다. 이 유인은 [19]가 말하는 공변량 조작과 구조적으로 같다.
- **정태 차등 수익배분의 설계·정당화**: Bhargava 외 (2022) [A14]가 규모별 차등 수익배분을 다룬다.
  우리 차등은 규모 기준이 아니라 **한계 유지가치 기준**이므로 차등의 근거가 다르다는 점을 밝히면 된다.
- **공정성 대 유지의 관계는 문헌에서 결론이 갈린다**: Kishimoto 외 [10]는 "fairness in itself is not the ultimate
  objective for many platforms, as users do not suddenly reward the platform simply because exposure is equalized"라고
  논증한다 [10]. 반대로 Chen & Xu 2026 프리프린트 [A17]는 공정 배분을 장기 이익에 대한 투자로 본다.
  → 우리 논문은 공정성을 목적에 넣지 않으므로 "유지 최적화가 결과적으로 배분 불균등을 낳을 수 있고, 그 후생·공정성 평가는
    범위 밖"이라고 명시하는 것이 안전하다. 양쪽 입장을 모두 인용할 수 있다.
- **확보하지 못한 것**: "차등 수수료의 지각된 공정성"에 대한 심사지 실증. 웹 검색이 거의 전부 마켓플레이스 SEO 블로그를
  반환했다(쿼리: `fairness differential commission rates marketplace sellers perceived fairness price discrimination
  platform backlash operations`). **학술 출처를 특정하지 못했으므로 인용하지 않는다.**

### E5. 배분 규칙을 고정 비율로 두는 것이 실무 표준인가 (규칙 기반 비교군의 정당화)

**그렇다. 심사지 논문에서 직접 확인했다.**

- **Alaei, Makhdoumi, Malekian, Pekec (2022), *Management Science* [20]** 저자 원고 초록 직접 인용:
  "compensates content providers (artists) through a **revenue-sharing allocation rule**. ... We study **two primary
  revenue allocation rules used by market-leading music streaming platforms** -- pro-rata and user-centric. With pro-rata,
  artists are paid **proportionally** to their share in the overall streaming volume, while with user-centric each user's
  subscription fee is divided **proportionally** among artists based on the consumption of that user." [20]
  → 시장 선도 플랫폼이 **비례(고정 공식) 배분 규칙**을 쓴다는 진술이 심사지에 있다. 우리 규칙 기반 비교군(고정 비율)의
    실무 표준성 근거로 인용할 수 있다.
- **계약이론 쪽**: Cachon & Lariviere (2005) [A15]의 수익공유 계약은 정의 자체가 매출의 **고정 비율** 배분이다.
- **주의**: [20]은 두 규칙을 비교하고 **임의 규칙 클래스에서의 최적 규칙까지** 다룬다 [20].
  따라서 "고정 비율이 실무 표준"은 인용할 수 있지만, "고정 비율이 최적이라고 여겨진다"고 쓰면 [20]에 의해 반박된다.
- **부수 이득**: [20]은 플랫폼의 아티스트 포트폴리오 선택이 NP-complete이고 Knapsack으로 환원된다고 보인다 [20].
  "고정 배분 규칙 + 참여자 집합 선택"만으로도 조합적으로 어렵다는 근거이므로 우리 MILP의 난이도 주장을 뒷받침한다.
- **확보하지 못한 것**: 물류·화물중개(freight brokerage)의 마진 배분이 고정 비율인지에 대한 학술 근거.
  검색이 업계 블로그만 반환했다. **인용하지 않는다.**

---

## 6. 우리 갭 주장에 미치는 영향 (G1 문제 설정 / G4 반응함수 강건성)

### G1 (개별 잉여 몫 결정 + 확률적 재참여 + 참여자 집합 전이 + 다품목 묶음 + 학습 VFA를 MILP에)

**약화하는 증거 (중요도 순)**
1. **[10] Kishimoto 외 2026 (ICLR 2026 표기)** — "양면 매칭 플랫폼에서 유지 최대화"라는 문제 설정을
   **먼저 공식적으로 선언**했다 [10]. G1의 "문제 설정 자체가 새롭다"는 주장을 가장 강하게 약화한다.
   반드시 인용하고 차이(금전 몫·다품목 묶음·MILP·V(s) 학습)를 명시해야 한다.
2. **[5] Klein & Kolb 2015 (Omega)** — 유지 반응이 있는 용량배분 MDP에서 "고객의 customer equity 기여분"을 도입했다 [5].
   우리 `c_i`의 개념적 신규성을 제거한다.
3. **[3] Ovchinnikov 외 2014 (MS)** — VIC로 "동적계획 안의 한계 개체 가치"를 이미 정식화하고, 용량 제약이 있으면
   그것이 CLV와 달라진다는 결과까지 냈다 [3]. 우리 `c_i`의 개념과 필요성 논거를 둘 다 선점한다.
4. **[4] Pfeifer & Ovchinnikov 2011** — "관계가 독립일 때만 CLV = WTS" [4].
   우리가 D 주제에서 쓰려던 "상호의존이면 개별 가치와 한계 가치가 다르다"는 논증의 선행 진술이다.
5. **[8][9] DiDi 계보** — "학습된 V의 차분을 개체별 계수로 조합최적화에 넣는다"는 구조가 이미 표준이며 실배포되었다 [8][9].
   우리 파이프라인의 골격이 신규가 아니다.
6. **[7] Afeche 외 2017 (M&SOM)** — "다른 유형에 의존하는 개체 가치(policy-dependent value)"를 이미 도출했다 [7].
7. **[20] Alaei 외 2022 (MS)** — 수익배분 규칙 + 참여 유지 + 조합 최적화(Knapsack, PTAS)를 한 편에 담았다 [20].
   "배분과 조합 최적화는 결합된 적이 없다"는 식의 주장은 불가능하다.
8. **[25][26][29]** — CLV·개체 가치를 0-1 대상 선택, 수리계획, POMDP에 넣은 선례 [25][26][29].
   "CLV 문헌의 결정변수는 연속량뿐"이라는 방어를 쓸 수 없다.

**강화하는 증거**
1. **금전 몫 + 매칭의 동시 결정이 없다.** [5][6][7]은 용량·서비스 품질(비금전), [8][9]는 매칭만, [10]은 노출만,
   [20]은 정태 규칙 선택, [3][4][22][27][28]은 연속 지출액이다.
   위 어느 것도 "개별 참여자에게 갈 금전 몫을 매칭과 같은 수학적 모형에서 결정"하지 않는다.
2. **양면 모두에 유지 가치를 두는 최적화 모형이 없다.** [10]이 양면 유지 이득을 함께 보지만
   배분 대상이 노출이고 금전·용량·묶음이 없다 [10].
3. **all-or-nothing 다품종 묶음이 없다.** 이번 조사에서 유지·CLV 문헌 중 묶음 완전충족 제약을 둔 것을 찾지 못했다.
   → G1을 **"다품목 all-or-nothing 묶음이 있는 5PL 미들마일 매칭에서 개별 금전 잉여 몫과 품목별 배정을 함께 정하는 동적 모형"**
     으로 좁히면 남는다. 그보다 넓게 주장하면 [3][5][10]에 걸린다.
4. **오지정 강건성 실험을 붙인 유지 최적화 연구가 없다.** [10][11]은 학습된 유지 곡선을 쓰지만 반응 오지정을 평가하지 않는다.

**권고**: G1의 문장을 "개별 유지 가치를 최적화 계수로 쓰는 것"에서
**"금전 잉여 몫 + 묶음 제약 + 양면 + 매칭을 하나의 MILP로"** 로 다시 써야 한다.
`c_i` 개념 자체는 선행연구 용어([3]의 VIC, [5]의 contribution to customer equity)로 **소개**하는 편이 방어에도 유리하다.

### G4 (반응함수 오지정 강건성이 다루어지지 않았다)

**약화하는 증거**
1. **[13] Besbes, Phillips, Zeevi 2010 (M&SOM)** — 오지정을 **성과 기준으로 검정하는 절차**가 이미 있다 [13].
   "오지정 강건성이 다루어지지 않았다"는 말은 성립하지 않는다. 특히 "로짓은 통계적으로 기각되어도 의사결정 성능으로는
   통과할 수 있다"는 결과가 우리 실험 2의 결과 해석을 미리 제약한다.
2. **[12] Nambiar 외 2019 (MS)** — 오지정 하 학습·최적화의 **알고리즘과 이론 보증**이 있다 [12].
3. **[14] Cooper 외 2006 (OR)** — 오지정과 최적화 되먹임의 자기강화 편향이 20년 전에 정식화되었다 [14](medium).
4. **[15][16][17][18]** — 강건·분포적 강건 MDP 계열이 존재한다(서지 확인).

**강화하는 증거**
1. **"참여 유지 반응"에 특화된 오지정 연구를 여전히 찾지 못했다.** [12][13][14]는 모두 **수요·구매 반응**이다.
   "배분받은 금전에서 다음 기간 재참여로"의 반응 오지정을 다룬 운영 문헌은 1차(A3)와 2차 모두에서 발견하지 못했다.
2. **생존 선택(survivorship) 편향이 결합된 사례가 없다.** 우리 설정은 오지정이 참여자 구성까지 바꾸므로
   [12]의 가격-예측오차 상관보다 경로가 하나 더 있다. 이 결합을 다룬 문헌을 찾지 못했다.

**권고**: G4를 "오지정 강건성 문헌이 없다"에서 **"수요 반응 오지정 문헌([12][13][14])은 두껍지만, 결정이 참여자 집합을
바꾸는 설정(반응 오지정과 생존 선택이 결합된 경우)에 그 방법론이 적용된 사례가 없다"** 로 다시 써야 한다.
동시에 실험 2를 "robust"로 부르지 않고 "sensitivity analysis"로 부르고, 4절의 빠진 항목(특히 되먹임 편향 측정)을
한계로 적거나 추가해야 한다.

---

## 7. 미해결 · 확인 불가 (도서관 접속으로 확인할 질문)

1. **Klein & Kolb (2015) Omega [5]** — 초록만 확보(ScienceDirect 403). 확인 질문:
   (a) "customer's contribution to customer equity"를 **최적화의 계수로 쓰는가**, 아니면 최적해의 사후 해석 지표로만 쓰는가?
       (전자면 우리 `c_i`의 신규성이 거의 사라진다. **최우선 확인 1순위.**)
   (b) 재구매 반응의 함수형(로짓인가 마르코프 전이확률 표인가)
   (c) 상태공간 규모와 해법(정확한 가치반복인지 근사인지)
   - OpenAlex 초록 색인 프로브 결과(초록 본문은 읽지 않았고 단어 존재만 확인):
     `markov`=있음, `opportunity cost`=있음, `customer lifetime value`=있음, `heterogeneous segments`=있음,
     `repurchase`=있음, `accept reject`=있음 / `integer`=없음, `bid price`=없음, `dynamic programming`=없음,
     `logistic`=없음, `retention probability`=없음. (대조: `zebra`=없음으로 프로브가 작동함을 확인)
2. **Ovchinnikov, Boulu-Reshef, Pfeifer (2014) MS [3]** — 초록 전문만. 확인 질문:
   (a) 유지확률의 **함수형**(지출액에 대해 오목인가, 로지스틱인가, 두 점 보간인가)
   (b) VIC를 최적화 안에서 계수로 쓰는가, 사후 지표로만 쓰는가?
   (c) VIC의 정의가 "고객 1명 추가"인지 "특정 고객 제거"인지(우리 leave-one-out과 기준점이 같은지)
3. **Buhl, Klein, Kolb, Landherr (2011) CR2M [6]** — 초록 미확보(Springer 리다이렉트). 확인 질문:
   용량배분 결정이 이산인가, CLV가 목적함수 계수로 들어가는가, 반응 함수형은 무엇인가.
4. **Afeche, Araghi, Baron (2017) M&SOM [7]** — 초록이 중간에서 절단되었다. 확인 질문:
   "policy-dependent value"의 정확한 정의와 그것이 leave-one-out 차분과 어떤 관계인가.
5. **Kishimoto 외 (2026) [10]** — ICLR 2026 게재를 **arXiv 주석(저자 신고)으로만 확인했다.** OpenReview에서 직접 확인 필요.
   추가 확인 질문: 다기간 유지 시뮬레이션에서 참여자 집합 전이를 어떻게 처리하는지, 오목성 가정이 깨지면 어떻게 되는지.
6. **Cooper, Homem-de-Mello, Kleywegt (2006) [14]** — 초록을 직접 확보하지 못했다
   (저자 PDF 링크 `http://www.menet.umn.edu/~billcoop/chk06.pdf`가 TLS 인증서 불일치로 실패).
   4절의 3번 항목은 현재 **검색요약 근거(medium)** 다. 인용 전 원문 확인 필요.
7. **Reinartz, Thomas, Kumar (2005) JM [22]** — 초록 미확보. 확인 질문: 1차 조건의 정확한 형태(한계 유지지출 = CLV인가)와
   반응 함수형. E1의 "FOC 선례"를 이 논문으로 인용할 수 있는지 판정하려면 필요하다.
8. **Blattberg & Deighton (1996)** — "한계 획득비용 = 한계 유지비용 = CLV"라는 최적성 조건이 이 논문에 있다는 진술을
   **검색 요약에서만 보았고 원문·URL을 확정하지 못했다.** E1의 핵심 선행일 수 있으므로 확인 필요.
   (확인 실패 경로: `https://link.springer.com/content/pdf/10.1057/palgrave.jt.5740142.pdf` → Springer 인증 리다이렉트)
9. **"Optimal revenue sharing in platform markets: a Stackelberg model"**(*Journal of Revenue and Pricing Management*,
   https://doi.org/10.1057/s41272-018-00180-4) — 초록 미확보. E1의 "수수료율 FOC" 선례 후보.
   확인 질문: 참여 반응이 1차 조건에 들어가는가, 동적인가.
10. **KDD 2022 "Optimizing Long-Term Efficiency and Fairness in Ride-Hailing via Joint Order Dispatching and Driver
    Repositioning"**, **"A vehicle value based ride-hailing order matching and dispatching algorithm"**(2024) —
    초록 미확보. E8의 유사 연구 후보. 내용 단정하지 않았다.
11. **차등 수수료의 지각된 공정성 실증** — 학술 출처를 특정하지 못했다(5절 E3). 검색이 SEO 블로그로 오염되어
    `site:` 제한 재검색이 필요하다.
12. **화물중개·물류에서 고정 비율 마진 배분의 실무 근거** — 학술 출처 없음(5절 E5).
13. **저널 사분위를 하나도 직접 확인하지 않았다.** 모두 "사분위 미확인"으로 표기했다.
14. **결과 0건·소수 건으로 확인한 쿼리(공백 신호로 남김)**
    - OpenAlex `title_and_abstract.search:customer lifetime value integer programming` → count=3, 전부 무관
      (공급망 네트워크 설계, 태양전지, 전력저장). 즉 **제목·초록 수준에서 "CLV + 정수계획"이라는 조합어는 거의 없다.**
      단 [25][26]처럼 다른 어휘로 존재하므로 "없다"로 읽으면 안 된다.
    - OpenAlex `title_and_abstract.search:customer lifetime value capacity control revenue management` → count=7,
      관련 있는 것은 [6] 하나뿐.
    - OpenAlex `title_and_abstract.search:customer equity capacity allocation markov` → **count=1, 곧 [5] 하나뿐.**
    - OpenAlex `title_and_abstract.search:retention matching platform long-term` → count=63이지만 전부 무관
      (신경과학·재료공학 등). 전문 검색 잡음이 크므로 이 결과는 공백 신호로 쓰지 않는다.

---

## 8. Coverage Status

**직접 확인한 것**
- Lewis 2005의 OpenAlex id(W2159517094)와 Pfeifer & Carraway 2000(W2077806964)을 찾아
  **피인용 81편·347편을 긁어 최적화 관련 항목을 필터링**했고, 그 결과에서
  [3][5][7][23][24][25][26][27][28][29]를 식별했다.
- 초록 전문을 직접 확보해 읽음:
  [1][3][4][5][7][8][9][10(본문 포함)][11][12][13][19][20][21][23][24][25][26][27][28][29].
- 저자 호스팅·리포지터리 PDF에서 pypdf로 초록을 직접 추출: [13](Wharton 사본), [20](U of T TSpace 원고).
- RePEc/IDEAS에서 출판사 제공 초록을 직접 확보: [3][4][5].
- arXiv API로 [10][11]의 저자·주석·초록을 확정했다([11]은 A 문서 [A34]의 저자 미확인 항목을 해결).
- Crossref로 [5][6][7][9][20][22][23][24][25][26][27][28][29] 및 [13][14][15][16][17]의
  저자·게재지·권호·연도를 검증했다.
- arXiv 2602.15752의 실재를 확인했고 **본문에서 Eq.9·14·15, Assumption 1, 4.1절을 직접 읽었다.**

**불확실한 것**
- [14](Cooper 외 2006)는 **검색요약 근거**다. 원문 확인 전까지 medium.
- [15][16][17][18]은 **서지만** 확인했다. robust MDP의 일반적 설명은 검색요약 근거이며 medium.
- [2][6][22]는 초록을 확보하지 못했거나 절단되었다.
- [5]가 "contribution to customer equity"를 최적화 계수로 쓰는지 여부가
  **E6 판정 강도를 좌우하는 가장 큰 불확실성**이다.
- 저널 사분위 미확인.
- [10]의 ICLR 2026 게재는 저자 신고(arXiv comment) 근거다.

**완료하지 못한 것**
- ScienceDirect·Springer·INFORMS·ACM 페이월(403·리다이렉트)로 [5][6][7][14][22]의 본문을 읽지 못했다.
- E3의 "차등 수수료 공정성 실증"과 E5의 "물류 업계 고정 비율" 학술 근거를 확보하지 못했다.
- E1의 Blattberg & Deighton 원전 확인 실패.
- D 주제(묶음 보완성)와의 연결 문헌([4]가 그 고리에 해당)은 E 조사 범위 밖이므로 지적만 하고 추적하지 않았다.

---

## 9. Sources

1. Lewis, M. (2005) *Research Note: A Dynamic Programming Approach to Customer Relationship Pricing*, Management Science 51(6):986-994 — https://doi.org/10.1287/mnsc.1050.0373
2. Pfeifer, P. E., Carraway, R. L. (2000) *Modeling customer relationships as Markov chains*, Journal of Interactive Marketing 14(2):43-55 — https://doi.org/10.1002/(sici)1520-6653(200021)14:2<43::aid-dir4>3.0.co;2-h
3. Ovchinnikov, A., Boulu-Reshef, B., Pfeifer, P. E. (2014) *Balancing Acquisition and Retention Spending for Firms with Limited Capacity*, Management Science 60(8):2002-2019 — https://doi.org/10.1287/mnsc.2013.1842 (초록: https://ideas.repec.org/a/inm/ormnsc/v60y2014i8p2002-2019.html)
4. Pfeifer, P. E., Ovchinnikov, A. (2011) *A Note on Willingness to Spend and Customer Lifetime Value for Firms with Limited Capacity*, Journal of Interactive Marketing 25(3):178-189 — https://doi.org/10.1016/j.intmar.2011.02.003 (초록: https://ideas.repec.org/a/eee/joinma/v25y2011i3p178-189.html)
5. Klein, R., Kolb, J. (2015) *Maximizing customer equity subject to capacity constraints*, Omega 55:111-125 — https://doi.org/10.1016/j.omega.2015.02.008 (초록: https://ideas.repec.org/a/eee/jomega/v55y2015icp111-125.html)
6. Buhl, H. U., Klein, R., Kolb, J., Landherr, A. (2011) *CR2M - an approach for capacity control considering long-term effects on the value of a customer for the company*, Journal of Management Control 22:187-204 — https://doi.org/10.1007/s00187-011-0133-8
7. Afeche, P., Araghi, M., Baron, O. (2017) *Customer Acquisition, Retention, and Service Access Quality: Optimal Advertising, Capacity Level, and Capacity Allocation*, M&SOM 19:674-691 — https://doi.org/10.1287/msom.2017.0635
8. Xu, Z., Li, Z., Guan, Q., Zhang, D., Li, Q., Nan, J., Liu, C., Bian, W., Ye, J. (2018) *Large-Scale Order Dispatch in On-Demand Ride-Hailing Platforms: A Learning and Planning Approach*, KDD 2018 — https://doi.org/10.1145/3219819.3219824
9. Qin, Z. (T.), Tang, X., Jiao, Y., Zhang, F., Xu, Z., Zhu, H., Ye, J. (2020) *Ride-Hailing Order Dispatching at DiDi via Reinforcement Learning*, INFORMS Journal on Applied Analytics 50:272-286 — https://doi.org/10.1287/inte.2020.1047
10. Kishimoto, R., Takehi, R., Tanaka, K., Nomura, M., Togashi, R., Tomita, Y., Saito, Y. (2026) *Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching*, arXiv:2602.15752v3 (주석: Published as a conference paper at ICLR 2026) — https://arxiv.org/abs/2602.15752
11. Ute, T., Ichimura, C., Saito, Y. (2026) *Not All Matches Are Equally Valuable: An Online Experiment of Retention-Focused Recommendation in a Job-Matching Platform*, arXiv:2609.01652 (RecSys in HR 2026 워크숍) — https://arxiv.org/abs/2609.01652
12. Nambiar, M., Simchi-Levi, D., Wang, H. (2019) *Dynamic Learning and Pricing with Model Misspecification*, Management Science 65(11):4980-5000 — https://doi.org/10.1287/mnsc.2018.3194
13. Besbes, O., Phillips, R., Zeevi, A. (2010) *Testing the Validity of a Demand Model: An Operations Perspective*, M&SOM 12(1) — https://doi.org/10.1287/msom.1090.0264 (저자 사본: https://faculty.wharton.upenn.edu/wp-content/uploads/2009/03/Testing_validity_ops_persp_ob_rp_az_posted.pdf)
14. Cooper, W. L., Homem-de-Mello, T., Kleywegt, A. J. (2006) *Models of the Spiral-Down Effect in Revenue Management*, Operations Research 54(5):968-987 — https://doi.org/10.1287/opre.1060.0304
15. Nilim, A., El Ghaoui, L. (2005) *Robust Control of Markov Decision Processes with Uncertain Transition Matrices*, Operations Research — https://doi.org/10.1287/opre.1050.0216
16. Iyengar, G. N. (2005) *Robust Dynamic Programming*, Mathematics of Operations Research — https://doi.org/10.1287/moor.1040.0129
17. Wiesemann, W., Kuhn, D., Rustem, B. (2013) *Robust Markov Decision Processes*, Mathematics of Operations Research — https://doi.org/10.1287/moor.1120.0566
18. Xu, H., Mannor, S. (2010) *Distributionally Robust Markov Decision Processes*, NeurIPS 2010 — https://papers.nips.cc/paper/3927-distributionally-robust-markov-decision-processes
19. Munro, E. (2023) *Treatment Allocation with Strategic Agents*, Management Science — https://doi.org/10.1287/mnsc.2022.01629 (프리프린트: https://arxiv.org/abs/2011.06528)
20. Alaei, S., Makhdoumi, A., Malekian, A., Pekec, S. (2022) *Revenue-Sharing Allocation Strategies for Two-Sided Media Platforms: Pro-Rata vs. User-Centric*, Management Science 68(12):8699-8721 — https://doi.org/10.1287/mnsc.2022.4307 (저자 원고: https://utoronto.scholaris.ca/bitstreams/54f9aed0-dfe9-4a62-b757-8b14bc41cece/download)
21. Li, H., Shen, Q., Bart, Y. (2021) *Dynamic Resource Allocation on Multi-Category Two-Sided Platforms*, Management Science — https://doi.org/10.1287/mnsc.2020.3586
22. Reinartz, W., Thomas, J. S., Kumar, V. (2005) *Balancing Acquisition and Retention Resources to Maximize Customer Profitability*, Journal of Marketing 69(1):63-79 — https://doi.org/10.1509/jmkg.69.1.63.55511
23. Aflaki, S., Popescu, I. (2014) *Managing Retention in Service Relationships*, Management Science 60(2):415-433 — https://doi.org/10.1287/mnsc.2013.1775
24. Lemmens, A., Gupta, S. (2020) *Managing Churn to Maximize Profits*, Marketing Science 39(5):956-973 — https://doi.org/10.1287/mksc.2020.1229
25. Bhaskar, T., Sundararajan, R., Krishnan, P. G. (2009) *A fuzzy mathematical programming approach for cross-sell optimization in retail banking*, Journal of the Operational Research Society 60:717-727 — https://doi.org/10.1057/palgrave.jors.2602609
26. Sundararajan, R., Bhaskar, T., Sarkar, A., Dasaratha, S., Bal, D., Marasanapalle, J. K., Zmudzka, B., Bak, K. (2011) *Marketing Optimization in Retail Banking*, Interfaces 41:485-505 — https://doi.org/10.1287/inte.1110.0597
27. Ching, W.-K., Ng, M. K., Wong, K.-K., Altman, E. (2004) *Customer lifetime value: stochastic optimization approach*, Journal of the Operational Research Society 55:860-868 — https://doi.org/10.1057/palgrave.jors.2601755
28. Li, S., Sun, B., Montgomery, A. L. (2011) *Cross-Selling the Right Product to the Right Customer at the Right Time*, Journal of Marketing Research 48(4):683-700 — https://doi.org/10.1509/jmkr.48.4.683
29. Montoya, R., Netzer, O., Jedidi, K. (2010) *Dynamic Allocation of Pharmaceutical Detailing and Sampling for Long-Term Profitability*, Marketing Science 29(5):909-924 — https://doi.org/10.1287/mksc.1100.0570
