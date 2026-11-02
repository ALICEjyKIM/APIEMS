# 주제 D 증거 수집: 묶음 보완성과 가치함수의 초모듈성

조사일: 2026-09-28 · 담당: researcher 서브에이전트 (본문을 메시지로 반환 → 부모가 저장)
범위 문서: `ref/literature-review.md` 주제 D(D1~D5) + "2차 조사 필수 추가"의 D6·D7
읽은 것: `CLAUDE.md`(모델 가정), `ref/literature-review.md`, `ref/_review/..._research_B.md` §2 B4(중복 회피),
`src/vfa/coef.py`, `src/match/milp_build.py`

---

## 0. 결론 먼저 (가장 중요한 산출물)

**초안의 논증 사슬 "묶음 보완성 → 가치함수 초모듈성 → 모듈러 근사의 체계적 과소평가"는 성립하지 않는다.**
세 번째 고리의 부호가 문헌과 반대다. 감사(reviewer)의 반론 1은 문헌으로 확증되고, 반론 2도 문헌으로 뒷받침된다.

핵심 근거 (Iyer–Jegelka–Bilmes ICML 2013 [1]의 supergradient 모듈러 상한 + Heo 외 2026 [3]의 상호작용 항 부호):

- **열모듈(대체재) 성분**이 leave-one-out 가산 대리값의 **다수 이탈 손실 과소평가**를 낳는다.
- **초모듈(보완재) 성분**이 **과대평가**를 낳는다.
- 즉 우리가 관측했다고 주장하려던 "과소평가"는 **보완성이 아니라 대체성(중복성)이 만드는 방향**이다.

우리 시장은 n_alt = 2로 같은 품목 두 공급자가 대체재, 다른 품목 공급자가 보완재이므로 가치함수는
**어느 쪽도 아니며**, 문헌(Bian 외 ICML 2017 [9]의 submodularity ratio γ와 generalized curvature α;
DS 분해 [10]; MPH 계층 [8])은 이런 혼재 함수를 다루는 표준 언어를 제공하지만
**모듈러 근사 편향의 순 부호를 결정해주는 형식 결과는 없다**. 부호는 측정으로만 정해진다.

추가로, **초모듈성이 선형 근사 실패를 함의한다는 수사는 OR 문헌에서 정면으로 반박된다**:
Iancu–Sharma–Sviridenko (Operations Research 2013) [17]은 DP 가치함수의 **초모듈성**이
(볼록성·격자 구조와 결합해) **아핀 결정규칙의 최적성을 보장하는 조건**이라고 보고한다.
그리고 Baldwin–Klemperer (Econometrica 2019) [11]은 "통념과 반대로 **순수 보완재** 부류에서
균형(선형 가격) 존재가 **순수 대체재**보다 더 많은 부류에서 보장된다"고 명시한다.

---

## 1. 증거 표

| # | 제목 | 저자·연도 | 게재지·등급 | 직접 확인한 내용 | 우리 논증과의 관계 | URL | 읽은 수준 |
|---|---|---|---|---|---|---|---|
| 1 | Fast Semidifferential-based Submodular Function Optimization | Iyer, Jegelka, Bilmes 2013 | ICML 2013 (ML 최상위) | supergradient 정의: `ĝ_Y(j) = f(j\|V∖{j})` (j∈Y), `ǧ_Y(j) = f(j\|Y∖{j})`. 이들이 정의하는 **모듈러 상한** `m_{g_Y}(X) = f(Y) + g_Y(X) − g_Y(Y) ≥ f(X)`, 그리고 `m_{g_Y}(Y) = f(Y)`로 **현재 집합에서만 tight** | **부호 판정의 핵심 근거.** 우리 c_i는 정확히 ĝ. **열모듈(대체) f에서 가산 대리값은 축소 집합의 가치를 과대평가 = 다수 이탈 손실을 과소평가** | http://proceedings.mlr.press/v28/iyer13.pdf | 전문 §3–§5 직접 추출 |
| 2 | Curvature and Optimal Algorithms for Learning and Minimizing Submodular Functions | Iyer, Jegelka, Bilmes 2013 | NeurIPS 2013 | 총 curvature `κ_f = 1 − min_j f(j\|V∖j)/f(j)`. **Lemma 3.1**: `f(X) ≤ Σ_{j∈X} f(j) ≤ [ \|X\| / (1 + (\|X\|−1)(1−κ_f(X))) ] · f(X)`. **Lemma 3.2**: 임의의 모듈러 상한에 대해 이 계수가 tight | **D3 직답.** 모듈러 근사 오차 크기가 curvature 하나로 규정되고 그 경계가 tight. κ=0이면 오차 0, κ=1이면 계수 \|X\| | https://arxiv.org/abs/1311.2110 | 전문 §2.1·§3 직접 추출 |
| 3 | Interaction-Aware Influence Functions for Group Attribution | Heo, Yun, Choi, Hwang, Ok, Kim 2026-05-15 | arXiv 프리프린트 (**미심사**) | 2차 전개 추정량 = 개별 영향 합 + 쌍 상호작용 κ(a,b). "the interaction term ... is non-negative whenever H_f is positive semidefinite". Proposition 1: 로지스틱 회귀에서 κ(a,b) = (σ_a−y_a)(σ_b−y_b)·⟨x_a,x_b⟩_M → **같은 클래스(중복적) 쌍에서 양, 다른 클래스(보완적) 쌍에서 음**. "the additive approximation underestimates the exact retraining effect on groups of **similar** examples ... leading to systematic underestimation" | **D6·D7의 핵심.** 가산 1차 근사 편향의 **방향이 제거 집합의 중복성/보완성 부호로 결정**된다는 형식 진술. 우리 주장의 부호를 뒤집는다 | https://arxiv.org/abs/2605.15675 | 전문 §3 + Prop.1 직접 추출 |
| 4 | On the Accuracy of Influence Functions for Measuring Group Effects | Koh, Ang, Teo, Liang 2019 | NeurIPS 2019 | 초록: 1차 Taylor 근사로 **큰 그룹** 제거 효과를 예측할 때 "the predicted effect ... correlates surprisingly well with its actual effect, **even if the absolute and relative errors are large**". 강한 상관은 "only under certain settings"에서만 | **D7 표준 선행.** 다수 동시 이탈에 1차 차분을 쓸 때의 오차를 정면으로 다룬다. **초록에 편향 방향 없음 → 부호 주장에 인용 금지** | https://arxiv.org/abs/1905.13289 | 초록 직접 확인 |
| 5 | Most Influential Subset Selection: Challenges, Promises, and Beyond | Hu, Hu, Zhao, Ma 2024 | **NeurIPS 2024** | "influence-based greedy heuristics ... can **provably fail even in linear regression**, with failure modes including the errors of influence function and the **non-additive structure of the collective influence**" | **D7 최강 인용.** "개별 1차 영향의 합"이 집합 효과를 못 잡는 실패가 **증명 가능**하다는 심사 통과 결과 | https://arxiv.org/abs/2409.18153 | 초록·요약 수준 (전문 미독) |
| 6 | On Second-Order Group Influence Functions for Black-Box Predictions | Basu, You, Feizi 2020 | ICML 2020 | 그룹 제거 시 파라미터 변화가 커서 "the first-order approximation can be loose"고, 2차 group influence function을 제안 | D7 보강 | https://arxiv.org/abs/1911.00418 | 초록 확인 |
| 7 | Capturing Complementarity in Set Functions by Going Beyond Submodularity/Subadditivity | Chen, Teng, Zhang 2019 | ITCS 2019 (이론 CS) | 초록: supermodular width·superadditive width가 "characterize the gap of monotone set functions from being submodular and subadditive". SMW 계층이 Feige–Izsak supermodular degree보다 strictly 표현력이 크다 | **D6.** "보완성의 정도"를 재는 형식 틀. **모듈러 근사 편향 부호는 다루지 않음** | https://arxiv.org/abs/1805.04436 (DOI: 10.4230/LIPIcs.ITCS.2019.24) | 초록 + 표지 확인 |
| 8 | A Unifying Hierarchy of Valuations with Complements and Substitutes | Feige, Feldman, Immorlica, Izsak, Lucier, Syrgkanis 2015 | AAAI 2015 | MPH(Maximum over Positive Hypergraphs) 계층. MPH-1이 monotone submodular과 XOS 포함, MPH-m이 모든 monotone 함수 포함 | **D6.** 대체·보완 혼재 가치함수의 표준 계층이 **이미 존재**. "순수 초모듈"이라 부를 수 없다는 근거 | https://arxiv.org/abs/1408.1211 (AAAI: https://ojs.aaai.org/index.php/AAAI/article/view/9314) | 초록 + 표지 확인 |
| 9 | Guarantees for Greedy Maximization of Non-submodular Functions with Applications | Bian, Buhmann, Krause, Tschiatschek 2017 | ICML 2017 | **Def.1** submodularity ratio γ (F가 열모듈 ⟺ γ=1), **Def.2** generalized curvature α (F가 **초모듈 ⟺ α=0**), 둘 다 [0,1] | **D6 최선의 형식 어휘.** 대체·보완 혼재 함수를 **두 모수로 양방향 정량화**. 단 모듈러 근사 편향 방향은 주지 않는다 | http://proceedings.mlr.press/v70/bian17a/bian17a.pdf | 전문 §2 정의 직접 추출 |
| 10 | Algorithms for Approximate Minimization of the Difference Between Submodular Functions | Iyer, Bilmes 2012 | UAI 2012 | 표지 확인. 임의의 집합함수를 두 열모듈 함수의 차(DS)로 쓸 수 있다는 결과를 **Narasimhan–Bilmes (2005)** 에 귀속 (이 귀속은 **검색 결과 수준**, 본문 정리 미독) | **반론 2의 형식 근거.** "대체 성분 − 보완 성분" 분해가 항상 가능 → **순 부호는 정의상 미결** | https://arxiv.org/abs/1207.0560 | 표지 확인, 본문 미독 |
| 11 | Understanding Preferences: "Demand Types", and the Existence of Equilibrium With Indivisibilities | Baldwin, Klemperer 2019 | **Econometrica** | 초록 원문: **"Contrary to popular belief, equilibrium is guaranteed for more classes of purely-complements than of purely-substitutes, preferences."** | **D4에 대한 반대 증거.** "보완성 ⇒ 선형(항목) 가격 실패"는 틀린 일반화 | https://doi.org/10.3982/ecta13693 | 초록 원문 확인(OpenAlex), 전문 미독 |
| 12 | Competitive Equilibrium in an Exchange Economy with Indivisibilities | Bikhchandani, Mamer 1997 | J. Economic Theory 74(2):385–413 | (**2차 출처 수준**) 선형 시장청산가격 존재의 **필요충분조건**: **정수계획 최적값 = LP 완화 최적값**(정수성 격차 0) | **D4의 형식 결과.** 실패의 뿌리는 "가치함수의 초모듈성"이 아니라 **배분 LP의 정수성 격차**다 → 우리 모듈러 근사 실패와 **같은 뿌리가 아니다** | https://www.sciencedirect.com/science/article/abs/pii/S0022053196922693 | **전문·초록 접근 실패(403)**, 검색 기반 2차 확인 |
| 13 | Walrasian Equilibrium with Gross Substitutes | Gul, Stacchetti 1999 | J. Economic Theory (피인용 651) | **서지만 확인.** 초록·전문 미확보 | D4 고전. maximal domain 정리의 정확한 진술 **미확인** | https://doi.org/10.1006/jeth.1999.2531 | 서지만 |
| 14 | On the maximal domain theorem: A corrigendum to "Walrasian equilibrium with gross substitutes" | Yang 2017 | J. Economic Theory | **서지만 확인.** 원 정리에 **정정이 있다는 사실** | D4 인용 시 경고 신호 — 원 정리를 그대로 쓰면 안 된다 | https://doi.org/10.1016/j.jet.2017.09.003 | 서지만 |
| 15 | Introduction to Combinatorial Auctions | Cramton, Shoham, Steinberg 2006 | MIT Press 서장 (2차) | "Items are complements when a set of items has greater utility than the sum of the utilities for the individual items". SAA는 "not a combinatorial auction, because bids ... are placed for individual items, rather than packages" | D4의 **서술적** 근거는 되지만 **선형 가격 비존재의 형식 정리는 이 문서에 없다**(키워드 검색으로 확인) | http://www.cramton.umd.edu/papers2005-2009/cramton-shoham-steinberg-combinatorial-auctions-introduction.pdf | 전문 키워드 검색 |
| 16 | Preservation of Supermodularity in Parametric Optimization | Chen, Long, Qi 2021 | **Operations Research** 69(1):1–12 | 초록 원문: 파라메트릭 최적화에서 초모듈성 보존의 필요충분조건(mostly sublattice 등), "**illustrate the use of our results in assemble-to-order systems**" | **D1·D2 최선 인용.** ATO에서 초모듈성을 쓰는 OR Q1 근거. **단 초모듈성이 정의된 격자는 파라미터 격자이고 참여자 부분집합 격자가 아니다** | https://doi.org/10.1287/opre.2020.1992 | 초록 원문 확인 |
| 17 | Supermodularity and Affine Policies in Dynamic Robust Optimization | Iancu, Sharma, Sviridenko 2013 | **Operations Research** 61(4):941–956 | 초록(작업논문판): "a set of unifying conditions (based on the interplay between the **convexity and supermodularity of the DP value functions**, and the lattice structure of the uncertainty sets) that, taken together, **guarantee the optimality of the class of affine decision rules**" | **우리 수사에 대한 반대 증거.** 초모듈성이 **아핀(선형) 규칙 최적성**의 조건으로 쓰인다 | https://doi.org/10.1287/opre.2013.1172 / 작업논문 https://optimization-online.org/wp-content/uploads/2012/06/3497.pdf | 작업논문 초록 직접 추출 |
| 18 | Submodular Function Maximization via the Multilinear Relaxation and Contention Resolution Schemes | Chekuri, Vondrák, Zenklusen 2011/2014 | STOC 2011 / SIAM J. Computing | "We have F(x) = E[f(x̂)] where x̂_i = 1 independently with probability x_i" (다선형 확장 정의). correlation gap = "the worst-case ratio between the multilinear extension F(x) = E[f(x̂)] and the concave closure f⁺(x)" | **우리 MILP 대리값의 정확한 정체를 규정.** `Σ c_i p_i`는 **다선형 확장 F의 p=1에서의 1차 Taylor**다 | https://theory.stanford.edu/~jvondrak/data/contention-res.pdf | 전문 §1·§3 직접 추출 |
| 19 | Price of Correlations in Stochastic Optimization | Agrawal, Ding, Saberi, Ye 2012 | **Operations Research** 60(1):150–162 | 검색 수준: 결합분포를 독립(곱)분포로 대체할 때 손실을 "price of correlations"로 정량화, correlation gap 개념 도입 | D3 보강(OR 저널 근거). **전문 미독 — 구체 상한 수치 인용 금지** | https://doi.org/10.1287/opre.1110.1011 | 서지 + 검색 수준 |
| 20 | Optimal Approximation for Submodular and Supermodular Optimization with Bounded Curvature | Sviridenko, Vondrák, Ward 2015/2017 | SODA 2015 / **Math. of Operations Research** 42(4) | 검색 수준: curvature c가 유계인 **열모듈 최대화·초모듈 최소화**의 최적 근사(1−c/e) | D3에서 **초모듈 쪽** curvature 경계의 존재 근거 | https://arxiv.org/abs/1311.4728 / https://doi.org/10.1287/moor.2016.0842 | 서지 + 검색 수준 |
| 21 | Supermodular Rank: Set Function Decomposition and Optimization | Sonthalia, Seigal, Montúfar 2023/2025 | SIAM J. Math. of Data Science | 초록: supermodular rank = 서로 다른 부분순서에 대한 초모듈 함수 합으로 분해할 때 필요한 최소 항 수 | D6 보강. 혼재 집합함수 분해 이론. **모듈러 근사 편향 부호는 다루지 않음** | https://arxiv.org/abs/2305.14632 / https://doi.org/10.1137/24M1643475 | 초록 확인 |
| 22 | On the Order Fill Rate in a Multi-Item, Base-Stock Inventory System | Song 1998 | **Operations Research** 46(6):831–845 | 검색 수준: 고객 주문이 여러 품목·수량으로 구성되고, **order fill rate**(요청한 모든 품목이 재고에서 충족될 확률)을 성과 지표로 둔다 | **D1.** 우리 all-or-nothing 주문 성립 조건과 같은 지표의 원전 후보. **전문 미독 → 가치함수 구조 주장 인용 금지** | https://doi.org/10.1287/opre.46.6.831 (INFORMS 403) | 서지 + 검색 수준 |
| 23 | Assemble-to-order systems: A review | Atan, Ahmadi, Stegehuis, de Kok, Adan 2017 | **EJOR** (피인용 88) | 서지만 확인. OpenAlex 초록 없음 | D1 리뷰 진입점. **내용 인용 불가** | https://doi.org/10.1016/j.ejor.2017.02.029 | 서지만 |
| 24 | Optimal and near-optimal control of a capacitated assemble-to-order system with component commonality and backordered demands | (저자 미확인) 2024/2025 | **IJPR** | 검색 수준: MDP 정식화, **가치함수의 component-wise convexity**가 부품별 단일 base-stock 임계와 제품별 rationing 임계를 함의 | D1·D2 최신 예. **전문 미독, 저자 미확인 → 인용 전 확인 필요** | https://www.tandfonline.com/doi/full/10.1080/00207543.2024.2443496 | 검색 수준 |
| 25 | Selecting a Portfolio of Suppliers Under Demand and Supply Risks | Federgruen, Yang 2008 | **Operations Research** 56(4):916–936 | 검색 수준 초록: 단일 품목·단일 수요 시즌, 각 공급원에 무작위 수율 | **D5의 가장 가까운 후보지만 적합도 낮다**(단일 품목·정적) | https://doi.org/10.1287/opre.1080.0551 / PDF https://business.columbia.edu/sites/default/files-efs/pubfiles/4115/federgruen_portfolio_suppliers.pdf | 검색 수준 |
| 26 | Data Shapley: Equitable Valuation of Data for Machine Learning | Ghorbani, Zou 2019 | ICML 2019 | 검색 수준: leave-one-out 점수가 부분집합 간 상호작용을 반영하지 못해 Shapley 계열보다 열등 | D7 보강(약). **전문 미독 — 수치·정리 인용 금지** | https://icml.cc/virtual/2019/poster/4290 | 검색 수준 |

---

## 2. D1~D7 직답

### D6 (최우선) — 대체재와 보완재가 동시에 있을 때 부호를 판정한 연구가 있는가

**"일부는 대체, 일부는 보완"인 집합함수의 모듈러 근사 편향 방향을 판정하는 형식 결과는 찾지 못했다.**
대신 두 가지가 확실히 확보된다.

1. **혼재를 다루는 형식 어휘는 이미 잘 정비되어 있다.** Bian 외 [9]는 임의의 단조 집합함수에 대해
   submodularity ratio γ∈[0,1] (열모듈 ⟺ γ=1)과 generalized curvature α∈[0,1] (**초모듈 ⟺ α=0**)를 정의한다.
   즉 대체·보완 혼재는 (γ<1, α>0)로 **두 모수 양방향** 표현된다. Chen–Teng–Zhang [7]의
   supermodular/superadditive width와 Feige 외 [8]의 MPH 계층도 같은 목적이다. Sonthalia 외 [21]의
   supermodular rank도 분해 관점이다.
2. **순 부호가 미결이라는 것 자체가 문헌의 귀결이다.** 임의의 집합함수가 두 열모듈 함수의 차(DS)로
   표현된다는 결과(Narasimhan–Bilmes 2005, [10]에서 귀속 확인)는 "대체 성분 − 보완 성분" 분해가 항상
   가능함을 뜻한다. **분해가 항상 가능하다면 두 항이 상쇄되는 방향은 함수마다 다르고 부류 차원에서
   결정될 수 없다** (추론).

**부호에 대해 가장 구체적인 문헌은 Heo 외 2026 [3]**(미심사 프리프린트)이다. 그룹 귀속에서 가산(1차)
근사의 오차가 쌍 상호작용 항 κ(a,b)이고, κ의 부호가 **중복적(대체적) 쌍에서 양, 보완적 쌍에서 음**임을
로지스틱 회귀에서 닫힌 형태로 보인다. 따라서 "가산 근사의 체계적 **과소**평가"는 **중복(대체)**에서
나오고 보완에서는 반대다. 이것이 감사 반론 1의 부호 논리와 정확히 일치한다.

### D7 — leave-one-out 차분이 다수 동시 이탈을 예측할 때의 오차

**있다. ML 데이터 귀속(influence function) 문헌에 정면으로 있다.**

- Koh 외 (NeurIPS 2019) [4]: 1차 Taylor 기반 influence function으로 **큰 그룹** 제거 효과를 예측할 때
  "절대·상대 오차가 크더라도" 실제 효과와 상관이 높고, 그 강한 상관은 "특정 설정에서만" 성립한다.
  **편향 방향은 초록에 없다.**
- Hu 외 (NeurIPS 2024) [5]: influence 기반 greedy가 **선형회귀에서도 증명 가능하게 실패**하며 실패 양식이
  (i) influence function의 오차와 (ii) **집합 영향의 비가산 구조**라고 보고한다.
  → 우리 `coef.py: raw()`가 1명 제거만 계산하고 MILP가 다수 이탈을 가산으로 예측하는 구조의
  **직접 선행 진단**이다.
- Basu 외 (ICML 2020) [6]: 그룹이 크면 1차 근사가 loose해지므로 2차 group influence를 제안.
- Heo 외 2026 [3]: 쌍 상호작용 항을 명시적으로 복원.
- 집합함수 이론 쪽 대응: Iyer 외 [1]의 모듈러 상·하한이 **현재 집합에서만 tight**하다는 진술.

**찾지 못한 것**: 이 진단을 **학습된 가치함수 + 정수계획 목적함수 계수**라는 맥락에서 한 연구.
즉 D7의 개념은 선행이 있으나 **우리 적용 맥락(ADP의 유지 가치 계수)은 비어 있다.**

### D3 — 모듈러 근사의 오차 크기 경계

**있다. 크기 경계가 명시적이고 tight하다.**

- [2] **Lemma 3.1**: 단조 열모듈 f에 대해
  `f(X) ≤ Σ_{j∈X} f(j) ≤ [ |X| / (1 + (|X|−1)(1−κ_f(X))) ] f(X)`,
  `κ_f = 1 − min_j f(j|V∖j)/f(j)`. **Lemma 3.2**: 임의의 κ에 대해 어떤 모듈러 상한도 이 계수보다
  나을 수 없는 f가 존재한다(tight). κ=0이면 오차 0, κ=1이면 계수 |X|.
- 초모듈 쪽: Sviridenko–Vondrák–Ward (MOR 2017) [20](검색 수준).
- 한계분포만 아는 대가: Agrawal 외 (OR 2012) [19]의 price of correlations / correlation gap.
  correlation gap의 정확한 정의는 [18]에서 직접 확인했다.

**실질적으로 중요한 점**: 이 경계들은 모두 **κ(또는 γ, α)를 알아야** 숫자가 된다. 우리 설정에서 κ를
추정하지 않았으므로 **경계의 수치를 인용할 수 없다.** "오차 크기가 curvature 한 개로 규정되며
tight하다"는 정성 사실은 인용 가능하고, **경험적으로 κ를 추정해 보고하는 것이 실행 가능한 후속 측정**이다.

### D4 — 조합경매의 선형 가격 실패가 우리 1차 분해 실패와 같은 뿌리인가

**같은 뿌리가 아니다. 유추에 불과하다. 그리고 "보완성 ⇒ 선형 가격 실패"라는 형태로 쓰면 틀린다.**

- Bikhchandani–Mamer (1997) [12]: 선형 시장청산가격 존재의 필요충분조건은
  **배분 정수계획의 최적값 = LP 완화 최적값**(정수성 격차 0)이다. 이는 **배분 다면체의 정수성** 문제이고,
  우리 문제(참여자 부분집합 격자 위 가치함수의 모듈러 근사 오차)와는 **다른 수학적 대상**이다.
  (2차 출처 수준 — 원문 403.)
- 반대 증거: Baldwin–Klemperer (Econometrica 2019) [11]은 **순수 보완재 선호 부류에서 균형이 보장되는
  경우가 순수 대체재보다 더 많다**고 명시한다.
- Gul–Stacchetti (1999) [13]의 maximal domain 정리는 **정정 논문(Yang 2017) [14]이 존재**하므로
  원문 확인 없이 인용하면 안 된다.
- Cramton 외 [15]는 서술적 동기만 제공하며 **형식 정리는 없다**(전문 키워드 검색 확인).

**판정: D4는 "강력한 인용"이 아니라 "느슨한 유추"다.** "문제의식이 닮아 있다" 수준으로만 쓰고
"같은 뿌리"라고 쓰면 안 된다.

### D1 — 다품목 all-or-nothing 주문을 동적 자원배분·재고에서 다룬 연구

**ATO 문헌에 있다. 다만 가치함수의 집합 위 초모듈성 형태로 진술되어 있지 않다.**

- Song (1998, OR) [22]: 다품목 주문에서 **order fill rate**를 성과 지표로 둔 원전 후보(검색 수준).
- Atan 외 (EJOR 2017) [23]: ATO 리뷰 진입점(서지만).
- Chen–Long–Qi (OR 2021) [16]: 파라메트릭 최적화에서 **초모듈성 보존**의 필요충분조건 + **ATO 적용**.
  **D1과 D2를 동시에 잇는 가장 좋은 인용.**
- IJPR 2024/2025 용량제약 ATO [24]: MDP 가치함수의 component-wise convexity(검색 수준).

**중요한 차이**: ATO 문헌의 구조 성질은 **부품 재고 수준 격자** 위의 볼록성·초모듈성이다.
우리가 필요한 것은 **참여자 부분집합 격자** 위의 초모듈성이고, 후자를 명시한 연구는 **찾지 못했다**.

### D2 — 초모듈성을 형식 진술하고 활용한 동적 자원배분 연구 / "보완재가 있으면 초모듈"의 출처

- 활용 사례는 있다: [16], [17].
- 그러나 **둘 다 격자 위 파라미터/상태 변수의 초모듈성**이며 참여자 집합의 초모듈성이 아니다.
- **"보완재가 있으면 가치함수가 초모듈"이라는 진술을 그대로 담은 1차 출처를 찾지 못했다.**
  집합함수 쪽에서 "보완성 ⟺ 초모듈성"의 정의적 등치는 [7][8][9]에서 표준이지만, 이는
  **가치평가 함수(valuation)** 이야기이고 **동적 프로그램의 가치함수**에 대한 진술이 아니다.
- **반대 방향으로 중요한 사실**: [17]에서 초모듈성은 **아핀 결정규칙 최적성의 조건**이다.
  "초모듈 ⇒ 선형 근사 실패"라는 수사는 OR 문헌에서 지지되지 않는다.

### D5 — 공급자 포트폴리오 옵션 가치 / "미래 주문 성사 가능성" 타깃의 선행 개념

**우리 설계의 직접 선행 개념을 찾지 못했다.**

- 가장 가까운 Federgruen–Yang (OR 2008) [25]은 **단일 품목·단일 시즌·정적**이다.
- 이중 소싱·공급중단 실물옵션 문헌은 여러 건 검색되었으나 **다품목 묶음 충족 확률을 가치의 원천으로
  두는 정식화**는 확인되지 않았다.
- 개념적으로 가장 가까운 것은 D1의 **order fill rate**(모든 품목 동시 가용 확률)이다.
  이것이 "미래 주문 성사 가능성" 타깃의 **가장 그럴듯한 선행 개념**이지만, 그 확률을 **신경망 학습
  타깃으로 두고 최적화 계수로 변환한 사례**는 찾지 못했다.

---

## 3. 논증 사슬 판정 (핵심 산출물)

우리 MILP 목적함수의 미래 항은 `Σ_i c_i p_i` (검증: `src/match/milp_build.py`의 `obj = obj + ck @ pv`,
`c_i`는 `src/vfa/coef.py: raw()`의 `V(φ(o)) − V(φ(o∖i))`). 결정과 무관한 상수를 더하면 이는
`V(S) − Σ_i (1−p_i) c_i`와 동등하다.

### 고리 A: "다품목 all-or-nothing 주문이 있으면 참여자 사이에 보완성이 생긴다"

- **(a) 인용 가능**: order fill rate를 다룬 ATO 문헌 [22][16][24]; 보완성 ⟺ 초모듈성의 정의 [7][8][9];
  묶음/보완재의 정의적 서술 [15].
- **(b) 직접 논증해야 하는 부분**: "**참여자(공급자) 부분집합 격자 위**의 가치함수가 보완적"이라는 구체적 진술.
  ATO는 부품 재고 수준 격자에서, 조합경매는 품목 묶음 valuation에서 말한다.
  **참여자 집합에 대한 초모듈성을 명시한 선행을 찾지 못했다.**
- **(c) 문헌이 반대 방향인 부분**: 없음. 이 고리는 무난하다.

### 고리 B: "따라서 가치함수가 초모듈적이다"

- **(a) 인용 가능**: 없음(우리 모형에 대해서는).
- **(b) 직접 논증해야 하는 부분**: 전부. **그리고 직접 논증해도 "초모듈"이라는 결론은 나오지 않는다.**
- **(c) 문헌이 반대 방향인 부분**: **결정적이다.** n_alt = 2에서 같은 품목 두 공급자는 용량 대체재이므로
  열모듈 방향, 다른 품목 공급자는 보완재이므로 초모듈 방향이다. [9]의 언어로 우리 V는
  **γ<1 이면서 α>0**인 혼재 함수다. [10]의 DS 분해가 항상 가능하다는 사실은 **부류 차원에서 순 부호를
  결정할 수 없음**을 뜻한다. **"가치함수가 초모듈이다"는 주장은 그대로 쓸 수 없다.**

### 고리 C: "따라서 1차 분해가 체계적으로 과소평가한다"

**여기서 부호가 반대다. 감사의 반론 1은 옳다.**

직접 확인한 형식 근거 [1]: submodular f, Y=V=S, `ĝ(j) = f(j|V∖{j})`에 대해
`m_ĝ(X) = f(S) + ĝ(X) − ĝ(S) ≥ f(X)`, `m_ĝ(S) = f(S)`.
우리 표기로 `m_ĝ(X) = V(S) − Σ_{j∈S∖X} c_j`.

여기서 다음이 따른다 (**(i)(ii)는 [1]에서 직접 인용 가능, (iii)(iv)는 우리 도출 = 추론**):

(i) **V가 열모듈(대체 지배)이면** `m_ĝ(X) ≥ V(X)` — 가산 대리값이 축소 집합의 가치를 **과대**평가,
    즉 **다수 이탈 손실을 과소평가**.
(ii) **V가 초모듈(보완 지배)이면** 부호가 반대 — 다수 이탈 손실을 **과대**평가.
(iii) 모듈러 함수의 기대는 선형이므로 `E[m_ĝ(R(p))] = V(S) − Σ_j (1−p_j) c_j` = 우리 MILP 대리값.
     정확한 대상은 **다선형 확장 `F(p) = E[V(R(p))]`** ([18]에서 정의 확인). 즉
     **우리 대리값은 F의 p=1에서의 1차 Taylor 전개다.**
(iv) 한계 유인 부호: `∂F/∂p_i = E[V(R∪i) − V(R∖i)]`. 열모듈 V에서는 leave-one-out 한계 c_i가
     **가능한 한계값 중 최소**이므로 `c_i ≤ ∂F/∂p_i` → **c_i가 진짜 한계 유지 가치를 과소평가**.
     초모듈 V에서는 반대로 **과대평가**.

- **(a) 인용 가능**: [1](모듈러 상·하한과 tight성), [18](다선형 확장·correlation gap 정의),
  [2][20](오차 크기 경계와 tight성), [5][4][6][3](1차 가산 근사가 집합 효과에 실패).
- **(b) 직접 논증해야 하는 부분**: (iii)(iv)의 우리 설정 특화 도출. 그리고 **우리 V의 실제 (γ, α) 또는
  순 부호를 측정하는 것.** 문헌은 부호를 주지 않는다.
- **(c) 문헌이 반대 방향인 부분**: **초안이 쓰려던 방향 그대로가 반대다.** 초모듈성은 과대평가를 낳는다 [1].
  과소평가는 열모듈성(대체·중복)이 낳는다 [1][3]. 게다가 [17]은 초모듈성을 **아핀 규칙 최적성의 조건**으로
  쓰고, [11]은 보완성이 선형 가격을 깨뜨린다는 통념이 **틀렸다**고 명시한다.
  Topaloglu–Powell(B조사)의 "대체가 있어도 분리형 근사가 고품질 해를 준다"도 같은 방향이다.

### 사슬 전체 판정

**현재 형태의 사슬은 성립하지 않는다.** 고리 B가 문헌적으로 지지되지 않고(혼재), 고리 C는 문헌과
**부호가 반대**다.

동시에 감사의 4-1 지적도 유효하다: 우리가 실제로 잰 "변환 오차"(모델 c vs MC 기준치 c)는
**둘 다 leave-one-out**이므로 이 사슬이 말하는 양이 아니다. 즉 **관측된 과소평가는 위 (i)(ii)(iv)
어느 쪽으로도 설명되지 않는다** — 그것은 모델 오차이고, 감사가 제시한 네 개의 경쟁 가설
(회귀 축소, 표본 불균형, 가드 shrink·절단, 정답 잡음)이 더 그럴듯하다.

### 쓸 수 있는 형태로 다시 적은 세 문장

1. 우리 MILP의 미래 항 `Σ c_i p_i`는 참여자 집합 위 가치함수의 **다선형 확장의 p=1 1차 Taylor 전개**이며,
   계수 c_i는 discrete supergradient ĝ다 [1][18].
2. 이 근사는 **현재 집합에서만 tight**하고, 다수 동시 이탈에서 어긋나며, 그 어긋남의 크기는 curvature로
   규정되고 그 경계는 tight하다 [1][2]. 1차 가산 근사가 집합 효과를 잡지 못하는 실패는 ML 데이터 귀속에서
   증명 가능하게 보고되어 있다 [5][3][6].
3. **우리 시장에서 그 어긋남의 부호는 이론적으로 미결이다**: 같은 품목의 대체 공급자(열모듈 방향)와
   묶음 주문의 보완 공급자(초모듈 방향)가 섞여 있어 (γ<1, α>0)의 혼재 함수이고, 임의 집합함수의
   DS 분해 가능성 때문에 부류 차원에서 부호가 정해지지 않는다 [9][10].
   **따라서 부호는 다중 제거 기준치로 측정해서 보고해야 한다.**

### 측정 제안 (문헌에서 도출된 구체 실행안)

(a) `Σ_{i∈A} c_i` 대 `V_mc(S) − V_mc(S∖A)`를 |A|=1,2,3에서 짝지어 비교해 부호와 크기를 직접 재라.
(b) 부호를 구조와 잇고 싶으면 A를 **같은 품목 대체 공급자 쌍** 대 **다른 품목 보완 공급자 쌍**으로 나눠
    비교하라 — [3]의 "중복 쌍은 양의 상호작용, 보완 쌍은 음의 상호작용"이 예측을 주므로 검증 가능한 가설이 된다.
(c) 원하면 `γ̂, α̂`(또는 `κ̂`)를 표본으로 추정해 [2]의 경계와 실제 오차를 대조하라.

---

## 4. 미해결·확인 불가 (남긴 쿼리 포함)

**결과 0건 / 찾지 못한 것 (모두 "없다"로 기록)**

1. "일부 대체·일부 보완인 집합함수의 **모듈러 근사 편향 방향**을 판정한 형식 결과" — 없음.
   쿼리: `set function neither submodular nor supermodular substitutes and complements mixed decomposition`,
   `leave-one-out marginal contribution underestimates group deletion effect set function first-order approximation`.
2. "**참여자(에이전트) 부분집합 격자** 위 가치함수의 초모듈성을 명시한 동적 플랫폼·매칭 연구" — 없음.
   쿼리: `"value function" supermodular "set of agents" subset lattice dynamic matching platform retention approximate dynamic programming`,
   `supermodular value function set of suppliers participation complementarity two-sided platform dynamic program 2024 2025`.
3. "다품목 묶음 성사 확률을 **학습 타깃**으로 두고 최적화 계수로 변환한 조달·플랫폼 연구" (D5) — 없음.
4. "1차 노드 삭제 차분의 오차를 **학습된 가치함수 + 정수계획 계수** 맥락에서 정량화한 연구" — 없음(B4 재확인).

**페이월·차단으로 확인 못 한 것 (도서관 접속으로 확인할 질문)**

- Gul & Stacchetti (1999) [13] 전문: maximal domain 정리의 정확한 진술과 Yang(2017) [14] 정정 이후의 유효 형태.
- Bikhchandani & Mamer (1997) [12] 원문: "경쟁균형 존재 ⟺ IP 최적값 = LP 완화 최적값"의 정리 번호와 가정.
- Song (1998) OR [22] 전문(INFORMS 403): order fill rate가 성분별 fill rate의 곱과 다른 이유(의존성)에 대한
  형식 진술이 있는가? **우리 all-or-nothing 보완성 논증에 직접 쓸 수 있는 문장이 있는지가 관건.**
- Atan 외 (EJOR 2017) [23] 전문: ATO 리뷰에 "가치함수의 부품 집합에 대한 초모듈성" 언급 절이 있는가?
- Chen–Long–Qi (OR 2021) [16] 전문: ATO 적용 절에서 초모듈성이 정의된 격자가 정확히 무엇인지.
- Iancu 외 (OR 2013) [17] 게재본: 초모듈성이 아핀 최적성에 쓰이는 정리의 정확한 가정.
- Iyer & Bilmes (UAI 2012) [10] 본문: "임의의 집합함수 = 두 단조 열모듈 함수의 차"의 정리 번호와
  Narasimhan–Bilmes 2005 귀속.
- Agrawal 외 (OR 2012) [19] 전문: 열모듈 부류의 correlation gap 상한 구체 수치. 지금은 인용 금지 상태.

**신뢰도 경고**

- [3] Heo 외 2026은 **미심사 arXiv 프리프린트**다. 핵심 부호 논증에 단독으로 걸면 안 된다.
  심사 통과 대체 인용은 [5] Hu 외 NeurIPS 2024와 [1] Iyer 외 ICML 2013이다.
- [24] IJPR 논문은 **저자 미확인**이다. 인용 전 확인 필요.
- [26] Ghorbani–Zou, [19] Agrawal 외, [20] Sviridenko 외, [22] Song, [25] Federgruen–Yang은
  **검색 결과 수준**만 확인했다. 내용 인용 시 전문 확인 필요.

---

## 5. G1~G5에 대한 영향

- **G3 (분해 오차 정량화)**: **약화된다.** 초모듈성 해석은 부호가 반대여서 삭제해야 한다.
  다만 **G3b(가산 대리값 `Σ c_i p_i` 대 다수 이탈 포함 진짜 기대가치)** 는 문헌적 위치가 오히려
  **명확해졌다**: [1]의 tight성, [18]의 다선형 확장, [5][3]의 비가산성 실패가 모두 이 양을 정확히 가리킨다.
  **G3b를 측정하면 "알려진 이론적 한계를 우리 응용 맥락에서 처음 정량화"라는 기여로는 설 수 있다.**
- **G3의 메커니즘 서술**: "묶음 보완성 때문에 과소평가"는 **폐기**. 대체·보완 혼재로 부호 미결임을 먼저 적고,
  부호를 **측정 결과로** 보고하는 서술로 바꿔야 한다.
