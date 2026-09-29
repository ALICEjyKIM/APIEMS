# 주제 F3 증거 수집: 구조 기반 가치 근사 — 신뢰도 구조와 Birnbaum 중요도

조사일: 2026-09-28 · 담당: researcher 서브에이전트
범위 문서: `ref/literature-review.md` "3차 조사 범위 (주제 F)" 중 F3(F3a~F3d)
읽은 것: `CLAUDE.md`(모델 가정), `ref/literature-review.md` 3차 조사 범위 전체,
`ref/_review/endogenous-participation-vfa-embedding_research_D.md` §1 증거표·§2 D7·§3(중복 회피·문맥), `src/vfa/coef.py`

D 조사의 출처 번호는 이 문서에서 `D[n]`으로 표기한다 (예: `D[1]` = Iyer·Jegelka·Bilmes ICML 2013, `D[18]` = Chekuri·Vondrák·Zenklusen 다선형 확장).

---

## 0. 결론 먼저

**대응은 성립한다. 다만 "정리로 증명"이 아니라 "동형 구조에서 이미 알려진 결과"로만 쓸 수 있다.**

1. 우리 `c_i = V(S) − V(S∖{i})`는 **Birnbaum 중요도를 p = 1 코너에서 평가한 값**이다. Birnbaum 중요도의 표준 정의
   `I_B(i) = ∂h(p)/∂p_i = h(1_i, p) − h(0_i, p)` [1][7]과 형태가 정확히 같다 (도출은 §3, 우리 추론).
2. 우리 묶음 구조(품목에 대해 직렬, 품목 내 공급자에 대해 병렬)는 신뢰도 이론의
   **parallel-in-series 시스템**(병렬 부분시스템들이 직렬로 연결)과 동형이다.
   그 구조에서 **joint reliability importance(JRI)의 부호 패턴은 이미 형식적으로 알려져 있다** [2]:
   **같은 부분시스템(= 같은 품목) 내 요소쌍은 JRI ≤ 0 = reliability substitutes,
   서로 다른 부분시스템(= 다른 품목) 요소쌍은 JRI ≥ 0 = reliability compliments.**
3. 이 부호 패턴은 **D 조사 결론("대체는 과소평가, 보완은 과대평가")과 완전히 일치한다.**
   신뢰도 쪽 공식으로 다시 확인된다: 병렬에서 `I_B(i) = Π_{j≠i}(1−p_j)`는 p = 1에서 **최소(0)**,
   직렬에서 `I_B(i) = Π_{j≠i} p_j`는 p = 1에서 **최대(1)** [1]. 즉 p = 1에서 재는 leave-one-out은
   대체 구조에서 과소평가, 보완 구조에서 과대평가다.
4. **신뢰도 계보가 D보다 더 주는 것**: D는 "혼재이므로 순 부호 미결"에서 멈췄으나,
   [2]는 우리 구조에서 **쌍별 부호 패턴을 결정**한다. 순 부호는 여전히 측정 대상이지만
   "같은 품목 쌍 vs 다른 품목 쌍"으로 나눠 재면 **부호가 반대로 나와야 한다는 이론적 예측**이 생긴다 (§3 (c) 측정 제안).
5. **깨지는 지점**: 우리 V는 이진 구조함수가 아니라 **수량·용량이 있는 연속 이윤**이고, 제거는 고장이 아니라
   **이탈 + 확률 0.25 충원**이며, p_i는 외생이 아니라 **잉여배분 결정변수**다. 따라서 위 결과는
   **구조적 유추이자 극단 사례(주문 단위에서 한 공급자가 단독 충족 가능, 이진 성립)** 에서만 정확하다.
   다상태(multistate)로 넘어가면 "측정치마다 순위가 달라질 수 있다"는 보고가 있다 [14].
   **그리고 [2]의 부호 정리는 "부분시스템이 요소를 공유하지 않는다"를 가정한다 — 우리는 공유한다(§4 마지막 항목).**
6. **F3b 선례 유무 직답**: Birnbaum 중요도를 **ADP 가치함수 계수**로 쓴 선례는 **찾지 못했다(없다고 판단)**.
   그러나 **우리 c_i와 형태가 같은 공급자 평가량**은 있다: Li & Nagurney [11]의
   `I(j) = [E(G) − E(G−j)] / E(G)`. 다만 정적 균형 지표이고 **사후 순위용**이며 최적화 계수가 아니다.

---

## 1. 증거 표

| # | 제목 | 저자·연도 | 게재지·등급 | 중요도 측정치 | 응용 영역 | 상호의존 종류 | 우리 `c_i`와의 관계 | URL | 읽은 수준 |
|---|------|-----------|-------------|---------------|-----------|---------------|---------------------|-----|-----------|
| 1 | Chapter 5: Component Importance (System Reliability Theory 강의자료, Ver. 0.1) | Rausand (RAMS Group, NTNU), 연도 미표기(교재 Wiley 2004 기반) | 대학 강의자료(**2차 출처**, 저자·소속 명시) | Birnbaum, improvement potential, RAW, RRW, criticality, Fussell-Vesely | 신뢰도 일반 | 직렬=보완, 병렬=대체 (예제로 대비) | **핵심.** `I_B(i|t) = ∂h(p)/∂p_i = h(1_i,p) − h(0_i,p)` (pivotal decomposition). 직렬 2요소 `I_B(1) = p_2`, 병렬 2요소 `I_B(1) = 1 − p_2`. 이 두 공식이 §3 부호 판정의 토대 | https://www.ntnu.edu/documents/624876/1277590549/chapt05.pdf/82cd565f-fa2f-43e4-a81a-095d95d39272 | **전문 텍스트 추출(pypdf), 정의·예제 verbatim 인용** |
| 2 | Multicomponent joint reliability importance of series-in-parallel and parallel-in-series systems | Jain, Dewan, Rani 2014 | **IJQRM** 31(7):858–876 (Emerald, 심사지) | JRI (다요소 확장 포함) | 신뢰도 구조 이론 | **혼재(구조별로 분리 판정)** | **가장 중요.** parallel-in-series(= 우리 구조)에서 같은 부분시스템 내 쌍 JRI 비양(substitutes), 다른 부분시스템 간 쌍 JRI 비음(compliments). series-in-parallel은 반대. **가정: 부분시스템 간 공통 요소 없음** | https://www.emerald.com/ijqrm/article/31/7/858/139105/Multicomponent-joint-reliability-importance-of (DOI 10.1108/IJQRM-06-2011-0087) | **초록 verbatim + 부호 진술 WebFetch 추출** (전문 미독) |
| 3 | Joint reliability-importance of two edges in an undirected network | Hong, Lie 1993 | **IEEE Trans. Reliability** (피인용 144) | JRI **원전** | 네트워크 신뢰도 | 해당 없음(측정치 정의) | JRI 개념의 원전. **부호 결과를 직접 확인하지 못했다** | https://doi.org/10.1109/24.210266 | **서지만** (Crossref·Semantic Scholar) |
| 4 | Joint reliability-importance of components | Armstrong 1995 | **IEEE Trans. Reliability** 44(3):408–412 (피인용 117) | JRI | 신뢰도 일반 | 혼재 | "reliability complements / substitutes" 용어의 출처로 2차 출처들이 귀속. **원문 미독 → 귀속은 2차 수준** | https://doi.org/10.1109/24.406574 | **서지만** |
| 5 | Analysis for joint importance of components in a coherent system | Gao, Cui, Li 2007 | **EJOR** 182(1):282–299 (피인용 88) | joint importance | 일관 시스템 | 혼재 | 일관 시스템 joint importance의 OR 저널 정통 출처. **초록조차 확보 못함 → 내용 인용 금지** | https://doi.org/10.1016/j.ejor.2006.07.022 | **서지만** |
| 6 | On Birnbaum type joint importance measures for multistate reliability systems | Chacko 2021 (게재 2023) | Communications in Statistics — Theory and Methods 52(9) | JBRAW, JBRRW, JBRFV (Birnbaum형 joint) | 다상태 시스템 | 혼재 | 초록: "Importance and joint importance measures in reliability engineering are used to identify the weak areas of a system". 다상태 확장 존재의 근거 | https://doi.org/10.1080/03610926.2021.1961000 | **초록 verbatim 확인** |
| 7 | On the Formalization of Importance Measures using HOL Theorem Proving | Ahmad, Murtza, Hasan, Tahar 2019 | arXiv 프리프린트(**미심사**) | Birnbaum, Fussell-Vesely, RAW, RRW | 정형 검증 | 해당 없음 | Birnbaum 정의의 기계검증형 진술: `I_B(i)(φ) = Pr{φ(1_i,x̄)} − Pr{φ(0_i,x̄)}`. **직렬·병렬 닫힌형 공식은 없음(확인)** | https://arxiv.org/abs/1904.01605 | **정의 절 직접 추출**(ar5iv HTML) |
| 8 | A mathematical definition and basic structures for supply chain reliability: A procurement capability perspective | Ha, Jun, Ok 2018 | **Computers & Industrial Engineering** 120:334–345 (피인용 34) | (측정치 아님: 신뢰도·가용도·hazard 정의) | **공급망 신뢰도·조달** | 혼재(구조모형 나열) | **F3a 최선 인용.** "the basic structural reliability models for various types of supply chains" — series, parallel, parallel-series, series–parallel 포함. 컴퓨터 조립 기업 사례 | https://doi.org/10.1016/j.cie.2018.04.036 | **초록 verbatim 확인** |
| 9 | Birnbaum importance analysis of supply chain fault risks based on binary decision diagram | Li, Xue 2019 | Procedia Manufacturing 30:106–111 (학회 proceedings, 등급 낮음) | **Birnbaum** | **공급망 위험(FTA/BDD)** | 혼재(고장木 AND/OR) | Birnbaum 중요도를 **공급망 위험 사건**에 적용해 약점 식별. 공급자 유지·최적화는 아님 | https://doi.org/10.1016/j.promfg.2019.02.016 | **초록 verbatim 확인**, 전문 403 |
| 10 | A heuristic application of systems reliability optimization in supplier selection problem of a make-to-order supply chain | Gupta, Deb 2024 | Int. J. System Assurance Eng. & Management 15:5742–5755 | **없음**(GA로 직접 최적화) | **공급자 선택** | **보완+대체 혼재(구조상)** | **F3a 직접 대응.** 공급자 선택을 "reliability maximization problem for a **parallel–series system**, constrained by cost and the maximum number of suppliers"로 정식화. 단계별 2공급자 병렬 = `1 − (1−p_1)(1−p_2)`. **정적·단일기간, 중요도 측정치 미사용** | https://doi.org/10.1007/s13198-024-02593-4 | **초록·정식화 WebFetch 추출** |
| 11 | Supply chain performance assessment and supplier and component importance identification in a general competitive multitiered supply chain network model | Li, Nagurney 2015(in press)/2017 | **J. Global Optimization** 67:223–250 | 자체 정의 importance indicator (Birnbaum이라 부르지 않음) | **공급자·부품 중요도 식별** | 혼재(균형 모형) | **F3b 최근접 선례.** Def 5.1: `I(j) = ΔE/E = [E(G) − E(G−j)] / E(G)`, "the relative supply chain network efficiency drop after j is removed from the whole supply chain". 다수 동시 제거판도 정의(식 23·25). **우리 c_i와 형태 동일**(정규화 차이). 단 정적 균형·**사후 순위 지표**, 최적화 계수 아님 | https://doi.org/10.1007/s10898-015-0371-7 · 슬라이드 https://supernet.isenberg.umass.edu/visuals/LN-INFORMS2015.pdf · SSRN https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2646922 | **슬라이드 전문 텍스트 추출(pypdf), 정의 4.1·4.2·5.1·5.2 verbatim 인용** |
| 12 | Recent advances in system reliability optimization driven by importance measures | Si, Zhao, Cai, Dui 2020 | Frontiers of Engineering Management (리뷰) | Birnbaum 및 IM 일반 | **신뢰도 최적화**(자원배분) | 혼재 | IM이 "allocating limited resources"에 쓰인 계보 리뷰. 대상: series-parallel, consecutive-k-out-of-n, 다상태, phased-mission, 네트워크. **dynamic programming 언급 없음(확인)**, 공급자 선택은 인용문헌 1건 | https://doi.org/10.1007/s42524-020-0112-6 · https://journal.hep.com.cn/fem/EN/10.1007/s42524-020-0112-6 | **초록·범위 WebFetch 확인** |
| 13 | Importance measures in reliability and mathematical programming | Zhu, Kuo 2012 | **Annals of Operations Research** 212:241–267 | IM 일반 | 수리계획 | 미확인 | 제목상 F3b의 정면 주제. **초록·전문 모두 확보 실패 → 내용 인용 금지** | https://doi.org/10.1007/s10479-012-1127-0 | **서지만** |
| 14 | Birnbaum criticality and importance measures for multistate systems with repairable components | Huseby, Kalinowska, Abrahamsen 2020 | **Probability in the Engineering and Informational Sciences** | Birnbaum 기반 4종 신규(forward/backward-looking) | 다상태·수리가능 시스템 | 혼재 | **대응 한계의 근거.** 초록: "Examples show that the different importance measures may result in **unequal rankings**". 제거 후 충원(수리)이 있는 우리 설정에서 이진 Birnbaum을 그대로 쓸 수 없다는 경고 | https://doi.org/10.1017/S0269964820000340 | **초록 verbatim 확인** |
| 15 | Reliability Importance Measures of the Components in a System Based on Semivalues and Probabilistic Values | Freixas, Puente 2002 | **Annals of Operations Research** (피인용 46) | semivalue·probabilistic value 기반 중요도 | 신뢰도 ↔ 협조게임 | 해당 없음 | Shapley 계열 중요도의 신뢰도 쪽 정통 출처. **초록 미확보 → 내용 인용 금지** | https://doi.org/10.1023/A:1016368606348 | **서지만** |
| 16 | On the extensions of Barlow-Proschan importance index and system signature to dependent lifetimes | Marichal, Mathonet 2011(arXiv)/2013 | J. Multivariate Analysis | Barlow-Proschan 지수 | 신뢰도 ↔ 게임이론 | 해당 없음 | "i.i.d.에서 Barlow-Proschan = Shapley 값"이라는 표준 진술의 출처 후보. **검색 요약 수준, 원문 미독 → 인용 금지** | https://arxiv.org/abs/1109.4860 | **서지 + 검색 수준** |
| 17 | Two new component importance measures for a flow network system | Aven, Østebø 1986 | Reliability Engineering (피인용 26) | 흐름망용 중요도 2종 | 흐름망(용량 있는 시스템) | 미확인 | **우리 용량·수량 구조의 올바른 일반화 방향.** 초록 미확보 → 존재만 기록 | https://doi.org/10.1016/0143-8174(86)90091-0 | **서지만** |

**원전 주의**: Birnbaum, Z. W. (1969) "On the importance of different components in a multicomponent system",
*Multivariate Analysis — II* (ed. Krishnaiah), Academic Press, pp. 581–592. 이 서지는 [1]이 본문에서
"Birnbaum (1969) proposed the following measure"로 귀속한 것과 검색 결과가 일치하지만, **원문에 직접 접근
가능한 URL을 찾지 못했다.** 따라서 증거 표에 별도 행으로 올리지 않았고, 인용 시 [1]을 경유한 2차 귀속임을 밝혀야 한다.

---

## 2. F3a~F3d 직답

### F3a — 직렬-병렬 신뢰도 구조를 가치함수 근사나 자원·공급자 평가에 쓴 선례

**공급망·조달 쪽에는 있다. 가치함수 근사(ADP)에는 없다.**

- **[8] Ha, Jun, Ok (2018, Computers & Industrial Engineering)** 이 F3a의 최선 인용이다. 공급망 신뢰도의
  수학적 정의와 관련 함수(hazard function, cumulative hazard function, availability, mean residual capacity)를
  요소 수준에서 세우고, 시스템 수준에서 **series, parallel, parallel-series, series–parallel, n-system**
  구조 신뢰도 모형을 제시한다. 관점이 명시적으로 **조달 역량(procurement capability)** 이다.
- **[10] Gupta & Deb (2024)** 는 우리 구조와 **동형인 정식화를 이미 쓴다**: 공급자 선택을
  "reliability maximization problem for a **parallel–series system**"으로 두고, 각 단계(부품/품목)에서
  선택된 2공급자의 단계 신뢰도를 `1 − (1−p_1)(1−p_2)`로, 전체를 단계들의 곱으로 둔다.
  즉 **품목에 대해 직렬, 품목 내 공급자에 대해 병렬** — CLAUDE.md의 우리 묶음 구조와 같다.
  다만 (i) 정적 단일기간, (ii) 중요도 측정치를 쓰지 않고 유전 알고리즘으로 직접 탐색,
  (iii) 유지·재참여 개념 없음.
- **[12] Si 외 (2020)** 리뷰는 중요도 측정치가 **자원 제약 하 배분**에 쓰인 계보를 정리하며
  대상 구조에 series-parallel을 포함한다. 그러나 이 리뷰에서 **dynamic programming 언급을 찾지 못했다**(WebFetch 확인).
- **찾지 못한 것**: 직렬-병렬 신뢰도 구조를 **동적 프로그램의 가치함수 근사에 특징(feature)이나 근사 형태로 쓴 연구.**
  남긴 쿼리: `reliability importance measure approximate dynamic programming value function approximation`,
  `"Birnbaum" OR "reliability importance" "approximate dynamic programming" OR "value function approximation" component 2024 2025`,
  `"importance measure" reliability "dynamic programming" component assignment optimization value function` → 해당 결과 없음.

### F3b (가장 중요) — Birnbaum 중요도를 공급자·자원 평가나 ADP 가치함수에 쓴 선례

**네 갈래로 나눠 답한다.**

**(1) 최적화에 쓴 선례: 풍부하다. 단 전부 정적 신뢰도 최적화의 휴리스틱 안내자다.**
[12]가 정리하는 계보에서 Birnbaum 중요도는 component assignment problem, redundancy allocation,
system upgrading, 유지보수 최적화의 **우선순위·탐색 안내**로 쓰인다. (검색 수준에서 Lin–Kuo 2002 LK 휴리스틱,
Yao–Zhu–Kuo 2011/2014의 Birnbaum 기반 유전 국소탐색 등이 반복 언급되지만 **원문 미독이므로 이들은 인용하지 않는다.**)
[13] Zhu & Kuo (2012, Annals of OR)가 제목상 이 주제의 정면이지만 **초록조차 확보하지 못해 내용을 쓸 수 없다.**

**(2) 공급자 평가에 쓴 선례: 이름은 다르지만 형태가 같은 것이 하나 있다 — [11] Li & Nagurney.**
정의 5.1(슬라이드 전문에서 직접 추출):
> "The importance of a supplier j, corresponding to a supplier node j ∈ G, I(j), for the whole competitive
> supply chain network, is measured by the relative supply chain network efficiency drop after j is removed
> from the whole supply chain: `I(j) = ΔE/E = [E(G) − E(G−j)] / E(G)`"

이것은 **정규화된 leave-one-out**이며 우리 `c_i = V(S) − V(S∖{i})`와 **형태가 정확히 같다.**
같은 논문은 "모든 공급자가 동시에 제거될 때"의 robustness 측도(식 23·25)도 정의하는데, 이는
**F3c·D7(다수 동시 이탈)의 공급망 쪽 대응물**이다.
**그러나 결정적 차이가 세 개 있다**: (i) 대상 E는 신뢰도가 아니라 균형 수요/가격 비의 효율 지표
(정의 4.1: `E(G) = Σ_i Σ_k d*_ik / ρ_ik(d*) / (I × n_R)`), (ii) 정적 Cournot–Bertrand 균형이고 동적 전이가 없음,
(iii) **사후 순위·진단 지표**로만 쓰이고 최적화 목적함수의 계수로 들어가지 않음.
그리고 저자들은 이를 Birnbaum 중요도라고 부르지 않는다.

**(3) ADP 가치함수 계수로 쓴 선례: 없다.** F3a에 적은 세 쿼리 전부 0건이었고, 추가로
`"importance measure" reliability Markov decision process reinforcement learning value function critical component selection`도
관련 결과 0건이었다. **따라서 "Birnbaum 중요도를 학습된 가치함수의 leave-one-out 계수로 정수계획 목적함수에 넣은
선례는 없다"고 단정한다.** (D 조사의 D7 결론 — "개념은 선행이 있으나 ADP 유지 가치 계수 맥락은 비어 있다" — 와 같다.)

**(4) Shapley 계열**: 신뢰도 중요도를 협조게임 값으로 보는 계보가 따로 있다 — [15] Freixas & Puente
(Annals of OR 2002, semivalue·probabilistic value 기반), [16] Marichal & Mathonet(Barlow–Proschan 지수의 확장).
"i.i.d. 수명에서 Barlow–Proschan 지수가 Shapley 값과 일치한다"는 진술은 여러 검색 결과에서 반복되지만
**원문에서 확인하지 못했으므로 인용하지 않는다.** 이 계보가 중요한 이유: 우리 `c_i`는 Shapley가 아니라
**Birnbaum형(leave-one-out)** 이므로, "왜 Shapley가 아닌가"를 물을 때 신뢰도 쪽에도 두 계열이 병존한다는
사실을 근거로 쓸 수 있다.

**우리 `c_i`와 정확히 같은 양인가?** — **형식적으로는 Birnbaum 중요도의 특수 경우다.**
`I_B(i) = h(1_i, p) − h(0_i, p)` [1][7]에서 다른 요소들의 p를 모두 1로 고정하면 `V(S) − V(S∖{i})`가 된다(§3 (a)).
차이는 두 가지: (i) 대상 함수가 이진 구조함수의 신뢰도가 아니라 **학습된 연속 가치함수**,
(ii) 평가점이 임의의 p가 아니라 **p = 1 코너로 고정**. 그래서 정확한 표현은
"우리 c_i는 Birnbaum 중요도를 전원 참여(p = 1) 상태에서 평가한 값"이다.

### F3c — 다수 요소 동시 고장에서 중요도 측정치가 어긋난다는 지적

**있다. 검색어 추정("joint reliability importance")이 맞았다.**

- 계보: [3] Hong & Lie (1993, IEEE TR, JRI 원전) → [4] Armstrong (1995, IEEE TR) → [5] Gao·Cui·Li (2007, EJOR) →
  다요소 확장 [2] Jain·Dewan·Rani (2014, IJQRM) → 다상태 확장 [6] Chacko (2021).
- 정의: `JRI(i,j) = ∂²h(p) / ∂p_i ∂p_j`. [2] 초록(직접 확인):
  "Joint reliability importance (JRI) of components is the effect of a change of their reliability on the system reliability."
- **우리 D7과의 정확한 대응**: JRI ≠ 0이면 1차(leave-one-out) 차분만으로는 두 요소 동시 이탈 효과를 못 맞춘다.
  이는 D 조사의 D7(`D[4]` Koh 외 2019, `D[5]` Hu 외 2024, `D[6]` Basu 외 2020, `D[3]` Heo 외 2026)과
  **같은 수학적 대상**이다. 신뢰도 쪽이 약 30년 먼저 같은 진단을 했고, **ML 쪽보다 부호 정보를 더 준다**(§3 (b)).
- [14] Huseby 외 (2020, PEIS)는 다상태·수리가능 시스템에서 "different importance measures may result in
  **unequal rankings**"라고 보고한다 — 측정치 선택이 결론을 바꾼다는 경고.
- [11] Li & Nagurney의 식 23·25(전체 공급자 동시 제거 robustness 측도)는 같은 문제의식의 공급망 쪽 표현이다.
- **검색 요약에서만 본 것(인용 금지)**: "기존 중요도 측정치는 주어진 크기의 임계 요소 **집합**을 동시에 식별할 수 없다",
  "JFI(joint failure importance)의 고장木 계산에 모순이 있고 약한 요소의 정확한 수치 순위를 주지 못한다".
  두 진술 모두 원문에서 확인하지 못했다.

**직렬-병렬 구조에서 joint importance의 부호** → §3 (b)에서 답한다. **결론: 다르다.**

### F3d — 이 계보가 우리에게 주는 것 (부호 논증이 뒷받침되는가)

**절반 뒷받침된다.**

**뒷받침되는 것 (인용 가능)**
1. **쌍별 상호의존 부호 패턴**: parallel-in-series 구조에서 같은 부분시스템 내 요소쌍은
   reliability substitutes(JRI ≤ 0), 다른 부분시스템 요소쌍은 reliability compliments(JRI ≥ 0) [2].
   우리 구조가 parallel-in-series이므로 **같은 품목 공급자쌍 = 대체, 다른 품목 공급자쌍 = 보완**이다.
   (단 [2]의 "부분시스템 간 공통 요소 없음" 가정 문제 — §4 마지막 항목.)
2. **직렬(보완) 구조에서 한 요소의 Birnbaum 중요도는 다른 요소들의 신뢰도가 높을수록 커지고
   전원 정상(p = 1)에서 최대가 된다**: `I_B(1) = p_2` [1]. 병렬(대체)은 `I_B(1) = 1 − p_2`이므로
   p = 1에서 최소(0)다 [1]. → **보완 구조에서 leave-one-out은 상한, 대체 구조에서는 하한.**
3. **G1의 현상 문장(증분 > 독립)의 교과서적 예시**: 위 공식들로부터 직렬 2요소에서는 증분 1 > 독립 0,
   병렬 2요소에서는 증분 0 < 독립 1이 바로 나온다(§3 (a), **우리 도출**).

**뒷받침되지 않는 것**
4. 우리 V가 이진 일관 시스템의 신뢰도라는 보장이 없다. 수량·용량·다기간·내생 p_i 때문에(§3 (e))
   위 결과들은 **구조적 유추이자 극단 사례에서만 정확**하다.
5. 신뢰도 문헌에서 "**묶음 all-or-nothing 조달에서 공급자의 증분 가치가 독립 가치를 넘는다**"를
   진술한 것을 찾지 못했다. 우리 문장 자체의 선행은 없다.
6. [13](Annals of OR, 제목상 정면 주제)의 내용을 확보하지 못했으므로, "수리계획과 중요도의 연결에
   이런 것이 없다"는 단정은 이 문헌을 읽은 뒤에야 안전하다.

**따라서 쓸 수 있는 형태**(논문에는 이 문장들만 쓴다):
> "우리 묶음 구조는 품목에 대해 직렬, 품목 내 공급자에 대해 병렬인 parallel-in-series 시스템과 동형이다.
> 그 구조에서 joint reliability importance의 부호는 같은 품목 공급자쌍에서 비양(reliability substitutes),
> 다른 품목 공급자쌍에서 비음(reliability compliments)으로 알려져 있다 [2]. 우리 `c_i`는 Birnbaum 중요도
> `∂h/∂p_i = h(1_i,p) − h(0_i,p)` [1]를 전원 참여 상태에서 평가한 값이며, 직렬 구조에서 이 값은
> 다른 요소가 모두 정상일 때 최대, 병렬 구조에서는 최소가 된다 [1]. 우리 시장은 두 종류 쌍을 모두 포함하므로
> 순 부호는 측정 대상이다."

---

## 2-B. 추가 증거 표 및 결론 보정 — OpenAlex 인용 그래프 역순회로 확보 (§0·§1·§2를 보정한다)

조사 후반에 OpenAlex 단일 레코드 경로(`works/doi:<DOI>`, `works/W<id>`)로 [2]의 참조 목록을 역순회해
**[2]보다 훨씬 강한 1차 출처를 찾았다.** 조사 순서상 뒤에 붙었지만 **내용상 §1 증거 표의 연장**이다. 아래 [19][20]이 이 조사의 실질적 핵심 근거이며,
§0·§2·§3에서 [2]에 기댄 부분은 [19][20]으로 대체하거나 보강해야 한다.

| # | 제목 | 저자·연도 | 게재지·등급 | 중요도 측정치 | 응용 영역 | 상호의존 종류 | 우리 `c_i`와의 관계 | URL | 읽은 수준 |
|---|------|-----------|-------------|---------------|-----------|---------------|---------------------|-----|-----------|
| 18 | ON THE IMPORTANCE OF DIFFERENT COMPONENTS IN A MULTICOMPONENT SYSTEM | Birnbaum 1968-05-20 (1969 챕터의 기술보고서판) | DTIC 기술보고서 AD0670563 (Univ. of Washington Lab of Statistical Research, ONR 지원) | **Birnbaum 중요도 원전** | 신뢰도 이론 | 해당 없음 | 우리 `c_i`의 개념 원전. **DOI가 등록돼 있어 서지가 검증된다**(DTIC citation 페이지로 302 리다이렉트) | https://doi.org/10.21236/ad0670563 → https://apps.dtic.mil/sti/citations/tr/AD0670563 | **서지 검증(Crossref·DOI 리다이렉트), 전문 403** |
| 19 | A Relationship Between Partial Derivatives of the Reliability Function of a Coherent System and its Minimal Path (Cut) Sets | El-Neweihi 1980 | **Mathematics of Operations Research** 5(4):553 (INFORMS, OR 최상위) | r차 편도함수(= r-요소 joint importance) | 일관 시스템 이론 | **부호로 구조를 특성화** | **이 조사 최강 인용.** 초록 verbatim: r차 편도함수의 부호가 최소 절단(경로)집합의 크기를 함의하며, "**When r = 2, a simple characterization is obtained for series (parallel) system.**" 즉 JRI 부호의 일률성이 직렬·병렬 여부를 결정한다 | https://doi.org/10.1287/moor.5.4.553 | **Crossref JATS 초록 verbatim 확보**, 전문 미독 |
| 20 | L-superadditive structure functions | Block, Griffith, Savits 1989 | **Advances in Applied Probability** 21:919–929 | (구조함수의 초모듈성) | 신뢰도 구조 이론 | **대체/보완을 구조와 직결** | **두 번째 최강 인용.** 초록 verbatim: "L-superadditive functions are also known under the names **supermodular**, quasi-monotone and superadditive"; "**For binary structure functions of binary values, El-Neweihi (1980) showed that L-superadditive structure functions must be series.**"; "**In the case of non-binary-valued structure functions this is no longer the case.**" → 신뢰도 이론과 D 조사의 초모듈성 언어를 **명시적으로 같은 것**으로 잇는다 | https://doi.org/10.2307/1427774 | **Crossref JATS 초록 verbatim 확보**, 전문 미독 |
| 21 | Joint reliability importance of k-out-of-n systems | Hong, Koo, Lie 2002 | **EJOR** 142(3):539–547 (피인용 47) | JRI | k-out-of-n 구조 | 혼재 | JRI의 OR 저널 계보. k-out-of-n은 직렬·병렬의 일반화이므로 우리 구조에 가깝다. **초록 미확보 → 내용 인용 금지** | https://doi.org/10.1016/S0377-2217(01)00306-X | **서지만** |
| 22 | Joint importance of multistate systems | Wu 2005 | **Computers & Industrial Engineering** 49(1):63–75 (피인용 70) | 다상태 joint importance | 다상태 시스템 | 혼재 | 우리 수량·용량 구조에 맞는 일반화 후보. **초록 미확보 → 내용 인용 금지** | https://doi.org/10.1016/j.cie.2005.02.001 | **서지만** |
| 23 | Reliability Importance and Invariant Optimal Allocation | Lin, Kuo 2002 | **Journal of Heuristics** 8:155–171 | Birnbaum 기반 배분 휴리스틱 | 신뢰도 배분 최적화 | 혼재 | F3b "최적화에 쓴 선례"의 검증된 서지(그간 검색 요약으로만 언급되던 LK 휴리스틱). **초록 미확보 → 내용 인용 금지** | https://doi.org/10.1023/A:1017908523107 | **서지만** |

**결론 보정 (중요)**

1. §0-2에서 "부호 패턴이 [2]에서 알려져 있다"고 적었으나, **더 정확하고 더 강한 진술은 [19][20]에 있다**:
   **이진 구조함수가 초모듈(L-superadditive)이면 반드시 직렬 시스템이어야 한다** ([20]이 [19]에 귀속).
   우리 구조는 순수 직렬이 아니므로(품목 내 공급자가 병렬), **이진 대응물로 볼 때 우리 구조함수는 초모듈일 수 없다.**
   → D 조사의 "고리 B(가치함수가 초모듈)는 지지되지 않는다"는 결론이 **정리 수준에서 확증된다.**
2. 그러나 [20]은 곧바로 **"비이진 값 구조함수에서는 이 제약이 성립하지 않는다"** 고 명시한다.
   우리 V는 연속 이윤(비이진)이므로 **이 금지 정리는 우리에게 그대로 적용되지 않는다.**
   즉 "우리 V가 초모듈일 수 있는가"는 **이진 이론이 금지하지만 다상태 이론은 열어둔다** — 여전히 미결이며 측정 대상이다.
   이것이 F3에서 가장 정직하고 유용한 결론이다.
3. [2]의 부분시스템별 부호 진술은 여전히 **쌍별 패턴**(같은 품목 = 대체, 다른 품목 = 보완)의 근거로 유용하지만,
   "부분시스템 간 공통 요소 없음" 가정 때문에 우리 구조에 직접 적용할 수 없다(§4 마지막 항목).
   **[19]는 그 가정 없이 일반 일관 시스템에 대한 진술이므로 더 안전하다.**

---

## 3. 부호 판정 절 (핵심 산출물)

### (a) 단일 요소 Birnbaum 중요도와 "독립 가치"의 관계

**인용 가능한 정의와 공식** ([1], 전문에서 직접 추출한 verbatim):

> "Birnbaum (1969) proposed the following measure of the reliability importance of component i at time t:
> `I_B(i|t) = ∂h(p(t))/∂p_i(t)`"
> "By pivotal decomposition ... Birnbaum's measure can therefore we written as
> `I_B(i|t) = ∂h(p(t))/∂p_i(t) = h(1_i, p(t)) − h(0_i, p(t))`
> where `h(1_i, p(t))` is the system reliability when we know that component i is functioning and
> `h(0_i, p(t))` is the system reliability when we know that component i is not functioning."
>
> 직렬 2요소: "The reliability of the series system is `h(p) = p_1 p_2`. ...
> Birnbaum's measure of component 1 is `I_B(1) = ∂h(p)/∂p_1 = p_2`"
> 병렬 2요소: "The reliability of the parallel system is `h(p) = p_1 + p_2 − p_1 p_2`. ...
> Birnbaum's measure of component 1 is `I_B(1) = ∂h(p)/∂p_1 = 1 − p_2`"

**우리 `c_i`와의 동일시** (이하 §3의 도출은 **우리 추론**이며, 인용 가능한 것은 위 정의·공식과 (b)의 [19][20][2]뿐이다)

집합함수 `V`의 다선형 확장을 `F(p) = E[V(R(p))]`라 하자 (정의는 `D[18]`에서 확인).
신뢰도함수 `h(p) = E[φ(X)]`(X_i ~ Bern(p_i) 독립)는 정확히 구조함수 `φ`의 다선형 확장이다. 그러면
`∂F/∂p_i = E[V(R ∪ i) − V(R ∖ i)]`이고, p = 1(다른 참여자 전원 확실히 존재)에서
`∂F/∂p_i |_{p=1} = V(S) − V(S∖{i}) = c_i`.
**즉 `c_i`는 Birnbaum 중요도를 p = 1 코너에서 평가한 값이다.**
(`D[18]`의 다선형 확장과 D §3-(iii)(iv)의 "우리 MILP 대리값 = F의 p = 1 1차 Taylor"라는 도출이 여기서
신뢰도 이론과 정확히 맞물린다. 즉 **우리 MILP의 미래항 `Σ c_i p_i`는 신뢰도함수를 전원 참여점에서
Birnbaum 중요도로 1차 선형화한 것**이다.)

**부호 판정 (a): 증분 vs 독립**

이진 구조함수로 보면 `V(A) = φ(1_A)`, `c_i = φ(1) − φ(1_{S∖i})`, 독립 가치 `= φ(1_{{i}}) − φ(0)`.

| 구조 | 구조함수 | `c_i` (증분) | 독립 가치 | 판정 |
|---|---|---|---|---|
| **직렬 2요소 (보완)** | `φ(A)=1 iff A=S` | `1 − 0 = 1` | `0 − 0 = 0` | **증분 > 독립** |
| **병렬 2요소 (대체)** | `φ(A)=1 iff A≠∅` | `1 − 1 = 0` | `1 − 0 = 1` | **증분 < 독립** |

Birnbaum 중요도를 p의 함수로 보면 방향이 더 명확하다:
- **직렬(보완)**: `I_B(i)(p) = Π_{j≠i} p_j` 는 p에 대해 **증가**, p = 1에서 **최대값 1**.
  → `c_i = I_B(i)|_{p=1}`은 가능한 한계값 중 **최대** → 전형적 한계값을 **과대평가**.
- **병렬(대체)**: `I_B(i)(p) = Π_{j≠i}(1 − p_j)` 는 p에 대해 **감소**, p = 1에서 **최소값 0**.
  → `c_i`는 가능한 한계값 중 **최소** → 전형적 한계값을 **과소평가**.

**이는 D 조사 §3 고리 C의 (iv)와 정확히 같은 결론이며, 신뢰도 쪽 공식은 그것을 닫힌형으로 보여준다.**
D는 열모듈/초모듈 일반론(`D[1]`)에서 부호를 얻었고, 여기서는 직렬·병렬의 명시적 공식에서 같은 부호를 얻는다.

### (b) joint importance의 부호 — 두 구조에서 다른가

**다르다. 그리고 이것이 인용 가능한 형식 결과다. 강도 순으로 세 층이 있다.**

**정의**: `JRI(i,j) = ∂²h(p) / ∂p_i ∂p_j` ([2]가 쓰는 표준 정의, 개념 원전 [3][4], 원전의 원전은 [18]).

**층 1 (가장 강함) — El-Neweihi (1980, Mathematics of Operations Research) [19]**
초록 verbatim:
> "A relationship between rth partial derivatives of the reliability function h(p) of a coherent system and
> its minimal cut (path) sets is studied (r ≥ 2). It is shown that `(−1)^r ∂^r h(p)/∂p_{i_1} … ∂p_{i_r}`
> (`(−1) ∂^r h(p)/∂p_{i_1} … ∂p_{i_r}`) is nonnegative for all `i_1, …, i_r` and all p implies that all minimal
> cut (path) sets have cardinality r − 1 or less, r = 2, …, n. **When r = 2, a simple characterization is
> obtained for series (parallel) system.** Interpretations of such characterization in terms of reliability
> importance of components are given and a simple proof of a known characterization of series system is obtained."

**우리 읽기 (r = 2 적용, 추론이지만 초록 문장에서 직접 따온다)**: r = 2에서 `∂²h/∂p_i∂p_j ≥ 0`이
모든 쌍과 모든 p에서 성립하면 모든 최소 절단집합의 크기가 1 이하 = **순수 직렬 시스템**이고,
`∂²h/∂p_i∂p_j ≤ 0`이 모든 쌍·모든 p에서 성립하면 모든 최소 경로집합의 크기가 1 이하 = **순수 병렬 시스템**이다.
→ **JRI의 부호가 일률적으로 양이면 직렬, 일률적으로 음이면 병렬이며, 그 밖의 구조에서는 부호가 섞인다.**

**층 2 — Block, Griffith, Savits (1989, Advances in Applied Probability) [20]**
초록 verbatim:
> "Such functions describe whether a system is more series-like or more parallel-like. **L-superadditive
> functions are also known under the names supermodular**, quasi-monotone and superadditive and have been
> studied by many authors. ... **For binary structure functions of binary values, El-Neweihi (1980) showed that
> L-superadditive structure functions must be series.** This continues to hold for binary-valued structure
> functions even if the component values are continuous (see Proposition 3.1). **In the case of
> non-binary-valued structure functions this is no longer the case.**"

이것이 신뢰도 이론과 D 조사의 집합함수 언어를 **명시적으로 같은 것으로 잇는 문장**이다
("L-superadditive = supermodular"). 그리고 두 방향을 동시에 준다:
- **이진 값 구조함수**: 초모듈 ⟹ 직렬. 우리 구조는 직렬이 아니므로 **초모듈일 수 없다.**
- **비이진 값 구조함수**: 이 제약이 **성립하지 않는다.** 우리 V는 연속 이윤이므로 **금지되지 않는다.**

**층 3 — Jain, Dewan, Rani (2014, IJQRM) [2]**: 구조별 쌍 분류.
parallel-in-series(= 우리 구조)에서 같은 부분시스템 내 쌍은 JRI 비양(reliability substitutes),
다른 부분시스템 간 쌍은 JRI 비음(reliability compliments). series-in-parallel은 반대.
해석: "one component becomes more important as the other one works"(compliments) /
"one component becomes more important as the other one fails"(substitutes).
**단 "부분시스템 간 공통 요소 없음" 가정 때문에 우리 구조에 직접 적용할 수 없다(§4).**

**2요소 직접 계산 (우리 검증, [1]의 h 공식에서 바로 나옴)**
- 직렬: `h = p_1 p_2` → `∂²h/∂p_1∂p_2 = +1 > 0`.
- 병렬: `h = p_1 + p_2 − p_1 p_2` → `∂²h/∂p_1∂p_2 = −1 < 0`.

**우리 구조에 대입 (우리 판정)**

| 공급자 쌍 | 신뢰도 구조상 관계 | JRI 부호 | 국소 모듈성 | D 조사 언어 |
|---|---|---|---|---|
| **같은 품목의 두 공급자** | 같은 병렬 부분시스템 내 | **≤ 0** (substitutes) | 열모듈(submodular) | 대체 → `c_i` 과소평가, 다수 이탈 손실 과소평가 |
| **다른 품목의 두 공급자** | 직렬로 연결된 서로 다른 부분시스템 | **≥ 0** (compliments) | 초모듈(supermodular) | 보완 → `c_i` 과대평가, 다수 이탈 손실 과대평가 |

`JRI(i,j) = ∂²h/∂p_i∂p_j`는 다선형 확장의 교차 2차 편도함수이므로 부호가 2차 차분
`Δ_{ij}V = V(S) − V(S∖i) − V(S∖j) + V(S∖{i,j})`의 부호와 같다(우리 도출).
**즉 신뢰도 이론의 JRI 부호와 D 조사의 초/열모듈성 부호는 같은 양이다** — 이 등치는 [20]의
"L-superadditive = supermodular" 문장으로 **문헌 안에서 확인된다.**

### (c) D 조사 결론과의 대조

| 항목 | D 조사 결론 (`D[1]`, `D[3]`, `D[9]`, `D[10]`) | 신뢰도 이론 ([1][19][20][2]) | 일치? |
|---|---|---|---|
| 대체(중복) 성분의 편향 방향 | 가산 대리값이 축소 집합 가치를 과대평가 = **다수 이탈 손실 과소평가**; `c_i ≤` 진짜 한계값 | 병렬 `I_B(i)=Π(1−p_j)`가 p=1에서 **최소**[1] → `c_i` 과소평가; JRI < 0 | **일치** |
| 보완 성분의 편향 방향 | **과대평가** | 직렬 `I_B(i)=Π p_j`가 p=1에서 **최대**[1] → `c_i` 과대평가; JRI > 0 | **일치** |
| "우리 가치함수는 초모듈이다"라는 주장 | **지지되지 않음**(혼재; γ<1 이면서 α>0) | **금지된다(이진 한정)**: 초모듈 이진 구조함수는 반드시 직렬 [20]→[19] | **일치, 그리고 D보다 강함** |
| 우리 시장의 순 부호 | 부류 차원에서 결정 불가, **측정으로만** | 쌍별 부호는 구조로 결정, **합은 결정되지 않음**; 비이진에서는 초모듈 금지도 풀린다 [20] | **일치** |
| G1의 "증분 > 독립" | 보완 성분 방향이며 대체 성분이 반대로 작용 | 직렬에서 증분 1 > 독립 0, 병렬에서 0 < 1 (우리 도출, [1] 공식 기반) | **일치** |

**신뢰도 계보가 D에 보태는 것 (D에는 없던 것)**
1. **부호 패턴이 구조로부터 결정된다**는 점. D는 "혼재"에서 멈췄지만 [19]는 "일률적 부호 ⟺ 순수 직렬/병렬"이라는
   특성화를 주므로, **혼재가 우연이 아니라 구조적 필연**임을 말할 수 있다.
2. **"초모듈"이라는 표현을 쓰면 안 되는 이유의 정리 수준 근거**: [20]→[19]의 "초모듈 이진 구조함수 ⟹ 직렬".
3. **그 제약이 우리에게 그대로 적용되지 않는 이유**: [20]의 "비이진 값에서는 성립하지 않는다".
   → 우리는 "이진 근사에서는 금지되지만 실제 연속 가치함수에서는 열려 있다"고 정확히 쓸 수 있다.
4. **용어**: `c_i` = Birnbaum 중요도(p=1), 묶음 = parallel-in-series, 같은 품목 쌍 = reliability substitutes,
   다른 품목 쌍 = reliability compliments. 네 용어 모두 확립된 문헌 용어다.

**측정 제안 (D의 측정 제안을 구체화 — 이 조사의 실행 산출물)**

`src/vfa/coef.py`의 `raw()`는 1명 제거만 계산한다. 2명 제거를 추가해
`Δ_{ij}V = V(S) − V(S∖i) − V(S∖j) + V(S∖{i,j})`를
**(A) 같은 품목을 공급하는 공급자 쌍**과 **(B) 겹치는 품목이 없는 공급자 쌍**으로 나눠 재면,
이론 예측은 **A는 ≤ 0, B는 ≥ 0**이다. 같은 것을 시뮬레이션 기준치(`mc_coefs`)에서도 재면
모델 오차와 구조 효과를 분리할 수 있다.
- 부호가 예측과 맞으면: JRI 언어로 우리 구조를 설명할 수 있고, 실험 3에서 "구조 정보"가 무엇인지 구체화된다.
- 부호가 반대면: (i) 용량 여유 1.3 때문에 같은 품목 두 공급자가 실제로는 대체가 아니거나
  (ii) 학습 모델이 구조를 못 담고 있다는 신호다.
- **이 측정은 실험 3의 "유지 가치 변환에서 구조 정보가 소실되었는가"에 직접 답한다.**
- 비용: 공급자 쌍 15개(공급자 6) × 추가 예측 1회. `coef.py: raw()`의 `phi_drops`/`enc` 경로를
  2명 제거로 확장하면 되고 MILP는 건드리지 않는다.

### (d) 형식 결과로 인용할 수 없는 것 — 명시

- **"직렬 구조에서 한 요소의 Birnbaum 중요도가 그 요소 단독 신뢰도를 넘어선다"는 형태의 정리: 없다.**
  신뢰도 문헌은 "요소의 독립(단독) 가치"라는 양을 정의하지 않는다. §3 (a)의 표는 **우리가 구조함수에서
  직접 계산한 것**이며, 인용 가능한 것은 그 계산에 쓰인 h와 `I_B` 공식뿐이다 [1].
- **"parallel-in-series 시스템에서 순(net) 편향의 부호"를 준 결과: 없다.** [19]는 일률적 부호의 특성화만,
  [2]는 쌍별 부호만 준다.
- **[19]의 r = 2 해석은 초록 문장에 근거한 우리 읽기다.** 전문을 읽지 않았으므로 정리 번호·정확한 가정을
  인용할 수 없다. 논문에 쓸 때는 전문 확인이 필요하다.
- Hong & Lie (1993) [3]와 Armstrong (1995) [4] 원문을 읽지 못했으므로 **부호 정리를 이 두 편에 귀속하면 안 된다.**
- [20]의 Proposition 3.1(연속 요소값·이진 시스템값)과 "비이진에서는 성립하지 않는다"의 **정확한 반례는 미확인**이다.

### (e) 대응이 깨지는 지점 (정직한 한계)

1. **이진 vs 다상태.** 우리 품목에는 **수량과 용량**이 있다. CLAUDE.md 기준 품목별 총 공급용량 151,
   품목당 공급자 2명(n_alt = 2), 안정 상태 품목별 수요 70. 주문 한 건의 평균 수량은 k = 1에서 21로
   총 용량의 1/4 이하이므로 **주문 한 건 수준에서는 한 공급자 단독 충족이 가능한 경우가 많아 병렬 근사가 그럴듯하다.**
   그러나 **기간 전체 수요(품목당 70) 대비로는 한 공급자만으로 부족할 수 있어** 이진 병렬이 깨진다.
   (이 문단은 CLAUDE.md 수치에서 한 **우리 추론**이고 문헌 주장이 아니다.)
   올바른 일반화는 다상태 시스템 [6][14][22] 또는 흐름망 신뢰도 [17]이며, [14]는 다상태에서
   **측정치마다 순위가 달라질 수 있다**고 보고한다. 동시에 [20]에 따르면 비이진으로 가면
   "초모듈 ⟹ 직렬" 금지가 풀리므로, **다상태로의 이동은 우리에게 불리하기만 한 것이 아니다.**
2. **제거 ≠ 고장.** 우리는 이탈 후 **빈 자리가 매 기간 확률 0.25로 충원**된다. 이는 수리가능(repairable)
   다상태 과정에 해당하고 [14], 이진 Birnbaum의 "영구 고장" 가정과 다르다.
3. **정적 신뢰도 ≠ 동적 가치.** h는 한 시점의 성립 확률이고 우리 V는 **다기간 누적 이윤**이다.
   재참여 동역학(로지스틱 반응, 프로필 변동)은 신뢰도 구조에 대응물이 없다.
4. **p_i가 외생이 아니다.** 신뢰도 이론에서 p_i는 주어진 값이고 중요도는 민감도다. 우리 MILP는
   **잉여배분으로 p_i를 선택**한다. 이 점에서는 신뢰도 배분 최적화 [12][23]에 가깝지만,
   그쪽은 비용-신뢰도 정적 배분이며 배분받은 잉여가 참여 확률을 통해 미래 참여자 집합을 바꾸는 구조는 아니다.
5. **주문자 쪽은 대응이 더 약하다.** 우리 `c_i`는 주문자에 대해서도 계산되지만, 신뢰도 구조에서
   주문자는 "시스템"이지 "요소"가 아니다. 주문자 노드를 요소로 볼 근거는 이 조사에서 찾지 못했다.
6. **일관성(coherence) 가정.** [19][20]은 coherent system(단조 + 모든 요소가 관련)을 가정한다.
   우리 V가 참여자 집합에 대해 단조인지는 확인되지 않았다 — 용량 과잉 시 공급자 추가가 이윤을 낮출 수도 있다
   (잉여 경쟁). **단조성이 깨지면 위 결과 전부가 적용 불가다. 이것을 먼저 수치로 확인해야 한다.**

---

## 4. 미해결·확인 불가 (남긴 쿼리 포함)

| 항목 | 상태 | 확인해야 할 질문 |
|---|---|---|
| **[19] El-Neweihi (1980, MOR) 전문** | **초록만 확보(Crossref JATS), 전문 미독** | **1순위.** r = 2 특성화의 정확한 정리 진술·가정(coherence, 독립성). 이 조사 최강 인용의 근거이므로 반드시 전문 확인 |
| **[20] Block·Griffith·Savits (1989, AAP) 전문** | **초록만 확보, 전문 미독** | **2순위.** Proposition 3.1의 정확한 진술과 "비이진에서는 성립하지 않는다"의 반례. 우리가 "연속 가치함수에서는 초모듈이 금지되지 않는다"고 쓰려면 필요 |
| [13] Zhu & Kuo (2012, Annals of OR) "Importance measures in reliability and mathematical programming" | **초록조차 확보 실패**(Springer 쿠키 리다이렉트, Crossref·S2 초록 없음) | 중요도를 **수리계획 목적함수 계수**로 쓰는가? F3b의 "없다" 단정을 깨뜨릴 수 있는 문헌 |
| [5] Gao·Cui·Li (2007, EJOR) | **초록 확보 실패** | 일관 시스템 joint importance의 **일반 부호 정리**가 있는가? |
| [2] Jain 외 (2014) 전문 | **전문 미독** | **부호 결과의 가정**: 초록에 "The subsystems do not have any component in common"이 명시돼 있다(§4 마지막 항목 참조) |
| [3] Hong & Lie (1993), [4] Armstrong (1995), [21] Hong·Koo·Lie (2002, EJOR) 원문 | **미독(IEEE·Elsevier 유료)** | "series → JRI > 0, parallel → JRI < 0"을 원전이 직접 진술하는가? 현재 귀속 가능한 것은 [19][2]뿐 |
| [22] Wu (2005, C&IE) "Joint importance of multistate systems" | **서지만** | 다상태에서 JRI 부호 결과가 보존되는가? (e)1의 핵심 질문 |
| [17] Aven & Østebø (1986) 흐름망 중요도 | **초록 미독** | 용량 있는 흐름망 중요도가 우리 수량 구조에 맞는가 |
| [15] Freixas & Puente (2002), [16] Marichal & Mathonet | **초록 미독** | Barlow–Proschan = Shapley 일치의 정확한 조건. "왜 Shapley가 아닌가" 논증에 필요 |
| [18] Birnbaum (1968, DTIC AD0670563) 전문 | **DOI·서지 검증 완료, 전문 403** | 원전 확인. DTIC 전문 PDF가 열리지 않았다(403) |
| "공급자 유지를 위한 금전 인센티브 + 신뢰도 중요도" 결합 | **0건** | 쿼리 `Birnbaum importance supplier retention critical supplier incentive procurement dynamic model` → 관련 결과 없음 |
| Birnbaum 중요도 × ADP/가치함수 근사 | **0건 (없다고 단정)** | 쿼리 4개(§2 F3a·F3b) 모두 0건 |
| 우리 V의 **참여자 집합에 대한 단조성** | **미확인 (수치 확인 필요)** | (e)6. 단조성이 깨지면 [19][20] 전부 적용 불가. `mc_coefs`의 `c_i`가 음수로 나오는 빈도로 즉시 확인 가능 |

### OpenAlex 순회 기록 (투명성)

- `search=` 및 `filter=` **목록 엔드포인트는 일일 무료 예산 소진으로 차단**되었다
  (응답: "Insufficient budget ... daily budget ... resets at midnight UTC", `retryAfter` 52614초).
  따라서 부모가 지시한 `filter=cites:W<id>` **전방 피인용 순회는 수행하지 못했다.**
- 대신 **단일 레코드 경로는 통과**했다: `works/doi:<DOI>`, `works/W<id>?select=...`.
  이를 이용해 **[2]의 `referenced_works` 22건 전부를 역순회**하여 [18][19][20][21][22]를 찾았다.
  이 역순회가 이 조사에서 가장 생산적인 단계였다.
- 전방 피인용은 **Semantic Scholar `/citations` 엔드포인트**로 대체했다([19]의 피인용 9건 확인 →
  [20] 및 Lin–Kuo (2002) [23] 발견).
- 미수행: [18] Birnbaum 원전과 [19][20]의 **전방 피인용 대량 순회**. 이것을 하면
  "신뢰도 중요도를 공급망·조달 최적화에 쓴 응용"을 더 체계적으로 걸러낼 수 있다. **후속 과제로 남긴다.**
- Crossref에 요청할 때 초기 몇 회에 사용자 이메일을 User-Agent의 `mailto`로 넣었다.
  지시를 받은 이후로는 넣지 않았다. **기록으로 남긴다.**

### 가장 위험한 미확인 항목 (강조)

[2]의 초록에 **"The subsystems do not have any component in common"** 이 명시돼 있다.
우리 시장에서는 **한 공급자가 여러 품목을 공급**(공급자당 평균 품목 수 = 품목 수 × n_alt / 공급자 수 = 2)하므로
**부분시스템이 요소를 공유한다.** 즉 [2]의 구조별 부호 진술을 우리 구조에 그대로 적용할 수 없다.
2요소 직접 계산(§3 (b))과 [19]의 일반 특성화는 여전히 유효하지만,
**공유 요소가 있는 parallel-in-series에서 쌍별 부호 패턴이 유지되는지는 확인되지 않았다.**
공급 편중도(conc)를 올리는 것이 바로 이 공유를 늘리는 조작이므로 **이 미확인 항목은 실험 3의 가설과 직결된다.**
[2] 전문 확인 또는 우리 쪽 직접 도출이 필요하다.

---

## 5. G1에 대한 영향

- G1의 현상 문장("보완형 상호의존 때문에 증분 가치가 독립 가치를 넘어선다")은 신뢰도 이론의
  **직렬 구조 예제로 정확히 재현되며**(§3 (a)), 대체형(병렬)에서 반대 부호가 되는 것도 같은 공식에서 나온다.
  **부호 반전 주장 자체는 유지할 수 있다.**
- 단 D 조사와 같은 제약이 그대로, 오히려 더 강하게 남는다: [20]→[19]에 따르면 **이진 구조에서 초모듈은
  순수 직렬만 허용**되므로, "우리 가치함수가 보완형(초모듈)이다"라고 쓰면 **정리와 충돌**한다.
  써야 하는 형태는 "**구조상 대체 쌍과 보완 쌍이 구별되며 부호가 반대다**"이다.
- 반면 [20]의 "비이진 값에서는 성립하지 않는다"는 **우리에게 숨 쉴 틈을 준다**: 연속 가치함수가
  초모듈일 가능성을 이론이 배제하지 않는다. 다만 이는 **가능성**이며 우리는 측정해야 한다.
- 신뢰도 계보가 G1에 보태는 실질은 **선례의 언어와 정리 수준 경계**다:
  `c_i` = Birnbaum 중요도(p=1) [1][18], 묶음 = parallel-in-series [2][10],
  같은 품목 쌍 = reliability substitutes / 다른 품목 쌍 = reliability compliments [2],
  "초모듈 = L-superadditive" [20], 그리고 **"일률적 부호는 순수 직렬/병렬만"** [19].
- "신뢰도 이론이 우리 부호 논증을 증명한다"고 쓰면 과장이다. §3 (d)의 다섯 항목이 인용 불가다.

---

## 6. Coverage Status

**직접 확인한 것 (전문 또는 verbatim 초록)**
- Birnbaum 중요도 정의와 직렬·병렬 닫힌형 공식 — [1] 전문 텍스트 추출(pypdf), [7] 정의 절
- JRI 부호의 구조 특성화 — [19] Crossref JATS 초록 verbatim, [20] Crossref JATS 초록 verbatim
- parallel-in-series / series-in-parallel 쌍별 부호 — [2] WebFetch 추출
- 공급자 leave-one-out 중요도 정의 — [11] 슬라이드 전문 텍스트 추출(정의 4.1·4.2·5.1·5.2)
- 공급망 신뢰도 구조모형과 공급자 선택의 parallel–series 정식화 — [8][10] 초록
- 다상태·수리가능 확장의 경고 — [14] 초록; 다상태 joint importance 존재 — [6] 초록
- Birnbaum 원전의 검증된 서지와 DOI — [18]

**불확실하게 남은 것**
- [19][20] 전문(정리 진술·가정) — 이 조사 최강 인용의 전문 미독. **논문 인용 전 필수 확인**
- [2]의 "부분시스템 간 공통 요소 없음" 가정이 우리 구조와 충돌 — 미해결(§4 강조 항목)
- [13] Zhu & Kuo (2012, Annals of OR) 내용 전무 — F3b 단정의 유일한 잔여 위험
- 우리 V의 참여자 집합에 대한 단조성 — 미확인. 깨지면 §3 전체가 적용 불가

**완료하지 못한 작업**
- OpenAlex `filter=cites:` **전방 피인용 순회**: 일일 무료 예산 소진으로 목록 엔드포인트 차단
  (`retryAfter` 52614초). [18][19][20]의 전방 피인용 대량 순회를 수행하지 못했다.
  Semantic Scholar `/citations`로 [19] 1건만 대체 수행했다. **UTC 자정 이후 재시도 가치가 높다.**
- DTIC 전문 PDF(AD0670563) 403.

**질문별 상태**: F3a = done · F3b = done(잔여 위험 [13]) · F3c = done · F3d = done(전문 확인 필요)
