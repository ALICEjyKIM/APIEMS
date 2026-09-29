# 3차 조사 F2: 묶음 구조의 ADP (네트워크 수익관리 계보) — 증거 수집

착수 2026-09-28. 과제: `ref/literature-review.md` "3차 조사 범위 (주제 F)" 중 **F2a~F2d**.
축: 찾은 문헌마다 자원 간 **상호의존의 종류**(대체형 / 보완형 / 혼재 / 해당 없음)를 분류한다.

중복 회피 확인: `..._research_B.md` §2 B3·B4(SPAR, Topaloglu–Powell, discrete semigradient)와
`..._research_D.md` §2 D1~D7(초모듈성 부호, Chen–Long–Qi, Song 1998, Iyer 외)는 다시 다루지 않는다.
이 파일은 **네트워크 수익관리(NRM) 쪽 계보**와 **ATO 근사 알고리즘**만 다룬다.

## 0. 결론 먼저

**F2b의 답은 있다. 그리고 매우 강하다.**
Bertsimas–Popescu (Transportation Science 2003)는 "여정이 여러 구간을 쓰므로 구간별 가산(additive)
bid price 근사가 묶음 효과(bundle effects)를 놓친다"를 **초록·서론·§3.2에서 반복해 명시**하고,
그 개선폭을 **고부하 시 평균 5~10%, 확장판 최대 20%**로 보고한다.
Zhang (M&SOM 2011)은 같은 문제의식을 **비분리형(nonlinear nonseparable) 근사**로 풀고
classical 분리형 DP 분해 대비 **최대 8%** 개선을 보고한다.

**F2d(부호 반전)의 답도 있다. 이것이 이번 조사의 가장 큰 수확이다.**
Bertsimas–Popescu의 Proposition 3은 가산 bid price가 참 기회비용의 **하한**임을 일반적으로 진술하고,
§4.1의 3구간 예제는 구간 2 용량의 참 기회비용이 그 구간 단독 운임 R2를 **초과**하며
2구간 여정 (13)의 기회비용이 그 여정 운임 R13을 **초과**함을 닫힌 형태로 보인다.
저자들은 같은 예제로 가치함수의 **decreasing differences(감소 차분 = 열모듈 방향)가 위반된다**고 적지만,
PDF 추출본의 부등호 방향이 그 주장과 어긋나 보여 **이 부분은 조판 원문 확인이 필요하다** (2절 F2d (다), 4.1절).
즉 "묶음이 있으면 자원의 한계 가치가 그 자원 단독 수익을 넘고, 가산 근사가 체계적으로 과소평가한다"는
우리 G1의 부호 문장은 **NRM 문헌에 이미 형식 결과와 반례로 존재한다.**

이것은 양날이다. 신규성 주장에는 불리하고(선례 존재), 논증 근거로는 매우 유리하다(인용 가능).
3절에 인용 가능한 원문 문장을 모으고 "우리가 쓸 수 있는 최선의 문장"을 지목했다. 4절은 미해결·확인 불가 목록이다.

또한 **부호 진술의 근거를 구분해 두었다**: Proposition 3의 하한 방향은 오목성만으로도 나올 수 있으므로,
보완성 고유의 증거로는 3구간 예제의 닫힌 형태만 쓴다 (2절 F2d 참조).

---

## 1. 증거 표

등급: OR/MS/M&SOM/TS/MOR = INFORMS 주요지(Q1), EJOR·POM = SSCI Q1, WP = 미심사 워킹페이퍼.
읽은 수준: **전문추출** = PDF 텍스트를 직접 추출해 해당 문장을 읽음 / 서지 = Crossref 서지만 / 검색 = 검색 스니펫만.

| # | 제목 | 저자·연도 | 게재지·등급 | 근사 형태 | 분리형 한계 지적 | 보고된 개선 폭 | 상호의존 종류 | 우리와의 차이 | URL | 읽은 수준 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Revenue Management in a Dynamic Network Environment | Bertsimas, Popescu 2003 | Transportation Science 37(3) 257–277, Q1 | **비가산** certainty equivalent control (CEC): V(n,t) ≈ LP(n, D_t) 값, 기회비용 = LP 차분 | **명시. 초록·서론·§3.2·§4 반복** | 고부하 평균 **5–10%**, 확장판 **최대 20%** (vs 가산 bid price) | **보완형**(다구간 여정 = 여러 자원 동시 소비) + 용량 경쟁(대체형) 혼재 | 자원 집합이 외생 용량 벡터이고 참여자 재참여 같은 결정 의존 전이가 없다 | https://doi.org/10.1287/trsc.37.3.257.16047 | **전문추출** (MIT 저자 사본) |
| 2 | An Improved Dynamic Programming Decomposition Approach for NRM | Zhang 2011 | M&SOM 13(1) 35–52, Q1 | **비선형·비분리형**: v_t(x) ≈ min_i { v̂_{t,i}(x_i) + Σ_{k≠i} π*_k x_k } | **명시.** classical 분해는 "network effect를 운임 안분(fare proration)으로만 포착" | 정책 **최대 8%** 개선(vs DCOMP), 계산시간 +30%, 상한도 더 타이트 | **보완형**(여러 구간 동시 소비) | 선택모형(MNLD) 기반 용량통제이고 잉여배분·참여자 유지가 없다 | https://doi.org/10.1287/msom.1100.0302 | **전문추출** (저자 사본) |
| 3 | An Approximate Dynamic Programming Approach to NRM | Farias, Van Roy 2007 | WP (미심사, MIT 사본) | **분리형 오목** 구간선형(자원별) + ALP·제약표본 | 부분. "아핀 근사는 J*의 오목성을 포착할 수 없다"를 명시 | DLP 대비 동질 도착 **~1%**, Markov 변조 도착 **최대 ~8%** | 보완형(여정=자원 묶음) | 분리형을 **쓰는** 쪽이고, 분리형의 보완성 손실 자체는 논점이 아니다 | https://web.mit.edu/vivekf/www/papers/ADP-rm-07-03.pdf | **전문추출** |
| 4 | An Analysis of Bid-Price Controls for NRM | Talluri, van Ryzin 1998 | Management Science 44(11) 1577–1593, Q1 | DLP 기반 정적 bid price(가산·분리형) | 확인 필요(페이월) | 확인 필요 | 보완형+대체형 혼재 | — | https://doi.org/10.1287/mnsc.44.11.1577 | 서지 (Crossref) |
| 5 | Dynamic Bid Prices in Revenue Management | Adelman 2007 | Operations Research 55(4) 647–661, Q1 | **아핀 근사** V_t(x) ≈ θ_t + Σ_l v_{lt} x_l, ALP 열생성 | 확인 필요(페이월) | 확인 필요 | 보완형+대체형 혼재 | — | https://doi.org/10.1287/opre.1060.0368 | 서지 (Crossref) |
| 6 | An ADP Approach to NRM with Customer Choice | Zhang, Adelman 2009 | Transportation Science 43(3) 381–394, Q1 | 아핀 근사 + 선택모형, DP 분해 상한 | 확인 필요(페이월). [2]가 "CDLP보다 타이트한 상한"이라고 귀속 | 확인 필요 | 보완형+대체형 혼재 | — | https://doi.org/10.1287/trsc.1090.0262 | 서지 + [2] 경유 2차 |
| 7 | Using Lagrangian Relaxation to Compute Capacity-Dependent Bid Prices in NRM | Topaloglu 2009 | Operations Research 57(3) 637–649, Q1 | **Lagrangian 완화 → 자원별 분리형** 가치함수, 곱·시간 의존 승수 | 확인 필요(페이월) | 확인 필요 | 보완형(여정이 여러 구간 소비 → 승수로 분해) | — | https://doi.org/10.1287/opre.1080.0597 | 서지 (Crossref) |
| 8 | Relaxations of Weakly Coupled Stochastic Dynamic Programs | Adelman, Mersereau 2008 | Operations Research 56(3) 712–727, Q1 | Lagrangian 완화 vs ALP 상한의 강도 비교 | 확인 필요(페이월) | 확인 필요 | 대체형(공통 자원 경쟁)이 주 | — | https://doi.org/10.1287/opre.1070.0445 | 서지 (Crossref) |
| 9 | Reductions of Approximate Linear Programs for NRM | Vossen, Zhang 2015 | Operations Research 63(6) 1352–1371, Q1 | 아핀·분리형 구간선형 ALP의 **동등 축소형** | 해당 없음(크기 축소가 논점) | 해당 없음 | 보완형+대체형 혼재 | — | https://doi.org/10.1287/opre.2015.1442 | 서지 (Crossref) |
| 10 | On the Choice-Based Linear Programming Model for NRM | Liu, van Ryzin 2008 | M&SOM 10(2) 288–310, Q1 | CDLP + **DP 분해**(자원별 분리형, 운임 안분) | 확인 필요(페이월). [2]가 분해의 network effect 한계를 이 논문에 귀속 | 확인 필요 | 보완형+대체형 혼재 | — | https://doi.org/10.1287/msom.1070.0169 | 서지 + [2] 경유 2차 |
| 11 | On a Piecewise-Linear Approximation for NRM | Kunnumkal, Talluri 2016 | Math. of OR 41(1) 72–91, Q1 | 분리형 구간선형 근사의 상한 강도 이론 | 확인 필요(페이월) | 확인 필요 | 보완형+대체형 혼재 | — | https://doi.org/10.1287/moor.2015.0716 | 서지 (Crossref) |
| 12 | On the Approximate LP Approach for NRM Problems | Tong, Topaloglu 2014 | INFORMS J. Computing 26(1) 121–134, Q1 | 분리형 구간선형 ALP 축소 | 확인 필요 | 확인 필요 | 보완형+대체형 혼재 | — | https://doi.org/10.1287/ijoc.2013.0551 | 서지 (Crossref) |
| 13 | Computing Time-Dependent Bid Prices in NRM Problems | Kunnumkal, Topaloglu 2010 | Transportation Science 44(1) 38–62, Q1 | 시간 의존 bid price, 분리형 | 확인 필요 | 확인 필요 | 보완형+대체형 혼재 | — | https://doi.org/10.1287/trsc.1090.0291 | 서지 (Crossref) |
| 14 | A New DP Decomposition Method for NRM with Customer Choice | Kunnumkal, Topaloglu 2010 | POM 19(5) 575–590, Q1 | Lagrangian 기반 분리형 분해, 운임 안분 개선 | 확인 필요 | 확인 필요 | 보완형+대체형 혼재 | — | https://doi.org/10.1111/j.1937-5956.2009.01118.x | 서지 (Crossref) |
| 15 | Network RM with Inventory-Sensitive Bid Prices and Customer Choice | Meissner, Strauss 2012 | EJOR 216(2) 459–468, Q1 | **분리형 오목** 근사 + 재고 집계 | 확인 필요 | 확인 필요 | 보완형+대체형 혼재 | — | https://doi.org/10.1016/j.ejor.2011.06.033 | 서지 (Crossref) |
| 16 | On Bid-price Controls for NRM | Ata, Akan 2015 | Stochastic Systems 5(2) 268–323 | 확산 극한 하 bid price 최적성 | 확인 필요 | 확인 필요 | 보완형+대체형 혼재 | — | https://doi.org/10.1287/12-ssy081 | 서지 (Crossref) |
| 17 | Equivalence of Piecewise-Linear Approximation and Lagrangian Relaxation for NRM / On a Piecewise-Linear Approximation for NRM | Kunnumkal, Talluri 2012 WP / 2016 | BGE WP 608 (미심사) → Math. of OR 41(1) 72–91, Q1 | 분리형 구간선형 ≡ Topaloglu(2009) Lagrangian | **명시(반대 방향).** Lagrangian이 **가장 타이트한 분리형 구간선형 상한**을 준다 = 분리형 계열의 **천장** | 동등성이므로 개선폭 0 (두 방법이 같은 LP) | 보완형+대체형 혼재 | 자원 격자 위 분리형의 천장을 규정. 우리는 참여자 집합 격자 위 분리형이다 | https://www.bse.eu/sites/default/files/working_paper_pdfs/608.pdf · https://doi.org/10.1287/moor.2015.0716 | **전문추출** (WP판) |
| 18 | Dynamic Basis Function Generation for Network Revenue Management | Adelman, Barz, Olivares-Nadal 2025 | INFORMS J. Computing (IJOC), Q1 | **비선형·비분리형** ridge 기저함수를 행생성으로 동적 추가 | **명시.** 가정 (i) 분리성은 문헌에 만연하다(pervasive) | 용량 희소 시 SPLA·NSEP보다 정책·상한 우위, 아핀(AA)보다는 큰 폭 우위 (수치는 본문 표) | 보완형+대체형 혼재 | 자원 용량 벡터 위 근사이고 참여자 집합·잉여배분이 없다 | https://doi.org/10.1287/ijoc.2023.0418 · https://arxiv.org/pdf/2502.16830 | **전문추출** (arXiv판) |
| 19 | Reductions of non-separable approximate linear programs for NRM | Laumer, Barz 2023 | EJOR 309(1) 252–270, Q1 | **비분리형** 기저함수(자원 여러 개에 동시 의존, 부분망 묶음) + ALP 축소 | **명시.** 자원 간 상호의존이 클 때 비분리성이 특히 중요 (검색 수준 진술) | 분리형 근사보다 상한이 종종 더 타이트 (검색 수준) | **보완형** (버스 노선의 중첩 구간 = 자원 공유가 큰 망) | 자원을 부분망으로 그룹화하는 것이지 참여자 집합 분해가 아니다 | https://doi.org/10.1016/j.ejor.2023.01.006 | 검색 (본문 403, 확인 필요) |
| 20 | Order-Optimal Correlated Rounding for Fulfilling Multi-Item E-Commerce Orders | Ma 2023 | M&SOM 25(5) 1324–1337, Q1 (+ ACM EC 2023) | LP 완화 + **상관 라운딩** (품목 간 결합 유지) | **명시.** DLP는 품목별 배차 빈도를 **독립적으로** 지정하고 다품목 주문 내 품목 간 **상관을 포착하지 못한다** | 근사비 **1+ln q** (q품목 주문), 희소망 **d**; 둘 다 tight. 기존 최선 **q/4** (Jasin–Sinha 2015) 대비 대폭 개선 | **보완형** (같은 주문의 품목을 한 물류센터에 모으는 것이 이득) | 목적이 배송비 절감(주문 분할 벌점)이고 **all-or-nothing 성립 조건이 아니다**. 유지·재참여도 없다 | https://doi.org/10.1287/msom.2023.1219 · https://arxiv.org/abs/2207.04774 | WebFetch 요약 수준 (초록·정리 진술) |
| 21 | Using Lagrangian Relaxation … (7번과 동일 논문, 저자 사본) | Topaloglu 2009 | Operations Research 57(3), Q1 | Lagrangian → 구간별 분리형 | **명시 (2차 귀속).** Bertsimas–Popescu(2003)는 여정이 소비하는 구간 용량의 **총** 기회비용을 더 정확히 포착하며 본질적으로 **비분리·오목** 가치함수 근사를 쓴다 | LR 수익이 DLP/RLP/DFD/RFD/LV 대비 평균 **8.9 / 3.7 / 5.5 / 3.5 / 4.6%** 높음. 상한은 DLP/RLP/LV 대비 **5.2 / 3.6 / 4.2%** 타이트 | 보완형+대체형 혼재 | 자원(구간) 분리이고 참여자 분리가 아니다 | https://people.orie.cornell.edu/huseyin/publications/revenue_man.pdf | **전문추출** |
| 22 | Relaxations of Weakly Coupled Stochastic Dynamic Programs (약결합 DP 원전) | Adelman, Mersereau 2008 | Operations Research 56(3) 712–727, Q1 | Lagrangian 완화 vs ALP 상한 | 확인 필요 | 확인 필요 | 대체형(공통 자원 경쟁)이 주 | NRM을 약결합 DP로 보는 틀의 원전. 우리 문제는 결합 제약이 **참여자 생존확률**이라 약결합 틀에 그대로 안 들어간다(추론) | https://doi.org/10.1287/opre.1070.0445 | 서지 + 21 경유 2차 |
| 23 | Asymptotically Optimal Inventory Control for Assemble-to-Order Systems | Reiman, Wan, Wang 2023 | Stochastic Systems 13(2) 128–180 | **VFA 아님**: 다단계 확률계획(SP) 하한 + 확산 척도 점근 최적 정책 | 해당 없음 (분리형 VFA를 아예 쓰지 않음) | 최장 리드타임 무한대에서 하한과의 격차 0으로 수렴 | **혼재**: 한 제품은 모든 부품 필요(보완) + 한 부품을 여러 제품이 씀(대체) | 근사 수단이 점근 SP이고 학습된 가치함수·참여자 집합이 없다 | https://doi.org/10.1287/stsy.2022.0099 · https://arxiv.org/pdf/1809.08271 | **전문추출** (초록·서론) |
| 24 | A Stochastic Programming Based Inventory Policy for ATO Systems with Application to the W Model | Dogru, Reiman, Wang 2010 | Operations Research 58(4) 849–864, Q1 | SP 기반 하한 + 정책 | 확인 필요 | 확인 필요 | 혼재 | 부품 재고 수준 위의 정책이고 참여자 집합이 아니다 | https://doi.org/10.1287/opre.1090.0772 | 서지 |
| 25 | Asymptotically Optimal Inventory Control for ATO Systems with Identical Lead Times | Reiman, Wang 2015 | Operations Research 63(3) 716–732, Q1 | SP 기반 4단계 점근 틀 | 확인 필요 | 확인 필요 | 혼재 | 같음 | https://doi.org/10.1287/opre.2015.1372 | 서지 |
| 26 | Joint Inventory Replenishment and Component Allocation Optimization in an ATO System | Akcay, Xu 2004 | Management Science 50(1) 99–116, Q1 | 2단계 확률계획, 주문 기반 배분 | 확인 필요 | 확인 필요 | 혼재 | 같음 | https://doi.org/10.1287/mnsc.1030.0167 | 서지 |
| 27 | A novel decomposition-based method for solving general-product structure ATO systems | ElHafsi, Fang, Hamouda 2020 | EJOR 286(1) 233–249, Q1 | 원 시스템을 여러 M형 부분시스템으로 **분해**하는 휴리스틱 | 확인 필요 | 확인 필요 | 혼재 | ATO 분해 근사의 최신 예. 여전히 재고 수준 격자 위 분해다 | https://doi.org/10.1016/j.ejor.2020.03.016 | 서지 |
| 28 | Order-Based Cost Optimization in Assemble-to-Order Systems | Lu, Song 2005 | Operations Research 53(1) 151–169, Q1 | 주문 기반(all-or-nothing) 비용 최적화 | 확인 필요 | 확인 필요 | **보완형** (주문 단위 충족) | **주의: 정정 논문 존재** (Bolandnazar–Huh–McCormick, OR 67(1) 163–166, 2019). 원문+정정 동시 확인 없이 인용 금지 | https://doi.org/10.1287/opre.1040.0146 · 정정 https://doi.org/10.1287/opre.2018.1789 | 서지 |
| 29 | Bid-Price Controls for NRM: Martingale Characterization of Optimal Bid Prices | Akan, Ata 2009 | Math. of OR 34(4) 912–936, Q1 | 최적 bid price의 마팅게일 특성화 | 확인 필요 | 확인 필요 | 보완형+대체형 혼재 | 확인 필요 | https://doi.org/10.1287/moor.1090.0411 | 서지 |
| 30 | A strong Lagrangian relaxation for general discrete-choice NRM | Kunnumkal, Talluri 2019 | Comput. Optim. Appl. 73 275–310 | 선택모형 하 강한 Lagrangian 완화 | 확인 필요 | 확인 필요 | 보완형+대체형 혼재 | 확인 필요 | https://doi.org/10.1007/s10589-019-00068-y | 서지 |

---

## 2. F2a~F2d 직답

### F2a — NRM이 자원(구간) 간 보완 관계를 가치함수 근사로 다룬 계보

**핵심 구조: 계보는 분리형 가족과 비분리형 가족으로 갈리고, 분리형 가족은 서로 동등하다는 것이 이미 증명되어 있다.**
이 동등성·천장 구조가 F2a의 가장 쓸모 있는 산출물이다.

**(1) DLP + 정적 bid price (분리형·선형).** 원전 Simpson(1989)·Williamson(1992), 분석 Talluri–van Ryzin (MS 1998) [4].
- Topaloglu [21] 의 명시 귀속: "Talluri and van Ryzin (1998) give a careful analysis of the policies that are based on
  bid prices and **point out that the idea of bid prices is equivalent to using linear value function approximations**
  in the dynamic programming formulation" [21]. → **bid price = 분리형 선형 VFA** 라는 등치가 이 문헌의 출발점이다.
- Bertsimas–Popescu [1] 의 귀속: 유체(fluid) 스케일링에서 가산 bid price 정책이 최적으로 수렴한다 (Talluri–van Ryzin 1998) [1].
- 실무 참조치: DLP·bid price 기법이 이전 운임등급 휴리스틱 대비 **1~2%** 증분 수익을 낳았다고 믿어진다 [3].

**(2) RLP (무작위 LP).** DLP 수요를 표본화해 평균. Topaloglu의 수치에서 RLP·RFD가 2·3위를 다투고,
LR 대비 평균 격차는 RLP 3.7%, RFD 3.5%다 [21].

**(3) 아핀 근사 (분리형·기울기 일정).** Adelman (OR 2007) [5].
Adelman–Barz–Olivares-Nadal [18] 의 정확한 요약:
"In the context of NRM, Adelman (2007) was the first to suggest an **additively separable value function approximation**
within the linear programming formulation ... This affine approximation implies two major assumptions:
(i) the different resources are **separable**, i.e., the marginal value of one unit of resource i is
**independent of the number of units available for resources j not equal to i**; and (ii) the marginal value of one
unit of resource i is constant" [18].

**(4) 분리형 구간선형 근사 (SPLA).** 가정 (ii)만 제거한다. Farias–Van Roy (2007 WP) [3] 제약표본,
Meissner–Strauss (EJOR 2012) [15] 재고 집계, Vossen–Zhang (OR 2015) [9] · Tong–Topaloglu (IJOC 2014) [12] 축소형 [18].
Farias–Van Roy의 동기 진술: "**Affine approximations are incapable of capturing this concavity of J\* in inventory
level.** This motivates us to consider a separable concave approximation architecture" [3].

**(5) Lagrangian 완화.** 약결합 DP 틀 (Hawkins 2003; Adelman–Mersereau OR 2008 [22]) 의 NRM 적용.
Topaloglu (OR 2009) [7]: "relaxing certain constraints that link the decisions for different flight legs by
associating Lagrange multipliers with them. In this case, **the network revenue management problem decomposes by the
flight legs** and we can concentrate on one flight leg at a time" [21].
즉 **여정이 여러 구간을 동시에 쓰는 결합을 승수로 끊어 분리형으로 만드는 것** 이 Lagrangian 완화의 정체다.
Topaloglu는 분리성을 쓰는 이유도 명시한다: 분리 구조가 없으면 의사결정 규칙 구현에 |C|^|L| · |T| 개의 수를 저장해야 하고,
분리 구조가 있으면 |C| · |L| · |T| 개로 줄어든다 [21]. (분리성은 **정확도 판단이 아니라 저장·계산 제약** 때문에 도입된 것이다.)

**(6) 【핵심】 분리형 가족의 동등성과 천장**
- 아핀 근사 (Adelman 2007) ≡ Kunnumkal–Topaloglu의 Lagrangian 완화: Tong–Topaloglu [12] 가 증명,
  Vossen–Zhang [9] 이 Dantzig–Wolfe 분해로 대안 증명 — 둘 다 Kunnumkal–Talluri [17] 가 귀속한다.
- 분리형 구간선형 (SPLA) ≡ Topaloglu(2009)의 곱·시간별 Lagrangian 완화: Kunnumkal–Talluri [17] **Proposition 2: V_PL = V_LR**.
- 결정적 진술: "This also shows that the Lagrangian relaxation approach yields **the tightest separable,
  piecewise-linear upper bound** to the value function of the network RM dynamic program" [17].
- **귀결 (인용 가능한 형태)**: 분리형 구간선형 계열에는 **알려진 최대 강도** 가 있고, 그 이상은 분리형을 버려야 얻는다.
  Adelman 외의 표현: "**Assumption (i), separability, is pervasive in the literature.**" [18]
- 단 [17] 은 선택모형 기반 NRM에서는 이 동등성이 **깨진다** 고 자기 논문 4절에서 밝힌다 [17]. 동등성은 모형 정식화에 의존한다.

**(7) DP 분해 (운임 안분).** 선택모형 NRM의 표준: CDLP + 자원별 분해 (Liu–van Ryzin, M&SOM 2008 [10]),
Zhang–Adelman (TS 2009) [6], Kunnumkal–Topaloglu (POM 2010) [14].
Zhang [2] 의 진술: classical 분해에서 "**the network effect is only captured through fare proration**" [2].
여정 운임을 구간별로 쪼개 나눠주는 것이 분리형 분해가 보완성을 다루는 유일한 수단이다.

**(8) 비분리형 가족.** Bertsimas–Popescu (TS 2003) [1] CEC (LP 값 자체를 VFA로),
Zhang (M&SOM 2011) [2] min 연산자, Laumer–Barz (EJOR 2023) [19] 부분망 기저함수,
Adelman–Barz–Olivares-Nadal (IJOC 2025) [18] ridge 기저함수 행생성.

**우리 위치 (추론).** 우리 목적항 (참여자별 c_i 곱하기 재참여확률의 구간선형 근사, 참여자에 대해 합) 은
이 계보의 **(3) 아핀·분리형 단계** 에 대응한다. 차이는 두 가지다.
(a) 분리 대상이 **자원 용량 벡터가 아니라 참여자 집합** 이다.
(b) 분리형 계수의 가중치가 **결정변수 (배분 잉여에 의존하는 재참여확률)** 다.
NRM의 분리형 계수는 상태에만 의존하고 결정 의존 가중치를 갖지 않는다.

### F2b — 분리형 근사의 한계를 명시적으로 지적한 연구: **있다. 4편 확인, 그중 1편이 최적 인용 후보**

1. **Bertsimas–Popescu (TS 2003) [1] — 최적 인용 후보.** 여정이 여러 구간을 쓴다는 사실 때문에 가산 bid price가
   실패한다는 것을 **bundle effects 라는 용어까지 붙여** 서론과 3.2절에서 두 번 진술하고 개선폭을 수치로 준다.
   원문 인용은 아래 3절에 모았다.
2. **Zhang (M&SOM 2011) [2].** classical 분리형 DP 분해가 network effect를 운임 안분으로만 포착한다고 지적하고,
   비분리형 근사 (구간별 함수와 나머지 구간의 선형항을 더한 뒤 구간에 대해 최소를 취하는 형태) 로
   정책 **최대 8%** 개선 (계산시간 +30%).
3. **Kunnumkal–Talluri [17].** 반대 방향의 한계 지적: 분리형 구간선형의 **상한 천장** 을 규정한다.
   분리형을 더 정교하게 해도 그 이상은 안 된다는 것을 보인 것이므로, 분리형을 버려야 한다는 논증의 가장 형식적인 근거다.
4. **Adelman–Barz–Olivares-Nadal (IJOC 2025) [18], Laumer–Barz (EJOR 2023) [19].** 최신 비분리형 계보.
   [18] 은 분리성 가정 (i) 을 **명시적 가정으로 분리해 적어** 주므로 인용에 가장 편하다.

**인접 문헌의 동형 결과.** Ma (M&SOM 2023) [20] 는 다품목 전자상거래 주문에서
"DLP는 품목별 배차 빈도를 **독립적으로** 지정하고 다품목 주문 내 품목 간 **상관을 포착하지 못한다**" 고 진술하고,
상관 라운딩으로 근사비를 **q/4 에서 1+ln q 로** 개선한다 (둘 다 tight).
자원이 아니라 **주문 안의 품목들** 을 묶는다는 점에서 우리 설정에 구조적으로 가장 가깝다.
단 목적이 배송비이고 **all-or-nothing 성립 조건이 아니다.**

### F2c — ATO에서 부품 간 보완을 **가치함수 근사 방법론** 으로 다룬 방식: **NRM식 분리형 VFA 계보는 없다**

확인한 범위에서 ATO 근사 방법론은 세 갈래이고, **어느 것도 자원별 분리형 가치함수 근사가 아니다.**
1. **정책 부류 제한 + 구조 결과**: base-stock, FIFO, no-holdback으로 정책을 한정하고 최적 구조를 증명.
   Reiman–Wan–Wang [23] 의 진단: "Restricting the consideration to these types of policies makes the problem more
   tractable, **but also sacrifices optimality**" [23].
2. **점근 (확산 척도) 확률계획 기반**: 다단계 SP로 하한을 세우고 그 해로 정책 파라미터를 정한 뒤 점근 최적성을 증명
   (Dogru–Reiman–Wang OR 2010 [24], Reiman–Wang OR 2015 [25], Reiman–Wan–Wang [23]).
   부품 간 보완은 **BOM (Bill of Materials)** 으로 SP 제약에 들어가고, 가치함수 근사 형태로 분해되지 않는다.
3. **부분시스템 분해 휴리스틱**: 일반 제품구조 ATO를 여러 M형 부분시스템으로 나눠 푸는 방식 (ElHafsi 외 EJOR 2020 [27]).
   분해 대상이 제품·부품 구조이고, 여전히 재고 수준 격자 위의 분해다.

**보완성의 정체에 대한 ATO의 진술**: "The complexity arises from the need to allocate components that are used by
multiple products" [23]. 즉 ATO의 상호의존은 **혼재** 다 — 한 제품은 모든 부품이 있어야 조립되므로 부품끼리 보완이고,
한 부품을 여러 제품이 쓰므로 제품끼리는 대체 (경쟁) 다.
**우리 설정 (품목 간 보완 + 같은 품목의 대체 공급자 간 대체) 과 같은 혼재 구조다.**

**F2c 결론**: ATO에서 분리형 가치함수 근사가 부품 보완성을 놓친다는 진술을 **직접 한 연구는 찾지 못했다.**
그 진술은 ATO가 아니라 **NRM 쪽 (F2b)** 에서 찾아야 하며, 실제로 거기에 있다.
(주의: Lu–Song OR 2005 [28] 은 주문 기반 최적화의 고전이지만 **정정 논문이 존재** 하므로 원문+정정 동시 확인 없이 인용하지 않는다.)

### F2d — 우리 부호 반전 (증분 가치 > 독립 가치) 을 명시적으로 말한 곳: **있다. Bertsimas–Popescu (TS 2003) [1]**

두 수준의 진술이 있다.

**(가) 일반 정리 — Proposition 3**
> "In any state (n, t), for any bid prices BP_j(n, t), BP_j(n − A_j, t), the following inequalities hold:
> **BP_j(n, t) <= OC_j^LP(n, t) <= BP_j(n − A_j, t).** (4)
> **Inequalities are strict if accepting class j must incur a change of basis in the LP dual.**" [1]

여기서 OC_j^LP(n,t) = LP(n, D_{t−1}) − LP(n − A_j, D_{t−1}) 는 여정 j가 소비하는 **자원 묶음 A_j의 증분 가치** 이고,
BP_j = 구간별 이중가격 v_l 의 여정 내 합 은 **구간별 독립 가치의 합** 이다.
즉 **묶음의 증분 가치 >= 독립 가치의 합** 이며, 등호는 이중 기저가 바뀌지 않을 때만 성립한다.
우리 부호 문장과 방향이 같다.

**주의 — 정직한 유보 (우리 추론).** 이 좌측 부등식은 **오목성만으로도** 나올 수 있다.
LP 값이 용량에 대해 오목이므로 후향 차분이 기울기보다 크고, 이를 좌표별로 더하면 같은 방향이 나온다.
따라서 Proposition 3을 "보완성 때문" 이라고 쓰면 과장이다. **보완성 고유의 증거는 아래 (나) 다.**

**(나) 3구간 예제 — 닫힌 형태의 부호 반전 (4.1절)**
허브 h, 출발지 o1·o2, 도착지 d. 구간 (1)=o1→h, (2)=o2→h, (3)=h→d. 여정 1, 2, 13, 23. 남은 용량 n=(1,1,1).
최적 이중해는 v1\* = R1, v2\* = R2, v3\* = R23 − R2. 이때 [1]:

> "OC_1^LP = LP(1,1,1,D) − LP(0,1,1,D) = R1 = BP(1),
> **OC_2^LP = LP(1,1,1,D) − LP(1,0,1,D) = R1 + R23 − R13 > BP(2) = R2,**
> **OC_13^LP = LP(1,1,1,D) − LP(0,1,0,D) = R1 + R23 − R2 = BP(13) > R13,**
> OC_23^LP = LP(1,1,1,D) − LP(1,0,0,D) = R23 = BP(23)." [1]

- 두 번째 줄: **구간 2 한 자리의 증분 가치가 그 구간 단독 운임 R2를 넘는다.**
- 세 번째 줄: **두 구간 묶음 {1,3} 의 증분 가치가 그 묶음으로 파는 여정의 운임 R13을 넘는다.**
  (명시 가정: 하위여정 운임이 더 싸다, 즉 R1 < R13, R2 < R23 [1].)
- 이것이 우리 G1의 "증분 가치 > 독립 가치" 와 **정확히 같은 종류의 진술** 이며,
  구간 2가 그만큼 값비싼 이유는 그것이 고가 조합 (여정 1 + 여정 23) 의 성립을 가능하게 하기 때문이다 = 보완형 상호의존.

**(다) 감소 차분 (열모듈 방향) 의 반례**
같은 절에서 저자들은 Karaesmen–van Ryzin의 **decreasing differences** (단변수에서 오목성으로 환원되는 교차 오목성) 를
정의하고 "**the decreasing differences property is violated in our example**" 이라 적으며
LP(1,1,1) − LP(0,1,1) = max(R1+R23, R2+R13) − R23 < R13 = LP(1,0,1) − LP(0,0,1) 을 제시한 뒤,
"Surprisingly, this says that **incremental revenues (opportunity costs) may decrease by decreasing capacity along a
certain direction**" 라고 논평한다 [1].

**주의 — 정직한 유보 (우리 확인 실패).** PDF 텍스트 추출본의 Definition 1
(f(s+di·ei+dj·ej) − f(s+di·ei) <= f(s+dj·ej) − f(s)) 과 제시된 부등식을 대조하면,
제시된 부등식은 (1,2) 쌍에 대해 감소 차분을 **만족** 하는 형태로 읽힌다.
부등호 방향이나 좌표 대응이 PDF 추출 과정에서 어긋났을 가능성이 있다. **조판 원문으로 방향을 확인해야 한다** (4절 미해결 목록).
다만 (나) 의 닫힌 형태는 그 자체로 충분하므로 우리 논증은 (다) 에 의존하지 않는다.

**F2d 최종 판정.** 증분 > 독립 이라는 **부호는 NRM에 이미 있다.**
따라서 G1을 "부호 반전 자체가 처음" 으로 쓰면 반박된다.
남는 차별점은 **부호가 아니라 부호가 얹히는 대상** 이다:
NRM의 부호 진술은 **외생 용량 벡터** 위의 것이고, 우리는 **결정에 따라 전이하는 참여자 집합** 위에서 같은 부호를 다룬다.
5절에 다시 정리한다.

---

## 3. F2b 전용: 인용 가능한 문장 모음 (원문)

출처를 논문 위치까지 적었다. 모두 PDF 전문에서 직접 추출한 것이다 (Bertsimas–Popescu는 MIT 저자 사본,
Zhang은 저자 사본, Farias–Van Roy는 MIT 워킹페이퍼, Topaloglu는 Cornell 저자 사본, Kunnumkal–Talluri는 BGE WP 608판,
Adelman 외는 arXiv판).

### 3.1 【최선의 문장】 Bertsimas & Popescu (2003), Transportation Science 37(3), 서론 Contributions 절

> "The most popular technique developed in the current literature is an additive bid-pricing approach. These are
> mechanisms whereby the opportunity cost of each itinerary is estimated as the sum of the shadow prices of the
> incident legs ... However, there are two obvious drawbacks to additive bid prices: (a) They are not well defined if
> there are multiple dual solutions. (b) **They are restrictive in their way of taking into account bundles by their
> predefined additive structure. In particular, they do not account for changes of a dual basis in response to
> accepting large-group and multileg itinerary requests.**" [1]

**우리가 논문에서 쓸 최선의 문장은 이것이다.** 이유:
(i) "묶음(bundle)" 과 "다구간(multileg)" 을 함께 명시하고, (ii) 실패 원인을 "미리 정해 둔 가산 구조" 로 지목하며,
(iii) Q1 저널 (Transportation Science) 의 본문 진술이고, (iv) 뒤이어 수치 개선폭까지 제시한다.

### 3.2 같은 논문 3.2절 — bundle effects 라는 이름

> "The main disadvantages that are apparent from the definition of leg-based additive bid prices is that (a) they are
> not uniquely defined (several sets of shadow prices may be optimal), and (b) **they provide an additive
> approximation of the opportunity costs, which are not necessarily additive due to bundle effects** (group or
> multileg itinerary requests may determine basis changes in the dual LP)." [1]

우리 문장에 그대로 대응시킬 수 있는 형태:
"기회비용은 반드시 가산적이지 않다 (묶음 효과)" → "미래 가치는 반드시 참여자별로 가산적이지 않다 (묶음 주문 효과)".

### 3.3 같은 논문 — 개선 폭 (수치)

> "We observe that the CEC algorithm performs very well in practice, giving results that are very close to optimum.
> **For high load factors, we observe an average 5%-10% improvement over existing policies (additive bid pricing).**"
> " ... simulate extensions of this algorithm that result in **significantly higher improvements (up to 20%)**." [1]

결론절의 조건 진술도 유용하다:
> "Computationally we observe that the CEC policy outperforms the BPC policy **when the load factors ... tend to be
> large. When the load factor is small, both policies perform optimally.**" [1]

→ **분리형의 손해는 자원이 빡빡할 때만 나타난다.** 우리 실험 3의 조건 설계 (공급 편중도, 대체 공급자 수) 를
정당화할 때 이 문장이 직접 근거가 된다.

### 3.4 Zhang (2011), M&SOM 13(1) — 분리형 분해가 network effect를 어떻게만 포착하는가

> "The approximation (6) is able to better capture the network effect, because after v_hat_{t,i}(.) are determined for
> each leg i, the value v_t(x) is approximated by a single minimum across the legs, whereas **in the decomposition in
> 3.3, the network effect is only captured through fare proration.**" [2]

> "To the best of our knowledge, our work is the first that adopts a **nonlinear nonseparable functional
> approximation**, which is shown to be efficiently solvable ... The functional approximation employed **can capture
> network effects better than classical dynamic programming decomposition** to the same problem." [2]

개선 폭:
> "it can **outperform DCOMP by as much as 8%** even in relatively large problem instances, which is quite remarkable
> given that DCOMP is often believed to be one of the strongest benchmarks available and is often the algorithm
> implemented in commercial applications." [2]
> "On average, DCOMP1 takes **30% more time** to solve than DCOMP." [2]

### 3.5 Kunnumkal & Talluri — 분리형 계열의 천장 (가장 형식적인 근거)

> "This also shows that the Lagrangian relaxation approach yields **the tightest separable, piecewise-linear upper
> bound** to the value function of the network RM dynamic program. **Proposition 2. V_PL = V_LR.**" [17]

논거 문장으로 쓸 때:
"분리형 구간선형 근사 계열에는 알려진 최대 강도가 있고 (Lagrangian 완화와 동등), 그보다 나아지려면 분리성을 버려야 한다."

### 3.6 Adelman, Barz & Olivares-Nadal (IJOC 2025) — 분리성을 가정으로 분리해 적은 문장

> "This affine approximation implies two major assumptions: (i) the different resources are separable, i.e., **the
> marginal value of one unit of resource i is independent of the number of units available for resources j not equal
> to i**; and (ii) the marginal value of one unit of resource i is constant ... **Assumption (i), separability, is
> pervasive in the literature.**" [18]

우리 방법의 가정을 대응시켜 쓸 때 가장 깔끔한 틀이다:
"(i) 참여자별 유지 가치가 다른 참여자의 잔류 여부와 무관하다, (ii) 그 값이 잉여 배분량과 무관하게 일정하다."
우리는 (ii) 를 구간선형 재참여확률로 완화했고 (i) 은 완화하지 않았다 — 이것이 우리 한계의 정확한 위치다.

### 3.7 Farias & Van Roy (2007 WP) — 아핀 근사의 한계 (분리성이 아니라 기울기 일정)

> "**Affine approximations are incapable of capturing this concavity of J\* in inventory level.** This motivates us to
> consider a separable concave approximation architecture which is the focus of this paper." [3]

주의: 이 문장은 분리성 (가정 i) 이 아니라 **기울기 일정 (가정 ii)** 에 대한 비판이다. 혼용하면 안 된다.

### 3.8 Topaloglu (OR 2009) — 분리성의 진짜 이유는 저장·계산

> "It is important to note that the separable structure ... plays a major role in implementing the decision rule ...
> efficiently. In particular, **if the separable structure did not exist**, then implementing the decision rule ...
> would require storing ... **|C|^|L| |T| numbers.** By using the separable structure ... |C| |L| |T| numbers." [21]

→ 분리형은 근사 품질이 좋아서 채택된 것이 아니라 **차원 저주 때문** 임을 원저자가 명시한다.
우리가 c_i 분해를 쓰는 이유 (MILP에 넣을 수 있게 하기 위함) 와 정확히 같은 동기다.

### 3.9 Ma (M&SOM 2023) — 다품목 주문에서 독립 근사가 무엇을 잃는가

WebFetch 요약 수준으로 확인한 진술 (원문 대조 필요):
DLP는 품목별 배차 빈도를 독립적으로 지정하고 **다품목 주문 내 품목 간 상관을 포착하지 못한다**.
각 품목의 LP 해를 따로 라운딩하면 같은 주문의 품목들을 소수 물류센터에 모아야 한다는 구조가 깨진다.
근사비: 일반 q품목 주문에서 **1+ln q**, 품목당 보관 센터가 d개 이하인 희소망에서 **d**, 둘 다 tight.
직전 최선은 Jasin–Sinha (2015) 의 **q/4** [20].

**우리가 쓸 최선의 문장 지목 (요약)**
- 한 문장만 인용한다면 **3.1** (Bertsimas–Popescu 서론 Contributions).
- 형식적 한계 근거로 한 문장 더 붙인다면 **3.5** (Kunnumkal–Talluri 천장).
- 우리 가정을 정직하게 적는 틀로는 **3.6** (Adelman 외의 가정 (i)/(ii) 분리).
- 조건부 손해 (자원이 빡빡할 때만) 로 실험 설계를 정당화하려면 **3.3** 의 load factor 문장.

---

## 4. 미해결·확인 불가 (INFORMS 페이월 목록과 확인할 구체적 질문)

### 4.1 전문을 읽지 못한 것 (페이월) 과 확인할 질문

| 대상 | 확인할 질문 | 왜 필요한가 |
|---|---|---|
| Talluri & van Ryzin, MS 1998 [4] | (a) "bid price = 선형 가치함수 근사" 라는 등치를 저자들이 **직접** 어느 절에서 어떤 문장으로 적는가 (Topaloglu [21] 의 2차 귀속만 확보). (b) 4.1절의 DLP 상한 정리 번호와 정확한 진술. (c) bid price 통제가 **일반적으로는 최적이 아니다** 는 반례를 제시하는가, 아니면 점근 최적성만 보이는가 | F2a 계보의 출발점을 1차 출처로 인용하려면 필요. 현재 우리 인용은 [21] 경유 2차다 |
| Adelman, OR 2007 [5] | (a) 저자가 분리성 가정을 **스스로** 어떻게 적는가 (현재는 Adelman 외 2025 [18] 의 자기 요약만 확보). (b) Theorem 1 (DLP 대비 상한 타이트) 의 정확한 진술. (c) 열생성 LP의 제약 수가 구간 수에 대해 어떻게 늘어나는가 | 아핀 근사를 1차 출처로 인용하려면 필요 |
| Topaloglu, OR 2009 [7] 게재본 | 저자 사본 (Cornell, 2007-11-22판) 은 읽었다. **게재본과 수치가 같은지** 확인 필요 (8.9 / 3.7 / 5.5 / 3.5 / 4.6%, 상한 5.2 / 3.6 / 4.2%) | 수치를 논문에 인용하려면 게재본 대조 필요 |
| Bertsimas & Popescu, TS 2003 [1] 조판 원문 | **4.1절 Definition 1 (decreasing differences) 의 부등호 방향과, 제시된 반례가 어느 좌표쌍에 대한 것인가.** 추출본으로는 제시 부등식이 감소 차분을 만족하는 형태로 읽혀 저자 주장과 어긋난다 | F2d (다) 의 인용 가능성이 여기에 달려 있다. (나) 의 닫힌 형태는 별도로 확인했으므로 논증 자체는 안전하다 |
| Kunnumkal & Talluri, MOR 2016 [17] 게재본 | BGE WP 608 (2012-11-05판) 은 읽었다. 게재본의 **Proposition 2 번호와 "tightest separable piecewise-linear upper bound" 문장이 그대로 남아 있는지** | 우리 핵심 인용이므로 게재본 확인 필수 |
| Laumer & Barz, EJOR 2023 [19] | (a) 비분리형 기저함수의 정확한 형태와 부분망 그룹화 기준. (b) 분리형 대비 상한·정책 개선폭의 **수치**. (c) "자원 간 상호의존이 클 때 비분리성이 중요하다" 는 진술의 원문 | 현재 검색 스니펫 수준이다. 본문 403으로 접근 실패 |
| Zhang & Adelman, TS 2009 [6] | DP 분해 상한이 CDLP 상한보다 타이트하다는 정리의 정확한 진술 (현재 Zhang [2] 경유 2차) | 선택모형 계보를 1차로 인용하려면 필요 |
| Vossen & Zhang, OR 2015 [9] / Tong & Topaloglu, IJOC 2014 [12] | 아핀 ≡ Lagrangian 동등성 증명의 정확한 진술 (현재 [17] 경유 2차) | 동등성 주장을 1차로 인용하려면 필요 |
| Ma, M&SOM 2023 [20] | 1+ln q 와 d 의 정리 번호, tight 성 증명 위치, 그리고 **all-or-nothing 조건이 정말 없는지** (있으면 우리와의 차이가 줄어든다) | arXiv판으로 확인 가능. 이번에는 WebFetch 요약 수준에서 멈췄다 |
| Akan & Ata, MOR 2009 [29] / Ata & Akan, Stochastic Systems 2015 [16] | bid price 통제의 최적성·차선성에 대한 확산 극한 결과가 **가산 구조의 손실** 을 정량화하는가 | F2b의 추가 근거가 될 가능성 |
| Adelman & Mersereau, OR 2008 [22] | 약결합 DP에서 결합 제약이 **확률** 일 때도 Lagrangian 완화가 적용되는가 (우리 설정: 결합이 참여자 생존확률) | 우리 방법을 약결합 DP 틀에 넣을 수 있는지 판단하는 데 필요 |
| Lu & Song, OR 2005 [28] + 정정 [OR 2019] | 정정이 어느 결과를 무효화했는가 | 인용 전 필수. 정정 확인 없이는 인용하지 않는다 |
| Capacity Allocation with Multiple Suppliers and Multiple Demand Classes, POM 2019 (10.1111/poms.13076) | **복수 공급자** 가 등장하는 용량배분 논문이다. 공급자 간 상호의존이 대체형뿐인지 확인 | 우리 설정 (복수 공급자) 과 가까울 가능성. 이번 조사에서는 서지만 확보 |

### 4.2 OpenAlex 인용 그래프 순회: 부분 성공 (시도한 쿼리 전부 기록)

**결과: `filter=` 목록 질의는 전부 실패, `works/doi:` 단일 조회는 성공, 역인용은 OpenCitations로 우회했다.**

성공한 것 (OpenAlex 단일 엔티티 GET, 스로틀 영향 없음):
- `works/doi:10.1287/opre.1060.0368` → **W2154706439** Adelman 2007, 피인용 258
- `works/doi:10.1287/opre.1080.0597` → **W2124897406** Topaloglu 2009, 피인용 164
- `works/doi:10.1287/trsc.37.3.257.16047` → **W2141904045** Bertsimas–Popescu 2003, 피인용 255, 참조 23편
- `works/doi:10.1287/msom.1100.0302` → **W2134648168** Zhang 2011, 피인용 62

실패한 것 (모두 HTTP 429, 본문은 "Insufficient budget ... free daily budget shared by everyone on your network IP
address ... $0 remaining"). 4회 재시도 + 대기 후에도 동일:
- `works?filter=cites:W2141904045,cites:W2134648168`
- `works?filter=cites:W2154706439,cites:W2124897406,from_publication_date:2020-01-01`
- (앞서) `works?search=dynamic bid prices network revenue management`
- (앞서) `works?search=Lagrangian relaxation network revenue management capacity control`

즉 이 IP의 OpenAlex 무키 **일일 예산** 이 소진된 상태이고, 초 단위 스로틀이 아니었다.
`search=` 뿐 아니라 `filter=` 목록 질의도 과금 대상이어서 막혔다. 단일 엔티티 조회는 통과했다.

**우회 결과 (OpenCitations, 키 불필요, 성공):**
- Bertsimas–Popescu (TS 2003) 피인용: **185편**
- Zhang (M&SOM 2011) 피인용: **50편**
- **교집합 15편** — 아래 4.3에 목록과 갭 판정을 적었다.

### 4.3 인용 그래프에서 관찰한 것 (갭 신호)

"비분리형 NRM" 의 핵을 이루는 두 논문 [1][2] 을 **동시에** 인용하는 15편은 다음과 같다 (OpenCitations + Crossref 서지).

| 연도 | 제목 | 게재지 | DOI |
|---|---|---|---|
| 2014 | A Network Airline Revenue Management Framework Based on Decomposition by Origins and Destinations | Transportation Science | 10.1287/trsc.2013.0469 |
| 2015 | Revenue management for operations with urgent orders | EJOR | 10.1016/j.ejor.2014.07.015 |
| 2017 | Dynamic Pricing for Network Revenue Management: A New Approach and Application in the Hotel Industry | INFORMS J. Computing | 10.1287/ijoc.2016.0713 |
| 2017 | Dynamic Programming Decomposition for Choice-Based Revenue Management with Flexible Products | Transportation Science | 10.1287/trsc.2017.0743 |
| 2018 | Decomposition methods for dynamic room allocation in hotel revenue management | EJOR | 10.1016/j.ejor.2018.05.027 |
| 2019 | Network Revenue Management with Cancellations and No-Shows | POM | 10.1111/poms.12907 |
| 2019 | An Approximate Dynamic Programming Approach to Dynamic Pricing for Network Revenue Management | POM | 10.1111/poms.13075 |
| 2019 | Capacity Allocation with Multiple Suppliers and Multiple Demand Classes | POM | 10.1111/poms.13076 |
| 2019 | Assign-to-Seat: Dynamic Capacity Control for Selling Train Tickets (WP) | SSRN | 10.2139/ssrn.3477339 |
| 2020 | An Approximation Algorithm for Network Revenue Management Under Nonstationary Arrivals | Operations Research | 10.1287/opre.2019.1931 |
| 2021 | Standardized cargo network revenue management with dual channels | EJOR | 10.1016/j.ejor.2021.02.046 |
| 2021 | A network group booking model for airline revenue management | J. Modelling in Management | 10.1108/jm2-06-2020-0175 |
| 2023 | Assign-to-Seat: Dynamic Capacity Control for Selling High-Speed Train Tickets | M&SOM | 10.1287/msom.2023.1188 |
| 2024 | Analyzing Mathematical Models in the Transition to Network-Based Revenue Management (리뷰) | IJSREM (저급 저널) | 10.55041/ijsrem30051 |
| 2025 | Dynamic Basis Function Generation for Network Revenue Management | INFORMS J. Computing | 10.1287/ijoc.2023.0418 |

**갭 판정 (제목 수준 관찰이므로 추론으로 표시).**
이 15편 중 **학습된 가치함수 (신경망), 참여자 유지·이탈, 잉여배분을 다룬 것은 한 편도 없다.**
전부 고전적 용량통제·동적가격·분해 방법론이다. 즉 "다구간 묶음의 비분리성" 을 정면으로 다루는 계보와
"내생적 참여자 구성 + 학습 VFA" 계보는 **인용 그래프에서 만나지 않는다.**
이것이 우리 위치를 지지하는 가장 구체적인 신호다.

단 이 관찰의 한계를 명시한다: (a) 제목만 보았고 본문은 보지 않았다, (b) OpenCitations 커버리지는 완전하지 않다
(Zhang 2011 의 OpenCitations 피인용 50편 대 OpenAlex 62편 — 약 20% 누락), (c) 두 논문을 동시에 인용하지 않으면서
같은 주제를 다루는 논문은 이 교집합에 안 잡힌다.

**추가 관찰 (Bertsimas–Popescu 피인용 185편 제목 키워드 스캔).**
학습·유지·묶음 관련 키워드 (neural, deep, reinforcement, learn, retention, churn, participation, surplus,
revenue sharing, graph, two-sided, platform, matching, assemble, multi-item, bundle, complement) 로 걸러 **5편만** 나왔다:

| 연도 | 제목 | 게재지 | DOI |
|---|---|---|---|
| 2016 | Pricing Personalized Bundles: A New Approach and An Empirical Study | M&SOM | 10.1287/msom.2015.0563 |
| 2021 | Online Matching with Reusable Network Resources and Decaying Rewards (WP) | SSRN | 10.2139/ssrn.3981123 |
| 2024 | **Reinforcement learning for freight booking control problems** | J. Revenue and Pricing Mgmt | 10.1057/s41272-023-00459-1 |
| 2024 | Modeling and Predicting Passenger Load Factor in Air Transportation (주제 무관) | Fractal and Fractional | 10.3390/fractalfract8040214 |
| 2025 | On Greedy-Like Policies in Online Matching with Reusable Network Resources and Decaying Rewards | Management Science | 10.1287/mnsc.2023.02588 |

**유지·이탈·잉여배분을 다룬 논문은 0편이다.** (제목 수준 관찰. 본문 확인 안 함.)

이 중 **Dumouchelle, Frejinger & Lodi (JRPM 2024) [31] 은 우리와 방법 구조가 가장 가깝고 반드시 인용해야 한다.**
전문에서 직접 확인한 내용:
- 2단계 구조: (1) 지도학습으로 **기간말 운영문제(EoHP)** 의 최적값을 예측하는 모형을 학습, (2) 그 예측을
  시뮬레이션 기반 RL 알고리즘 안에 넣어 수락·거절 정책을 계산한다 [31].
- 자기 갭 진술: "we are unaware of any works that introduce **predictions of optimal discrete optimization problem
  values within an RL algorithm**" [31].
- 화물 맥락에서 **bid price 자체가 불가능해지는 이유** 를 명시한다: 운영문제가 (3차원) 패킹·VRP 같은 MILP이면
  "the loss of duality to derive bid-price policies because the LP relaxation of packing problems whose constraints
  are dependent on 3-dimensional objects is extremely poor" [31]. 또 DLP형 정식화가 실은 LP가 아니라 MILP이면
  "the dual solution is not available, so it cannot be used to compute bid-price policies" [31].
- 상태 표현을 두 가지로 비교한다: **DQN-L** (수락 건수·사용 용량·기간의 선형 상태 + 완전연결망) 대 **DQN-S**
  (수락된 요청들의 특징을 쓰는 **순열 불변** 집합 기반 모형) [31].
  → 우리 실험 3의 "요약 지표 MLP 대 구조 인식 모형" 비교와 같은 종류의 대조군이다.
  **어느 쪽이 이겼는지는 결과 표를 읽지 않았으므로 여기 적지 않는다** (4.1 확인 목록에 추가).

**우리와 [31] 의 차이 (핵심 차별점).**
[31] 은 (a) 용량이 **외생 고정** 이고, (b) 결정이 **수락·거절** 뿐이며 (잉여배분이 없다),
(c) 미래 참여자 구성이 결정에 따라 바뀌지 않고, (d) 학습된 예측을 **RL 정책** 에 넣되
**MILP 목적함수 계수로 분해해 넣지 않는다.** 우리는 네 가지 모두 다르다.

---

## 5. 상호의존 종류 분류표 (3차 조사 공통 축)

| 종류 | 이 조사에서 해당하는 문헌 | 근거 |
|---|---|---|
| **보완형이 명시적으로 논점인 것** | [1] Bertsimas–Popescu (bundle effects, 다구간 여정), [2] Zhang (network effect), [19] Laumer–Barz (중첩 구간이 많은 버스망), [20] Ma (같은 주문의 품목을 모으는 이득), [28] Lu–Song (주문 단위 충족) | 본문에 bundle, network effect, correlation within a multi-item order, order-based 가 직접 나온다 |
| **혼재 (보완 + 대체)** | [3][4][5][6][7][9][10][11][12][13][14][15][16][17][18][21][29][30] NRM 전체; [23][24][25][26][27] ATO 전체 | NRM: 여정이 여러 구간을 동시에 쓰므로(보완) + 여러 여정이 같은 구간을 다투므로(대체). ATO: 한 제품이 모든 부품 필요(보완) + 한 부품을 여러 제품이 씀(대체) [23] |
| **대체형이 주** | [22] Adelman–Mersereau 약결합 DP (공통 자원을 다투는 부문제들) | 결합 제약이 공통 자원 용량이다 (서지 수준 판단, 본문 미확인) |
| **해당 없음** | 없음 | 이번 F2 범위의 문헌은 전부 다자원 문제다 |

**중요한 정정 (2차 조사 대비).** 3차 조사 착수 문서는 "기존 선례들의 상호의존은 용량 경쟁, 즉 대체형으로 보인다" 고
가정했다. **F2 범위에서는 그 가정이 성립하지 않는다.** NRM은 처음부터 혼재이고, 보완 쪽 손실을 정면으로 다룬
문헌군이 존재한다. G1을 쓸 때 이 사실을 반영해야 한다.

---

## 6. G1에 대한 영향 (이 조사가 바꾼 것)

**약화되는 부분.**
G1의 현상 문장 "보완형 상호의존 때문에 증분 가치가 독립 가치를 넘어서고, 이 부호 반전이 결정을 질적으로 다르게 만든다" 의
**부호 부분은 선례가 있다.** Bertsimas–Popescu [1] 는 (a) 일반 부등식 (Prop 3), (b) 닫힌 형태의 예제
(구간 하나의 기회비용이 그 구간 단독 운임을 초과, 두 구간 묶음의 기회비용이 그 여정 운임을 초과) 를 모두 제시한다.
게다가 "가산 근사가 묶음 효과를 놓친다" 는 진술도, 그것을 개선한 비분리형 근사의 **수치 개선폭** 도 이미 있다
([1] 5~10%·최대 20%, [2] 최대 8%). 리뷰어가 "이미 있다" 고 말할 근거는 [1][2][17][18][19][20] 이다.

**강화되는 부분 (그리고 G1을 다시 좁히는 방향).**
1. **분리형의 천장이 형식적으로 알려져 있다** [17]. 따라서 "우리 1차 분해가 구조 정보를 잃는다" 는 음의 결과는
   변칙이 아니라 **분리형 근사 계열의 알려진 한계를 새 영역에서 실증한 것** 으로 위치 지을 수 있다. 방어에 유리하다.
2. **부호 진술이 얹히는 대상이 다르다.** NRM·ATO의 모든 결과는 **외생 용량 벡터(또는 부품 재고 수준) 격자** 위의
   초/열모듈성이다. 우리는 **결정에 따라 전이하는 참여자 집합** 위에서 같은 부호를 다룬다.
   (D 조사도 같은 결론에 도달했다: 참여자 부분집합 격자 위의 초모듈성을 명시한 연구는 찾지 못했다.)
3. **결정 의존 가중치가 없다.** NRM의 분리형 계수는 상태에만 의존한다. 우리 목적항의 가중치는 **결정변수** 다
   (배분 잉여 → 재참여확률). 이 구조는 F2 범위 어디에도 없었다.
4. **인용 그래프가 비어 있다** (4.3). 비분리형 NRM 계보를 인용하는 논문 중 유지·이탈·잉여배분을 다룬 것은 0편이다.

**따라서 G1 재서술 제안 (F2 반영판).**
"다구간 여정·다품목 주문이 만드는 보완형 상호의존 때문에 자원 묶음의 증분 가치가 개별 자원 가치의 합을 넘어선다는 것은
네트워크 수익관리에서 이미 알려져 있고 (Bertsimas–Popescu 2003), 분리형 근사로는 그 이상 개선할 수 없다는 천장도
증명되어 있다 (Kunnumkal–Talluri 2016). 그러나 그 결과들은 모두 **외생적으로 주어진 용량 벡터** 위의 것이다.
우리는 같은 부호 문제가 **플랫폼의 잉여배분 결정이 다음 기간 참여자 집합을 바꾸는** 환경에서 어떻게 나타나는지를 보인다."

이 형태는 "부호 반전이 처음" 이라는 깨지기 쉬운 주장을 버리고, **선례를 인용해 문제를 정당화한 뒤 대상의 차이로
기여를 좁힌다.** 리뷰어의 "이미 있다" 공격을 정면으로 흡수한다.

**추가로 실행 가능한 측정 (문헌에서 도출).**
[1] 의 조건 진술 "load factor가 클 때만 분리형의 손해가 나타나고, 작으면 두 정책이 모두 최적에 가깝다" 는
우리 실험 3의 조건 축(공급 편중도, 대체 공급자 수)과 직접 대응한다.
우리 실험 3 v1에서 GNN과 MLP의 차이가 없었다면, **부하율(수요/공급용량 비)을 축으로 추가해** 재확인하는 것이
문헌 근거가 있는 후속 측정이다. (우리 설정의 여유는 1.3으로 고정되어 있어 부하율이 낮은 쪽이다.)

---

## 7. 출처 목록

1. Bertsimas, D., Popescu, I. (2003) Revenue Management in a Dynamic Network Environment. Transportation Science 37(3) 257–277 — https://doi.org/10.1287/trsc.37.3.257.16047 (전문 사본: https://www.mit.edu/~dbertsim/papers/Revenue%20Management/Revenue%20Management%20in%20a%20Dynamic%20Network%20Environment.pdf)
2. Zhang, D. (2011) An Improved Dynamic Programming Decomposition Approach for Network Revenue Management. M&SOM 13(1) 35–52 — https://doi.org/10.1287/msom.1100.0302 (전문 사본: https://danzhang.com/papers/DPDecomp_authorcopy.pdf)
3. Farias, V. F., Van Roy, B. (2007) An Approximate Dynamic Programming Approach to Network Revenue Management. 워킹페이퍼 (미심사) — https://web.mit.edu/vivekf/www/papers/ADP-rm-07-03.pdf
4. Talluri, K. T., van Ryzin, G. J. (1998) An Analysis of Bid-Price Controls for Network Revenue Management. Management Science 44(11) 1577–1593 — https://doi.org/10.1287/mnsc.44.11.1577
5. Adelman, D. (2007) Dynamic Bid Prices in Revenue Management. Operations Research 55(4) 647–661 — https://doi.org/10.1287/opre.1060.0368
6. Zhang, D., Adelman, D. (2009) An Approximate Dynamic Programming Approach to Network Revenue Management with Customer Choice. Transportation Science 43(3) 381–394 — https://doi.org/10.1287/trsc.1090.0262
7. Topaloglu, H. (2009) Using Lagrangian Relaxation to Compute Capacity-Dependent Bid Prices in Network Revenue Management. Operations Research 57(3) 637–649 — https://doi.org/10.1287/opre.1080.0597
8. (결번 — 표 작성 중 항목 병합)
9. Vossen, T. W. M., Zhang, D. (2015) Reductions of Approximate Linear Programs for Network Revenue Management. Operations Research 63(6) 1352–1371 — https://doi.org/10.1287/opre.2015.1442
10. Liu, Q., van Ryzin, G. (2008) On the Choice-Based Linear Programming Model for Network Revenue Management. M&SOM 10(2) 288–310 — https://doi.org/10.1287/msom.1070.0169
11. Kunnumkal, S., Talluri, K. (2016) On a Piecewise-Linear Approximation for Network Revenue Management. Mathematics of Operations Research 41(1) 72–91 — https://doi.org/10.1287/moor.2015.0716
12. Tong, C., Topaloglu, H. (2014) On the Approximate Linear Programming Approach for Network Revenue Management Problems. INFORMS Journal on Computing 26(1) 121–134 — https://doi.org/10.1287/ijoc.2013.0551
13. Kunnumkal, S., Topaloglu, H. (2010) Computing Time-Dependent Bid Prices in Network Revenue Management Problems. Transportation Science 44(1) 38–62 — https://doi.org/10.1287/trsc.1090.0291
14. Kunnumkal, S., Topaloglu, H. (2010) A New Dynamic Programming Decomposition Method for the Network Revenue Management Problem with Customer Choice Behavior. POM 19(5) 575–590 — https://doi.org/10.1111/j.1937-5956.2009.01118.x
15. Meissner, J., Strauss, A. (2012) Network revenue management with inventory-sensitive bid prices and customer choice. EJOR 216(2) 459–468 — https://doi.org/10.1016/j.ejor.2011.06.033
16. Ata, B., Akan, M. (2015) On bid-price controls for network revenue management. Stochastic Systems 5(2) 268–323 — https://doi.org/10.1287/12-ssy081
17. Kunnumkal, S., Talluri, K. (2012) Equivalence of Piecewise-Linear Approximation and Lagrangian Relaxation for Network Revenue Management. Barcelona GSE Working Paper 608 (미심사판, 게재본은 11번) — https://www.bse.eu/sites/default/files/working_paper_pdfs/608.pdf
18. Adelman, D., Barz, C., Olivares-Nadal, A. V. (2025) Dynamic Basis Function Generation for Network Revenue Management. INFORMS Journal on Computing — https://doi.org/10.1287/ijoc.2023.0418 (프리프린트: https://arxiv.org/pdf/2502.16830)
19. Laumer, S., Barz, C. (2023) Reductions of non-separable approximate linear programs for network revenue management. EJOR 309(1) 252–270 — https://doi.org/10.1016/j.ejor.2023.01.006
20. Ma, W. (2023) Order-Optimal Correlated Rounding for Fulfilling Multi-Item E-Commerce Orders. M&SOM 25(5) 1324–1337 — https://doi.org/10.1287/msom.2023.1219 (프리프린트: https://arxiv.org/abs/2207.04774)
21. Topaloglu, H. (2007) 위 7번의 저자 사본 (2007-11-22판) — https://people.orie.cornell.edu/huseyin/publications/revenue_man.pdf
22. Adelman, D., Mersereau, A. J. (2008) Relaxations of Weakly Coupled Stochastic Dynamic Programs. Operations Research 56(3) 712–727 — https://doi.org/10.1287/opre.1070.0445
23. Reiman, M. I., Wan, H., Wang, Q. (2023) Asymptotically Optimal Inventory Control for Assemble-to-Order Systems. Stochastic Systems 13(2) 128–180 — https://doi.org/10.1287/stsy.2022.0099 (프리프린트: https://arxiv.org/pdf/1809.08271)
24. Dogru, M. K., Reiman, M. I., Wang, Q. (2010) A Stochastic Programming Based Inventory Policy for Assemble-to-Order Systems with Application to the W Model. Operations Research 58(4) 849–864 — https://doi.org/10.1287/opre.1090.0772
25. Reiman, M. I., Wang, Q. (2015) Asymptotically Optimal Inventory Control for Assemble-to-Order Systems with Identical Lead Times. Operations Research 63(3) 716–732 — https://doi.org/10.1287/opre.2015.1372
26. Akcay, Y., Xu, S. H. (2004) Joint Inventory Replenishment and Component Allocation Optimization in an Assemble-to-Order System. Management Science 50(1) 99–116 — https://doi.org/10.1287/mnsc.1030.0167
27. ElHafsi, M., Fang, J., Hamouda, E. (2020) A novel decomposition-based method for solving general-product structure assemble-to-order systems. EJOR 286(1) 233–249 — https://doi.org/10.1016/j.ejor.2020.03.016
28. Lu, Y., Song, J.-S. (2005) Order-Based Cost Optimization in Assemble-to-Order Systems. Operations Research 53(1) 151–169 — https://doi.org/10.1287/opre.1040.0146 · **정정**: Bolandnazar, M., Huh, W. T., McCormick, S. T. (2019) Technical Note—Error Noted in Order-Based Cost Optimization in Assemble-to-Order Systems. Operations Research 67(1) 163–166 — https://doi.org/10.1287/opre.2018.1789
29. Akan, M., Ata, B. (2009) Bid-Price Controls for Network Revenue Management: Martingale Characterization of Optimal Bid Prices. Mathematics of Operations Research 34(4) 912–936 — https://doi.org/10.1287/moor.1090.0411
30. Kunnumkal, S., Talluri, K. (2019) A strong Lagrangian relaxation for general discrete-choice network revenue management. Computational Optimization and Applications 73 275–310 — https://doi.org/10.1007/s10589-019-00068-y
31. Dumouchelle, J., Frejinger, E., Lodi, A. (2024) Reinforcement learning for freight booking control problems. Journal of Revenue and Pricing Management — https://doi.org/10.1057/s41272-023-00459-1 (전문: https://arxiv.org/pdf/2102.00092, 코드: https://github.com/jdumouchelle/RLforBookingControl)

### 7.1 4.3 표에만 등장하는 서지 (제목·게재지 수준만 확인)

32. A Network Airline Revenue Management Framework Based on Decomposition by Origins and Destinations. Transportation Science (2014) — https://doi.org/10.1287/trsc.2013.0469
33. Revenue management for operations with urgent orders. EJOR (2015) — https://doi.org/10.1016/j.ejor.2014.07.015
34. Dynamic Pricing for Network Revenue Management: A New Approach and Application in the Hotel Industry. INFORMS J. Computing (2017) — https://doi.org/10.1287/ijoc.2016.0713
35. Dynamic Programming Decomposition for Choice-Based Revenue Management with Flexible Products. Transportation Science (2017) — https://doi.org/10.1287/trsc.2017.0743
36. Decomposition methods for dynamic room allocation in hotel revenue management. EJOR (2018) — https://doi.org/10.1016/j.ejor.2018.05.027
37. Network Revenue Management with Cancellations and No-Shows. POM (2019) — https://doi.org/10.1111/poms.12907
38. Ke, J., Zhang, D., Zheng, H. (2019) An Approximate Dynamic Programming Approach to Dynamic Pricing for Network Revenue Management. POM 28(11) 2719–2737 — https://doi.org/10.1111/poms.13075
39. Capacity Allocation with Multiple Suppliers and Multiple Demand Classes. POM (2019) — https://doi.org/10.1111/poms.13076
40. An Approximation Algorithm for Network Revenue Management Under Nonstationary Arrivals. Operations Research (2020) — https://doi.org/10.1287/opre.2019.1931
41. Standardized cargo network revenue management with dual channels. EJOR (2021) — https://doi.org/10.1016/j.ejor.2021.02.046
42. Assign-to-Seat: Dynamic Capacity Control for Selling High-Speed Train Tickets. M&SOM (2023) — https://doi.org/10.1287/msom.2023.1188
43. Pricing Personalized Bundles: A New Approach and An Empirical Study. M&SOM (2016) — https://doi.org/10.1287/msom.2015.0563
44. Simchi-Levi, D., Zheng, Z., Zhu, F. (2025) On Greedy-Like Policies in Online Matching with Reusable Network Resources and Decaying Rewards. Management Science — https://doi.org/10.1287/mnsc.2023.02588

---

## Coverage Status

**직접 확인한 것 (PDF 전문 텍스트를 추출해 해당 문장을 읽음)**
- [1] Bertsimas–Popescu TS 2003: 초록, 서론 Contributions, 1절 문헌검토, 2.1·2.2절, 3.1·3.2절, 4절 Proposition 3, 4.1절 예제와 감소차분 논의, 5절 오버부킹, 결론절, 개선폭 수치
- [2] Zhang M&SOM 2011: 초록, 서론, 2절 문헌검토, 식 (6) 비분리형 근사, 수치 결과 (8%, CPU +30%, 표 1·2)
- [3] Farias–Van Roy WP 2007: 초록, 1절, 4.1·4.2절 (아핀·분리형 오목 구조), 6.1절 rALP, 수치 (약 1%, 약 8%), 실무 1~2% 참조치
- [21] Topaloglu OR 2009 저자 사본: 초록, 서론 (Talluri–van Ryzin·Bertsimas–Popescu·Adelman 귀속), 3절 분리 구조의 저장 이유, 5절 상한·정책 수치
- [17] Kunnumkal–Talluri BGE WP 608: 초록, 1절 문헌, 3절 Proposition 2 및 tightest separable 문장, 4절 선택모형 예외
- [18] Adelman–Barz–Olivares-Nadal arXiv 2502.16830: 초록, 2.1절 기저함수 계보 전체, 6절 기저함수 성질, 7절 벤치마크 설계 (AA·SPLA·NSEP)
- [23] Reiman–Wan–Wang arXiv 1809.08271: 초록, 1절 서론 (정책 부류 제한의 대가, 보완·대체 혼재의 원인)
- [31] Dumouchelle–Frejinger–Lodi arXiv 2102.00092: 1.1.1절 DLP·bid price, 1.2절 기여·갭 진술, 2절 DQN-L 대 DQN-S, 3.1.3·3.2.3절 예측과제, duality 상실 진술

**요약·서지 수준만 확인한 것**: [4][5][6][9][10][11][12][13][14][15][16][19][22][24][25][26][27][28][29][30] 및 32~44.
[20] Ma는 WebFetch 요약 수준이다 (근사비 1+ln q, d, tight성, 선행 q/4 확보. 원문 대조는 미완).

**확인 불가·미완 (4.1 표에 논문별 확인 질문을 적었다)**
- INFORMS 페이월: pubsonline.informs.org 전문은 403으로 접근 불가 (Talluri–van Ryzin 1998, Adelman 2007, Vossen–Zhang 2015, Liu–van Ryzin 2008, Kunnumkal–Talluri 2016 게재본, Akan–Ata 2009 등).
- ScienceDirect 403: **Laumer–Barz EJOR 2023 본문 접근 실패.** F2b의 최신 핵이므로 도서관 접속 최우선 대상이다.
- [1] 4.1절 decreasing differences 부등호 방향: 조판 원문 확인 필요 (PDF 추출본과 저자 주장이 어긋나 보인다).
- [31] DQN-L 대 DQN-S 성능 비교 결과: 결과 표를 읽지 않았으므로 어느 쪽이 나았는지 적지 않았다.
- OpenAlex: 단일 엔티티 조회 (works/doi 경로) 는 성공해 4개 ID를 확보했다. 그러나 **목록 질의 (filter=cites 및 search) 는 일일 예산 소진 (HTTP 429, Insufficient budget) 으로 4회 재시도·대기 후에도 실패했다.** 실패한 쿼리 전부를 4.2에 기록했다. 역인용 순회는 OpenCitations API로 우회해 수행했고 커버리지 누락을 4.3에 명시했다.
- Semantic Scholar API: 무키 429로 즉시 차단되어 사용하지 않았다.

**과제 처리 상태**: F2a done · F2b done · F2c done (단정적 부정 답) · F2d done (선례 있음 판정).

---

## 부록: 증거 표 추가 항목 (1절 표의 31행)

| # | 제목 | 저자·연도 | 게재지·등급 | 근사 형태 | 분리형 한계 지적 | 보고된 개선 폭 | 상호의존 종류 | 우리와의 차이 | URL | 읽은 수준 |
|---|---|---|---|---|---|---|---|---|---|---|
| 31 | Reinforcement learning for freight booking control problems | Dumouchelle, Frejinger, Lodi 2024 | J. Revenue and Pricing Management (Springer); 프리프린트 arXiv 2102.00092 | **학습된 예측을 RL에 삽입**: 지도학습으로 기간말 운영문제(EoHP) 최적값을 예측, 그 예측을 DQN 안에서 사용. 상태 표현 2종 (선형 DQN-L 대 순열불변 집합 DQN-S) | **간접.** 화물 맥락에서는 운영문제가 MILP라 **이중해가 없어 bid price 자체를 만들 수 없다**고 지적한다 | 결과 표 미독 (수치 인용하지 않음) | 보완형+대체형 혼재 (요청이 여러 용량 차원을 동시에 소비) | 용량이 외생 고정, 결정이 수락·거절뿐 (잉여배분 없음), 참여자 구성이 결정에 무관, 학습 예측을 **MILP 목적함수 계수로 분해하지 않고** RL 정책에 넣는다 | https://doi.org/10.1057/s41272-023-00459-1 · https://arxiv.org/pdf/2102.00092 | **전문추출** (결과 표 제외) |

이 항목은 1절 표 작성 이후 4.3절의 피인용 스캔에서 발견했으므로 여기에 따로 적는다.
방법 구조가 우리와 가장 가깝고 **화물(freight) 도메인**이므로 관련연구 절에 반드시 포함해야 한다.
또한 저자군(Dumouchelle, Lodi)은 C 조사에서 확인한 Neur2RO·Neur2SP 계보와 같다 — 즉 F2와 C의 두 계보가
이 논문에서 만난다. C 조사 산출물과 교차 확인이 필요하다.
