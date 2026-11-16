# 양면경매 전제, 증명해야 하나? 근거와 범위 정리

2026-11-16 작성 (참고용). 용어는 작성 당시 표기 그대로 두었다. 당기이윤 최대화 정책 = 한 기간 최적 정책, 후속이윤 반영 정책 = 기간별 평균 이후 손실 반영 정책.

## 결론

양면경매가 개별 경매보다 낫다는 것을 처음부터 증명할 필요는 없다. 대신 세 가지를 한다.

1. 현실 근거: 같은 원자재를 두고 판매 경매와 구매 역경매가 따로, 반복적으로 열린다는 산업 사례를 보인다.
2. 선행연구 근거: 묶음 수요가 있을 때 따로 거래하면 노출 문제(exposure problem)로 효율이 떨어지고, 묶음 단위 통합 청산이 이를 해소한다는 근거를 대어 전제가 그럴듯하다는 것을 보인다.
3. 범위와 실험 분리: 통합의 우월성 자체는 범위 밖으로 두되, 실험에서는 통합 효과(ΔSW)와 다기간 padding 효과(ΔΠ)를 나눠 본다.

## 증명까지는 필요 없는 이유

연구 질문이 다르다. 이 연구의 질문은 "따로 하는 경매와 양면경매 중 어느 쪽이 나은가"가 아니고 "양면 청산 플랫폼이 있을 때, 참여자가 떠나고 들어오는 상황에서 padding을 어떻게 운영해야 하는가"이다.

양면경매가 더 나은지는 별도 논문 주제다. 대부분의 메커니즘 설계 논문도 "이런 플랫폼이 있다고 하자"에서 출발하고, 그 전제가 그럴듯하다는 근거만 댄다.

## 현실 근거: 인도 합금철 시장

같은 페로크롬을 두고 판매자는 경매로 팔고(FACOR), 구매자는 역경매로 산다(SAIL-Salem). 두 경매는 따로, 반복적으로 열린다.

| 근거 | 내용 | 출처 |
|---|---|---|
| 판매 경매 | Vedanta-FACOR가 페로크롬 판매 경매를 2024년 1월·2월 연속 실시 | BigMint, Ferro Alloy Monthly Review, Feb-2024 · Mar-2024, p.16 |
| 구매 역경매 | SAIL 7개 공장 페로실리콘 15,150톤이 2026년 3월 6일 종료된 역경매에서 전량 낙찰 (톤당 약 107,800루피, 세금 제외) | BigMint (2026.03) |
| 구매 역경매 | SAIL-Salem 고탄소 페로크롬 7,300톤 구매 입찰, 역경매 자격 규칙에 따라 최고가 입찰 제외 | BigMint (2025.12) |
| 구매 역경매 | Salem Steel Plant 저탄소 페로크롬 구매(GEM/2026/R/681740)가 역경매 "Yes"로 표시 | Infralens (GeM 기록) |
| 반복성 | 월간 표에서 두 달 연속 반복. Salem 페로크롬 구매도 2025년 12월 → 2026년 3월(2,500톤 이상) 반복 | BigMint (2026.03) |
| 보강 (원문 확인 필요) | CAG 보고서의 "Salem 페로크롬 가격을 2개월마다 온라인 역경매로 결정" 규정 (35쪽) | CAG Report No. 10 of 2025 보도자료 |

"최고가 입찰 제외" 규칙: SAIL의 구매 역경매에서는 역경매 전에 가장 비싼 입찰자를 규칙에 따라 제외한다(페로망간 사례). 참여자를 매번 일부 밀어낸다는 점에서 padding과 성격이 비슷하다. 논문에서는 "유사한 제외 규칙이 실제 조달에서도 쓰인다"는 정도로만 쓴다.

주의
- GEM/2026/R/672743(Salem 페로크롬 4,300톤), GEM/2026/R/645449(RINL 실리코망간)는 아직 원문 미확인이다.
- BigMint 표는 주요 경매만 고른 목록이라 건수를 시장 전체 빈도로 쓰지 않는다.
- 이 사례만으로는 "두 경매를 통합하면 더 좋다"는 결론이 나오지 않는다.

## 왜 통합인가: 서론에 넣을 근거

가장 강한 근거는 노출 문제다. 다품목 묶음 주문이라는 이 연구의 설정에 직접 맞으므로 서론에서 가장 앞에 둔다.

| 순서 | 근거 | 내용 | 인용할 문헌 |
|---|---|---|---|
| 1 | 노출 문제 해소 | 여러 거래가 모두 성립해야 의미가 있는데 따로 진행하면, 일부만 성사된 참여자가 손해를 본다. 실험에서 연속 양면경매의 효율은 노출이 클 때 20%, 작을 때 55%였고, 묶음 단위 청산(패키지 시장)은 82%·89%였다. | Goeree & Lindsay (2020), Review of Economic Studies, 87(5), 2230–2255 |
| 2 | 시장 두께 | 따로 열리는 경매는 그때그때 참여자가 적다. 한곳에 모으면 거래 상대가 많아져 매칭이 좋아진다. | Roth (2008), The Economic Journal, 118(527) |
| 3 | 효율 손실 축소 | McAfee 방식 trade reduction은 거래 하나만 취소해 거래이익이 최대치의 1 − 1/k다 (k = 효율적인 거래 수). 시장이 커질수록 진실성을 보장하는 비용이 작아진다. | McAfee (1992); Satterthwaite & Williams (1989), Review of Economic Studies |

인도 합금철 시장에 대입하면: 판매자들이 품목별로 따로 경매를 열면, 여러 합금철이 함께 필요한 구매자는 품목마다 다른 경매에 들어가야 한다. 하나는 사고 하나는 못 사면 손해를 본다. 묶음 양면경매는 필요한 품목을 한 번에 입찰하고 다 확보될 때만 거래하므로 이 위험이 없다.

간접 근거 (참고용): Ketcham et al.(1984)과 Alós-Ferrer et al.(2022)은 양면경매가 판매자 가격제시(posted offer)보다 효율이 높음을 실험으로 보였다. 다만 "판매 경매와 구매 역경매가 따로 열리는 구조"와는 다른 비교이고, Alós-Ferrer et al.의 수치는 원문 미확인이다. Søberg(2002)처럼 뚜렷한 차이가 없다는 결과도 있다.

## 범위 명시

논문의 한계 또는 범위 문단에 다음 문장을 넣는다.

> 본 연구는 양면 청산 구조를 전제로 하며, 개별 단면 경매 대비 양면 청산의 우월성 자체는 분석 범위 밖으로 둔다.

선행연구를 인용한 뒤 "다기간 B2B 거래에서는 단기 효율뿐 아니라 참여자 유지와 장기 성과를 함께 봐야 한다"는 문장을 쓸 때는, 선행연구가 입증한 결론이 아니라 이 연구가 제기하는 필요성으로 쓴다.

## 효과를 나눠 검증하는 실험 설계

연구의 기여는 ΔΠ(다기간 padding 효과)다. ΔSW(통합 효과)와 섞이면 무엇 때문에 좋아졌는지 해석이 흐려지므로 두 비교를 분리한다.

| 단계 | 구조 | 확인할 질문 | 현재 상태 |
|---|---|---|---|
| ① | 분리된 판매 경매 + 구매 역경매 | 기존 구조에서 얼마나 거래되는가? | 없음 (저널 단계) |
| ② | 통합 양면경매 (고정 padding) | 통합만으로 거래량·사회후생이 늘어나는가? | 고정 padding 정책으로 있음 |
| ③ | 통합 + 단일기간 padding 조절 | 당기 이윤만 보고 padding을 조절하면? | 당기이윤 최대화 정책으로 있음 |
| ④ | 통합 + 다기간 재참여 고려 | 재참여를 고려하면 장기 이윤이 더 나아지는가? | 후속이윤 반영 정책으로 있음 |

ΔSW = SW(② Integrated) − SW(① Separate)

ΔΠ = Π(④ Multi-period) − Π(③ Single-period)

① 분리 구조 만드는 법: 같은 참여자를 판매자 주도 경매(판매자 1명 → 구매자 여럿)와 구매자 주도 역경매(구매자 1명 → 공급자 여럿)로 나눠 배정한다. 묶음 구매자는 품목별로 따로 사야 하므로 노출 위험이 드러난다. 참여자 집합, 가치·원가, 정보 조건은 ②와 똑같이 둔다.

## APIEMS와 저널 적용

- APIEMS (11/29~12/2): 동기 부여 슬라이드에 인도 합금철 사례 1장 + 노출 문제·시장 두께 근거 각 한 줄. 추가 실험 없음.
- 저널: ① 분리 구조 기준선을 추가 실험 또는 리비전 대응용으로 준비. CAG 보고서·GeM 미확인 항목 원문 확인.

## Sources

- Goeree, J. K., & Lindsay, L. (2020). The Exposure Problem and Market Design. Review of Economic Studies, 87(5), 2230–2255.
- Roth, A. E. (2008). What Have We Learned from Market Design? The Economic Journal, 118(527), 285–310.
- McAfee, R. P. (1992). A dominant strategy double auction. Journal of Economic Theory.
- Satterthwaite, M. A., & Williams, S. R. (1989). The rate of convergence to efficiency in the buyer's bid double auction as the market becomes large. Review of Economic Studies.
- Segal-Halevi, Hassidim & Aumann, SBBA (arXiv)
- Ketcham, Smith & Williams (1984), Review of Economic Studies; Alós-Ferrer, Buckenmaier & Kirchsteiger (2022), Experimental Economics (간접 근거, 원문 미확인)
- BigMint, Ferro Alloy Monthly Review, Feb-2024 · Mar-2024, p.16
- BigMint: SAIL books over 15,000 t of ferro silicon at reverse auction (2026.03)
- BigMint: SAIL-SSP issues purchase tender for 7,300 t of HC ferro chrome (2025.12)
- BigMint: SAIL-SSP floats purchase tender for over 2,500 t of HC ferro chrome (2026.03)
- Infralens: GeM bid GEM/2026/R/681740
- CAG Report No. 10 of 2025 보도자료

저널·권·호 서지사항은 인용 전에 원문으로 확인한다.
