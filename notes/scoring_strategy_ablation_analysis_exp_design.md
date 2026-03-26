# Scoring Strategy Ablation 실험 분석

## 1. 실험 목적

MoAS의 scoring 방식에 대한 설계 선택을 정당화하기 위한 ablation study.
핵심 수식:

```
z^{MoAS}_c(x) = z_{main,c}(x) + Σ_m α_m(x) · z_{m,c}(x)
s(x) = max_c z^{MoAS}_c(x)
```

---

## 2. 최종 실험 구성

| 이름 | α 값 | 설명 |
|------|-------|------|
| **Main Only** | α = 0 | modality logit 미사용, z_main만으로 scoring |
| **Uniform Sum** | (1, 1, 1) | modality logit을 full-scale로 합산 |
| **Uniform Average** | (1/3, 1/3, 1/3) | modality logit의 평균을 합산 |
| **Adaptive-Entropy** | softmax(entropy 기반) | entropy 기반 sample-wise adaptive weighting |
| **Adaptive-Energy (Ours)** | softmax(-E/τ) | energy 기반 sample-wise adaptive weighting |

### 용어 선정 근거

- **Main Only**: modality logit을 전혀 사용하지 않음을 명확히 전달. 기존 module ablation (tab_ablation)의 "w/o MoAS"와 동일한 설정이므로, 본문에서 "this corresponds to the w/o MoAS setting in Table X"로 명시하여 일관성 확보.
- **Uniform Sum / Uniform Average**: 둘 다 균등(uniform) 가중치이지만, z_main 대비 modality logit의 기여 스케일이 다름. "Sum vs Average"로 스케일 차이를 직관적으로 전달. 기존의 "No Weight / Equal Weight"가 갖던 모호함을 해소.
- **Adaptive-Entropy / Adaptive-Energy**: "Adaptive-"를 공통 접두어로 사용하여 fixed(Uniform) 계열과의 대비를 명확히 하고, 뒤의 지표명으로 reliability 측정 방식의 차이를 표현.

---

## 3. 실험이 검증하는 비교 축

이 테이블은 아래에서 위로 하나씩 설계 선택을 제거하는 구조로 읽을 수 있다:

```
Main Only
  ↓  modality logit 추가 자체의 효과
Uniform Sum
  ↓  기여 스케일(normalization)의 효과
Uniform Average
  ↓  adaptive weighting의 효과 (fixed → adaptive)
Adaptive-Entropy
  ↓  reliability 지표의 효과 (entropy → energy)
Adaptive-Energy (Ours)
```

| 비교 | 검증 포인트 |
|------|-------------|
| Main Only → Uniform Sum | modality logit을 z_main에 추가하는 것 자체가 novelty detection에 도움이 되는가? |
| Uniform Sum → Uniform Average | modality logit의 기여 스케일을 조절하면 성능이 달라지는가? (z_main 대비 modality 비중) |
| Uniform Average → Adaptive-Energy | 고정 균등 가중치 대비, sample-wise adaptive weighting이 얼마나 효과적인가? |
| Adaptive-Entropy → Adaptive-Energy | 같은 adaptive 방식에서, reliability 지표로 energy가 entropy보다 우수한가? |

---

## 4. 타당성 분석

### 4.1 실험 방향: 타당

MoAS의 핵심 기여가 "adaptive energy-based modality weighting"이므로, 가중치 전략 비교 ablation은 적절한 방향이다.

### 4.2 Uniform Sum과 Uniform Average의 구분

(1,1,1)과 (1/3,1/3,1/3)은 모달리티 간 상대 비율은 동일하지만, **z_main 대비 modality logit의 기여 스케일이 다르다**. MaxLogit은 절대값에 의존하므로 결과가 달라진다.

- Uniform Sum: modality logit이 z_main과 동등한 스케일로 합산 → modality 쪽 기여가 큼
- Uniform Average: modality logit 기여가 1/3로 축소 → z_main이 상대적으로 지배적

이 비교는 **modality contribution scale**의 효과를 보여준다. 논문에서 기술할 때 "weighting strategy"가 아니라 "modality contribution scale" 관점에서 분석하는 것이 정확하다.

### 4.3 Adaptive-Entropy 추가의 의의

기존 3가지 설정(Main Only, Uniform Sum, Uniform Average)만으로는 "왜 energy인가?"라는 질문에 답할 수 없었다. Adaptive-Entropy를 추가함으로써:

- 같은 adaptive framework 내에서 **reliability 지표만 다른** 비교가 가능
- Energy가 모든 class의 logit을 종합적으로 반영하는 반면, entropy는 softmax 정규화 과정에서 정보가 손실될 수 있다는 가설을 실험으로 검증

---

## 5. 추가 비교 후보 (rebuttal/supplementary 용)

리뷰어가 추가 비교를 요구할 경우를 대비해 미리 실험을 돌려둘 수 있는 후보:

### 축 1: Reliability 지표 추가

| 전략 | 수식 | 의미 |
|------|------|------|
| **Adaptive-MaxLogit** | softmax(max_c z_{m,c}) | 가장 confident한 class의 logit만 사용 |
| **Adaptive-MSP** | softmax(max_c softmax(z_m)_c) | 최대 softmax 확률 기반 |

### 축 2: Weight 계산 방식

| 전략 | 방식 | 의미 |
|------|------|------|
| **Hard Selection (Top-1)** | energy 최소인 modality만 α=1, 나머지 0 | Winner-takes-all |

### 축 3: Integration 방식

| 전략 | 수식 | 의미 |
|------|------|------|
| **Modality Only** | `Σ α_m z_m` (z_main 제거) | z_main 없이 modality logit만으로 scoring |
