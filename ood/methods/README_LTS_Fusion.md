# LTS Fusion OOD Detection Method

## 개요
LTS (Large-scale Temperature Scaling) Fusion은 기존의 개별 모달리티 특징 대신 **fusion features**를 사용하여 OOD 탐지를 수행하는 방법입니다.

## 주요 특징
- **Fusion Features 활용**: 멀티모달 네트워크에서 추출된 융합 특징을 사용
- **LTS Scaling**: 융합 특징에서 상위 percentile 특징을 선택하여 스케일링 팩터 계산
- **Energy-based Detection**: 스케일링된 로짓을 사용하여 최종 OOD 점수 계산

## 구현 파일
- `lts_fusion.py`: LTSFusionDetector 클래스 구현
- `mmeabase.py`: 평가 프레임워크에 통합
- `__init__.py`: 모듈 export 설정

## 사용법

### 1. 설정 파일에서 활성화
```yaml
enable_ood: true
ood_methods: ["LTS_Fusion"]
```

### 2. 코드에서 직접 사용
```python
from ood import LTSFusionDetector

# 탐지기 초기화
detector = LTSFusionDetector(model, device='cuda', temperature=1.0, percentile=65)

# OOD 점수 계산
scores = detector.compute_scores_with_fusion_features(logits, fusion_features)
```

## 파라미터
- `temperature`: 온도 스케일링 파라미터 (기본값: 1.0)
- `percentile`: LTS에서 사용할 상위 특징 비율 (기본값: 65%)

## 실행 예시

### TBN 네트워크 사용
```bash
python main.py --config scripts/TBN/5.uestc-mmea-lts-fusion.yaml
```

### TSN 네트워크 사용  
```bash
python main.py --config scripts/TSN/3.uestc-mmea-lts-fusion.yaml
```

## 기존 방법과의 차이점

| 방법 | 사용 특징 | 특징 |
|------|-----------|------|
| LTS_Individual | 개별 모달리티 특징 | 각 모달리티별 가중치 계산 |
| **LTS_Fusion** | **융합 특징** | **통합된 멀티모달 표현 활용** |

## 장점
1. **통합된 표현**: 이미 융합된 특징을 사용하여 모달리티 간 상호작용 정보 활용
2. **단순한 구조**: 개별 모달리티 가중치 계산 불필요
3. **효율성**: 융합 특징은 이미 네트워크에서 계산되므로 추가 연산 최소화

## 로그 출력 예시
```
🚀 [LTS_Fusion] Starting computation with fusion features
  📥 Input logits shape: torch.Size([64, 24])
  📥 Fusion features shape: torch.Size([64, 512])
  ⚡ Computing LTS scale from fusion features:
    📈 Percentile: 65%, Selected features: 179/512
    📊 S1 (all features sum): 12.3456
    📊 S2 (top-k features sum): 8.7654
    🎯 LTS Scale - Avg: 2.1234, Min: 1.5678, Max: 3.4567
  📊 Original logits mean: 0.1234
  📊 Scaled logits mean: 0.2345
  📊 Effective scaling factor: 1.9012
  🎯 Final OOD scores - Avg: -5.6789, Min: -12.3456, Max: -2.1098
✅ [LTS_Fusion] Computation completed
```

## 주의사항
- 융합 특징이 추출되지 않은 경우 일반 Energy 방법으로 fallback
- 멀티모달 네트워크에서 최적의 성능 발휘
- 단일 모달리티에서도 사용 가능하지만 융합의 이점은 제한적
