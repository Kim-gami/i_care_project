# 👶 I-Care — 영유아 위험 감지 알림 서비스

> RTSP 카메라 스트림 기반 실시간 영유아 위험 감지 및 Telegram 알림 시스템  
> 🏆 원내 최우수상 수상

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![YOLOv11](https://img.shields.io/badge/YOLOv11-00FFFF?style=flat)
![TensorRT](https://img.shields.io/badge/TensorRT-76B900?style=flat&logo=nvidia&logoColor=white)

---

## 📌 문제 정의

- 영유아 안전사고의 87.8%가 주택, 그 중 41.6%가 침실에서 발생
- 수면 중 담요·베개로 인한 얼굴 가림, 이물질 근접 등 보호자가 즉시 인지하지 못하는 위험 상황 존재
- 보호자의 지속적 관찰을 대체할 수 있는 실시간 감지 시스템 필요

---

## 🏗 서비스 아키텍처

```
RTSP 카메라 스트림 (Tapo C210)
        ↓ GStreamer 기반 GPU 영상처리
Jetson Xavier + TensorRT 추론 엔진
   ├─ Night.engine (야간: 코·입 감지)
   └─ Day.engine  (주간: 위험 물체 감지)
        ↓ 위험 감지 판단 로직
Telegram Bot API → 보호자 즉시 알림
```

---

## ⚡ 핵심 성과

| 클래스 | 개선 전 mAP@50 | 개선 후 mAP@50 | 향상폭 |
|--------|--------------|--------------|-------|
| bottle_cap | 0.815 | 0.888 | +0.073 |
| chopstick | 0.789 | 0.890 | +0.101 |
| fork | 0.755 | 0.923 | **+0.168** |

- **주간 모델 mAP@50: 0.907 / 야간 모델 mAP@50: 0.927**
- TensorRT + FP16 적용으로 실시간 추론 가능 수준 확보
- Telegram 명령 기반 Night/Day 모드 실시간 전환 구현

---

## 🛠 기술 스택

| 분류 | 기술 |
|------|------|
| 모델 | YOLOv11s, PyTorch |
| 최적화 | TensorRT, ONNX, FP16 |
| 영상처리 | GStreamer, OpenCV |
| 데이터 | Roboflow, OpenImagesV7 |
| 하드웨어 | Jetson Xavier, Tapo C210 (RTSP) |
| 알림 | Telegram Bot API |

---

## 🔍 주요 구현 내용

**1. 데이터 구축 및 품질 관리**
- 공개 데이터셋(OpenImagesV7) + 직접 수집 데이터를 Roboflow로 라벨링
- 소형 물체 감지 품질 저하 원인을 라벨링 기준 불일치로 파악하고 기준 재정비
- copy-paste 증강, NMS·loss 비중 조정으로 성능 개선

**2. 주/야간 이중 모델 구조**
- 조도 환경에 따라 감지 성능이 달라지는 문제를 주/야간 모델 분리로 해결
- Night mode: 코·입이 동시에 미감지될 경우 위험으로 판단
- Day mode: 손과 위험 물체의 거리 근접 시 위험으로 판단

**3. 엣지 디바이스 최적화**
- PyTorch 모델 → ONNX → TensorRT 변환 파이프라인 구축
- FP16 정밀도 적용으로 연산량 감소 및 처리 속도 개선
- GStreamer 기반 영상처리를 CPU에서 GPU로 전환하여 추론 병목 해소

**4. Telegram 기반 서비스 연동**
- 위험 감지 시 Telegram Bot을 통해 보호자에게 즉시 알림 전송
- Telegram 명령어로 Night/Day 추론 엔진 실시간 전환 기능 구현

---

## 🚀 실행 방법

```bash
# Jetson Xavier 환경에서
cd /home/nvidia/Desktop/i_care
python3 i_care_main.py --show
```

---

## 👥 팀 구성

- 3인 팀
- 본인 담당: 서비스 기획·데이터 수집 및 라벨링·모델 학습 및 튜닝·TensorRT 최적화·위험 감지 로직·Telegram 엔진 전환 기능 개발
