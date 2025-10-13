# n_model_tunning.py
# YOLO11 학습/검증/추론/내보내기 통합 스크립트 (소물체 리콜 개선 확장판 + 학습 하이퍼 옵션 추가)
# ---------------------------------------------------------------
# 신규 포인트
#  - 클래스별 임계값: --class-thresh "mouth:0.18,nose:0.25,pacifier:0.16"
#  - NMS/후처리: --agnostic-nms, --max-det, --tta
#  - ROI 2단계 추론(선택): 얼굴 → ROI 확장 → 코/입 재추론
#    예) --roi-face-weights face_yolo.pt --roi-scale 1.2 --roi-imgsz 640
#  - 학습 안정화 하이퍼(추가됨): save-period/label-smoothing/mosaic/close-mosaic/등
# ---------------------------------------------------------------

import argparse
import os
from typing import Dict, Tuple, Union, List, Optional

import torch
from ultralytics import YOLO

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

# ===== 기본 설정 =====
DEFAULT_MODEL = "yolo11n.pt"
RUNS_DIR = os.environ.get("YOLO_RUNS_DIR", "runs")


# ---------- 유틸 ----------
def check_cuda(device_arg: Union[str, int, None]) -> str:
    if device_arg is None:
        return "0" if torch.cuda.is_available() else "cpu"
    if isinstance(device_arg, int):
        return str(device_arg)
    s = str(device_arg).strip()
    return "cpu" if s.lower() == "cpu" else s


def parse_class_thresh(arg: Optional[str], names: List[str] | Dict[int, str]) -> Dict[int, float]:
    """
    "mouth:0.18,nose:0.25,pacifier:0.16" → {cls_id: conf}
    names가 list이든 dict이든 모두 처리 가능
    """
    mapping: Dict[int, float] = {}
    if not arg:
        return mapping

    # names를 역매핑(이름 -> id)으로 변환
    name2id: Dict[str, int] = {}
    if isinstance(names, dict):
        for i, n in names.items():   # i: class id, n: class name
            name2id[str(n)] = int(i)
    else:  # list/tuple
        for i, n in enumerate(names):
            name2id[str(n)] = i

    pairs = [p.strip() for p in arg.split(",") if p.strip()]
    for p in pairs:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        k = k.strip()
        try:
            confv = float(v)
        except ValueError:
            continue
        if k.isdigit():
            mapping[int(k)] = confv
        elif k in name2id:
            mapping[name2id[k]] = confv
    return mapping


def apply_classwise_threshold(result, class_conf: Dict[int, float]):
    """Ultralytics Result에 클래스별 conf 필터 적용 (in-place mask)"""
    if not class_conf:
        return result
    if result.boxes is None or result.boxes.cls is None or result.boxes.conf is None:
        return result

    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    conf = result.boxes.conf.detach().cpu().numpy()
    keep = np.ones((len(cls),), dtype=bool)

    for i, (c, cf) in enumerate(zip(cls, conf)):
        if c in class_conf and cf < class_conf[c]:
            keep[i] = False

    # 마스크 적용
    mask = torch.from_numpy(keep).to(result.boxes.cls.device)
    result.boxes.cls = result.boxes.cls[mask]
    result.boxes.conf = result.boxes.conf[mask]
    result.boxes.xyxy = result.boxes.xyxy[mask]
    if hasattr(result.boxes, "id") and result.boxes.id is not None:
        result.boxes.id = result.boxes.id[mask]
    return result


def draw_results(frame, result, names: Optional[List[str]] = None):
    if result is None or result.boxes is None:
        return frame
    boxes = result.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    clss = boxes.cls.detach().cpu().numpy().astype(int)
    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
        cls_name = str(cls)
        if names and 0 <= cls < len(names):
            cls_name = names[cls]
        label = f"{cls_name} {conf:.2f}"
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 180, 255), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def crop_with_scale(xyxy: Tuple[int, int, int, int], scale: float, W: int, H: int, pad: int = 0) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1)
    h = (y2 - y1)
    s = max(w, h) * scale
    nx1 = int(max(0, cx - s / 2 - pad))
    ny1 = int(max(0, cy - s / 2 - pad))
    nx2 = int(min(W - 1, cx + s / 2 + pad))
    ny2 = int(min(H - 1, cy + s / 2 + pad))
    return nx1, ny1, nx2, ny2


# ---------- 명령들 ----------
def train(args):
    device = check_cuda(args.device)
    model = YOLO(args.model or DEFAULT_MODEL)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=RUNS_DIR,
        name=args.name,
        workers=args.workers,
        seed=args.seed,
        cache=args.cache,
        patience=args.patience,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        cos_lr=args.cos_lr,
        optimizer=args.optimizer,
        amp=args.amp,
        resume=args.resume,
        val=True
    )

    # ---- 추가된 학습 하이퍼 전달 ----
    train_kwargs.update(dict(
        save_period=args.save_period,
        label_smoothing=args.label_smoothing,
        mosaic=args.mosaic,
        copy_paste=args.copy_paste,
        close_mosaic=args.close_mosaic,
        fliplr=args.fliplr,
        hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
        translate=args.translate,
        degrees=args.degrees,
        perspective=args.perspective,
        scale=args.scale,
    ))

    # ---- 소형물체 프리셋 ----
    if args.smallobj:
        smallobj_overrides = dict(
            imgsz=1280,              # 입력 해상도 상향
            mosaic=0.15,             # 모자이크 약화
            mixup=0.0,               # mixup 비권장
            copy_paste=0.5,          # 작은 물체 복제
            close_mosaic=10,         # 후반부 모자이크 끄기
            degrees=0.0, shear=0.0, perspective=0.0,
            translate=0.05,          # 이동 축소
            scale=0.5,               # 단일 float
            hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
            fliplr=0.5,
            # 학습 안정화
            patience=50,             # 조기종료 여유
            lr0=0.0015               # 초기 LR 낮춤
        )
        train_kwargs.update(smallobj_overrides)

        # 사용자가 직접 imgsz 지정했다면 그 값 우선 적용
        if args.imgsz != 640:
            train_kwargs["imgsz"] = args.imgsz

    print(f"[INFO] Start training on device={device}")
    results = model.train(**train_kwargs)
    print("[INFO] Train done. Best weights:", model.trainer.best)
    return results


def validate(args):
    device = check_cuda(args.device)
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=device,
        batch=args.batch,
        project=RUNS_DIR,
        name=args.name or "val",
        split="val"
    )
    print("[INFO] Val metrics:", metrics)
    return metrics


def predict(args):
    device = check_cuda(args.device)
    model = YOLO(args.weights)

    # 클래스별 임계값 파싱
    names = getattr(model.model, "names", getattr(model, "names", None))
    if names is None:
        names = []
    class_thresh = parse_class_thresh(args.class_thresh, names)

    # 공통 예측 인자
    common = dict(
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=device,
        agnostic_nms=args.agnostic_nms,
        max_det=args.max_det,
        vid_stride=args.vid_stride,
        augment=args.tta,  # test-time augmentation
        verbose=False
    )

    # 스트림 여부 판별
    is_cam = str(args.source).isdigit() and len(str(args.source)) <= 2
    is_stream_url = str(args.source).startswith(("rtsp://", "rtmp://", "http://", "https://"))

    # ===== ROI 파이프라인 세팅 (선택) =====
    face_model = None
    if args.roi_face_weights:
        face_model = YOLO(args.roi_face_weights)

    if is_cam or is_stream_url:
        if cv2 is None:
            raise RuntimeError("OpenCV가 필요합니다. `pip install opencv-python` 후 다시 실행하세요.")
        cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
        if not cap.isOpened():
            raise RuntimeError(f"소스 열기 실패: {args.source}")

        print("[INFO] Streaming... 'q' 종료")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] 프레임 획득 실패")
                break
            H, W = frame.shape[:2]

            if face_model is None:
                # 단일 단계 추론
                res_list = model.predict(source=frame, **common)
                res = res_list[0]
                res = apply_classwise_threshold(res, class_thresh)
                out = draw_results(frame.copy(), res, names)
            else:
                # 2단계 ROI 추론: 얼굴 → ROI 확장 → 코/입 재추론
                faces = face_model.predict(source=frame, imgsz=args.roi_imgsz, conf=args.roi_conf, iou=args.roi_iou,
                                           device=device, agnostic_nms=args.agnostic_nms, max_det=args.roi_max_det,
                                           verbose=False)
                out = frame.copy()
                if len(faces) and faces[0].boxes is not None and len(faces[0].boxes) > 0:
                    for fbox in faces[0].boxes.xyxy.detach().cpu().numpy().astype(int):
                        x1, y1, x2, y2 = fbox
                        cx1, cy1, cx2, cy2 = crop_with_scale((x1, y1, x2, y2), args.roi_scale, W, H, pad=args.roi_pad)
                        crop = frame[cy1:cy2, cx1:cx2]
                        if crop.size == 0:
                            continue
                        res_c = model.predict(source=crop, imgsz=args.roi_imgsz or args.imgsz, conf=args.conf,
                                              iou=args.iou, device=device, agnostic_nms=args.agnostic_nms,
                                              max_det=args.max_det, augment=args.tta, verbose=False)[0]
                        res_c = apply_classwise_threshold(res_c, class_thresh)
                        # 좌표 복원
                        if res_c.boxes is not None and len(res_c.boxes) > 0:
                            rxy = res_c.boxes.xyxy.detach().cpu().numpy()
                            rxy[:, [0, 2]] += cx1
                            rxy[:, [1, 3]] += cy1
                            res_c.boxes.xyxy = torch.from_numpy(rxy).to(res_c.boxes.xyxy.device)
                            out = draw_results(out, res_c, names)
                else:
                    pass

            cv2.imshow("YOLO11 (predict)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # 파일/폴더/비디오 (오프라인)
        res_list = model.predict(source=args.source, save=args.save, save_txt=args.save_txt, save_crop=args.save_crop,
                                 project=RUNS_DIR, name=args.name or "predict", **common)
        if class_thresh:
            save_dir = None
            for r in res_list:
                r = apply_classwise_threshold(r, class_thresh)
                plotted = r.plot()
                if hasattr(r, "save_dir") and hasattr(r, "path"):
                    save_dir = r.save_dir
                    out_path = os.path.join(r.save_dir, os.path.basename(r.path))
                    cv2.imwrite(out_path, plotted)
            if save_dir:
                print(f"[INFO] Predict done with class-thresh. Outputs: {save_dir}")
        else:
            print("[INFO] Predict done. Outputs saved to runs/predict/*")
        return res_list


def export_(args):
    device = check_cuda(args.device)
    model = YOLO(args.weights)
    fmt = args.format.lower()
    out = model.export(
        format=fmt,
        imgsz=args.imgsz,
        device=device,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset if fmt == "onnx" else None,
        half=args.half
    )
    print("[INFO] Exported:", out)
    return out


# ---------- 파서 ----------
def build_parser():
    parser = argparse.ArgumentParser(description="YOLO11 trainer/predictor/exporter (small-object tuned)")

    # (선택) 메인 공통 인자 — 그대로 둬도 되고, 없어도 됩니다.
    parser.add_argument("--device", type=str, default=None, help="예: 0 또는 'cpu'")
    parser.add_argument("--imgsz", type=int, default=640, help="입력 해상도 (전역, 서브커맨드에도 중복 추가)")
    parser.add_argument("--name", type=str, default=None, help="runs 하위 이름 (전역, 서브커맨드에도 중복 추가)")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---------------- train ----------------
    p_train = sub.add_parser("train", help="학습")
    # 공통 인자도 서브커맨드에 다시 추가 (중복 OK, 서브가 우선 파싱)
    p_train.add_argument("--imgsz", type=int, default=640, help="입력 해상도")
    p_train.add_argument("--name", type=str, default=None, help="runs 하위 이름")

    p_train.add_argument("--model", type=str, default=DEFAULT_MODEL, help="베이스 가중치(.pt)")
    p_train.add_argument("--data", type=str, required=True, help="data.yaml 경로")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch", type=int, default=16)
    p_train.add_argument("--workers", type=int, default=8)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--cache", type=str, default="ram", help="ram/disk/True/False")
    p_train.add_argument("--patience", type=int, default=50, help="early stopping patience")
    p_train.add_argument("--lr0", type=float, default=0.01)
    p_train.add_argument("--lrf", type=float, default=0.01)
    p_train.add_argument("--weight_decay", type=float, default=0.0005)
    p_train.add_argument("--cos_lr", action="store_true")
    p_train.add_argument("--optimizer", type=str, default="auto")
    p_train.add_argument("--amp", action="store_true", help="AMP(fp16) 사용")
    p_train.add_argument("--resume", action="store_true", help="중단 위치에서 재개")

    # 추가 학습 하이퍼
    p_train.add_argument("--save-period", type=int, default=0)
    p_train.add_argument("--label-smoothing", type=float, default=0.0)
    p_train.add_argument("--mosaic", type=float, default=0.2)
    p_train.add_argument("--copy-paste", type=float, default=0.0)
    p_train.add_argument("--close-mosaic", type=int, default=15)
    p_train.add_argument("--fliplr", type=float, default=0.5)
    p_train.add_argument("--hsv-h", type=float, default=0.015)
    p_train.add_argument("--hsv-s", type=float, default=0.6)
    p_train.add_argument("--hsv-v", type=float, default=0.3)
    p_train.add_argument("--translate", type=float, default=0.08)
    p_train.add_argument("--degrees", type=float, default=0.0)
    p_train.add_argument("--perspective", type=float, default=0.0)
    p_train.add_argument("--scale", type=float, default=0.5, help="단일 float만 허용 (예: 0.5)")

    p_train.add_argument("--smallobj", action="store_true",
                         help="소형물체 친화 하이퍼를 일괄 적용(imgsz/augment/lr 등)")

    # ---------------- val ----------------
    p_val = sub.add_parser("val", help="검증")
    p_val.add_argument("--imgsz", type=int, default=640, help="입력 해상도")
    p_val.add_argument("--name", type=str, default=None, help="runs 하위 이름")
    p_val.add_argument("--weights", type=str, required=True)
    p_val.add_argument("--data", type=str, required=True)
    p_val.add_argument("--batch", type=int, default=16)

    # ---------------- predict ----------------
    p_pred = sub.add_parser("predict", help="추론")
    p_pred.add_argument("--imgsz", type=int, default=640, help="입력 해상도")
    p_pred.add_argument("--name", type=str, default=None, help="runs 하위 이름")
    p_pred.add_argument("--weights", type=str, required=True)
    p_pred.add_argument("--source", type=str, required=True, help="이미지/폴더/비디오/숫자(웹캠)/RTSP")
    p_pred.add_argument("--conf", type=float, default=0.25)
    p_pred.add_argument("--iou", type=float, default=0.45)
    p_pred.add_argument("--agnostic-nms", action="store_true")
    p_pred.add_argument("--max-det", type=int, default=300)
    p_pred.add_argument("--vid-stride", type=int, default=1, help="비디오 프레임 간격")
    p_pred.add_argument("--tta", action="store_true", help="test-time augmentation 켜기")
    p_pred.add_argument("--class-thresh", type=str, default=None,
                        help="클래스별 conf (예: 'mouth:0.18,nose:0.25,pacifier:0.16')")
    p_pred.add_argument("--save", action="store_true", help="결과 이미지/비디오 저장")
    p_pred.add_argument("--save_txt", action="store_true", help="YOLO txt 결과 저장")
    p_pred.add_argument("--save_crop", action="store_true", help="감지 영역 크롭 저장")

    # ROI (얼굴→코/입) 2단계 추론 옵션
    p_pred.add_argument("--roi-face-weights", type=str, default=None, help="얼굴 검출 가중치(.pt) 지정 시 2단계 ROI 추론")
    p_pred.add_argument("--roi-scale", type=float, default=1.2, help="얼굴 박스 확대 배수")
    p_pred.add_argument("--roi-pad", type=int, default=8, help="ROI 여백(px)")
    p_pred.add_argument("--roi-imgsz", type=int, default=640, help="ROI 재추론 입력 크기")
    p_pred.add_argument("--roi-conf", type=float, default=0.25, help="얼굴 검출 conf")
    p_pred.add_argument("--roi-iou", type=float, default=0.45, help="얼굴 검출 iou")
    p_pred.add_argument("--roi-max-det", type=int, default=50, help="얼굴 최대 감지 개수")

    # ---------------- export ----------------
    p_exp = sub.add_parser("export", help="내보내기")
    p_exp.add_argument("--imgsz", type=int, default=640, help="입력 해상도")
    p_exp.add_argument("--name", type=str, default=None, help="runs 하위 이름")
    p_exp.add_argument("--weights", type=str, required=True)
    p_exp.add_argument("--format", type=str, default="onnx", help="onnx/engine/torchscript/openvino/coreml 등")
    p_exp.add_argument("--dynamic", action="store_true")
    p_exp.add_argument("--simplify", action="store_true")
    p_exp.add_argument("--opset", type=int, default=12)
    p_exp.add_argument("--half", action="store_true")

    return parser




def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "val":
        validate(args)
    elif args.cmd == "predict":
        predict(args)
    elif args.cmd == "export":
        export_(args)
    else:
        raise ValueError(f"알 수 없는 명령: {args.cmd}")


if __name__ == "__main__":
    main()
