# n_model_tunning.py
import argparse, os
from typing import Dict, Tuple, Union, List, Optional

import torch
from ultralytics import YOLO

try:
    import cv2, numpy as np
except Exception:
    cv2 = None; np = None

DEFAULT_MODEL = "yolo11s.pt"
RUNS_DIR = os.environ.get("YOLO_RUNS_DIR", "runs")

def check_cuda(device_arg: Union[str, int, None]) -> str:
    if device_arg is None: return "0" if torch.cuda.is_available() else "cpu"
    if isinstance(device_arg, int): return str(device_arg)
    s = str(device_arg).strip()
    return "cpu" if s.lower() == "cpu" else s

def parse_class_thresh(arg: Optional[str], names: List[str] | Dict[int, str]) -> Dict[int, float]:
    m: Dict[int, float] = {}
    if not arg: return m
    name2id: Dict[str, int] = {}
    if isinstance(names, dict):
        for i, n in names.items(): name2id[str(n)] = int(i)
    else:
        for i, n in enumerate(names): name2id[str(n)] = i
    for p in [x.strip() for x in arg.split(",") if x.strip()]:
        if ":" not in p: continue
        k, v = p.split(":", 1)
        try: confv = float(v)
        except ValueError: continue
        k = k.strip()
        if k.isdigit(): m[int(k)] = confv
        elif k in name2id: m[name2id[k]] = confv
    return m

def apply_classwise_threshold(result, class_conf: Dict[int, float]):
    if not class_conf or result.boxes is None: return result
    if result.boxes.cls is None or result.boxes.conf is None: return result
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    conf = result.boxes.conf.detach().cpu().numpy()
    keep = np.ones((len(cls),), dtype=bool)
    for i, (c, cf) in enumerate(zip(cls, conf)):
        if c in class_conf and cf < class_conf[c]: keep[i] = False
    mask = torch.from_numpy(keep).to(result.boxes.cls.device)
    result.boxes.cls = result.boxes.cls[mask]
    result.boxes.conf = result.boxes.conf[mask]
    result.boxes.xyxy = result.boxes.xyxy[mask]
    if hasattr(result.boxes, "id") and result.boxes.id is not None:
        result.boxes.id = result.boxes.id[mask]
    return result

def draw_results(frame, result, names: Optional[List[str]] = None):
    if result is None or result.boxes is None: return frame
    boxes = result.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    clss = boxes.cls.detach().cpu().numpy().astype(int)
    for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
        cls_name = str(cls)
        if names and 0 <= cls < len(names): cls_name = names[cls]
        label = f"{cls_name} {conf:.2f}"
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 180, 255), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return frame

def crop_with_scale(xyxy: Tuple[int, int, int, int], scale: float, W: int, H: int, pad: int = 0) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1) * scale
    nx1 = int(max(0, cx - s / 2 - pad)); ny1 = int(max(0, cy - s / 2 - pad))
    nx2 = int(min(W - 1, cx + s / 2 + pad)); ny2 = int(min(H - 1, cy + s / 2 + pad))
    return nx1, ny1, nx2, ny2

def train(args):
    device = check_cuda(args.device)
    model = YOLO(args.model or DEFAULT_MODEL)
    train_kwargs = dict(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
        project=RUNS_DIR, name=args.name, workers=args.workers, seed=args.seed, cache=args.cache,
        patience=args.patience, lr0=args.lr0, lrf=args.lrf, weight_decay=args.weight_decay,
        cos_lr=args.cos_lr, optimizer=args.optimizer, amp=args.amp, resume=args.resume, val=True
    )
    train_kwargs.update(dict(
        save_period=args.save_period, label_smoothing=args.label_smoothing, mosaic=args.mosaic,
        copy_paste=args.copy_paste, close_mosaic=args.close_mosaic, fliplr=args.fliplr,
        hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v, translate=args.translate,
        degrees=args.degrees, perspective=args.perspective, scale=args.scale,
    ))
    if args.smallobj:
        smallobj_overrides = dict(
            imgsz=1280, mosaic=0.15, mixup=0.0, copy_paste=0.5, close_mosaic=10,
            degrees=0.0, shear=0.0, perspective=0.0, translate=0.05, scale=0.5,
            hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, fliplr=0.5, patience=50, lr0=0.0015
        )
        train_kwargs.update(smallobj_overrides)
        if args.imgsz != 640: train_kwargs["imgsz"] = args.imgsz
    print(f"[INFO] Start training on device={device}")
    results = model.train(**train_kwargs)
    print("[INFO] Train done. Best weights:", model.trainer.best)
    return results

def validate(args):
    device = check_cuda(args.device)
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data, imgsz=args.imgsz, device=device, batch=args.batch,
        project=RUNS_DIR, name=args.name or "val", split="val"
    )
    print("[INFO] Val metrics:", metrics); return metrics

def predict(args):
    device = check_cuda(args.device)
    model = YOLO(args.weights)
    names = getattr(model.model, "names", getattr(model, "names", []))
    class_thresh = parse_class_thresh(args.class_thresh, names)
    common = dict(
        imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=device,
        agnostic_nms=args.agnostic_nms, max_det=args.max_det,
        vid_stride=args.vid_stride, augment=args.tta, verbose=False
    )
    is_cam = str(args.source).isdigit() and len(str(args.source)) <= 2
    is_stream = str(args.source).startswith(("rtsp://", "rtmp://", "http://", "https://"))
    face_model = YOLO(args.roi_face_weights) if args.roi_face_weights else None

    if is_cam or is_stream:
        if cv2 is None: raise RuntimeError("OpenCV 필요: pip install opencv-python")
        cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
        if not cap.isOpened(): raise RuntimeError(f"소스 열기 실패: {args.source}")
        print("[INFO] Streaming... 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret: print("[WARN] 프레임 획득 실패"); break
            H, W = frame.shape[:2]
            if face_model is None:
                r = model.predict(source=frame, **common)[0]
                r = apply_classwise_threshold(r, class_thresh)
                out = draw_results(frame.copy(), r, names)
            else:
                faces = face_model.predict(
                    source=frame, imgsz=args.roi_imgsz, conf=args.roi_conf, iou=args.roi_iou,
                    device=device, agnostic_nms=args.agnostic_nms, max_det=args.roi_max_det, verbose=False
                )
                out = frame.copy()
                if len(faces) and faces[0].boxes is not None and len(faces[0].boxes) > 0:
                    for fbox in faces[0].boxes.xyxy.detach().cpu().numpy().astype(int):
                        x1, y1, x2, y2 = fbox
                        cx1, cy1, cx2, cy2 = crop_with_scale((x1, y1, x2, y2), args.roi_scale, W, H, pad=args.roi_pad)
                        crop = frame[cy1:cy2, cx1:cx2]
                        if crop.size == 0: continue
                        rc = model.predict(
                            source=crop, imgsz=args.roi_imgsz or args.imgsz, conf=args.conf, iou=args.iou,
                            device=device, agnostic_nms=args.agnostic_nms, max_det=args.max_det,
                            augment=args.tta, verbose=False
                        )[0]
                        rc = apply_classwise_threshold(rc, class_thresh)
                        if rc.boxes is not None and len(rc.boxes) > 0:
                            rxy = rc.boxes.xyxy.detach().cpu().numpy()
                            rxy[:, [0, 2]] += cx1; rxy[:, [1, 3]] += cy1
                            rc.boxes.xyxy = torch.from_numpy(rxy).to(rc.boxes.xyxy.device)
                            out = draw_results(out, rc, names)
            cv2.imshow("YOLO11 (predict)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        cap.release(); cv2.destroyAllWindows()
    else:
        res_list = model.predict(
            source=args.source, save=args.save, save_txt=args.save_txt, save_crop=args.save_crop,
            project=RUNS_DIR, name=args.name or "predict", **common
        )
        if class_thresh:
            save_dir = None
            for r in res_list:
                r = apply_classwise_threshold(r, class_thresh)
                plotted = r.plot()
                if hasattr(r, "save_dir") and hasattr(r, "path"):
                    save_dir = r.save_dir
                    out_path = os.path.join(r.save_dir, os.path.basename(r.path))
                    cv2.imwrite(out_path, plotted)
            if save_dir: print(f"[INFO] Outputs: {save_dir}")
        else:
            print("[INFO] Predict done.")
        return res_list

def export_(args):
    device = check_cuda(args.device)
    model = YOLO(args.weights)
    fmt = args.format.lower()
    out = model.export(
        format=fmt, imgsz=args.imgsz, device=device, dynamic=args.dynamic,
        simplify=args.simplify, opset=args.opset if fmt == "onnx" else None, half=args.half
    )
    print("[INFO] Exported:", out); return out

def build_parser():
    p = argparse.ArgumentParser(description="YOLO11 trainer/predictor/exporter (small-object tuned)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--name", type=str, default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="학습")
    t.add_argument("--imgsz", type=int, default=640); t.add_argument("--name", type=str, default=None)
    t.add_argument("--model", type=str, default=DEFAULT_MODEL)
    t.add_argument("--data", type=str, required=True)
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--cache", type=str, default="ram")
    t.add_argument("--patience", type=int, default=50)
    t.add_argument("--lr0", type=float, default=0.01)
    t.add_argument("--lrf", type=float, default=0.01)
    t.add_argument("--weight_decay", type=float, default=0.0005)
    t.add_argument("--cos_lr", action="store_true")
    t.add_argument("--optimizer", type=str, default="auto")
    t.add_argument("--amp", action="store_true")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--save-period", type=int, default=0)
    t.add_argument("--label-smoothing", type=float, default=0.0)
    t.add_argument("--mosaic", type=float, default=0.2)
    t.add_argument("--copy-paste", type=float, default=0.0)
    t.add_argument("--close-mosaic", type=int, default=15)
    t.add_argument("--fliplr", type=float, default=0.5)
    t.add_argument("--hsv-h", type=float, default=0.015)
    t.add_argument("--hsv-s", type=float, default=0.6)
    t.add_argument("--hsv-v", type=float, default=0.3)
    t.add_argument("--translate", type=float, default=0.08)
    t.add_argument("--degrees", type=float, default=0.0)
    t.add_argument("--perspective", type=float, default=0.0)
    t.add_argument("--scale", type=float, default=0.5)
    t.add_argument("--smallobj", action="store_true")

    v = sub.add_parser("val", help="검증")
    v.add_argument("--imgsz", type=int, default=640); v.add_argument("--name", type=str, default=None)
    v.add_argument("--weights", type=str, required=True)
    v.add_argument("--data", type=str, required=True)
    v.add_argument("--batch", type=int, default=16)

    pr = sub.add_parser("predict", help="추론")
    pr.add_argument("--imgsz", type=int, default=640); pr.add_argument("--name", type=str, default=None)
    pr.add_argument("--weights", type=str, required=True)
    pr.add_argument("--source", type=str, required=True)
    pr.add_argument("--conf", type=float, default=0.25)
    pr.add_argument("--iou", type=float, default=0.45)
    pr.add_argument("--agnostic-nms", action="store_true")
    pr.add_argument("--max-det", type=int, default=300)
    pr.add_argument("--vid-stride", type=int, default=1)
    pr.add_argument("--tta", action="store_true")
    pr.add_argument("--class-thresh", type=str, default=None)
    pr.add_argument("--save", action="store_true")
    pr.add_argument("--save_txt", action="store_true")
    pr.add_argument("--save_crop", action="store_true")
    pr.add_argument("--roi-face-weights", type=str, default=None)
    pr.add_argument("--roi-scale", type=float, default=1.2)
    pr.add_argument("--roi-pad", type=int, default=8)
    pr.add_argument("--roi-imgsz", type=int, default=640)
    pr.add_argument("--roi-conf", type=float, default=0.25)
    pr.add_argument("--roi-iou", type=float, default=0.45)
    pr.add_argument("--roi-max-det", type=int, default=50)

    ex = sub.add_parser("export", help="내보내기")
    ex.add_argument("--imgsz", type=int, default=640); ex.add_argument("--name", type=str, default=None)
    ex.add_argument("--weights", type=str, required=True)
    ex.add_argument("--format", type=str, default="onnx")
    ex.add_argument("--dynamic", action="store_true")
    ex.add_argument("--simplify", action="store_true")
    ex.add_argument("--opset", type=int, default=12)
    ex.add_argument("--half", action="store_true")
    return p

def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    args = build_parser().parse_args()
    if args.cmd == "train": train(args)
    elif args.cmd == "val": validate(args)
    elif args.cmd == "predict": predict(args)
    elif args.cmd == "export": export_(args)
    else: raise ValueError(f"Unknown cmd: {args.cmd}")

if __name__ == "__main__":
    main()
