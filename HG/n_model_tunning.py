# train_night_face.py
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO


# ===== 기본 설정 =====
DATA_ROOT = Path("datasets/night_face")
YAML_PATH = DATA_ROOT / "night_face.yaml"
RUN_NAME_PREFIX = "night_smallobj"
BASE_WEIGHTS = "yolov8n.pt"         # 필요시 외부 가중치 경로로 바꿔도 됨
CLASSES = ["nose_mouth"]            # 단일 클래스
DEFAULT_IMGSZ = 960                 # 작은 객체 → 해상도 ↑
DEFAULT_DEVICE = None               # None이면 자동 감지(없으면 cpu)


# ===== 유틸 =====
def _natural_int(v: str) -> int:
    v = int(v)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return v


def now_run_name() -> str:
    ts = datetime.now().strftime("%y%m%d-%H%M%S")
    return f"{RUN_NAME_PREFIX}-{ts}"


def write_yaml():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    content = {
        "path": str(DATA_ROOT),
        "train": "images/train",
        "val": "images/val",
        "names": {0: CLASSES[0]},
        "nc": 1,
    }
    YAML_PATH.write_text(yaml.safe_dump(content, allow_unicode=True, sort_keys=False))
    print(f"[OK] wrote {YAML_PATH}")


def count_files():
    def nimg(p: Path) -> int:
        return sum(len(list((p).glob(ext))) for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"))
    def nlab(p: Path) -> int:
        return len(list(p.glob("*.txt")))

    tr_img = nimg(DATA_ROOT / "images/train")
    va_img = nimg(DATA_ROOT / "images/val")
    tr_lab = nlab(DATA_ROOT / "labels/train")
    va_lab = nlab(DATA_ROOT / "labels/val")
    stats = {"train_imgs": tr_img, "val_imgs": va_img, "train_labels": tr_lab, "val_labels": va_lab}
    print("[DATA] ", stats)
    return stats


def quick_check(strict_labels=True):
    req = [
        DATA_ROOT / "images/train",
        DATA_ROOT / "images/val",
        DATA_ROOT / "labels/train",
        DATA_ROOT / "labels/val",
    ]
    miss = [str(p) for p in req if not p.exists()]
    if miss:
        raise SystemExit("[ERR] 아래 경로가 없습니다:\n- " + "\n- ".join(miss))

    stats = count_files()
    if strict_labels and stats["train_labels"] == 0:
        raise SystemExit("[ERR] labels/train/*.txt 라벨이 1개도 없습니다. 최소 1개 이상 필요합니다.")
    if stats["train_imgs"] == 0 or stats["val_imgs"] == 0:
        raise SystemExit("[ERR] 이미지가 부족합니다. train/val 이미지가 모두 1장 이상 필요합니다.")


def auto_device(dev_arg: str | None):
    if dev_arg is not None:
        return dev_arg
    # CUDA가 없으면 자동으로 cpu
    try:
        import torch
        if torch.cuda.is_available():
            return "0"  # 첫 GPU
    except Exception:
        pass
    return "cpu"


# ===== 서브커맨드 =====
def cmd_prepare(_args):
    write_yaml()
    quick_check(strict_labels=False)
    print("[OK] prepare done.")


def cmd_train(args):
    write_yaml()
    quick_check(strict_labels=True)

    device = auto_device(args.device)
    run_name = args.name or now_run_name()

    # 베이스 가중치 선택: --weights가 주어지면 우선 사용
    model_path = args.weights or BASE_WEIGHTS
    model = YOLO(model_path)

    print("[INFO] training...")
    overrides = dict(
        data=str(YAML_PATH),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        project="runs/detect",
        name=run_name,
        exist_ok=True,        # 동일 이름 허용
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        # 저장/로그 강화
        save=True,
        save_period=1,        # ✅ 매 에폭 저장 → 최소 last는 항상 생성
        plots=True,           # 학습 곡선 이미지 저장
        val=True,
        save_json=False,
        cos_lr=True,
        # 작은 객체 친화 설정
        multi_scale=True,
        mosaic=1.0,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,
        hsv_v=0.6,
        fliplr=0.5,
        degrees=5.0, translate=0.05, scale=0.5,
        # 손실 가중치(단일 클래스)
        box=8.0, cls=0.2, dfl=1.5,
        amp=False,            # 환경 호환
    )
    results = model.train(**overrides)

    # 저장 경로 안내
    save_dir = getattr(results, "save_dir", None)
    print(f"[OK] train done. save_dir={save_dir or f'runs/detect/{run_name}'}")
    print(f"[TIP] best.pt/last.pt는 위 경로의 weights/ 폴더에 저장됩니다.")


def cmd_resume(args):
    # Ultralytics는 run_dir 지정으로 resume 가능
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"[ERR] run_dir가 없습니다: {run_dir}")
    device = auto_device(args.device)
    model = YOLO(run_dir / "weights" / "last.pt")
    print("[INFO] resuming...")
    results = model.train(resume=True, device=device)
    print("[OK] resume done.", getattr(results, "save_dir", run_dir))


def _resolve_weights(args) -> Path:
    # 명시한 가중치 우선, 아니면 최근 run의 best→last 순으로 탐색
    if args.weights:
        w = Path(args.weights)
        if not w.exists():
            raise SystemExit(f"[ERR] 지정한 가중치가 없음: {w}")
        return w
    # 최근 생성된 RUN_NAME_PREFIX 디렉터리 탐색
    root = Path("runs/detect")
    cand = sorted([p for p in root.glob(f"{RUN_NAME_PREFIX}-*") if (p / "weights").exists()], key=os.path.getmtime, reverse=True)
    if not cand:
        raise SystemExit("[ERR] 가중치를 찾지 못했습니다. --weights로 명시해 주세요.")
    best = cand[0] / "weights" / "best.pt"
    last = cand[0] / "weights" / "last.pt"
    return best if best.exists() else last


def cmd_predict(args):
    device = auto_device(args.device)
    ckpt = _resolve_weights(args)
    model = YOLO(str(ckpt))
    print(f"[INFO] predict: src={args.source}, weights={ckpt}, device={device}")
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        save=True,
        name=f"{RUN_NAME_PREFIX}_pred",
        augment=(not args.no_tta),  # TTA 유사
        iou=0.5,
        max_det=300,
        project="runs/detect",
        exist_ok=True,
    )
    print(f"[OK] results -> runs/detect/{RUN_NAME_PREFIX}_pred/")


def cmd_validate(args):
    device = auto_device(args.device)
    ckpt = _resolve_weights(args)
    model = YOLO(str(ckpt))
    print(f"[INFO] validate: weights={ckpt}")
    metrics = model.val(
        data=str(YAML_PATH),
        imgsz=args.imgsz,
        device=device,
        plots=True,
    )
    # 간단 요약 출력
    summary = {
        "precision": float(getattr(metrics, "box", {}).get("mp", getattr(metrics, "results_dict", {}).get("metrics/precision(B)", 0.0))),
        "recall": float(getattr(metrics, "box", {}).get("mr", getattr(metrics, "results_dict", {}).get("metrics/recall(B)", 0.0))),
        "mAP50": float(getattr(metrics, "box", {}).get("map50", getattr(metrics, "results_dict", {}).get("metrics/mAP50(B)", 0.0))),
        "mAP50-95": float(getattr(metrics, "box", {}).get("map", getattr(metrics, "results_dict", {}).get("metrics/mAP50-95(B)", 0.0))),
    }
    print("[VAL]", json.dumps(summary, ensure_ascii=False, indent=2))
    print("[OK] validation done.")


def cmd_export(args):
    ckpt = _resolve_weights(args)
    model = YOLO(str(ckpt))
    fmt = args.format  # 'onnx', 'engine', 'openvino', 'torchscript' 등
    print(f"[INFO] export: weights={ckpt}, format={fmt}")
    model.export(format=fmt, dynamic=args.dynamic, half=args.half, imgsz=args.imgsz)
    print("[OK] export done.")


# ===== 메인 =====
def main():
    ap = argparse.ArgumentParser(description="Robust trainer for night nose+mouth (small objects)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # prepare
    ap_prep = sub.add_parser("prepare", help="YAML 생성 및 경로 점검")
    ap_prep.set_defaults(func=cmd_prepare)

    # train
    ap_tr = sub.add_parser("train", help="학습")
    ap_tr.add_argument("--epochs", type=_natural_int, default=100)
    ap_tr.add_argument("--batch", type=_natural_int, default=8)
    ap_tr.add_argument("--imgsz", type=_natural_int, default=DEFAULT_IMGSZ)
    ap_tr.add_argument("--device", default=DEFAULT_DEVICE)        # 'cpu' 또는 '0'
    ap_tr.add_argument("--patience", type=int, default=20)
    ap_tr.add_argument("--workers", type=int, default=2)
    ap_tr.add_argument("--seed", type=int, default=0)
    ap_tr.add_argument("--weights", default=None, help="시작 가중치(.pt 등)")
    ap_tr.add_argument("--name", default=None, help="실행 이름(미지정 시 자동)")
    ap_tr.set_defaults(func=cmd_train)

    # resume
    ap_rs = sub.add_parser("resume", help="이전 run 재개")
    ap_rs.add_argument("--run-dir", required=True, help="예: runs/detect/night_smallobj-YYMMDD-HHMMSS")
    ap_rs.add_argument("--device", default=DEFAULT_DEVICE)
    ap_rs.set_defaults(func=cmd_resume)

    # predict
    ap_pr = sub.add_parser("predict", help="추론")
    ap_pr.add_argument("--source", default=str(DATA_ROOT / "images/val"))
    ap_pr.add_argument("--conf", type=float, default=0.2)
    ap_pr.add_argument("--imgsz", type=_natural_int, default=DEFAULT_IMGSZ)
    ap_pr.add_argument("--device", default=DEFAULT_DEVICE)
    ap_pr.add_argument("--no-tta", action="store_true")
    ap_pr.add_argument("--weights", default=None, help="사용할 가중치 경로(미지정 시 최신 run의 best/last 자동 탐색)")
    ap_pr.set_defaults(func=cmd_predict)

    # validate
    ap_va = sub.add_parser("validate", help="검증 및 지표 산출")
    ap_va.add_argument("--imgsz", type=_natural_int, default=DEFAULT_IMGSZ)
    ap_va.add_argument("--device", default=DEFAULT_DEVICE)
    ap_va.add_argument("--weights", default=None)
    ap_va.set_defaults(func=cmd_validate)

    # export
    ap_ex = sub.add_parser("export", help="가중치 내보내기")
    ap_ex.add_argument("--format", default="onnx", help="onnx/engine/openvino/torchscript 등")
    ap_ex.add_argument("--imgsz", type=_natural_int, default=DEFAULT_IMGSZ)
    ap_ex.add_argument("--dynamic", action="store_true")
    ap_ex.add_argument("--half", action="store_true")
    ap_ex.add_argument("--weights", default=None)
    ap_ex.set_defaults(func=cmd_export)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
