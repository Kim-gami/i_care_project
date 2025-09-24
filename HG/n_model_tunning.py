# train_night_nosemouth.py (YOLO11n, 소물체·야간 보수 세팅 + 고정 검증 + 스윕)
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO

# ==== 기본 ====
RUN_NAME_PREFIX = "night_smallobj"
DEFAULT_IMGSZ = 960
BASE_WEIGHTS = "yolo11n.pt"   # 필요시 yolov8s.pt 등으로 교체 가능

# ==== 유틸 ====
def ts_name() -> str:
    return f"{RUN_NAME_PREFIX}-{datetime.now().strftime('%y%m%d-%H%M%S')}"

def make_yaml(data_dir: Path) -> Path:
    data_dir = data_dir.resolve()
    yaml_path = data_dir / "night_face.yaml"
    cfg = {
        "path": str(data_dir),  # 절대경로
        "train": str((data_dir / "images/train").resolve()),
        "val":   str((data_dir / "images/val").resolve()),
        "names": {0: "nose_mouth"},
        "nc": 1,
    }
    yaml_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yaml_path

def scan_counts(data_dir: Path):
    p = data_dir
    exts = ("*.jpg","*.jpeg","*.png","*.bmp")
    def nimg(d): return sum(len(list((d).glob(e))) for e in exts)
    def nlab(d): return len(list(d.glob("*.txt")))
    return {
        "train_imgs": nimg(p/"images/train"),
        "val_imgs":   nimg(p/"images/val"),
        "train_labels": nlab(p/"labels/train"),
        "val_labels":   nlab(p/"labels/val"),
    }

def auto_device(dev: str | None) -> str:
    if dev is not None:
        return dev
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

# ==== 커맨드 ====
def cmd_train(args):
    os.environ["WANDB_MODE"] = "disabled"
    data_dir = Path(args.data_dir)

    # 필수 경로 체크
    req = [data_dir/"images/train", data_dir/"images/val", data_dir/"labels/train", data_dir/"labels/val"]
    miss = [str(r) for r in req if not r.exists()]
    if miss:
        raise SystemExit("[ERR] 경로 없음:\n- " + "\n- ".join(miss))

    stats = scan_counts(data_dir)
    if stats["train_imgs"] == 0 or stats["val_imgs"] == 0 or stats["train_labels"] == 0:
        raise SystemExit(f"[ERR] 데이터 부족/라벨 0\n{stats}")

    yaml_path = make_yaml(data_dir)
    device = auto_device(args.device)
    run_name = args.name or ts_name()

    model = YOLO(args.weights or BASE_WEIGHTS)
    print("[INFO] training start...")

    # 기본(보수) 세팅
    overrides = dict(
        data=str(yaml_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        project="runs/detect",
        name=run_name,
        exist_ok=True,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        save=True, save_period=1, plots=True, val=True, save_json=False,
        cos_lr=True,

        # 증강/도메인
        multi_scale=True,              # 다양한 스케일
        mosaic=0.3, close_mosaic=10,   # 강도↓, 막판 10epoch 끔
        mixup=0.0, copy_paste=0.0,
        hsv_h=0.0, hsv_s=0.1, hsv_v=0.6,
        fliplr=0.5, flipud=0.0,
        degrees=0.0, translate=0.05, scale=0.1, shear=0.0,

        # 손실 가중 (회귀 비중↑)
        box=8.0, cls=0.2, dfl=1.5,

        # 최적화
        lr0=0.005, lrf=0.1, weight_decay=0.0002,
        amp=True,
    )

    # 초-보수 프로파일(소물체/야간 특화) 토글
    if args.strict_smallobj:
        overrides.update({
            "multi_scale": False,
            "mosaic": 0.0, "close_mosaic": 0,
            "mixup": 0.0, "copy_paste": 0.0,
            "hsv_h": 0.0, "hsv_s": 0.05, "hsv_v": 0.3,
            "fliplr": 0.5, "flipud": 0.0,
            "degrees": 0.0, "translate": 0.02, "scale": 0.0, "shear": 0.0,
        })

    results = model.train(**overrides)
    save_dir = getattr(results, "save_dir", None)
    print(f"[OK] train done. save_dir={save_dir or f'runs/detect/{run_name}'}")
    print("[TIP] weights는 위 경로의 weights/{best.pt,last.pt}")

def cmd_predict(args):
    os.environ["WANDB_MODE"] = "disabled"
    device = auto_device(args.device)
    weights = args.weights

    if not weights:
        root = Path("runs/detect")
        cands = sorted([p for p in root.glob(f"{RUN_NAME_PREFIX}-*") if (p/"weights").exists()],
                       key=os.path.getmtime, reverse=True)
        if not cands:
            raise SystemExit("[ERR] --weights 미지정 & 최근 run 없음")
        best = cands[0]/"weights"/"best.pt"
        last = cands[0]/"weights"/"last.pt"
        weights = str(best if best.exists() else last)
    else:
        if not Path(weights).exists():
            raise SystemExit(f"[ERR] 가중치 없음: {weights}")

    model = YOLO(weights)
    print(f"[INFO] predict: weights={weights}")
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
        save=True,
        name=f"{RUN_NAME_PREFIX}_pred",
        iou=0.5, max_det=300,
        project="runs/detect",
        augment=not args.no_tta,
        exist_ok=True,
    )
    print(f"[OK] results -> runs/detect/{RUN_NAME_PREFIX}_pred/")

def cmd_validate(args):
    os.environ["WANDB_MODE"] = "disabled"

    if args.data_dir:
        yaml_path = make_yaml(Path(args.data_dir))
    elif args.data_yaml:
        yaml_path = args.data_yaml
    else:
        raise SystemExit("[ERR] --data-dir 또는 --data-yaml 중 하나 필요")

    device = auto_device(args.device)
    if not Path(args.weights).exists():
        raise SystemExit(f"[ERR] weights 없음: {args.weights}")

    model = YOLO(args.weights)
    metrics = model.val(
        data=str(yaml_path),
        imgsz=args.imgsz,
        device=device,
        plots=True,
        conf=args.conf,     # 고정 threshold
        iou=args.iou,       # 고정 IoU
        rect=True           # 종횡비 유지 검증
    )
    rd = getattr(metrics, "results_dict", {})
    summary = {
        "precision": float(rd.get("metrics/precision(B)", 0.0)),
        "recall": float(rd.get("metrics/recall(B)", 0.0)),
        "mAP50": float(rd.get("metrics/mAP50(B)", 0.0)),
        "mAP50-95": float(rd.get("metrics/mAP50-95(B)", 0.0)),
    }
    print("[VAL]", json.dumps(summary, ensure_ascii=False, indent=2))

def cmd_validate_sweep(args):
    import csv
    os.environ["WANDB_MODE"] = "disabled"

    if args.data_dir:
        yaml_path = make_yaml(Path(args.data_dir))
    elif args.data_yaml:
        yaml_path = args.data_yaml
    else:
        raise SystemExit("[ERR] --data-dir 또는 --data-yaml 중 하나 필요")

    device = auto_device(args.device)
    if not Path(args.weights).exists():
        raise SystemExit(f"[ERR] weights 없음: {args.weights}")

    model = YOLO(args.weights)
    conf_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    rows = []
    for c in conf_list:
        m = model.val(data=str(yaml_path), imgsz=args.imgsz, device=device,
                      plots=False, conf=c, iou=args.iou, rect=True)
        rd = getattr(m, "results_dict", {})
        rows.append({
            "conf": c,
            "precision": float(rd.get("metrics/precision(B)", 0.0)),
            "recall": float(rd.get("metrics/recall(B)", 0.0)),
            "mAP50": float(rd.get("metrics/mAP50(B)", 0.0)),
            "mAP50-95": float(rd.get("metrics/mAP50-95(B)", 0.0)),
        })
    print("\nCONF |  PREC   | RECALL  |  mAP50  | mAP50-95")
    for r in rows:
        print(f"{r['conf']:.2f}  | {r['precision']:.4f} | {r['recall']:.4f} | {r['mAP50']:.4f} | {r['mAP50-95']:.4f}")
    if args.out_csv:
        out = Path(args.out_csv)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["conf","precision","recall","mAP50","mAP50-95"])
            w.writeheader(); w.writerows(rows)
        print(f"[OK] saved sweep csv -> {out}")

# ==== 엔트리 ====
def main():
    ap = argparse.ArgumentParser("Night nose+mouth detector trainer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # train
    tr = sub.add_parser("train")
    tr.add_argument("--data-dir", required=True, help="데이터 루트( images/, labels/ 포함 폴더 )")
    tr.add_argument("--epochs", type=int, default=100)
    tr.add_argument("--batch", type=int, default=16)
    tr.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    tr.add_argument("--device", default=None)   # 'cpu' or '0'
    tr.add_argument("--patience", type=int, default=30)
    tr.add_argument("--workers", type=int, default=2)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--weights", default=None)
    tr.add_argument("--name", default=None)
    tr.add_argument("--strict-smallobj", action="store_true",
                    help="소물체·야간 보수 프로파일(증강 최소화, multi_scale OFF)")
    tr.set_defaults(func=cmd_train)

    # predict
    pr = sub.add_parser("predict")
    pr.add_argument("--weights", default=None, help="미지정 시 최근 run 자동 선택")
    pr.add_argument("--source", required=True, help="파일/폴더/비디오/웹캠(0)")
    pr.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    pr.add_argument("--conf", type=float, default=0.2)
    pr.add_argument("--device", default=None)
    pr.add_argument("--no-tta", action="store_true")
    pr.set_defaults(func=cmd_predict)

    # validate
    va = sub.add_parser("validate")
    va.add_argument("--weights", required=True)
    va.add_argument("--data-dir", default=None)
    va.add_argument("--data-yaml", default=None)
    va.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    va.add_argument("--device", default=None)
    va.add_argument("--conf", type=float, default=0.25)
    va.add_argument("--iou", type=float, default=0.5)
    va.set_defaults(func=cmd_validate)

    # validate-sweep
    sw = sub.add_parser("validate-sweep")
    sw.add_argument("--weights", required=True)
    sw.add_argument("--data-dir", default=None)
    sw.add_argument("--data-yaml", default=None)
    sw.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    sw.add_argument("--device", default=None)
    sw.add_argument("--iou", type=float, default=0.5)
    sw.add_argument("--out-csv", default=None)
    sw.set_defaults(func=cmd_validate_sweep)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
