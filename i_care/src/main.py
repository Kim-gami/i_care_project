#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, argparse, asyncio, threading
import numpy as np
import cv2
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
from telegram import Bot
from dotenv import load_dotenv
load_dotenv("/home/nvidia/Desktop/i_care/.env")

# 카메라
RTSP_URL = os.getenv("RTSP_URL")
USER = os.getenv("RTSP_USER")
PASSWORD = os.getenv("PASSWORD")

# 엔진
NIGHT_ENGINE = os.path.expanduser(os.getenv("NIGHT_ENGINE"))
DAY_ENGINE   = os.path.expanduser(os.getenv("DAY_ENGINE"))

# 텔레그램
TOKEN   = os.getenv("TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID"))

# 로그 전송, 엔진 파서
def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

async def tg_send(text: str):
    try:
        await Bot(token=TOKEN).send_message(chat_id=CHAT_ID, text=text)
    except Exception as e:
        log(f"텔레그램 전송 실패: {e}")

def parse_profile_from_weights(path: str, default="night"):
    p = (path or "").lower()
    if "day" in p: return "day"
    if "night" in p: return "night"
    return default

# 상태
class Telegram:
    def __init__(self, init_profile="night"):
        self.desired_profile = init_profile
        self.last_cmd = ""
        self.lock = threading.Lock()

    def set_profile(self, profile: str, src=""):
        with self.lock:
            self.desired_profile = profile
            self.last_cmd = f"{src}->{profile}@{time.strftime('%H:%M:%S')}"

def start_telegram_poller(state: Telegram):
    from telegram.ext import Application, CommandHandler, MessageHandler, filters

    async def cmd_day(update, context):
        state.set_profile("day", "/day")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="☀️ DAY 모드로 전환")

    async def cmd_night(update, context):
        state.set_profile("night", "/night")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="🌙 NIGHT 모드로 전환")

    async def cmd_status(update, context):
        with state.lock:
            prof = state.desired_profile
            last = state.last_cmd
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"ℹ️ profile={prof}, last_cmd={last}")

    async def on_text(update, context):
        txt = (update.message.text or "").lower()
        if "day" in txt:   await cmd_day(update, context)
        elif "night" in txt: await cmd_night(update, context)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="명령어: /day /night /status")

    async def run():
        app = Application.builder().token(TOKEN).build()
        app.add_handler(CommandHandler("day", cmd_day))
        app.add_handler(CommandHandler("night", cmd_night))
        app.add_handler(CommandHandler("status", cmd_status))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
        try:
            await app.initialize()
            await app.start()
            await app.updater.start_polling(drop_pending_updates=True, poll_interval=1.0, timeout=10)
            while True:
                await asyncio.sleep(3600)
        except Exception as e:
            log(f"텔레그램 에러 : {e}")
        finally:
            try:
                await app.updater.stop(); await app.stop(); await app.shutdown()
            except Exception:
                pass

    def thread_target():
        asyncio.run(run())

    th = threading.Thread(target=thread_target, daemon=True)
    th.start()
    return th

# 감시 및 알림 트리거
class NightWatcher:
    def __init__(self, absence_sec=10.0, cooldown_sec=30.0):
        self.nose = "nose"
        self.mouth = "mouth"
        self.paci = "pacifier"
        self.abs = float(absence_sec)
        self.cool = float(cooldown_sec)
        now = time.monotonic()
        self.last_seen = {"nose": now, "mouth": now, "paci": now}
        self.last_fired = 0.0

    def update(self, names):
        now = time.monotonic()
        if self.nose in names:
            self.last_seen["nose"] = now
        if self.mouth in names:
            self.last_seen["mouth"] = now
        if self.paci in names:
            self.last_seen["paci"] = now

    def should_fire(self):
        now = time.monotonic()
        na = (now - self.last_seen["nose"]) >= self.abs
        ma = (now - self.last_seen["mouth"]) >= self.abs
        pa = (now - self.last_seen["paci"]) >= self.abs
        if na and (ma or pa) and (now - self.last_fired >= self.cool):
            self.last_fired = now
            return True, "아이의 코와 입 감지 실패 확인 요망"
        return False, ""

class DayWatcher:
    def __init__(self,
                 hand={"hand"},
                 others={"bottle_cap", "chopstick", "coin", "fork", "ring"},
                 ratio=0.08, hold=1.0, cooldown=10.0):
        self.hand = set(hand)
        self.others = set(others)
        self.r = float(ratio)
        self.h = float(hold)
        self.cool = float(cooldown)
        self.since = None
        self.last_fired = 0.0

    @staticmethod
    def _edge(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        dx = max(0.0, max(bx1 - ax2, ax1 - bx2))
        dy = max(0.0, max(by1 - ay2, ay1 - by2))
        return (dx * dx + dy * dy) ** 0.5

    def update_and_check(self, dets, W, H):
        diag = (W * W + H * H) ** 0.5
        th = self.r * diag

        hands = [d for d in dets if d["name"] in self.hand]
        others = [d for d in dets if d["name"] in self.others]
        now = time.monotonic()

        if not hands or not others:
            self.since = None
            return False, ""

        min_d = 1e9
        for h in hands:
            for o in others:
                d = self._edge(h["xyxy"], o["xyxy"])
                if d < min_d:
                    min_d = d

        if min_d <= th:
            if self.since is None:
                self.since = now
            elapsed = now - self.since

            if elapsed >= self.h and (now - self.last_fired >= self.cool):
                self.last_fired = now
                self.since = now
                return True, "아이의 주변에 위험물체가 있습니다."
        else:
            self.since = None

        return False, ""

#영상 GStreamer
from gi.repository import GstVideo

def make_pipeline(latency=100, qsize=4, max_appsink=2, out_w=640, out_h=360):
    return (
        f"rtspsrc location={RTSP_URL} user-id={USER} user-pw={PASSWORD} "
        f"latency={latency} protocols=tcp do-rtsp-keep-alive=true ! "
        f"rtph264depay ! h264parse ! queue leaky=2 max-size-buffers={qsize} ! "
        f"nvv4l2decoder ! queue leaky=2 max-size-buffers={qsize} ! "
        f"nvvidconv ! video/x-raw,format=RGBA,width={out_w},height={out_h} ! "
        f"appsink name=mysink max-buffers={max_appsink} drop=true sync=false"
    )


def open_pipeline():
    try:
        p = Gst.parse_launch(make_pipeline())
        sink = p.get_by_name("mysink")
        if not sink:
            p.set_state(Gst.State.NULL)
            return None, None, None
        p.set_state(Gst.State.PLAYING)
        bus = p.get_bus()
        msg = bus.timed_pop_filtered(
            3 * Gst.SECOND,
            Gst.MessageType.ERROR | Gst.MessageType.ASYNC_DONE | Gst.MessageType.STATE_CHANGED
        )
        if msg and msg.type == Gst.MessageType.ERROR:
            p.set_state(Gst.State.NULL)
            return None, None, None
        log("RTSP 연결 성공")
        return p, sink, bus
    except Exception as e:
        log(f"RTSP 파이프라인 생성 실패: {e}")
        return None, None, None

def map_sample_to_bgr(sample):
    buf = sample.get_buffer()
    caps = sample.get_caps()
    s = caps.get_structure(0)
    W = s.get_value("width")
    H = s.get_value("height")
    fmt = s.get_value("format")

    vinfo = GstVideo.VideoInfo()
    vinfo.from_caps(caps)
    stride = vinfo.stride[0]

    ok, m = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    try:
        data = np.frombuffer(m.data, np.uint8).reshape(H, stride)
        if fmt == "RGBA":
            rgba = data[:, :W * 4].reshape(H, W, 4)
            return rgba[:, :, :3][:, :, ::-1]
        else:
            raise RuntimeError(f"포맷 문제: {fmt}")
    finally:
        buf.unmap(m)


def load_yolo(weights):
    from ultralytics import YOLO
    m = YOLO(weights)
    return m, "exported"

# -------------------------------------------------------
def run(args):
    # 텔레그램
    profile = parse_profile_from_weights(args.weights, default="night")
    state = Telegram(profile)
    start_telegram_poller(state)
    log("Telegram poller started. (/status, /day, /night)")

    # 영상
    Gst.init(None)
    pipe, sink, bus = open_pipeline()
    if not pipe:
        log("❌ RTSP 연결 실패"); return

    # 모델, 로직
    model = None; cur_profile = None
    absence = NightWatcher(absence_sec=10.0, cooldown_sec=30.0)
    prox    = DayWatcher(ratio=0.08, hold=1.0, cooldown=30.0)

    # 모델 로드
    def weights_for(p): return NIGHT_ENGINE if p == "night" else DAY_ENGINE
    desired = state.desired_profile
    try:
        model, _ = load_yolo(weights_for(desired))
        cur_profile = desired
        log(f"{cur_profile.upper()}모델 활성화 ({weights_for(cur_profile)})")
        asyncio.run(tg_send(f"{cur_profile.upper()}모델 활성화"))
    except Exception as e:
        log(f"모델 로드 실패: {e}")

    if args.show:
        cv2.namedWindow("RTSP (test)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RTSP (test)", 1280, 720)

    last_log = time.time(); fps_cnt = 0; no_frame = 0

    while True:
        # 텔레그램 커맨드
        with state.lock:
            desired = state.desired_profile
        if desired != cur_profile:
            try:
                model, _ = load_yolo(weights_for(desired))
                cur_profile = desired
                log(f"{cur_profile.upper()}모델 활성화 ({weights_for(cur_profile)})")
                asyncio.run(tg_send(f"{cur_profile.upper()}모델 활성화"))
            except Exception as e:
                log(f"모델 로드 실패: {e}"); time.sleep(1)

        # 자동 재연결
        sample = sink.emit("try-pull-sample", 1 * Gst.SECOND)
        if sample is None:
            msg = bus.timed_pop_filtered(0, Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg and msg.type in (Gst.MessageType.ERROR, Gst.MessageType.EOS):
                log("GStreamer 오류 -> 자동 재연결")
                pipe.set_state(Gst.State.NULL); time.sleep(1)
                pipe, sink, bus = open_pipeline(); continue
            no_frame += 1
            if no_frame > 50:
                log("프레임 누락 -> 자동 재연결")
                pipe.set_state(Gst.State.NULL); time.sleep(1)
                pipe, sink, bus = open_pipeline(); no_frame = 0
            time.sleep(0.01); continue
        no_frame = 0

        # 프레임 변환
        frame = map_sample_to_bgr(sample)
        if frame is None:
            continue
        H, W = frame.shape[:2]

        # 추론
        out = frame
        det_msg = ""
        present = []
        dets = []

        # 추론
        if model is not None:
            try:
                res = model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    verbose=False
                )
                r0 = res[0]
                if args.show:
                    out = r0.plot()
                if r0.boxes is not None and len(r0.boxes) > 0:
                    names = r0.names
                    cls   = r0.boxes.cls.cpu().numpy().astype(int)
                    confs = r0.boxes.conf.cpu().numpy()
                    xyxy  = r0.boxes.xyxy.cpu().numpy()
                    for c, cf, box in zip(cls, confs, xyxy):
                        if cf < args.conf: 
                            continue
                        name = names[c]
                        present.append(name)
                        dets.append({"name": name, "conf": float(cf), "xyxy": tuple(box.tolist())})
                    det_msg = ", ".join([f"{names[c]}:{cf:.2f}" for c, cf in zip(cls, confs)])
                else:
                    det_msg = "(no dets)"
            except Exception as e:
                det_msg = f"(infer err: {e})"

        # 트리거
        if cur_profile == "night":
            absence.update(present)
            fired, reason = absence.should_fire()
            if fired:
                txt = f"경고 : {reason}"
                log(txt); asyncio.run(tg_send(txt))
        else:
            fired, reason = prox.update_and_check(dets, W, H)
            if fired:
                txt = f"경고 : {reason}"
                log(txt); asyncio.run(tg_send(txt))

        # 로그
        fps_cnt += 1; now = time.time()
        if now - last_log >= 2.0:
            log(f"FPS ≈ {fps_cnt/(now-last_log):.1f} ({W}x{H}) {det_msg}")
            fps_cnt = 0; last_log = now

        if args.show:
            cv2.imshow("RTSP (test)", out)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    # 종료
    try: pipe.set_state(Gst.State.NULL)
    except: pass
    if args.show: cv2.destroyAllWindows()


# -----------------------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--weights", type=str, default="weights/night/night_weight.engine")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", type=str, default="0")
    args=ap.parse_args()
    run(args)

if __name__=="__main__":
    main()
