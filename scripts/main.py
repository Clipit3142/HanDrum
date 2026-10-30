import time
from pathlib import Path
import cv2
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import itertools
import sound
import tracking

MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
if not MODEL.exists():
    raise SystemExit(
        f"missing model: {MODEL}\n"
        "curl -fSL --create-dirs -o models/hand_landmarker.task \\\n"
        "  https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )

dev = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=dev).view(3, 1, 1)

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4), (0,5),(5,6),(6,7),(7,8),
               (5,9),(9,10),(10,11),(11,12), (9,13),(13,14),(14,15),(15,16),
               (13,17),(17,18),(18,19),(19,20),(0,17)]

THUMB, INDEX, MIDDLE, RING, PINKY = 4, 8, 12, 16, 20

def find_camera(max_index=8, retries=5, delay=1.5):
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            continue
        for attempt in range(retries):
            ok, frame = cap.read()
            if ok and frame is not None:
                print(f"index {index} working after {attempt+1} attempt(s)")
                return cap, index
            time.sleep(delay)
        cap.release()
    return None, None

def to_frame(t):
    # normalized CHW tensor -> HWC BGR uint8
    t = t * STD + MEAN 
    t = t.clamp_(0, 1).mul_(255).round_()
    t = t.permute(1, 2, 0).to(torch.uint8)   # CHW -> HWC, still RGB
    t = t.flip(-1).contiguous()              # RGB -> BGR, fused into one copy
    return t.cpu().numpy()

def to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR -> RGB

def landmarks(frame:mp.ImageFormat.SRGB): 
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) # wrapper
    ts_ms = int((time.perf_counter() - t0) * 1000)  # packet timestamp (int ms)
    return landmarker.detect_for_video(mp_img, ts_ms) # push

def draw(frame, res, fps):
    h, w = frame.shape[:2]
    for hand, handed in zip(res.hand_landmarks, res.handedness):
        pts = [(int(p.x * w), int(p.y * h)) for p in hand]
        for a, b in CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)

        label = f"{handed[0].category_name} {handed[0].score:.2f}"
        cv2.putText(frame, label, pts[0], cv2.FONT_HERSHEY_SIMPLEX,
                     0.7, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, f"{fps:5.1f} fps  {dev}  hands={len(res.hand_landmarks)}",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("model view", frame)

opts = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(opts)

cap, idx = find_camera()
if not cap.isOpened():
    raise SystemExit("could not open camera 0")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow("model view", cv2.WINDOW_NORMAL)
print(f"device={dev}  model={MODEL.name}  —  press q or esc to quit")

fps, last = 0.0, time.perf_counter()
t0 = time.perf_counter()

r = ["snare","kick","floor-tom","perc2"]
l = ["hh-close","hh-semi","ride","crash"]

mixer = sound.Mixer()
kit = sound.Kit(mixer, "acoustic")
Rtrackers = [tracking.Tracker(i*4+8, THUMB, kit, r[i]) for i in range(0,4)]
Ltrackers = [tracking.Tracker(i*4+8, THUMB, kit, l[i]) for i in range(0,4)]
"""Ltrackers = []
Rtrackers = [tracking.Tracker(8, THUMB, kit, r[0])]"""

try:
    while True:

        ok, frame = cap.read() # HWC, BGR, uint8
        if not ok: print("frame grab failed"); break

        rgb = to_rgb(frame)
        res = landmarks(rgb)

        hands = [(h[0].category_name, h[0].score,
          torch.tensor([[p.x, p.y, p.z] for p in lmk], device=dev))
         for lmk, h in zip(res.hand_landmarks, res.handedness)]

        by_side = {name: joints for name, score, joints in hands}

        left  = by_side.get("Left")
        right = by_side.get("Right")

        now = time.perf_counter()
        dt, last = now - last, now
        if dt > 0:
            fps = 0.9 * fps + 0.1 / dt

        for rt,lt in itertools.zip_longest(Rtrackers, Ltrackers):
            if rt: rt.update(right, dt)
            if lt: lt.update(left, dt)



        
        #draw(frame.copy(), res, fps)
        #if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break


finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
