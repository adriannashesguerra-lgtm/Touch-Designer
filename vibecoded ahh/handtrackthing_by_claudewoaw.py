"""
hand_tracker_touchdesigner.py

Detects up to 6 hands (i.e. "3 pairs") using MediaPipe's HandLandmarker in
LIVE_STREAM mode, and streams the landmark + handedness data out over OSC
so TouchDesigner (or any other OSC-capable app) can pick it up in real time.

------------------------------------------------------------------
SETUP
------------------------------------------------------------------
1. Install dependencies:

       pip install mediapipe opencv-python python-osc

2. Download the HAND LANDMARK model (not the gesture model this time --
   we just want raw landmarks here, not gesture classification):

       https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

   Save it next to this script, or point MODEL_PATH at wherever you keep it.

3. In TouchDesigner:
       - Add an "OSC In" CHOP.
       - Set "Network Port" to match OSC_PORT below (default 7000).
       - Turn OFF "Split Strings" if present, and set the CHOP to receive
         Value messages.
       - Hands are addressed by SIDE, not by detection order, so a given
         address always means "a left hand" -- it won't jump to a different
         physical hand between frames. With MAX_HANDS_PER_SIDE = 3 you get:

             /hands/count                 -> total hands visible right now
             /hands/left/count            -> how many left hands are visible (0-3)
             /hands/right/count           -> how many right hands are visible (0-3)
             /hand/left/0/landmarks       -> 63 floats: x0,y0,z0, x1,y1,z1, ... x20,y20,z20
             /hand/left/0/confidence      -> 0.0-1.0 handedness confidence
             /hand/right/0/landmarks
             /hand/right/0/confidence
             ... up to /hand/left/2 and /hand/right/2 for 3 pairs

         If a slot has no hand in it right now, its landmarks message is
         empty and confidence is 0.0, so you can tell "no hand" apart from
         "hand present".

         Slots are now STABLE across frames -- /hand/left/1 keeps referring
         to roughly the same physical hand as it moves, using wrist-position
         nearest-neighbor tracking (see HandSideTracker below). Worth knowing
         its limits: it tracks HANDS, not PEOPLE (MediaPipe never tells us
         which two hands belong to one body, so /hand/left/1 and
         /hand/right/1 are not guaranteed to be the same person); two same-
         side hands crossing paths can swap slot IDs; and a hand that leaves
         frame for longer than MAX_MISSED_FRAMES gets treated as new and may
         land in a different slot when it returns.


4. Run this script on your machine (not in a sandbox -- it needs a webcam):

       python hand_tracker_touchdesigner.py

------------------------------------------------------------------
NOTES
------------------------------------------------------------------
- If you actually wanted GESTURE classification (thumbs up, victory, etc.)
  as well as tracking, that's a separate MediaPipe task (GestureRecognizer,
  like in your original file). You can run both in parallel and just add
  a second OSC address like /hand/0/gesture -- ask if you want that version.
- If OSC ever feels too chatty/slow for your patch, the other common route
  into TouchDesigner is a Spout/NDI video texture (send the annotated frame
  as a texture) instead of raw OSC data -- happy to build that variant too.
"""


# SWITCHES WHICH CAMERA THE CODE IS USING DEPENDING ON HOW MUCH CAMERAS U HAVE CONNECTED
# ex : 0 = integrated camera, 1 = OBS virtual camera
cam_using = 1  # if its not showing the preview make sure virtual camara is turned on in OBS 

import time
import cv2
import mediapipe as mp
from pathlib import Path
from typing import Optional, List

from pythonosc import udp_client

# === CONFIGURATION ===
current_directory = Path(__file__).parent
MODEL_PATH = str(current_directory / "hand_landmarker.task")  # update if stored elsewhere

MAX_HANDS = 6              # "3 pairs" of hands, total across both sides
MAX_HANDS_PER_SIDE = 3     # up to 3 left hands + 3 right hands at once
MIN_HANDEDNESS_SCORE = 0.7 # drop low-confidence hands instead of risking a flipped label
INVERT_HANDEDNESS = True   # set True if testing shows Left/Right are swapped for your pipeline
                            # (e.g. raising your right hand reports as "Left"). This can happen
                            # when something upstream (webcam driver, OBS capture settings, etc.)
                            # mirrors the image an extra time, canceling out TD's flip1. Rather than
                            # track down exactly where, this flips the label back at the source.
MAX_MATCH_DISTANCE = 0.15  # normalized image-space distance a wrist can move frame-to-frame and still count as "the same hand" -- raise if fast motion causes ID swaps, lower if two hands ever "steal" each other's slot
MAX_MISSED_FRAMES = 15     # ~0.5s at 30fps grace period before a slot is freed for reuse (tolerates brief occlusion without reassigning IDs)
OSC_IP = "127.0.0.1"       # change to TouchDesigner's machine IP if it's remote
OSC_PORT = 7000            # must match the OSC In CHOP's Network Port in TD
SHOW_PREVIEW = True        # local OpenCV preview window, turn off if you don't need it

# === SHORTCUTS TO THE TASKS API ===
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
Image = mp.Image

# Standard MediaPipe hand connections, used only for the local preview draw
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17)                                 # Palm base
]

# === Shared state, updated by the async result callback, read by the main loop ===
_latest_hands: List[dict] = []  # each item: {"handedness": str, "score": float, "landmarks": [(x, y, z), ...]}


class HandSideTracker:
    """Tracks up to `max_slots` hands of ONE side (left or right) across
    frames, using nearest-neighbor matching on wrist position (landmark 0).
    This is what gives each OSC slot a stable identity -- e.g. /hand/left/1
    keeps referring to roughly the same physical hand from frame to frame,
    instead of jumping around based on MediaPipe's raw detection order.

    Limitations (inherent to proximity-based tracking without a body/pose
    model -- not something more code alone can fully fix):
      - It tracks HANDS, not PEOPLE. It cannot guarantee /hand/left/1 and
        /hand/right/1 belong to the same person.
      - If two hands of the SAME side pass close to each other (e.g. two
        performers' left hands cross), their slot IDs can swap.
      - A hand that leaves frame for longer than max_missed_frames is
        treated as new and may come back in a different slot.
    """

    def __init__(self, max_slots=MAX_HANDS_PER_SIDE,
                 max_match_distance=MAX_MATCH_DISTANCE,
                 max_missed_frames=MAX_MISSED_FRAMES):
        self.max_slots = max_slots
        self.max_match_distance = max_match_distance
        self.max_missed_frames = max_missed_frames
        self.tracks = {}  # slot_id -> {"position": (x, y), "hand": dict, "missed": int}

    def update(self, detections: List[dict]) -> dict:
        """detections: this frame's hands for ONE side only (already
        confidence-filtered). Returns {slot_id: hand_dict} for every slot
        that currently has a hand."""
        det_positions = [d["landmarks"][0][:2] for d in detections]  # wrist (x, y)

        # Score every (existing track, new detection) pair within range, then
        # greedily accept the closest pairs first. Simple and fast, which is
        # plenty at this scale (max 3 tracks per side).
        candidates = []
        for tid, track in self.tracks.items():
            tx, ty = track["position"]
            for di, (dx, dy) in enumerate(det_positions):
                dist = ((tx - dx) ** 2 + (ty - dy) ** 2) ** 0.5
                if dist <= self.max_match_distance:
                    candidates.append((dist, tid, di))
        candidates.sort(key=lambda c: c[0])

        matched_tracks, matched_dets = set(), set()
        for dist, tid, di in candidates:
            if tid in matched_tracks or di in matched_dets:
                continue
            self.tracks[tid]["position"] = det_positions[di]
            self.tracks[tid]["hand"] = detections[di]
            self.tracks[tid]["missed"] = 0
            matched_tracks.add(tid)
            matched_dets.add(di)

        # Age out tracks not matched this frame; free the slot once they've
        # been missing too long so it can be reused.
        for tid in list(self.tracks.keys()):
            if tid not in matched_tracks:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed_frames:
                    del self.tracks[tid]

        # Leftover detections are "new" hands -- give each the lowest free slot.
        used_ids = set(self.tracks.keys())
        for di in range(len(detections)):
            if di in matched_dets:
                continue
            free_id = next((i for i in range(self.max_slots) if i not in used_ids), None)
            if free_id is None:
                continue  # more hands on this side right now than we have slots for
            self.tracks[free_id] = {"position": det_positions[di], "hand": detections[di], "missed": 0}
            used_ids.add(free_id)

        return {tid: t["hand"] for tid, t in self.tracks.items() if t["missed"] == 0}


def result_callback(result, output_image, timestamp_ms: int):
    """Callback for LIVE_STREAM results. Just stashes the latest hands for
    the main loop to send over OSC / draw."""
    global _latest_hands

    hands = []
    if result and result.hand_landmarks:
        for i, hand_landmarks in enumerate(result.hand_landmarks):
            handedness, score = "Unknown", 0.0
            if result.handedness and len(result.handedness) > i:
                category = result.handedness[i][0]
                handedness, score = category.category_name, category.score
                if INVERT_HANDEDNESS and handedness in ("Left", "Right"):
                    handedness = "Right" if handedness == "Left" else "Left"
            hands.append({
                "handedness": handedness,
                "score": score,
                "landmarks": [(lm.x, lm.y, lm.z) for lm in hand_landmarks],
            })
    _latest_hands = hands


def send_hands_osc(client: udp_client.SimpleUDPClient, left_slots: dict, right_slots: dict, total_hands: int):
    """left_slots / right_slots: {slot_id: hand_dict} from each side's
    HandSideTracker. Sends each slot's landmarks to a stable
    /hand/{side}/{slot_id}/... address."""
    client.send_message("/hands/count", total_hands)
    client.send_message("/hands/left/count", len(left_slots))
    client.send_message("/hands/right/count", len(right_slots))

    for side, slots in (("left", left_slots), ("right", right_slots)):
        for n in range(MAX_HANDS_PER_SIDE):
            if n in slots:
                hand = slots[n]
                flat = [coord for lm in hand["landmarks"] for coord in lm]  # 63 floats
                client.send_message(f"/hand/{side}/{n}/confidence", hand["score"])
                client.send_message(f"/hand/{side}/{n}/landmarks", flat)
            else:
                # Slot not currently filled -- send empty markers so TD can
                # tell "no data yet" apart from "hand left frame".
                client.send_message(f"/hand/{side}/{n}/confidence", 0.0)
                client.send_message(f"/hand/{side}/{n}/landmarks", [])


def main():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=result_callback,
    )

    osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    left_tracker = HandSideTracker()
    right_tracker = HandSideTracker()

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(cam_using)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FRAME_COUNT, 60)

        if not cap.isOpened():
            print("Error: could not open camera")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # NOTE: this frame is already mirrored by the time it gets here --
                # it comes from TD (videodevin1 -> flip1 -> syphonSpoutOut) via
                # OBS Virtual Camera, and TD's flip1 already does the mirroring.
                # Do NOT flip again here, or you'll un-mirror it and MediaPipe's
                # Left/Right handedness will come out swapped again.
                frame_flipped = frame
                rgb_frame = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = int(time.time() * 1000)
                landmarker.detect_async(mp_image, timestamp_ms)

                hands = _latest_hands
                left_dets = [h for h in hands if h["handedness"] == "Left" and h["score"] >= MIN_HANDEDNESS_SCORE]
                right_dets = [h for h in hands if h["handedness"] == "Right" and h["score"] >= MIN_HANDEDNESS_SCORE]
                left_slots = left_tracker.update(left_dets)
                right_slots = right_tracker.update(right_dets)
                send_hands_osc(osc_client, left_slots, right_slots, total_hands=len(hands))

                if SHOW_PREVIEW:
                    h, w = frame_flipped.shape[:2]
                    tracked_count = len(left_slots) + len(right_slots)
                    cv2.putText(
                        frame_flipped, f"Tracked hands: {tracked_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
                    )
                    for side, slots in (("L", left_slots), ("R", right_slots)):
                        for slot_id, hand in slots.items():
                            pts = [(int(x * w), int(y * h)) for (x, y, _z) in hand["landmarks"]]
                            for a, b in HAND_CONNECTIONS:
                                if a < len(pts) and b < len(pts):
                                    cv2.line(frame_flipped, pts[a], pts[b], (0, 0, 0), 6, cv2.LINE_AA)
                                    cv2.line(frame_flipped, pts[a], pts[b], (0, 255, 0), 3, cv2.LINE_AA)
                            for (x_px, y_px) in pts:
                                cv2.circle(frame_flipped, (x_px, y_px), 6, (0, 0, 0), -1)
                                cv2.circle(frame_flipped, (x_px, y_px), 4, (0, 0, 255), -1)
                            # Label the slot ID above the wrist so you can visually
                            # confirm it stays put as the hand moves around.
                            label_pos = (pts[0][0] - 20, pts[0][1] + 30)
                            cv2.putText(frame_flipped, f"{side}{slot_id}", label_pos,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
                            cv2.putText(frame_flipped, f"{side}{slot_id}", label_pos,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

                    cv2.imshow("Hand Tracker -> TouchDesigner (OSC)", frame_flipped)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
