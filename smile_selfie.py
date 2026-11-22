"""
smile_selfie.py
Opens the webcam (or a video file), detects faces with a Haar cascade,
checks if the person is smiling with the trained LeNet model, and saves a
selfie automatically when a smile is held for a few frames.

Keys (click on the video window first):
  q  quit
  r  start / stop RECORDING the window to an MP4  (saved in recordings/)
  g  show / hide heatmap (LayerCAM - where the model is looking)

Run:
  python smile_selfie.py                    # use the webcam
  python smile_selfie.py --video clip.mp4   # use a video file instead
  python smile_selfie.py --model smile_lenet_v1.h5   # use a different model file
"""

import os
import time
import argparse
import subprocess
import cv2
import numpy as np
import tensorflow as tf
from gradcam import GradCAM, overlay_heatmap, preprocess, tight_box, remove_duplicate_faces

# ---------------- Settings ----------------
MODEL_PATH = "smile_lenet.h5"
SAVE_DIR = "selfies"
RECORD_DIR = "recordings"
SMILE_THRESHOLD = 0.80   # how confident the model must be
SMILE_FRAMES = 5         # smile must last this many frames in a row
COOLDOWN = 3             # seconds to wait between two selfies

os.makedirs(SAVE_DIR, exist_ok=True)

# Load model, Grad-CAM and OpenCV's built-in face detector
model = None      # loaded in main(), so --model can choose the file
cam = None
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ======================================================
# Recording the window to a video
# ======================================================
class Recorder:
    """Writes every displayed frame to a temporary .avi file. When recording
    stops, it converts it to an H.264 .mp4 (plays in browsers and on GitHub)."""

    def __init__(self):
        self.writer = None

    @property
    def is_recording(self):
        return self.writer is not None

    def start(self, frame):
        os.makedirs(RECORD_DIR, exist_ok=True)
        h, w = frame.shape[:2]
        self.temp_path = os.path.join(RECORD_DIR, "_temp_recording.avi")
        self.writer = cv2.VideoWriter(self.temp_path, cv2.VideoWriter_fourcc(*"MJPG"), 20, (w, h))
        self.frames = 0
        self.start_time = time.time()
        print("Recording started... press 'r' again to stop")

    def write(self, frame):
        self.writer.write(frame)
        self.frames += 1

    def seconds(self):
        return time.time() - self.start_time

    def stop(self):
        self.writer.release()
        self.writer = None
        # real frames-per-second, so the video plays at the correct speed
        real_fps = max(1.0, self.frames / max(self.seconds(), 0.1))
        out_path = os.path.join(RECORD_DIR, time.strftime("demo_%Y%m%d_%H%M%S.mp4"))
        convert_to_mp4(self.temp_path, out_path, real_fps)


def convert_to_mp4(src, dst, fps):
    """Convert to H.264 MP4 using the ffmpeg that comes with 'imageio-ffmpeg'."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([
            ffmpeg, "-y", "-loglevel", "error",
            "-r", f"{fps:.2f}", "-i", src,                    # input at the real speed
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",       # H.264 needs even width/height
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", dst,
        ], check=True)
        os.remove(src)
        print(f"Recording saved: {dst}  ({fps:.1f} fps)")
    except ImportError:
        avi = dst.replace(".mp4", ".avi")
        os.replace(src, avi)
        print(f"Recording saved: {avi}")
        print("Tip: 'pip install imageio-ffmpeg' to get an MP4 that plays on GitHub.")
    except Exception as e:
        avi = dst.replace(".mp4", ".avi")
        os.replace(src, avi)
        print(f"MP4 conversion failed ({e}). Saved as: {avi}")


# ======================================================
# Helpers
# ======================================================
def save_selfie(frame):
    filename = time.strftime("selfie_%Y%m%d_%H%M%S.jpg")
    path = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(path, frame)
    print("Selfie saved:", path)


def draw_status(frame, show_cam, recorder):
    h, w = frame.shape[:2]
    # key help (bottom right)
    help_text = f"g: heatmap {'ON' if show_cam else 'OFF'} | r: record | q: quit"
    cv2.putText(frame, help_text, (w - 330, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    # recording indicator (top right)
    if recorder.is_recording:
        cv2.circle(frame, (w - 95, 25), 8, (0, 0, 255), -1)
        cv2.putText(frame, f"REC {recorder.seconds():.0f}s", (w - 80, 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# ======================================================
# Main loop
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="path to a video file (default: webcam)")
    parser.add_argument("--model", default=MODEL_PATH, help=f"model file (default: {MODEL_PATH})")
    args = parser.parse_args()

    global model, cam
    model = tf.keras.models.load_model(args.model)
    cam = GradCAM(model)
    print("Model loaded:", args.model)

    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        print("Could not open", args.video or "the webcam")
        return
    mirror = args.video is None          # mirror only the live webcam

    print("\nSmile Selfie running. CLICK ON THE VIDEO WINDOW first, then press:")
    print("   g = heatmap on/off   r = record on/off   q = quit\n")

    smile_count = 0
    last_capture = 0
    flash_until = 0
    show_cam = False
    recorder = Recorder()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if mirror:
            frame = cv2.flip(frame, 1)              # mirror view
        clean_frame = frame.copy()                  # selfie without boxes/text
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80))
        faces = remove_duplicate_faces(faces)       # one box per face

        someone_smiling = False

        for (x, y, w, h) in faces:
            top, bottom, left, right = tight_box(x, y, w, h)
            face = gray[top:bottom, left:right]
            if face.size == 0:
                continue
            face_input = preprocess(face)

            if show_cam:
                # Grad-CAM also gives the prediction, so no extra model call needed
                prob, heatmap, _ = cam.explain(face_input)
                frame[top:bottom, left:right] = overlay_heatmap(frame[top:bottom, left:right], heatmap)
            else:
                prob = float(model(face_input, training=False)[0][1])

            smiling = prob >= SMILE_THRESHOLD
            if smiling:
                someone_smiling = True

            color = (0, 255, 0) if smiling else (0, 0, 255)
            label = f"Smiling {prob*100:.0f}%" if smiling else f"Not smiling {(1-prob)*100:.0f}%"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Count consecutive smiling frames (avoids random false triggers)
        smile_count = smile_count + 1 if someone_smiling else 0

        now = time.time()
        if smile_count >= SMILE_FRAMES and now - last_capture > COOLDOWN:
            save_selfie(clean_frame)
            last_capture = now
            flash_until = now + 0.3
            smile_count = 0

        # Small white flash effect after capture
        if now < flash_until:
            frame = cv2.addWeighted(frame, 0.4, np.full_like(frame, 255), 0.6, 0)
            cv2.putText(frame, "SNAP!", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

        # Progress bar: how many smiling frames in a row so far
        bar = int(200 * min(smile_count, SMILE_FRAMES) / SMILE_FRAMES)
        h_img = frame.shape[0]
        cv2.rectangle(frame, (10, h_img - 30), (210, h_img - 10), (255, 255, 255), 1)
        cv2.rectangle(frame, (10, h_img - 30), (10 + bar, h_img - 10), (0, 255, 0), -1)

        draw_status(frame, show_cam, recorder)

        # Record exactly what is shown on screen
        if recorder.is_recording:
            recorder.write(frame)

        cv2.imshow("Smile Selfie (press q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):           # q, Q or Esc
            break
        elif key in (ord("g"), ord("G")):             # works with Caps Lock too
            show_cam = not show_cam
            print("Heatmap", "ON" if show_cam else "OFF")
        elif key in (ord("r"), ord("R")):
            if recorder.is_recording:
                recorder.stop()
            else:
                recorder.start(frame)

    if recorder.is_recording:          # stop cleanly if you quit while recording
        recorder.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
