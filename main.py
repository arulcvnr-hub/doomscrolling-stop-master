import cv2
import mediapipe as mp
import numpy as np
import math
import subprocess
import threading
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

VIDEO_PATH = '/Users/jayeshvishwakarma/Documents/Stuffs/doomscrolling-stop/Video-75425.mp4'

QT_OPEN_SCRIPT = f'''
tell application "QuickTime Player"
    activate
    set theDoc to open POSIX file "{VIDEO_PATH}"
    tell theDoc to play
end tell
'''

QT_CLOSE_SCRIPT = '''
tell application "QuickTime Player"
    if (count of documents) > 0 then
        close every document
    end if
end tell
'''

def start_video():
    subprocess.Popen(['osascript', '-e', QT_CLOSE_SCRIPT],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait()
    time.sleep(0.1)
    subprocess.Popen(['osascript', '-e', QT_OPEN_SCRIPT],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def stop_video():
    subprocess.Popen(['osascript', '-e', QT_CLOSE_SCRIPT],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def dist3d(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0,  -150.0, -125.0)
], dtype=np.float64)
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

def get_head_pitch(landmarks, img_w, img_h):
    image_points = np.array([
        (landmarks[i].x * img_w, landmarks[i].y * img_h)
        for i in LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0

    rot_mat, _ = cv2.Rodrigues(rotation_vec)
    pitch = math.degrees(math.asin(-rot_mat[2][1]))
    return pitch

def eye_looking_down(landmarks):
    try:
        results = []
        for top_idx, bottom_idx, iris_idx in [(159, 145, 468), (386, 374, 473)]:
            top    = landmarks[top_idx]
            bottom = landmarks[bottom_idx]
            iris   = landmarks[iris_idx]
            eye_h  = dist3d(top, bottom)
            if eye_h < 0.005:
                return None
            iris_rel = (iris.y - top.y) / (bottom.y - top.y + 1e-6)
            results.append(iris_rel)

        avg = sum(results) / len(results)
        return avg > 0.52, avg
    except:
        return None

def draw_face_box(image, landmarks, img_w, img_h):
    xs = [int(lm.x * img_w) for lm in landmarks]
    ys = [int(lm.y * img_h) for lm in landmarks]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = max_x - min_x
    height = max_y - min_y
    
    size = max(width, height)
    center_x = min_x + width // 2
    center_y = min_y + height // 2
    
    pad = 10
    sq_min_x = center_x - size // 2 - pad
    sq_max_x = center_x + size // 2 + pad
    sq_min_y = center_y - size // 2 - pad
    sq_max_y = center_y + size // 2 + pad

    cv2.rectangle(image,
                  (sq_min_x, sq_min_y),
                  (sq_max_x, sq_max_y),
                  (0, 0, 0), 2)

cap = cv2.VideoCapture(1)
cv2.namedWindow('Doomscroller Ctrl', cv2.WINDOW_NORMAL)

FRAMES_TO_PLAY = 15
FRAMES_TO_STOP = 35

at_screen_frames = 0
away_frames      = 0
no_face_frames   = 0
video_playing    = False

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)
    img_h, img_w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = face_mesh.process(rgb)
    rgb.flags.writeable = True

    display = frame.copy()
    looking_at_screen = not video_playing

    if results.multi_face_landmarks:
        no_face_frames = 0
        lms = results.multi_face_landmarks[0].landmark

        draw_face_box(display, lms, img_w, img_h)

        pitch = get_head_pitch(lms, img_w, img_h)
        eye_result = eye_looking_down(lms)

        if eye_result is None:
            looking_at_screen = not video_playing
        else:
            _, iris_val = eye_result
            # Only trigger "away" (doomscrolling) if explicitly looking down.
            # Relax the pitch threshold so looking up or slightly around doesn't trigger it.
            head_tilted_down = pitch > 20 or pitch < -20
            eyes_looking_down = iris_val > 0.62
            
            is_looking_down = head_tilted_down or eyes_looking_down
            looking_at_screen = not is_looking_down

            color_p = (0, 0, 255) if head_tilted_down else (0, 200, 0)
            color_e = (0, 0, 255) if eyes_looking_down else (0, 200, 0)
            cv2.putText(display, f'Pitch: {pitch:+.1f}', (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_p, 2)
            cv2.putText(display, f'Eye: {iris_val:.2f}', (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_e, 2)

        state_label = 'AT SCREEN' if looking_at_screen else 'AWAY'
        label_color = (0, 200, 0) if looking_at_screen else (0, 0, 255)
        cv2.putText(display, state_label, (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

    else:
        no_face_frames += 1
        looking_at_screen = True if no_face_frames >= 30 else False
        cv2.putText(display, 'No face', (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

    if looking_at_screen:
        at_screen_frames += 1
        away_frames = 0
    else:
        away_frames += 1
        at_screen_frames = 0

    if not video_playing and away_frames >= FRAMES_TO_PLAY:
        video_playing = True
        away_frames = 0
        threading.Thread(target=start_video, daemon=True).start()

    elif video_playing and at_screen_frames >= FRAMES_TO_STOP:
        video_playing = False
        at_screen_frames = 0
        threading.Thread(target=stop_video, daemon=True).start()

    status_color = (0, 0, 255) if video_playing else (0, 200, 0)
    cv2.circle(display, (img_w - 25, 25), 12, status_color, -1)
    cv2.putText(display, 'REC' if video_playing else 'IDLE',
                (img_w - 70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if video_playing:
        label = 'DOOMSCROLLING ALARM'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, th + 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)
        tx = (img_w - tw) // 2
        cv2.putText(display, label, (tx, th + 8),
                    font, font_scale, (0, 255, 80), thickness, cv2.LINE_AA)

    cv2.imshow('Doomscroller Ctrl', display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

stop_video()
cap.release()
cv2.destroyAllWindows()