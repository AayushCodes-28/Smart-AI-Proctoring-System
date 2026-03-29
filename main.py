!pip install scikit-learn matplotlib joblib pandas numpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib

np.random.seed(42)
print(" Libraries imported successfully!")


N = 3000

gaze_off_normal = np.random.beta(2, 18, size=int(N*0.8)) * 0.3
head_yaw_normal = np.random.normal(3, 2, size=int(N*0.8)).clip(0, 30)
blink_normal = np.random.normal(12, 4, size=int(N*0.8)).clip(1, 40)
phone_normal = np.random.poisson(0.05, size=int(N*0.8))
hand_normal = np.random.poisson(0.5, size=int(N*0.8))
stress_normal = np.random.beta(2, 6, size=int(N*0.8))

gaze_off_mal = np.random.beta(6, 4, size=int(N*0.2)) * 0.9
head_yaw_mal = np.random.normal(15, 8, size=int(N*0.2)).clip(0, 90)
blink_mal = np.random.normal(18, 6, size=int(N*0.2)).clip(1, 60)
phone_mal = np.random.poisson(1.5, size=int(N*0.2)).clip(0, 30)
hand_mal = np.random.poisson(2, size=int(N*0.2))
stress_mal = np.random.beta(5, 2, size=int(N*0.2))

gaze = np.concatenate([gaze_off_normal, gaze_off_mal])
head = np.concatenate([head_yaw_normal, head_yaw_mal])
blink = np.concatenate([blink_normal, blink_mal])
phone = np.concatenate([phone_normal, phone_mal])
hand = np.concatenate([hand_normal, hand_mal])
stress = np.concatenate([stress_normal, stress_mal])
labels = np.concatenate([np.zeros(int(N*0.8)), np.ones(int(N*0.2))])

df = pd.DataFrame({
    "gaze_off_pct": gaze,
    "head_yaw_std": head,
    "blink_rate": blink,
    "phone_detected_count": phone,
    "hand_to_face_count": hand,
    "expression_stress": stress,
    "label": labels.astype(int)
}).sample(frac=1).reset_index(drop=True)

print(" Synthetic dataset created!")
df.head()


X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(" Train-test split done!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)
print(" Model training complete!")


y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=["normal", "malicious"]))

print(" Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"ROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")


fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Synthetic Data)")
plt.legend()
plt.grid(True)
plt.show()

feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n Feature Importances:")
display(feat_imp)


joblib.dump(rf, "cheat_detector_rf.pkl")
print(" Model saved as cheat_detector_rf.pkl")

model = joblib.load("cheat_detector_rf.pkl")
print(" Model reloaded successfully!")


sample_input = pd.DataFrame({
    "gaze_off_pct": [0.8],
    "head_yaw_std": [25],
    "blink_rate": [20],
    "phone_detected_count": [2],
    "hand_to_face_count": [3],
    "expression_stress": [0.85]
})

prediction = model.predict(sample_input)[0]
prob = model.predict_proba(sample_input)[0][1]

print("Prediction:", "Malicious" if prediction == 1 else "Normal")
print(f"Probability of malicious: {prob:.2f}")


!pip install mediapipe opencv-python==4.11.0.86 numpy==1.26.4

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

print(" MediaPipe and OpenCV installed!")

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def extract_features_from_video(video_path, max_frames=600):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    gaze_off_frames = 0
    blink_frames = 0
    total_frames = 0
    head_angles = []

    prev_eye_ratio = None
    blink_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret or total_frames > max_frames:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        total_frames += 1

        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0].landmark

        nose_tip = np.array([landmarks[1].x, landmarks[1].y])
        left_eye = np.array([landmarks[33].x, landmarks[33].y])
        right_eye = np.array([landmarks[263].x, landmarks[263].y])

        head_yaw = np.degrees(np.arctan2(right_eye[0] - left_eye[0],
                                         right_eye[1] - left_eye[1]))
        head_angles.append(head_yaw)

        def eye_ratio(idx_top, idx_bottom, idx_left, idx_right):
            top = np.array([landmarks[idx_top].x, landmarks[idx_top].y])
            bottom = np.array([landmarks[idx_bottom].x, landmarks[idx_bottom].y])
            left = np.array([landmarks[idx_left].x, landmarks[idx_left].y])
            right = np.array([landmarks[idx_right].x, landmarks[idx_right].y])
            vertical = np.linalg.norm(top - bottom)
            horizontal = np.linalg.norm(left - right)
            return vertical / horizontal

        left_ratio = eye_ratio(159, 145, 133, 33)
        right_ratio = eye_ratio(386, 374, 362, 263)
        eye_open_ratio = (left_ratio + right_ratio) / 2

        if prev_eye_ratio is not None:
            if eye_open_ratio < 0.18 and prev_eye_ratio >= 0.18:
                blink_counter += 1
        prev_eye_ratio = eye_open_ratio

        if abs(nose_tip[0] - 0.5) > 0.2:
            gaze_off_frames += 1

    cap.release()

    blink_rate = (blink_counter / total_frames) * 60
    gaze_off_pct = gaze_off_frames / total_frames
    head_yaw_std = np.std(head_angles)

    features = {
        "gaze_off_pct": gaze_off_pct,
        "head_yaw_std": head_yaw_std,
        "blink_rate": blink_rate,
        "phone_detected_count": 0,
        "hand_to_face_count": 0,
        "expression_stress": 0.5
    }

    return pd.DataFrame([features])

from base64 import b64decode
from google.colab import output
RECORD_SECONDS = 10
print(f"Recording for {RECORD_SECONDS} seconds...")

js = f"""
(async () => {{
  const stream = await navigator.mediaDevices.getUserMedia({{video:true, audio:false}});
  const mediaRecorder = new MediaRecorder(stream);
  let chunks = [];
  mediaRecorder.ondataavailable = e => chunks.push(e.data);
  mediaRecorder.start();
  await new Promise(r => setTimeout(r, {int(RECORD_SECONDS*1000)}));
  mediaRecorder.stop();
  await new Promise(r => mediaRecorder.onstop = r);
  const blob = new Blob(chunks, {{ type: 'video/webm' }});
  const arrayBuffer = await blob.arrayBuffer();
  let binary = '';
  const bytes = new Uint8Array(arrayBuffer);
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {{
    let sub = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode.apply(null, sub);
  }}
  return btoa(binary);
}})()
"""
print("Please allow camera permission in the browser popup (if prompted).")
video_b64 = output.eval_js(js)

# write to file
video_bytes = b64decode(video_b64)
with open("recorded_video.webm", "wb") as f:
    f.write(video_bytes)
print("Saved raw recording to: recorded_video.webm")


!ffmpeg -y -i recorded_video.webm -c:v libx264 recorded_video.mp4
print("Saved converted file: recorded_video.mp4")


!pip install mediapipe opencv-python==4.11.0.86 numpy==1.26.4

import joblib
import pandas as pd
import os

import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def extract_features_from_video(video_path, max_frames=600):
    cap = cv2.VideoCapture(video_path)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    gaze_off_frames = 0
    blink_frames = 0
    total_frames = 0
    head_angles = []

    prev_eye_ratio = None
    blink_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret or total_frames > max_frames:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        total_frames += 1

        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0].landmark

        nose_tip = np.array([landmarks[1].x, landmarks[1].y])
        left_eye = np.array([landmarks[33].x, landmarks[33].y])
        right_eye = np.array([landmarks[263].x, landmarks[263].y])

        head_yaw = np.degrees(np.arctan2(right_eye[0] - left_eye[0],
                                         right_eye[1] - left_eye[1]))
        head_angles.append(head_yaw)

        def eye_ratio(idx_top, idx_bottom, idx_left, idx_right):
            top = np.array([landmarks[idx_top].x, landmarks[idx_top].y])
            bottom = np.array([landmarks[idx_bottom].x, landmarks[idx_bottom].y])
            left = np.array([landmarks[idx_left].x, landmarks[idx_left].y])
            right = np.array([landmarks[idx_right].x, landmarks[idx_right].y])
            vertical = np.linalg.norm(top - bottom)
            horizontal = np.linalg.norm(left - right)
            return vertical / horizontal

        left_ratio = eye_ratio(159, 145, 133, 33)
        right_ratio = eye_ratio(386, 374, 362, 263)
        eye_open_ratio = (left_ratio + right_ratio) / 2

        if prev_eye_ratio is not None:
            if eye_open_ratio < 0.18 and prev_eye_ratio >= 0.18:
                blink_counter += 1
        prev_eye_ratio = eye_open_ratio

        if abs(nose_tip[0] - 0.5) > 0.2:
            gaze_off_frames += 1

    cap.release()

    blink_rate = (blink_counter / total_frames) * 60
    gaze_off_pct = gaze_off_frames / total_frames
    head_yaw_std = np.std(head_angles)

    features = {
        "gaze_off_pct": gaze_off_pct,
        "head_yaw_std": head_yaw_std,
        "blink_rate": blink_rate,
        "phone_detected_count": 0,
        "hand_to_face_count": 0,
        "expression_stress": 0.5
    }

    return pd.DataFrame([features])


video_file = "recorded_video.mp4"
assert os.path.exists(video_file), "Video file not found. Please run A1 & A2 first."

features_df = extract_features_from_video(video_file, max_frames=600)
print("Extracted features:")
display(features_df)

model_path = "cheat_detector_rf.pkl"
assert os.path.exists(model_path), "Model not found. Please train it again if deleted."
model = joblib.load(model_path)

prediction = model.predict(features_df)[0]
probability = model.predict_proba(features_df)[0][1]

print("🧠 Model Prediction:", "MALICIOUS" if prediction == 1 else "NORMAL")
print(f"Malicious probability: {probability:.2f}")

import cv2
import mediapipe as mp
import numpy as np

def annotate_video(input_path, output_path="annotated_output.mp4", max_frames=300):
    mp_face = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        count = 0
        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                for landmarks in result.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=landmarks,
                        connections=mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
                    )
            out.write(frame)

    cap.release()
    out.release()
    print(f" Annotated video saved as {output_path}")

# Run this in Colab after recording
annotate_video("recorded_video.mp4")

from IPython.display import Video
Video("annotated_output.mp4", embed=True, width=640, height=480)



