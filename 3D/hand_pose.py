import cv2
import mediapipe as mp
import time
import numpy as np

# Inisialisasi MediaPipe untuk Tangan dan Deteksi Wajah
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Konfigurasi deteksi tangan
hands = mp_hands.Hands(
    static_image_mode=False,  # False untuk deteksi real-time
    max_num_hands=2,         # Maksimal dua tangan yang dideteksi
    min_detection_confidence=0.5,  # Kepercayaan minimal untuk deteksi
    min_tracking_confidence=0.5    # Kepercayaan minimal untuk pelacakan
)

# Konfigurasi deteksi wajah
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Membuka Webcam
cap = cv2.VideoCapture(0)

# Variabel untuk menghitung FPS (Frame Per Second)
prev_time = 0

# Fungsi untuk mendeteksi gerakan tangan (gestur)
def detect_gesture(hand_landmarks):
    # Koordinat ujung jari untuk mendeteksi gerakan
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    # Menghitung jarak antara ujung jari telunjuk dan ibu jari
    distance_thumb_index = np.linalg.norm([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y])

    # Jika jarak antara jari telunjuk dan ibu jari kecil (tangan mengepal)
    if distance_thumb_index < 0.05:
        return "Tangan Mengepal", (0, 0, 255)  # Warna merah
    
    # Jika hanya jari telunjuk yang terangkat (gestur menunjuk)
    if index_tip.y < thumb_tip.y:
        return "Menunjuk", (0, 255, 0)  # Warna hijau
    
    # Jika semua jari terpisah (tangan terbuka)
    if (index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y):
        return "Tangan Terbuka", (255, 255, 0)  # Warna kuning

    return "Tidak Ada Gerakan", (255, 255, 255)  # Tidak ada gerakan (putih)

# Membuka window dengan mode full-screen
cv2.namedWindow("Deteksi Tangan & Wajah dengan Gestur", cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Membalik frame secara horizontal untuk efek mirror
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Proses frame untuk deteksi tangan
    hand_results = hands.process(rgb_frame)
    # Proses frame untuk deteksi wajah
    face_results = face_detection.process(rgb_frame)

    # Jika tangan terdeteksi
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Gambar landmark tangan di frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

            # Deteksi gestur tangan dan tampilkan hasilnya
            gesture, color = detect_gesture(hand_landmarks)
            cv2.putText(frame, f"Gestur: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Tampilkan log gerakan tangan di terminal
            print(f"Gestur Tangan Terdeteksi: {gesture}")

    # Jika wajah terdeteksi
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Gambar kotak pembatas di sekitar wajah
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Tampilkan teks "Wajah Terdeteksi"
            cv2.putText(frame, "Ridwan, 21 th", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Tampilkan log deteksi wajah di terminal
            print(f"Wajah Terdeteksi di: x={x}, y={y}, width={w}, height={h}")

    # Menghitung FPS untuk memberikan informasi tentang performa
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Tampilkan FPS di atas video
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

    # Tampilkan frame di jendela full-screen
    cv2.imshow("Deteksi Tangan & Wajah dengan Gestur", frame)

    # Tekan 'q' untuk keluar dari aplikasi
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan sumber daya
cap.release()
cv2.destroyAllWindows()
