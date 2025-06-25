import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import math

class EyeController:
    def __init__(self):
        # Inisialisasi MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Setup kamera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Landmark indices untuk mata
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Iris landmarks (estimasi)
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # Konfigurasi layar - Parameter lebih responsif
        self.screen_width, self.screen_height = pyautogui.size()
        self.movement_sensitivity = 3  # Dikurangi untuk kontrol lebih halus
        self.smoothing_factor = 0.6  # Dikurangi untuk respon lebih cepat
        
        # Dead zone untuk mengurangi noise
        self.dead_zone_x = 0.05
        self.dead_zone_y = 0.05
        
        # State tracking
        self.prev_cursor_pos = None
        self.calibration_mode = True
        self.calibration_data = {
            'center': [],
            'left': [],
            'right': [],
            'up': [],
            'down': []
        }
        # Kalibrasi lebih cepat
        self.calibration_step = 0
        self.calibration_steps = ['center', 'left', 'right', 'up', 'down']
        self.frames_per_calibration = 30  # Dikurangi untuk kalibrasi lebih cepat
        self.current_calibration_frames = 0
        
        # Blink detection - Parameter yang lebih sensitif
        self.blink_threshold = 0.3
        self.blink_frames = 0
        self.blink_counter = 0
        self.last_blink_time = 0
        self.double_blink_threshold = 0.8  # seconds
        
        # Eye aspect ratio untuk deteksi kedip - Lebih sensitif
        self.ear_threshold = 0.3  # Dinaikkan untuk deteksi lebih mudah
        self.ear_consecutive_frames = 2  # Dikurangi untuk respon lebih cepat
        self.ear_counter = 0
        self.ear_history = []  # History EAR untuk smoothing
        self.ear_history_size = 5
        
        # Baseline EAR untuk kalibrasi otomatis
        self.baseline_ear = None
        self.baseline_frames = 0
        self.baseline_collection_frames = 60
        
        # Disable failsafe
        pyautogui.FAILSAFE = False
        
    def get_eye_aspect_ratio(self, eye_landmarks):
        """Menghitung Eye Aspect Ratio untuk deteksi kedip - Metode yang lebih akurat"""
        if len(eye_landmarks) < 6:
            return 0
            
        # Ambil 6 titik penting mata
        p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]
        
        # Jarak vertikal (2 pengukuran)
        vertical_1 = np.linalg.norm(p2 - p6)
        vertical_2 = np.linalg.norm(p3 - p5)
        
        # Jarak horizontal
        horizontal = np.linalg.norm(p1 - p4)
        
        # EAR formula dengan pembagi yang lebih stabil
        if horizontal > 0:
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        else:
            ear = 0
            
        return ear
    
    def extract_eye_landmarks(self, landmarks, eye_indices, img_width, img_height):
        """Ekstrak koordinat landmarks mata"""
        eye_points = []
        for idx in eye_indices:
            if idx < len(landmarks.landmark):
                x = int(landmarks.landmark[idx].x * img_width)
                y = int(landmarks.landmark[idx].y * img_height)
                eye_points.append([x, y])
        return np.array(eye_points)
    
    def get_iris_position(self, eye_landmarks):
        """Estimasi posisi iris dalam mata - Metode yang lebih akurat"""
        if len(eye_landmarks) < 6:
            return None
            
        # Hitung center mata menggunakan convex hull
        hull = cv2.convexHull(eye_landmarks)
        moments = cv2.moments(hull)
        
        if moments['m00'] != 0:
            eye_center_x = int(moments['m10'] / moments['m00'])
            eye_center_y = int(moments['m01'] / moments['m00'])
            eye_center = np.array([eye_center_x, eye_center_y])
        else:
            eye_center = np.mean(eye_landmarks, axis=0)
        
        # Hitung bounding box mata yang lebih akurat
        x_coords = eye_landmarks[:, 0]
        y_coords = eye_landmarks[:, 1]
        
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        
        eye_width = max_x - min_x
        eye_height = max_y - min_y
        
        # Estimasi posisi iris berdasarkan distribusi intensitas
        # (Ini adalah estimasi sederhana, untuk akurasi tinggi perlu deep learning)
        
        return {
            'center': eye_center,
            'width': eye_width,
            'height': eye_height,
            'bbox': (min_x, min_y, max_x, max_y)
        }
    
    def calculate_gaze_direction(self, left_iris, right_iris):
        """Menghitung arah pandangan berdasarkan posisi iris"""
        if left_iris is None or right_iris is None:
            return None
            
        # Rata-rata posisi kedua iris
        avg_iris_x = (left_iris['center'][0] + right_iris['center'][0]) / 2
        avg_iris_y = (left_iris['center'][1] + right_iris['center'][1]) / 2
        
        # Normalisasi posisi iris dalam bounding box mata
        left_bbox = left_iris['bbox']
        right_bbox = right_iris['bbox']
        
        # Hitung posisi relatif iris dalam mata (0-1)
        left_relative_x = (left_iris['center'][0] - left_bbox[0]) / (left_bbox[2] - left_bbox[0])
        left_relative_y = (left_iris['center'][1] - left_bbox[1]) / (left_bbox[3] - left_bbox[1])
        
        right_relative_x = (right_iris['center'][0] - right_bbox[0]) / (right_bbox[2] - right_bbox[0])
        right_relative_y = (right_iris['center'][1] - right_bbox[1]) / (right_bbox[3] - right_bbox[1])
        
        # Rata-rata relatif position
        avg_relative_x = (left_relative_x + right_relative_x) / 2
        avg_relative_y = (left_relative_y + right_relative_y) / 2
        
        return {
            'x': avg_relative_x,
            'y': avg_relative_y,
            'absolute_x': avg_iris_x,
            'absolute_y': avg_iris_y
        }
    
    def calibrate_gaze(self, gaze_data):
        """Kalibrasi sistem tracking mata"""
        if self.calibration_mode and gaze_data:
            current_step = self.calibration_steps[self.calibration_step]
            
            # Tambahkan data kalibrasi
            self.calibration_data[current_step].append([gaze_data['x'], gaze_data['y']])
            self.current_calibration_frames += 1
            
            # Cek apakah sudah cukup frame untuk step ini
            if self.current_calibration_frames >= self.frames_per_calibration:
                self.calibration_step += 1
                self.current_calibration_frames = 0
                
                # Cek apakah kalibrasi selesai
                if self.calibration_step >= len(self.calibration_steps):
                    self.calibration_mode = False
                    self.process_calibration_data()
                    print("Kalibrasi selesai! Eye controller siap digunakan.")
            
            return True
        return False
    
    def process_calibration_data(self):
        """Proses data kalibrasi untuk mapping"""
        self.calibration_points = {}
        
        for step, data in self.calibration_data.items():
            if data:
                # Hitung rata-rata posisi untuk setiap arah
                avg_x = np.mean([d[0] for d in data])
                avg_y = np.mean([d[1] for d in data])
                self.calibration_points[step] = [avg_x, avg_y]
        
        print("Calibration points:", self.calibration_points)
    
    def map_gaze_to_screen(self, gaze_data):
        """Mapping arah pandangan ke koordinat layar - Lebih responsif"""
        if self.calibration_mode or not hasattr(self, 'calibration_points'):
            return None
            
        # Gunakan interpolasi berdasarkan data kalibrasi
        center_x, center_y = self.calibration_points.get('center', [0.5, 0.5])
        
        # Hitung offset dari center
        offset_x = gaze_data['x'] - center_x
        offset_y = gaze_data['y'] - center_y
        
        # Dead zone untuk mengurangi noise
        if abs(offset_x) < self.dead_zone_x:
            offset_x = 0
        if abs(offset_y) < self.dead_zone_y:
            offset_y = 0
        
        # Mapping ke koordinat layar dengan scaling yang lebih agresif
        scale_factor_x = self.screen_width * 0.8  # 80% dari lebar layar
        scale_factor_y = self.screen_height * 0.8  # 80% dari tinggi layar
        
        screen_x = self.screen_width // 2 + (offset_x * scale_factor_x * self.movement_sensitivity)
        screen_y = self.screen_height // 2 + (offset_y * scale_factor_y * self.movement_sensitivity)
        
        # Batasi dalam area layar
        screen_x = max(0, min(self.screen_width - 1, int(screen_x)))
        screen_y = max(0, min(self.screen_height - 1, int(screen_y)))
        
        return (screen_x, screen_y)
    
    def smooth_cursor_movement(self, new_pos):
        """Smoothing gerakan cursor"""
        if self.prev_cursor_pos is None:
            self.prev_cursor_pos = new_pos
            return new_pos
        
        smooth_x = int(self.prev_cursor_pos[0] * self.smoothing_factor + 
                      new_pos[0] * (1 - self.smoothing_factor))
        smooth_y = int(self.prev_cursor_pos[1] * self.smoothing_factor + 
                      new_pos[1] * (1 - self.smoothing_factor))
        
        self.prev_cursor_pos = (smooth_x, smooth_y)
        return (smooth_x, smooth_y)
    
    def detect_blink(self, left_ear, right_ear):
        """Deteksi kedipan mata - Algoritma yang lebih sensitif"""
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Tambahkan ke history untuk smoothing
        self.ear_history.append(avg_ear)
        if len(self.ear_history) > self.ear_history_size:
            self.ear_history.pop(0)
        
        # Hitung rata-rata EAR dari history
        smooth_ear = np.mean(self.ear_history)
        
        # Kalibrasi baseline EAR otomatis
        if self.baseline_ear is None:
            if self.baseline_frames < self.baseline_collection_frames:
                self.baseline_frames += 1
                return False
            else:
                # Set baseline dari rata-rata history
                self.baseline_ear = np.mean(self.ear_history)
                print(f"Baseline EAR dikalibrasi: {self.baseline_ear:.3f}")
        
        # Dynamic threshold berdasarkan baseline
        dynamic_threshold = self.baseline_ear * 0.7  # 70% dari baseline
        
        # Deteksi blink dengan threshold dinamis
        if smooth_ear < dynamic_threshold:
            self.ear_counter += 1
            if self.ear_counter == 1:  # Frame pertama blink terdeteksi
                print(f"Blink detected! EAR: {smooth_ear:.3f}, Threshold: {dynamic_threshold:.3f}")
        else:
            if self.ear_counter >= self.ear_consecutive_frames:
                current_time = time.time()
                
                # Deteksi double blink
                if current_time - self.last_blink_time < self.double_blink_threshold:
                    self.double_blink_detected()
                    print("Double blink detected!")
                
                self.last_blink_time = current_time
                self.blink_counter += 1
                self.ear_counter = 0
                return True
            
            self.ear_counter = 0
        
        return False
    
    def double_blink_detected(self):
        """Aksi ketika double blink terdeteksi"""
        if not self.calibration_mode:
            pyautogui.click()
            print("Double blink detected - Mouse clicked!")
    
    def draw_eye_overlay(self, img, left_eye, right_eye, left_iris, right_iris):
        """Gambar overlay mata dan iris"""
        # Gambar outline mata
        if len(left_eye) > 0:
            cv2.polylines(img, [left_eye], True, (0, 255, 0), 1)
        if len(right_eye) > 0:
            cv2.polylines(img, [right_eye], True, (0, 255, 0), 1)
        
        # Gambar iris center
        if left_iris:
            center = tuple(map(int, left_iris['center']))
            cv2.circle(img, center, 3, (255, 0, 0), -1)
        
        if right_iris:
            center = tuple(map(int, right_iris['center']))
            cv2.circle(img, center, 3, (255, 0, 0), -1)
    
    def draw_calibration_ui(self, img):
        """Gambar UI kalibrasi"""
        h, w = img.shape[:2]
        
        if self.calibration_mode:
            current_step = self.calibration_steps[self.calibration_step]
            progress = self.current_calibration_frames / self.frames_per_calibration
            
            # Instruksi kalibrasi
            instructions = {
                'center': "Lihat ke tengah layar",
                'left': "Lihat ke kiri",
                'right': "Lihat ke kanan", 
                'up': "Lihat ke atas",
                'down': "Lihat ke bawah"
            }
            
            instruction = instructions.get(current_step, "")
            cv2.putText(img, f"Kalibrasi: {instruction}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Progress bar
            bar_width = 300
            bar_height = 20
            bar_x = (w - bar_width) // 2
            bar_y = 60
            
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
            
            # Step indicator
            step_text = f"Step {self.calibration_step + 1}/{len(self.calibration_steps)}"
            cv2.putText(img, step_text, (bar_x, bar_y + bar_height + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
    def draw_ui_elements(self, img, gaze_data=None, screen_pos=None):
        """Gambar elemen UI"""
        h, w = img.shape[:2]
        
        if not self.calibration_mode:
            # Status aktif
            cv2.putText(img, "Eye Controller Aktif", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Koordinat gaze dan screen
            if gaze_data:
                gaze_text = f"Gaze: ({gaze_data['x']:.2f}, {gaze_data['y']:.2f})"
                cv2.putText(img, gaze_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if screen_pos:
                screen_text = f"Screen: ({screen_pos[0]}, {screen_pos[1]})"
                cv2.putText(img, screen_text, (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Blink counter dan EAR info
            blink_text = f"Blinks: {self.blink_counter}"
            cv2.putText(img, blink_text, (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # EAR info untuk debugging
            if hasattr(self, 'current_ear'):
                ear_text = f"EAR: {self.current_ear:.3f}"
                cv2.putText(img, ear_text, (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if self.baseline_ear:
                baseline_text = f"Baseline: {self.baseline_ear:.3f}"
                cv2.putText(img, baseline_text, (10, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Kontrol
        controls = [
            "Double blink: Click mouse",
            "Press 'c': Recalibrate", 
            "Press 'r': Reset blink detection",
            "Press SPACE: Manual click",
            "Press 'q': Quit"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(img, control, (10, h - 60 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def run(self):
        """Fungsi utama aplikasi"""
        print("=== Eye Controller ===")
        print("Instruksi Kalibrasi:")
        print("1. Posisikan wajah dengan nyaman di depan kamera")
        print("2. Ikuti instruksi kalibrasi (lihat ke 5 arah)")
        print("3. Setelah kalibrasi selesai, gerakkan mata untuk kontrol cursor")
        print("4. Double blink untuk klik mouse")
        print("5. Tekan 'c' untuk kalibrasi ulang, 'q' untuk keluar")
        print()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Ekstrak landmarks mata
                    left_eye = self.extract_eye_landmarks(face_landmarks, self.LEFT_EYE, w, h)
                    right_eye = self.extract_eye_landmarks(face_landmarks, self.RIGHT_EYE, w, h)
                    
                    if len(left_eye) >= 6 and len(right_eye) >= 6:
                        # Hitung EAR untuk deteksi blink
                        left_ear = self.get_eye_aspect_ratio(left_eye[:6])
                        right_ear = self.get_eye_aspect_ratio(right_eye[:6])
                        
                        # Simpan EAR untuk debugging
                        self.current_ear = (left_ear + right_ear) / 2.0
                        
                        # Deteksi blink
                        blink_detected = self.detect_blink(left_ear, right_ear)
                        
                        # Visual feedback untuk blink
                        if blink_detected:
                            cv2.putText(frame, "BLINK!", (w//2 - 50, 50), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        
                        # Estimasi posisi iris
                        left_iris = self.get_iris_position(left_eye)
                        right_iris = self.get_iris_position(right_eye)
                        
                        # Hitung arah pandangan
                        gaze_data = self.calculate_gaze_direction(left_iris, right_iris)
                        
                        screen_pos = None
                        if gaze_data:
                            # Kalibrasi atau kontrol
                            if not self.calibrate_gaze(gaze_data):
                                # Mode kontrol normal
                                screen_pos = self.map_gaze_to_screen(gaze_data)
                                if screen_pos:
                                    smooth_pos = self.smooth_cursor_movement(screen_pos)
                                    pyautogui.moveTo(smooth_pos[0], smooth_pos[1])
                                    screen_pos = smooth_pos
                        
                        # Gambar overlay
                        self.draw_eye_overlay(frame, left_eye, right_eye, left_iris, right_iris)
                        self.draw_ui_elements(frame, gaze_data, screen_pos)
            
            # Gambar UI kalibrasi
            if self.calibration_mode:
                self.draw_calibration_ui(frame)
            
            cv2.imshow('Eye Controller', frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                            elif key == ord('c'):
                # Reset kalibrasi
                self.calibration_mode = True
                self.calibration_step = 0
                self.current_calibration_frames = 0
                self.calibration_data = {step: [] for step in self.calibration_steps}
                self.prev_cursor_pos = None
                # Reset blink calibration
                self.baseline_ear = None
                self.baseline_frames = 0
                self.ear_history = []
                print("Memulai kalibrasi ulang...")
            elif key == ord('r'):
                # Reset hanya blink detection
                self.baseline_ear = None
                self.baseline_frames = 0
                self.ear_history = []
                self.blink_counter = 0
                print("Reset deteksi blink...")
            elif key == ord(' '):
                # Manual click untuk testing
                if not self.calibration_mode:
                    pyautogui.click()
                    print("Manual click!")
        
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        controller = EyeController()
        controller.run()
    except KeyboardInterrupt:
        print("\nAplikasi dihentikan oleh user.")
    except Exception as e:
        print(f"Error: {e}")
        print("Pastikan dependencies telah terinstall:")
        print("pip install opencv-python mediapipe pyautogui numpy")

if __name__ == "__main__":
    main()