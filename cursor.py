import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import math

class ForeheadCursor:
    def __init__(self):
        # Inisialisasi MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Inisialisasi drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Setup kamera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Konfigurasi pointer
        self.pointer_color = (0, 255, 0)  # Hijau
        self.pointer_radius = 8
        self.trail_points = []  # Untuk jejak pointer
        self.max_trail_length = 10
        
        # Kalibrasi area gerakan
        self.screen_width, self.screen_height = pyautogui.size()
        self.movement_sensitivity = 3
        self.smoothing_factor = 0.7
        self.prev_cursor_pos = None
        
        # Status tracking
        self.calibration_mode = True
        self.calibration_frames = 0
        self.calibration_positions = []
        self.center_point = None
        
        # Dwell Click Configuration
        self.dwell_enabled = True
        self.dwell_time = 2.0  # detik untuk dwell click
        self.dwell_threshold = 15  # pixel threshold untuk mendeteksi "diam"
        self.dwell_start_time = None
        self.dwell_position = None
        self.is_dwelling = False
        self.last_click_time = 0
        self.click_cooldown = 1.0  # cooldown setelah click (detik)
        
        # UI Animation
        self.dwell_progress = 0.0
        self.animation_angle = 0.0
        
        # Disable pyautogui failsafe
        pyautogui.FAILSAFE = False
        
    def get_forehead_point(self, landmarks, img_width, img_height):
        """Mendapatkan titik tengah dahi dari landmarks wajah"""
        # Indeks landmark untuk area dahi (bagian atas wajah)
        forehead_indices = [10, 151, 9, 10]  # Titik-titik di area dahi
        
        if landmarks:
            # Ambil koordinat landmark dahi
            forehead_points = []
            for idx in forehead_indices:
                if idx < len(landmarks.landmark):
                    x = int(landmarks.landmark[idx].x * img_width)
                    y = int(landmarks.landmark[idx].y * img_height)
                    forehead_points.append((x, y))
            
            if forehead_points:
                # Hitung titik tengah dahi
                avg_x = sum(p[0] for p in forehead_points) // len(forehead_points)
                avg_y = sum(p[1] for p in forehead_points) // len(forehead_points)
                return (avg_x, avg_y - 30)  # Offset ke atas untuk posisi dahi yang lebih akurat
        
        return None
    
    def calibrate_movement_area(self, forehead_pos):
        """Kalibrasi area gerakan untuk mapping yang lebih akurat"""
        if self.calibration_mode:
            self.calibration_positions.append(forehead_pos)
            self.calibration_frames += 1
            
            if self.calibration_frames >= 30:  # Kalibrasi selama 30 frame
                # Hitung titik tengah dari posisi kalibrasi
                avg_x = sum(p[0] for p in self.calibration_positions) // len(self.calibration_positions)
                avg_y = sum(p[1] for p in self.calibration_positions) // len(self.calibration_positions)
                self.center_point = (avg_x, avg_y)
                self.calibration_mode = False
                print("Kalibrasi selesai! Sekarang Anda dapat menggunakan pointer.")
            
            return True
        return False
    
    def map_to_screen_coordinates(self, forehead_pos, img_width, img_height):
        """Mapping posisi dahi ke koordinat layar"""
        if not self.center_point:
            return None
            
        # Hitung offset dari titik tengah
        offset_x = forehead_pos[0] - self.center_point[0]
        offset_y = forehead_pos[1] - self.center_point[1]
        
        # Mapping ke koordinat layar dengan sensitivitas
        screen_x = self.screen_width // 2 + (offset_x * self.movement_sensitivity)
        screen_y = self.screen_height // 2 + (offset_y * self.movement_sensitivity)
        
        # Batasi dalam area layar
        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))
        
        return (screen_x, screen_y)
    
    def smooth_cursor_movement(self, new_pos):
        """Menghaluskan gerakan cursor untuk mengurangi jitter"""
        if self.prev_cursor_pos is None:
            self.prev_cursor_pos = new_pos
            return new_pos
        
        # Smoothing menggunakan interpolasi linear
        smooth_x = int(self.prev_cursor_pos[0] * self.smoothing_factor + 
                      new_pos[0] * (1 - self.smoothing_factor))
        smooth_y = int(self.prev_cursor_pos[1] * self.smoothing_factor + 
                      new_pos[1] * (1 - self.smoothing_factor))
        
        self.prev_cursor_pos = (smooth_x, smooth_y)
        return (smooth_x, smooth_y)
    
    def calculate_distance(self, pos1, pos2):
        """Menghitung jarak euclidean antara dua titik"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def update_dwell_click(self, current_pos):
        """Update logika dwell click"""
        current_time = time.time()
        
        # Skip jika masih dalam cooldown setelah click
        if current_time - self.last_click_time < self.click_cooldown:
            self.dwell_start_time = None
            self.is_dwelling = False
            self.dwell_progress = 0.0
            return False
        
        if not self.dwell_enabled:
            return False
            
        # Jika belum ada posisi dwell atau posisi berubah signifikan
        if (self.dwell_position is None or 
            self.calculate_distance(current_pos, self.dwell_position) > self.dwell_threshold):
            
            # Reset dwell
            self.dwell_position = current_pos
            self.dwell_start_time = current_time
            self.is_dwelling = True
            self.dwell_progress = 0.0
            return False
        
        # Jika masih dalam area dwell
        if self.is_dwelling:
            elapsed_time = current_time - self.dwell_start_time
            self.dwell_progress = min(elapsed_time / self.dwell_time, 1.0)
            
            # Jika waktu dwell tercapai
            if elapsed_time >= self.dwell_time:
                self.perform_dwell_click()
                return True
                
        return False
    
    def perform_dwell_click(self):
        """Melakukan click otomatis"""
        pyautogui.click()
        print("Dwell click activated!")
        
        # Reset dwell state
        self.dwell_start_time = None
        self.is_dwelling = False
        self.dwell_progress = 0.0
        self.last_click_time = time.time()
    
    def draw_dwell_indicator(self, img, position):
        """Menggambar indikator lingkaran berputar untuk dwell click"""
        if not self.is_dwelling or self.dwell_progress <= 0:
            return
            
        # Update animation angle
        self.animation_angle += 8  # kecepatan rotasi
        if self.animation_angle >= 360:
            self.animation_angle = 0
            
        # Radius lingkaran indicator
        outer_radius = 25
        inner_radius = 20
        
        # Warna berdasarkan progress
        progress_color = (
            int(255 * (1 - self.dwell_progress)),  # Red menurun
            int(255 * self.dwell_progress),        # Green meningkat
            0                                       # Blue tetap 0
        )
        
        # Gambar lingkaran luar (background)
        cv2.circle(img, position, outer_radius, (100, 100, 100), 2)
        
        # Gambar arc progress
        if self.dwell_progress > 0:
            # Hitung sudut akhir berdasarkan progress
            end_angle = int(360 * self.dwell_progress)
            
            # Gambar arc menggunakan ellipse
            axes = (outer_radius - 2, outer_radius - 2)
            cv2.ellipse(img, position, axes, -90, 0, end_angle, progress_color, 3)
        
        # Gambar lingkaran berputar (loading indicator)
        if self.dwell_progress < 1.0:
            # Titik-titik berputar
            for i in range(8):
                angle = (self.animation_angle + i * 45) * math.pi / 180
                dot_x = int(position[0] + (outer_radius - 5) * math.cos(angle))
                dot_y = int(position[1] + (outer_radius - 5) * math.sin(angle))
                
                # Opacity berdasarkan posisi
                alpha = 1.0 - (i / 8.0)
                color_intensity = int(255 * alpha)
                dot_color = (color_intensity, color_intensity, color_intensity)
                
                cv2.circle(img, (dot_x, dot_y), 2, dot_color, -1)
        
        # Teks progress di tengah
        progress_text = f"{int(self.dwell_progress * 100)}%"
        text_size = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = position[0] - text_size[0] // 2
        text_y = position[1] + text_size[1] // 2
        
        cv2.putText(img, progress_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_pointer_trail(self, img, current_pos):
        """Menggambar jejak pointer untuk efek visual"""
        self.trail_points.append(current_pos)
        
        # Batasi panjang jejak
        if len(self.trail_points) > self.max_trail_length:
            self.trail_points.pop(0)
        
        # Gambar jejak dengan opacity yang menurun
        for i, point in enumerate(self.trail_points):
            alpha = (i + 1) / len(self.trail_points)
            radius = int(self.pointer_radius * alpha)
            color_intensity = int(255 * alpha)
            color = (0, color_intensity, 0)
            cv2.circle(img, point, radius, color, -1)
    
    def draw_ui_elements(self, img):
        """Menggambar elemen UI pada layar"""
        h, w = img.shape[:2]
        
        # Status text
        if self.calibration_mode:
            status_text = f"Kalibrasi... {self.calibration_frames}/30"
            color = (0, 255, 255)  # Kuning
        else:
            dwell_status = "ON" if self.dwell_enabled else "OFF"
            status_text = f"Pointer Aktif - Dwell Click: {dwell_status} ({self.dwell_time}s)"
            color = (0, 255, 0)  # Hijau
        
        cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Instruksi
        instructions = [
            "Tekan 'c' untuk kalibrasi ulang",
            "Tekan 'd' untuk toggle dwell click",
            "Tekan '+/-' untuk ubah waktu dwell",
            "Tekan 'q' untuk keluar",
            "Tekan SPACE untuk klik manual"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(img, instruction, (10, h - 125 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Info dwell click
        if self.dwell_enabled and not self.calibration_mode:
            dwell_info = f"Diam {self.dwell_time}s untuk klik otomatis"
            cv2.putText(img, dwell_info, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def run(self):
        """Fungsi utama untuk menjalankan aplikasi"""
        print("=== Aplikasi Dahi Pointer Cursor dengan Dwell Click ===")
        print("Instruksi:")
        print("1. Posisikan wajah Anda di tengah kamera")
        print("2. Tunggu proses kalibrasi selesai (30 frame)")
        print("3. Gerakkan kepala untuk mengontrol pointer")
        print("4. Diamkan pointer untuk klik otomatis (dwell click)")
        print("5. Tekan 'd' untuk toggle dwell click")
        print("6. Tekan '+' atau '-' untuk mengubah waktu dwell")
        print("7. Tekan 'c' untuk kalibrasi ulang")
        print("8. Tekan SPACE untuk klik manual")
        print("9. Tekan 'q' untuk keluar")
        print()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontal untuk efek mirror
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Konversi ke RGB untuk MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Dapatkan posisi dahi
                    forehead_pos = self.get_forehead_point(face_landmarks, w, h)
                    
                    if forehead_pos:
                        # Kalibrasi jika masih dalam mode kalibrasi
                        if self.calibrate_movement_area(forehead_pos):
                            # Gambar lingkaran kalibrasi
                            cv2.circle(frame, forehead_pos, self.pointer_radius + 5, (0, 255, 255), 2)
                        else:
                            # Mode normal - kontrol cursor
                            screen_pos = self.map_to_screen_coordinates(forehead_pos, w, h)
                            
                            if screen_pos:
                                # Smoothing gerakan cursor
                                smooth_pos = self.smooth_cursor_movement(screen_pos)
                                
                                # Gerakkan cursor mouse
                                pyautogui.moveTo(smooth_pos[0], smooth_pos[1])
                                
                                # Update dwell click
                                self.update_dwell_click(forehead_pos)
                                
                                # Gambar pointer dengan trail
                                self.draw_pointer_trail(frame, forehead_pos)
                                
                                # Gambar dwell indicator jika aktif
                                if self.dwell_enabled:
                                    self.draw_dwell_indicator(frame, forehead_pos)
                                
                                # Gambar pointer utama
                                cv2.circle(frame, forehead_pos, self.pointer_radius, self.pointer_color, -1)
                                cv2.circle(frame, forehead_pos, self.pointer_radius + 2, (255, 255, 255), 2)
                                
                                # Tampilkan koordinat
                                coord_text = f"Screen: ({smooth_pos[0]}, {smooth_pos[1]})"
                                cv2.putText(frame, coord_text, (10, 85), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # Reset dwell jika wajah tidak terdeteksi
                self.dwell_start_time = None
                self.is_dwelling = False
                self.dwell_progress = 0.0
            
            # Gambar UI elements
            self.draw_ui_elements(frame)
            
            # Tampilkan frame
            cv2.imshow('Dahi Pointer Cursor with Dwell Click', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Reset kalibrasi
                self.calibration_mode = True
                self.calibration_frames = 0
                self.calibration_positions = []
                self.center_point = None
                self.prev_cursor_pos = None
                # Reset dwell
                self.dwell_start_time = None
                self.is_dwelling = False
                self.dwell_progress = 0.0
                print("Kalibrasi ulang...")
            elif key == ord('d'):
                # Toggle dwell click
                self.dwell_enabled = not self.dwell_enabled
                status = "ON" if self.dwell_enabled else "OFF"
                print(f"Dwell click: {status}")
                # Reset dwell state
                self.dwell_start_time = None
                self.is_dwelling = False
                self.dwell_progress = 0.0
            elif key == ord('+') or key == ord('='):
                # Increase dwell time
                self.dwell_time = min(5.0, self.dwell_time + 0.5)
                print(f"Dwell time: {self.dwell_time}s")
            elif key == ord('-'):
                # Decrease dwell time
                self.dwell_time = max(1.0, self.dwell_time - 0.5)
                print(f"Dwell time: {self.dwell_time}s")
            elif key == ord(' '):
                # Klik mouse manual
                if not self.calibration_mode:
                    pyautogui.click()
                    print("Manual mouse click!")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Fungsi main untuk menjalankan aplikasi"""
    try:
        app = ForeheadCursor()
        app.run()
    except KeyboardInterrupt:
        print("\nAplikasi dihentikan oleh user.")
    except Exception as e:
        print(f"Error: {e}")
        print("Pastikan Anda telah menginstall dependencies yang diperlukan:")
        print("pip install opencv-python mediapipe pyautogui numpy")

if __name__ == "__main__":
    main()