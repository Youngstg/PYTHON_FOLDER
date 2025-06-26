import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

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
            status_text = "Pointer Aktif - Gerakkan kepala untuk mengontrol cursor"
            color = (0, 255, 0)  # Hijau
        
        cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Instruksi
        instructions = [
            "Tekan 'c' untuk kalibrasi ulang",
            "Tekan 'q' untuk keluar",
            "Tekan SPACE untuk klik mouse"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(img, instruction, (10, h - 80 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Fungsi utama untuk menjalankan aplikasi"""
        print("=== Aplikasi Dahi Pointer Cursor ===")
        print("Instruksi:")
        print("1. Posisikan wajah Anda di tengah kamera")
        print("2. Tunggu proses kalibrasi selesai (30 frame)")
        print("3. Gerakkan kepala untuk mengontrol pointer")
        print("4. Tekan SPACE untuk klik mouse")
        print("5. Tekan 'c' untuk kalibrasi ulang")
        print("6. Tekan 'q' untuk keluar")
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
                    # Gambar face mesh (opsional, untuk debugging)
                    # self.mp_drawing.draw_landmarks(
                    #     frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                    #     None, self.mp_drawing_styles.get_default_face_mesh_contours_style())
                    
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
                                
                                # Gambar pointer dengan trail
                                self.draw_pointer_trail(frame, forehead_pos)
                                
                                # Gambar pointer utama
                                cv2.circle(frame, forehead_pos, self.pointer_radius, self.pointer_color, -1)
                                cv2.circle(frame, forehead_pos, self.pointer_radius + 2, (255, 255, 255), 2)
                                
                                # Tampilkan koordinat
                                coord_text = f"Screen: ({smooth_pos[0]}, {smooth_pos[1]})"
                                cv2.putText(frame, coord_text, (10, 60), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Gambar UI elements
            self.draw_ui_elements(frame)
            
            # Tampilkan frame
            cv2.imshow('Dahi Pointer Cursor', frame)
            
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
                print("Kalibrasi ulang...")
            elif key == ord(' '):
                # Klik mouse
                if not self.calibration_mode:
                    pyautogui.click()
                    print("Mouse clicked!")
        
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