import cv2
import numpy as np
import time
import mediapipe as mp

class HeadRotationRemote:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Head rotation parameters
        self.rotation_threshold = 15  # Degree threshold untuk trigger
        self.last_command = "CENTER"
        self.command_count = {"LEFT": 0, "RIGHT": 0, "CENTER": 0}
        self.rotation_history = []
        self.history_size = 5  # Untuk smoothing
        
        # Key facial landmarks untuk mendeteksi rotasi
        self.face_landmarks = [
            10,   # Nose tip
            151,  # Chin
            33,   # Left eye outer corner
            263,  # Right eye outer corner
            61,   # Left mouth corner
            291,  # Right mouth corner
        ]
        
        print("=== HEAD ROTATION REMOTE CONTROL ===")
        print("Instruksi:")
        print("- Hadapkan wajah ke kamera")
        print("- GELENGKAN kepala ke KIRI untuk kontrol KIRI")
        print("- GELENGKAN kepala ke KANAN untuk kontrol KANAN")
        print("- Kembali ke posisi tengah untuk CENTER")
        print("- Tekan 'q' untuk keluar")
        print("=====================================\n")

    def calculate_head_rotation(self, landmarks, frame_width, frame_height):
        """Menghitung rotasi kepala berdasarkan landmark wajah"""
        if not landmarks:
            return 0
        
        # Ambil koordinat landmark penting
        nose_tip = landmarks[10]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        chin = landmarks[151]
        
        # Konversi ke pixel coordinates
        nose_x = int(nose_tip.x * frame_width)
        nose_y = int(nose_tip.y * frame_height)
        
        left_eye_x = int(left_eye.x * frame_width)
        left_eye_y = int(left_eye.y * frame_height)
        
        right_eye_x = int(right_eye.x * frame_width)
        right_eye_y = int(right_eye.y * frame_height)
        
        left_mouth_x = int(left_mouth.x * frame_width)
        right_mouth_x = int(right_mouth.x * frame_width)
        
        chin_x = int(chin.x * frame_width)
        chin_y = int(chin.y * frame_height)
        
        # Hitung garis mata (eye line)
        eye_center_x = (left_eye_x + right_eye_x) // 2
        eye_center_y = (left_eye_y + right_eye_y) // 2
        
        # Hitung garis mulut
        mouth_center_x = (left_mouth_x + right_mouth_x) // 2
        
        # Hitung garis tengah wajah (dari mata ke dagu)
        face_center_line_angle = np.arctan2(chin_y - eye_center_y, chin_x - eye_center_x)
        
        # Hitung rotasi berdasarkan asimetri mata dan mulut
        eye_asymmetry = (right_eye_x - left_eye_x)
        mouth_nose_offset = mouth_center_x - nose_x
        
        # Kombinasi beberapa indikator untuk rotasi yang lebih akurat
        rotation_indicator = (mouth_nose_offset * 0.7) + (eye_asymmetry * 0.3)
        
        # Konversi ke derajat (normalisasi berdasarkan lebar wajah)
        face_width = abs(right_eye_x - left_eye_x)
        if face_width > 0:
            rotation_degrees = (rotation_indicator / face_width) * 45  # Max 45 derajat
        else:
            rotation_degrees = 0
            
        return rotation_degrees, (nose_x, nose_y), (eye_center_x, eye_center_y), (chin_x, chin_y)

    def smooth_rotation(self, rotation):
        """Smooth rotation dengan history untuk mengurangi noise"""
        self.rotation_history.append(rotation)
        if len(self.rotation_history) > self.history_size:
            self.rotation_history.pop(0)
        
        return sum(self.rotation_history) / len(self.rotation_history)

    def determine_direction(self, rotation_degrees):
        """Tentukan arah berdasarkan derajat rotasi"""
        if rotation_degrees > self.rotation_threshold:
            return "RIGHT"
        elif rotation_degrees < -self.rotation_threshold:
            return "LEFT"
        else:
            return "CENTER"

    def draw_face_info(self, frame, landmarks, rotation_degrees, direction):
        """Gambar informasi wajah dan rotasi pada frame"""
        if not landmarks:
            return frame
            
        frame_height, frame_width = frame.shape[:2]
        
        # Gambar landmark penting
        for idx in self.face_landmarks:
            landmark = landmarks[idx]
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        # Gambar pointer di jidat (antara mata dan rambut)
        nose_tip = landmarks[10]
        forehead_x = int(nose_tip.x * frame_width)
        forehead_y = int(nose_tip.y * frame_height) - 40
        
        cv2.circle(frame, (forehead_x, forehead_y), 8, (255, 0, 0), -1)
        cv2.putText(frame, "HEAD POINTER", (forehead_x-50, forehead_y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Tampilkan informasi rotasi
        cv2.putText(frame, f"Rotation: {rotation_degrees:.1f}°", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Direction: {direction}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Threshold: ±{self.rotation_threshold}°", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        
        # Gambar indikator arah
        center_x, center_y = frame_width // 2, 120
        
        if direction == "LEFT":
            cv2.arrowedLine(frame, (center_x, center_y), (center_x - 60, center_y), (0, 0, 255), 8)
            cv2.putText(frame, "GELENG KIRI", (center_x - 80, center_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif direction == "RIGHT":
            cv2.arrowedLine(frame, (center_x, center_y), (center_x + 60, center_y), (0, 0, 255), 8)
            cv2.putText(frame, "GELENG KANAN", (center_x + 10, center_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.circle(frame, (center_x, center_y), 25, (0, 255, 0), 4)
            cv2.putText(frame, "TENGAH", (center_x - 30, center_y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Gambar rotation meter
        meter_x, meter_y = 520, 50
        meter_width = 100
        cv2.rectangle(frame, (meter_x - meter_width//2, meter_y - 10), 
                     (meter_x + meter_width//2, meter_y + 10), (100, 100, 100), 2)
        
        # Indikator rotasi pada meter
        rotation_pos = int((rotation_degrees / 45) * (meter_width // 2))
        rotation_pos = max(-meter_width//2, min(meter_width//2, rotation_pos))
        cv2.circle(frame, (meter_x + rotation_pos, meter_y), 5, (0, 255, 255), -1)
        
        return frame

    def send_control_command(self, direction, rotation_degrees):
        """Kirim perintah kontrol dan tampilkan di terminal"""
        if direction != self.last_command:
            self.command_count[direction] += 1
            timestamp = time.strftime("%H:%M:%S")
            
            print(f"[{timestamp}] ROTASI KEPALA: {rotation_degrees:.1f}° → KONTROL: {direction}")
            
            if direction == "LEFT":
                print(">>> AKSI: GELENG KIRI TERDETEKSI <<<")
                print("    🡸 Simulasi: Remote LEFT (Channel Down / Volume Down)")
            elif direction == "RIGHT":
                print(">>> AKSI: GELENG KANAN TERDETEKSI <<<")
                print("    🡺 Simulasi: Remote RIGHT (Channel Up / Volume Up)")
            else:
                print(">>> AKSI: POSISI TENGAH <<<")
                print("    ⚬ Simulasi: Remote NETRAL")
            
            print("-" * 60)
            self.last_command = direction

    def run(self):
        """Jalankan sistem head rotation tracking"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Tidak dapat membaca dari kamera")
                    break
                
                # Flip frame horizontal untuk efek mirror
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Deteksi face mesh
                results = self.face_mesh.process(frame_rgb)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Hitung rotasi kepala
                        rotation_data = self.calculate_head_rotation(
                            face_landmarks.landmark, frame.shape[1], frame.shape[0]
                        )
                        
                        if len(rotation_data) == 4:
                            rotation_degrees, nose_pos, eye_pos, chin_pos = rotation_data
                            
                            # Smooth rotation
                            smooth_rotation = self.smooth_rotation(rotation_degrees)
                            
                            # Tentukan arah
                            direction = self.determine_direction(smooth_rotation)
                            
                            # Kirim perintah kontrol
                            self.send_control_command(direction, smooth_rotation)
                            
                            # Gambar informasi pada frame
                            frame = self.draw_face_info(frame, face_landmarks.landmark, 
                                                      smooth_rotation, direction)
                else:
                    cv2.putText(frame, "WAJAH TIDAK TERDETEKSI", (200, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Tampilkan statistik
                y_pos = 400
                for cmd, count in self.command_count.items():
                    cv2.putText(frame, f"{cmd}: {count}x", (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_pos += 25
                
                # Tampilkan frame
                cv2.imshow('Head Rotation Remote Control', frame)
                
                # Keluar jika tekan 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nProgram dihentikan oleh user")
        
        finally:
            self.cleanup()

    def cleanup(self):
        """Bersihkan resources"""
        print("\n=== STATISTIK ROTASI KEPALA ===")
        total_commands = sum(self.command_count.values())
        for cmd, count in self.command_count.items():
            percentage = (count / total_commands * 100) if total_commands > 0 else 0
            print(f"{cmd}: {count} kali ({percentage:.1f}%)")
        print("==============================")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Program selesai. Terima kasih!")

def main():
    print("Checking dependencies...")
    try:
        import mediapipe
        print(f"✓ MediaPipe Version: {mediapipe.__version__}")
        print(f"✓ OpenCV Version: {cv2.__version__}")
    except ImportError:
        print("❌ MediaPipe tidak terinstall!")
        print("Install dengan: pip install mediapipe")
        return
    
    # Inisialisasi dan jalankan head rotation remote
    remote = HeadRotationRemote()
    remote.run()

if __name__ == "__main__":
    main()