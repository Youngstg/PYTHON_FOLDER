import cv2
import numpy as np
import time

class HeadTrackingRemote:
    def __init__(self):
        # Initialize face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Control parameters
        self.center_x = 320  # Center of frame
        self.threshold = 50  # Sensitivity threshold
        self.last_command = "CENTER"
        self.command_count = {"LEFT": 0, "RIGHT": 0, "CENTER": 0}
        
        print("=== HEAD TRACKING REMOTE CONTROL ===")
        print("Instruksi:")
        print("- Hadapkan wajah ke kamera")
        print("- Putar kepala ke kiri untuk kontrol KIRI")
        print("- Putar kepala ke kanan untuk kontrol KANAN")
        print("- Tekan 'q' untuk keluar")
        print("=====================================\n")

    def detect_head_direction(self, frame):
        """Deteksi arah kepala berdasarkan posisi wajah dan mata"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        direction = "CENTER"
        
        for (x, y, w, h) in faces:
            # Gambar kotak di sekitar wajah
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Hitung titik tengah wajah
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # Gambar pointer (lingkaran) di jidat
            forehead_y = y + int(h * 0.3)  # Posisi jidat
            cv2.circle(frame, (face_center_x, forehead_y), 8, (0, 255, 0), -1)
            cv2.putText(frame, "POINTER", (face_center_x-30, forehead_y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Deteksi mata untuk konfirmasi arah
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            # Gambar kotak di mata
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)
            
            # Tentukan arah berdasarkan posisi wajah relatif terhadap center
            offset = face_center_x - self.center_x
            
            if offset < -self.threshold:
                direction = "LEFT"
            elif offset > self.threshold:
                direction = "RIGHT"
            else:
                direction = "CENTER"
            
            # Tampilkan informasi pada frame
            cv2.line(frame, (self.center_x, 0), (self.center_x, frame.shape[0]), (255, 255, 255), 1)
            cv2.putText(frame, f"Offset: {offset}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Direction: {direction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Gambar indikator arah
            if direction == "LEFT":
                cv2.arrowedLine(frame, (100, 100), (50, 100), (0, 0, 255), 5)
            elif direction == "RIGHT":
                cv2.arrowedLine(frame, (540, 100), (590, 100), (0, 0, 255), 5)
            else:
                cv2.circle(frame, (320, 100), 20, (0, 255, 0), 3)
        
        return direction, frame

    def send_control_command(self, direction):
        """Kirim perintah kontrol dan tampilkan di terminal"""
        if direction != self.last_command:
            self.command_count[direction] += 1
            timestamp = time.strftime("%H:%M:%S")
            
            print(f"[{timestamp}] KONTROL: {direction}")
            
            if direction == "LEFT":
                print(">>> AKSI: Gerak ke KIRI <<<")
                print("    (Simulasi: TV Channel Down / Volume Down)")
            elif direction == "RIGHT":
                print(">>> AKSI: Gerak ke KANAN <<<")
                print("    (Simulasi: TV Channel Up / Volume Up)")
            else:
                print(">>> AKSI: NETRAL <<<")
                print("    (Simulasi: Tidak ada aksi)")
            
            print("-" * 50)
            self.last_command = direction

    def run(self):
        """Jalankan sistem head tracking"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Tidak dapat membaca dari kamera")
                    break
                
                # Flip frame horizontal untuk efek mirror
                frame = cv2.flip(frame, 1)
                
                # Deteksi arah kepala
                direction, processed_frame = self.detect_head_direction(frame)
                
                # Kirim perintah kontrol
                self.send_control_command(direction)
                
                # Tampilkan statistik di frame
                y_pos = 400
                for cmd, count in self.command_count.items():
                    cv2.putText(processed_frame, f"{cmd}: {count}", (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_pos += 25
                
                # Tampilkan frame
                cv2.imshow('Head Tracking Remote Control', processed_frame)
                
                # Keluar jika tekan 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nProgram dihentikan oleh user")
        
        finally:
            self.cleanup()

    def cleanup(self):
        """Bersihkan resources"""
        print("\n=== STATISTIK PENGGUNAAN ===")
        for cmd, count in self.command_count.items():
            print(f"{cmd}: {count} kali")
        print("===========================")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Program selesai. Terima kasih!")

def main():
    # Cek apakah OpenCV terinstall dengan benar
    print(f"OpenCV Version: {cv2.__version__}")
    
    # Inisialisasi dan jalankan head tracking remote
    remote = HeadTrackingRemote()
    remote.run()

if __name__ == "__main__":
    main()