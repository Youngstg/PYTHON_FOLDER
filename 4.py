import cv2
import numpy as np
import time
import mediapipe as mp
from pynput.keyboard import Key, Listener as KeyListener
from pynput import keyboard as pynput_keyboard
import threading

class GameHeadController:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Higher FPS for gaming
        
        # Gaming control parameters
        self.rotation_threshold = 12  # Lebih sensitif untuk gaming
        self.last_direction = "CENTER"
        self.current_direction = "CENTER"
        self.action_count = {"LEFT": 0, "RIGHT": 0, "CENTER": 0}
        
        # Smoothing untuk gaming (lebih responsif)
        self.rotation_history = []
        self.history_size = 3  # Lebih kecil untuk response cepat
        
        # Key control
        self.keyboard_controller = pynput_keyboard.Controller()
        self.is_pressing_left = False
        self.is_pressing_right = False
        
        # Gaming mode settings
        self.continuous_press = True  # Hold key saat geleng
        self.key_press_duration = 0.1  # Durasi press jika tidak continuous
        
        print("🎮 === GAME HEAD CONTROLLER - SUBWAY SURFERS ===")
        print("🎯 Optimized for Gaming!")
        print("📋 Controls:")
        print("   • GELENG KIRI  → ← Arrow Key (Move Left)")
        print("   • GELENG KANAN → → Arrow Key (Move Right)")
        print("   • POSISI TENGAH → Release Keys")
        print("🎮 Tips:")
        print("   • Pastikan game window aktif")
        print("   • Gerakan kepala yang smooth")
        print("   • Threshold: ±12° (sensitif)")
        print("================================================\n")

    def calculate_head_rotation(self, landmarks, frame_width, frame_height):
        """Hitung rotasi kepala dengan akurasi tinggi untuk gaming"""
        if not landmarks:
            return 0
        
        # Key landmarks untuk rotasi
        nose_tip = landmarks[10]      # Ujung hidung
        left_eye_outer = landmarks[33]   # Sudut luar mata kiri
        right_eye_outer = landmarks[263] # Sudut luar mata kanan
        left_mouth = landmarks[61]    # Sudut kiri mulut
        right_mouth = landmarks[291]  # Sudut kanan mulut
        left_cheek = landmarks[116]   # Pipi kiri
        right_cheek = landmarks[345]  # Pipi kanan
        
        # Konversi ke koordinat pixel
        nose_x = nose_tip.x * frame_width
        left_eye_x = left_eye_outer.x * frame_width
        right_eye_x = right_eye_outer.x * frame_width
        left_mouth_x = left_mouth.x * frame_width
        right_mouth_x = right_mouth.x * frame_width
        left_cheek_x = left_cheek.x * frame_width
        right_cheek_x = right_cheek.x * frame_width
        
        # Hitung beberapa indikator rotasi
        eye_asymmetry = right_eye_x - left_eye_x
        mouth_asymmetry = right_mouth_x - left_mouth_x
        cheek_asymmetry = right_cheek_x - left_cheek_x
        
        # Nose position relative to eye center
        eye_center_x = (left_eye_x + right_eye_x) / 2
        nose_offset = nose_x - eye_center_x
        
        # Combined rotation indicator
        rotation_indicator = (
            nose_offset * 0.4 +
            (mouth_asymmetry - eye_asymmetry) * 0.3 +
            (cheek_asymmetry - eye_asymmetry) * 0.3
        )
        
        # Convert to degrees
        face_width = abs(right_eye_x - left_eye_x)
        if face_width > 0:
            rotation_degrees = (rotation_indicator / face_width) * 60  # Max 60 degrees
        else:
            rotation_degrees = 0
            
        return rotation_degrees

    def smooth_rotation(self, rotation):
        """Smoothing khusus untuk gaming - lebih responsif"""
        self.rotation_history.append(rotation)
        if len(self.rotation_history) > self.history_size:
            self.rotation_history.pop(0)
        
        # Weighted average - prioritas ke data terbaru
        weights = [0.2, 0.3, 0.5]  # Bobot terbaru lebih besar
        if len(self.rotation_history) == 3:
            return sum(w * r for w, r in zip(weights, self.rotation_history))
        else:
            return sum(self.rotation_history) / len(self.rotation_history)

    def determine_direction(self, rotation_degrees):
        """Tentukan arah dengan threshold gaming"""
        if rotation_degrees > self.rotation_threshold:
            return "RIGHT"
        elif rotation_degrees < -self.rotation_threshold:
            return "LEFT"
        else:
            return "CENTER"

    def execute_game_control(self, direction):
        """Kontrol game nyata - kirim arrow keys"""
        try:
            # Release previous keys first
            if self.is_pressing_left:
                self.keyboard_controller.release(Key.left)
                self.is_pressing_left = False
                
            if self.is_pressing_right:
                self.keyboard_controller.release(Key.right)
                self.is_pressing_right = False
            
            # Press new key based on direction
            if direction == "LEFT":
                self.keyboard_controller.press(Key.left)
                self.is_pressing_left = True
                print("🎮 GAME: ← LEFT ARROW PRESSED")
                
            elif direction == "RIGHT":
                self.keyboard_controller.press(Key.right)
                self.is_pressing_right = True
                print("🎮 GAME: → RIGHT ARROW PRESSED")
                
            elif direction == "CENTER":
                print("🎮 GAME: ⚬ KEYS RELEASED")
            
        except Exception as e:
            print(f"❌ Control Error: {e}")

    def draw_gaming_interface(self, frame, rotation_degrees, direction):
        """Interface khusus untuk gaming"""
        height, width = frame.shape[:2]
        
        # Gaming HUD Style
        # Background panel
        cv2.rectangle(frame, (0, 0), (width, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (width, 100), (0, 255, 0), 2)
        
        # Title
        cv2.putText(frame, "SUBWAY SURFERS HEAD CONTROL", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Rotation info
        cv2.putText(frame, f"Head Rotation: {rotation_degrees:.1f}°", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Threshold indicator
        cv2.putText(frame, f"Threshold: ±{self.rotation_threshold}°", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        
        # Direction indicator (Big and Clear)
        center_x, center_y = width // 2, 200
        
        if direction == "LEFT":
            # Big left arrow
            cv2.arrowedLine(frame, (center_x, center_y), (center_x - 100, center_y), (0, 0, 255), 15)
            cv2.putText(frame, "← MOVING LEFT", (center_x - 150, center_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # Key indicator
            cv2.rectangle(frame, (50, 250), (150, 300), (0, 0, 255), -1)
            cv2.putText(frame, "LEFT", (70, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        elif direction == "RIGHT":
            # Big right arrow
            cv2.arrowedLine(frame, (center_x, center_y), (center_x + 100, center_y), (0, 0, 255), 15)
            cv2.putText(frame, "MOVING RIGHT →", (center_x + 20, center_y + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            # Key indicator
            cv2.rectangle(frame, (width-150, 250), (width-50, 300), (0, 0, 255), -1)
            cv2.putText(frame, "RIGHT", (width-140, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        else:
            # Center position
            cv2.circle(frame, (center_x, center_y), 40, (0, 255, 0), 8)
            cv2.putText(frame, "⚬ CENTER", (center_x - 60, center_y + 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Rotation meter (Visual feedback)
        meter_x, meter_y, meter_w = width - 150, 120, 120
        cv2.rectangle(frame, (meter_x - meter_w//2, meter_y - 10), 
                     (meter_x + meter_w//2, meter_y + 10), (100, 100, 100), 2)
        
        # Rotation position on meter
        rotation_pos = int((rotation_degrees / 60) * (meter_w // 2))
        rotation_pos = max(-meter_w//2, min(meter_w//2, rotation_pos))
        cv2.circle(frame, (meter_x + rotation_pos, meter_y), 8, (0, 255, 255), -1)
        
        # Threshold lines
        threshold_pos = int((self.rotation_threshold / 60) * (meter_w // 2))
        cv2.line(frame, (meter_x - threshold_pos, meter_y - 15), 
                (meter_x - threshold_pos, meter_y + 15), (0, 0, 255), 2)
        cv2.line(frame, (meter_x + threshold_pos, meter_y - 15), 
                (meter_x + threshold_pos, meter_y + 15), (0, 0, 255), 2)
        
        # Action counter
        y_pos = height - 80
        cv2.putText(frame, "Actions:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for cmd, count in self.action_count.items():
            y_pos += 20
            cv2.putText(frame, f"{cmd}: {count}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Gaming tips
        cv2.putText(frame, "Tips: Keep game window active, smooth head movements", 
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame

    def run(self):
        """Main game loop"""
        print("🚀 Starting Game Head Controller...")
        print("🎮 Ready for Subway Surfers!")
        print("📱 Open your game and start playing!")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Camera error")
                    break
                
                # Flip untuk mirror effect
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process face
                results = self.face_mesh.process(frame_rgb)
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Calculate rotation
                        rotation_degrees = self.calculate_head_rotation(
                            face_landmarks.landmark, frame.shape[1], frame.shape[0]
                        )
                        
                        # Smooth for gaming
                        smooth_rotation = self.smooth_rotation(rotation_degrees)
                        
                        # Determine direction
                        direction = self.determine_direction(smooth_rotation)
                        
                        # Execute control if direction changed
                        if direction != self.current_direction:
                            self.execute_game_control(direction)
                            self.action_count[direction] += 1
                            self.current_direction = direction
                        
                        # Draw gaming interface
                        frame = self.draw_gaming_interface(frame, smooth_rotation, direction)
                        
                else:
                    # No face detected
                    cv2.putText(frame, "NO FACE - PLACE FACE IN CAMERA", (100, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Release all keys when no face
                    if self.is_pressing_left or self.is_pressing_right:
                        if self.is_pressing_left:
                            self.keyboard_controller.release(Key.left)
                            self.is_pressing_left = False
                        if self.is_pressing_right:
                            self.keyboard_controller.release(Key.right)
                            self.is_pressing_right = False
                        print("🎮 GAME: Keys released (no face)")
                
                # Show frame
                cv2.imshow('Game Head Controller - Subway Surfers (Press Q to quit)', frame)
                
                # Quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️ Game controller stopped")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        # Release any pressed keys
        try:
            if self.is_pressing_left:
                self.keyboard_controller.release(Key.left)
            if self.is_pressing_right:
                self.keyboard_controller.release(Key.right)
        except:
            pass
        
        print("\n🎮 === GAMING SESSION STATS ===")
        total_actions = sum(self.action_count.values())
        print(f"Total Actions: {total_actions}")
        for direction, count in self.action_count.items():
            if total_actions > 0:
                percentage = (count / total_actions) * 100
                print(f"   {direction}: {count} ({percentage:.1f}%)")
        print("==============================")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Game controller closed successfully!")
        print("🎮 Thanks for playing!")

def main():
    print("🔍 Checking gaming setup...")
    
    try:
        import pynput
        print("✓ pynput (keyboard control)")
    except ImportError:
        print("❌ pynput not found!")
        print("📦 Install: pip install pynput")
        return
    
    try:
        import mediapipe
        print("✓ mediapipe (face tracking)")
    except ImportError:
        print("❌ mediapipe not found!")
        print("📦 Install: pip install mediapipe")
        return
    
    print("✅ All dependencies OK!")
    print("\n🎮 Starting Game Head Controller...")
    
    # Start the game controller
    controller = GameHeadController()
    controller.run()

if __name__ == "__main__":
    main()