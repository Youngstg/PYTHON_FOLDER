import cv2
import numpy as np
import pyautogui
import time
from collections import deque

class EyeCursorController:
    def __init__(self):
        # Initialize face and eye cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Eye position tracking
        self.eye_positions = deque(maxlen=5)  # Smooth cursor movement
        
        # Blink detection
        self.blink_counter = 0
        self.blink_threshold = 2
        self.last_blink_time = 0
        self.blink_cooldown = 1.0  # 1 second cooldown between blink actions
        self.consecutive_blinks = 0
        self.blink_window = 1.5  # Time window for detecting consecutive blinks
        self.blink_times = deque(maxlen=5)
        
        # Calibration
        self.calibrated = False
        self.calibration_points = []
        self.eye_bounds = None
        
        # Sensitivity settings
        self.cursor_sensitivity = 2.0
        self.smooth_factor = 0.3
        
        # Disable pyautogui failsafe
        pyautogui.FAILSAFE = False
        
    def detect_eyes(self, frame):
        """Detect eyes in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        eyes = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            detected_eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in detected_eyes:
                eyes.append((x + ex, y + ey, ew, eh))
                
        return eyes
    
    def get_pupil_position(self, eye_region):
        """Get pupil position within eye region"""
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_eye, (7, 7), 0)
        
        # Find the darkest point (pupil)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        
        return min_loc
    
    def is_eye_closed(self, eye_region):
        """Simple blink detection based on eye region analysis"""
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate the variance of pixel intensities
        # Closed eyes have less variance
        variance = np.var(gray_eye)
        
        # Threshold for closed eye detection
        return variance < 200
    
    def calibrate(self, eye_center):
        """Simple calibration to map eye movement to screen"""
        if not self.calibrated:
            self.calibration_points.append(eye_center)
            
            if len(self.calibration_points) >= 30:  # Collect 30 points
                points = np.array(self.calibration_points)
                
                # Calculate eye movement bounds
                min_x, min_y = np.min(points, axis=0)
                max_x, max_y = np.max(points, axis=0)
                
                self.eye_bounds = {
                    'min_x': min_x, 'max_x': max_x,
                    'min_y': min_y, 'max_y': max_y
                }
                
                self.calibrated = True
                print("Kalibrasi selesai! Gerakkan mata untuk mengontrol kursor.")
                
        return self.calibrated
    
    def map_to_screen(self, eye_pos):
        """Map eye position to screen coordinates"""
        if not self.calibrated or not self.eye_bounds:
            return None
            
        # Normalize eye position
        norm_x = (eye_pos[0] - self.eye_bounds['min_x']) / (self.eye_bounds['max_x'] - self.eye_bounds['min_x'])
        norm_y = (eye_pos[1] - self.eye_bounds['min_y']) / (self.eye_bounds['max_y'] - self.eye_bounds['min_y'])
        
        # Clamp values
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))
        
        # Map to screen
        screen_x = int(norm_x * self.screen_width)
        screen_y = int(norm_y * self.screen_height)
        
        return (screen_x, screen_y)
    
    def smooth_cursor_movement(self, new_pos):
        """Smooth cursor movement using moving average"""
        if new_pos is None:
            return
            
        self.eye_positions.append(new_pos)
        
        if len(self.eye_positions) >= 3:
            # Calculate weighted average
            positions = list(self.eye_positions)
            avg_x = sum(pos[0] for pos in positions) / len(positions)
            avg_y = sum(pos[1] for pos in positions) / len(positions)
            
            # Move cursor smoothly
            current_x, current_y = pyautogui.position()
            new_x = int(current_x + (avg_x - current_x) * self.smooth_factor)
            new_y = int(current_y + (avg_y - current_y) * self.smooth_factor)
            
            pyautogui.moveTo(new_x, new_y)
    
    def handle_blink(self):
        """Handle blink detection and actions"""
        current_time = time.time()
        
        # Add blink time to queue
        self.blink_times.append(current_time)
        
        # Count blinks in the last 1.5 seconds
        recent_blinks = [t for t in self.blink_times if current_time - t <= self.blink_window]
        
        # Check for double blink
        if len(recent_blinks) >= 2:
            if current_time - self.last_blink_time > self.blink_cooldown:
                print("Double blink detected! Performing left click...")
                pyautogui.click()
                self.last_blink_time = current_time
                self.blink_times.clear()  # Clear blink history
    
    def run(self):
        """Main execution loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Tidak dapat membuka kamera")
            return
        
        print("Memulai deteksi mata...")
        print("Tekan 'q' untuk keluar")
        print("Lakukan kalibrasi dengan menggerakkan mata ke berbagai arah...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect eyes
            eyes = self.detect_eyes(frame)
            
            if len(eyes) >= 2:  # At least 2 eyes detected
                # Use the first two eyes (left and right)
                left_eye = eyes[0]
                right_eye = eyes[1]
                
                # Extract eye regions
                left_eye_region = frame[left_eye[1]:left_eye[1]+left_eye[3], 
                                       left_eye[0]:left_eye[0]+left_eye[2]]
                right_eye_region = frame[right_eye[1]:right_eye[1]+right_eye[3], 
                                        right_eye[0]:right_eye[0]+right_eye[2]]
                
                # Check for blinks
                left_closed = self.is_eye_closed(left_eye_region)
                right_closed = self.is_eye_closed(right_eye_region)
                
                if left_closed and right_closed:
                    cv2.putText(frame, "BLINK DETECTED", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    self.handle_blink()
                
                # Get pupil positions for cursor control
                if not left_closed and not right_closed:
                    left_pupil = self.get_pupil_position(left_eye_region)
                    right_pupil = self.get_pupil_position(right_eye_region)
                    
                    # Calculate average eye position
                    avg_eye_x = (left_eye[0] + left_pupil[0] + right_eye[0] + right_pupil[0]) // 2
                    avg_eye_y = (left_eye[1] + left_pupil[1] + right_eye[1] + right_pupil[1]) // 2
                    
                    eye_center = (avg_eye_x, avg_eye_y)
                    
                    # Calibration phase
                    if not self.calibrated:
                        self.calibrate(eye_center)
                        cv2.putText(frame, f"Calibrating... {len(self.calibration_points)}/30", 
                                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    else:
                        # Map to screen coordinates and move cursor
                        screen_pos = self.map_to_screen(eye_center)
                        self.smooth_cursor_movement(screen_pos)
                        
                        cv2.putText(frame, "TRACKING ACTIVE", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw eye rectangles
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            
            # Show frame
            cv2.imshow('Eye Cursor Control', frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import pyautogui
    except ImportError:
        print("Installing pyautogui...")
        import subprocess
        subprocess.check_call(["pip", "install", "pyautogui"])
        import pyautogui
    
    controller = EyeCursorController()
    controller.run()