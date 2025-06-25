import cv2
import mediapipe as mp
import numpy as np
import os
import platform
import time

class HandGestureDetector:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Gesture detection variables
        self.gesture_detected = False
        self.gesture_start_time = None
        self.required_hold_time = 2.0  # Hold gesture for 2 seconds
        
    def detect_middle_finger_gesture(self, landmarks):
        """
        Detect middle finger up gesture (other fingers down)
        Returns True if gesture is detected
        Thumb position is ignored (can be up or down)
        """
        # Get landmark positions
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        index_mcp = landmarks[5]
        
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        middle_mcp = landmarks[9]
        
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        ring_mcp = landmarks[13]
        
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        pinky_mcp = landmarks[17]
        
        # Check if middle finger is extended (tip higher than pip and mcp)
        middle_extended = (middle_tip.y < middle_pip.y) and (middle_tip.y < middle_mcp.y)
        
        # Check if other fingers are folded/down (EXCLUDING THUMB)
        # Index finger: tip should be lower than pip (folded)
        index_folded = index_tip.y > index_pip.y
        
        # Ring finger: tip should be lower than pip (folded)
        ring_folded = ring_tip.y > ring_pip.y
        
        # Pinky finger: tip should be lower than pip (folded)
        pinky_folded = pinky_tip.y > pinky_pip.y
        
        # Additional check: middle finger should be significantly higher than other fingertips
        # (EXCLUDING THUMB from comparison)
        middle_highest = (middle_tip.y < index_tip.y - 0.015) and \
                        (middle_tip.y < ring_tip.y - 0.015) and \
                        (middle_tip.y < pinky_tip.y - 0.015)
        
        # Combine all conditions (THUMB IS IGNORED)
        gesture_detected = (middle_extended and 
                          index_folded and 
                          ring_folded and 
                          pinky_folded and
                          middle_highest)
        
        return gesture_detected
    
    def shutdown_system(self):
        """
        Shutdown the system based on the operating system
        """
        system = platform.system()
        
        try:
            if system == "Windows":
                # Windows shutdown command with 10 second delay
                os.system("shutdown /s /t 10 /c \"System shutdown initiated by gesture\"")
                print("Windows shutdown initiated in 10 seconds...")
                
            elif system == "Darwin":  # macOS
                # macOS shutdown command with 1 minute delay
                os.system("sudo shutdown -h +1 \"System shutdown initiated by gesture\"")
                print("macOS shutdown initiated in 1 minute...")
                
            elif system == "Linux":
                # Linux shutdown command with 1 minute delay
                os.system("shutdown -h +1 \"System shutdown initiated by gesture\"")
                print("Linux shutdown initiated in 1 minute...")
                
            else:
                print(f"Unsupported operating system: {system}")
                return False
            
            return True
        except Exception as e:
            print(f"Error shutting down system: {e}")
            return False
    
    def cancel_shutdown(self):
        """
        Cancel pending shutdown (in case user wants to abort)
        """
        system = platform.system()
        
        try:
            if system == "Windows":
                os.system("shutdown /a")
                print("Windows shutdown cancelled")
                
            elif system == "Darwin":  # macOS
                os.system("sudo killall shutdown")
                print("macOS shutdown cancelled")
                
            elif system == "Linux":
                os.system("shutdown -c")
                print("Linux shutdown cancelled")
                
        except Exception as e:
            print(f"Error cancelling shutdown: {e}")
    
    def run_detection(self):
        """
        Main detection loop
        """
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Hand gesture detection started...")
        print("Show middle finger gesture and hold for 2 seconds to activate shutdown")
        print("Press 'q' to quit")
        print("Press 'c' to cancel pending shutdown")
        
        shutdown_initiated = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            results = self.hands.process(rgb_frame)
            
            current_time = time.time()
            gesture_detected_now = False
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Check for gesture
                    if self.detect_middle_finger_gesture(hand_landmarks.landmark):
                        gesture_detected_now = True
                        
                        if not self.gesture_detected:
                            # Gesture just started
                            self.gesture_detected = True
                            self.gesture_start_time = current_time
                        
                        # Calculate hold time
                        hold_time = current_time - self.gesture_start_time
                        
                        # Display countdown
                        countdown = max(0, self.required_hold_time - hold_time)
                        cv2.putText(frame, f"Shutdown in: {countdown:.1f}s", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(frame, "Gesture Detected!", 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Check if held long enough
                        if hold_time >= self.required_hold_time and not shutdown_initiated:
                            cv2.putText(frame, "INITIATING SHUTDOWN!", 
                                      (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            cv2.imshow('Hand Gesture Detection', frame)
                            cv2.waitKey(1000)  # Show message for 1 second
                            
                            # Shutdown system
                            if self.shutdown_system():
                                print("System shutdown initiated...")
                                shutdown_initiated = True
                            else:
                                print("Failed to initiate system shutdown")
            
            # Reset gesture detection if not detected
            if not gesture_detected_now:
                self.gesture_detected = False
                self.gesture_start_time = None
            
            # Display status
            if shutdown_initiated:
                cv2.putText(frame, "SHUTDOWN PENDING - Press 'c' to cancel", 
                          (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display instructions
            cv2.putText(frame, "Show middle finger to activate shutdown", 
                      (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit, 'c' to cancel shutdown", 
                      (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Hand Gesture Detection', frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and shutdown_initiated:
                self.cancel_shutdown()
                shutdown_initiated = False
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

def main():
    """
    Main function to run the hand gesture detector
    """
    try:
        detector = HandGestureDetector()
        detector.run_detection()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()