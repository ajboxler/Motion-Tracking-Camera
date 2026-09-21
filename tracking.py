# This project uses YOLO to detect a person in the XIAO ESP32-CAM web camera server, and sends 
# commands back to the ESP32 to actuate a DC motor according to the persons position in frame 

from ultralytics import YOLO
import cv2
import requests
import threading
import time


# Address for motor control endpoints 
ESP32_IP = "http://172.20.10.8:82"

# ESP32 stream address through port 81 
STREAM_URL = "http://172.20.10.8:81/stream"

# Size of frame for YOLO preprocessing 
IMGSZ = 160

# Frame skipping to run object detection on every other frame 
SKIP = 2

# Threshold for bounding box to minimize over correction 
OFFSET_THRESH = 30

# Minimum time between command attempts, in seconds 
MIN_CMD_INTERVAL = 0.12 

# Track the most recently attempted command and when it was sent 
last_cmd = None
last_cmd_time = 0.0

def send_command_async(cmd):
    # Schedule an HTTP GET without wiaintg for it in the main video loop. 

    global last_cmd, last_cmd_time

    now = time.monotonic()

    # Don't spam same command or send too often 
    if cmd == last_cmd and (now - last_cmd_time) < MIN_CMD_INTERVAL:
        return
    if (now - last_cmd_time) < MIN_CMD_INTERVAL:
        return

    last_cmd = cmd
    last_cmd_time = now

    def _worker():
        # Insert command into its HTTP endpoint 
        url = f"{ESP32_IP}/{cmd}"
        
        try:
            # Send the command from background thread 
            # Wait 50 ms for connection/read, print it if recieved 
            requests.get(url, timeout=0.05)
            print(f"Sent command: {cmd}")
        
        except Exception as e:
            # Print request failures if they happen 
            print(f"Failed to send {cmd}: {e}")

    threading.Thread(target=_worker, daemon=True).start()



# Model and video setup 

print("Loading YOLO model...")
model = YOLO("yolo11n.pt")

print("Opening video stream...")
cap = cv2.VideoCapture(STREAM_URL)

# Try to reduce internal buffering lag
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Could not open video stream from ESP32.")


frame_count = 0

print("Starting main loop. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    h, w = frame.shape[:2]
    frame_center_x = w // 2


    # FRAME SKIP: only run YOLO every SKIP frames 
 
    if frame_count % SKIP != 0:
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue


    # YOLO inderence (fast) 

    results = model(frame, imgsz=IMGSZ, verbose=False)
    boxes = results[0].boxes

    # If no detections, stop motor and just show frame 
    if boxes is None or len(boxes) == 0:
        send_command_async("stop")
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue


    # Pick one object to track 
    # Just using the first box, you can add class filtering 
  
    box = boxes[0]
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Draw bounding box 
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Compute object center and offset 
    object_center_x = (x1 + x2) // 2
    object_center_y = (y1 + y2) // 2
    offset_from_center = object_center_x - frame_center_x

    # Draw center point + info text 
    cv2.circle(frame, (object_center_x, object_center_y), 5, (255, 0, 0), -1)
    cv2.line(frame, (frame_center_x, 0), (frame_center_x, h), (0, 255, 255), 1)
    cv2.putText(
        frame,
        f"Offset: {offset_from_center}",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    print("Offset from center:", offset_from_center)

   
    # Deciding the command: LEFT / RIGHT / STOP 
    
    if offset_from_center < -OFFSET_THRESH:
        cmd = "left"
    elif offset_from_center > OFFSET_THRESH:
        cmd = "right"
    else:
        cmd = "stop"

    send_command_async(cmd)

    cv2.putText(
        frame,
        f"CMD: {cmd}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
