# ESP32-CAM Motion Tracking

A wireless camera system that uses computer vision to control a motorized camera mount and keep a detected object near the center of the frame.

I built this project to explore how visual feedback can guide physical movement. An ESP32-CAM streams video to a laptop, where a Python program detects objects using YOLO. The program compares the object's position with the center of the image and sends movement commands back to the ESP32 over Wi-Fi.

One of the main challenges was getting the camera to respond to movement without constantly overcorrecting. The project uses a center tolerance and several approaches to reducing processing delays to make the feedback loop more responsive.

## How it works

1. The ESP32-CAM streams video over Wi-Fi.
2. OpenCV reads the stream, and YOLO runs on selected frames.
3. The program uses the first detected bounding box to calculate the object's horizontal position.
4. Python sends a `left`, `right`, or `stop` command over HTTP.
5. The ESP32 controls the motor through a motor driver, rotating the camera and changing the next image the program receives.

Object detection runs on the laptop. The ESP32 handles video streaming and motor commands.

### Control logic

The horizontal error is the difference between the bounding-box center and the frame center:

```python
object_center_x = (x1 + x2) / 2
offset = object_center_x - frame_width / 2
```

| Detection result | Requested command |
| --- | --- |
| Object is more than 30 pixels left of center | `left` |
| Object is more than 30 pixels right of center | `right` |
| Object is within 30 pixels of center | `stop` |
| No object is detected on a processed frame | `stop` |

The 30-pixel tolerance creates a **deadband** around the image center. Small position changes inside that region do not trigger another correction. Motor direction must match the camera mount so each command moves the object toward the center of the image.

## Hardware and software

| Component | Role |
| --- | --- |
| ESP32-CAM | Captures video and receives wireless commands |
| DC motor and camera mount | Rotate the camera horizontally |
| Motor driver and suitable power supply | Drive the motor from the ESP32's control signals |
| Laptop on the same Wi-Fi network | Runs object detection and control logic |
| Python with Ultralytics YOLO | Detects objects using `yolo11n.pt` |
| OpenCV | Reads and displays video frames |
| Requests and Python threading | Send HTTP commands in background threads |

Motor wiring, pin assignments, and power requirements depend on the driver and ESP32 firmware used for the build.

## Setup

### 1. Prepare the ESP32-CAM

Upload firmware configured for your camera board and Wi-Fi network. The firmware must provide the video stream and these motor-control routes:

| Route | Purpose |
| --- | --- |
| `http://<ESP32_IP>:81/stream` | Camera stream |
| `http://<ESP32_IP>/left` | Rotate left |
| `http://<ESP32_IP>/right` | Rotate right |
| `http://<ESP32_IP>/stop` | Stop the motor |

The motor routes are custom firmware behavior and must be implemented alongside the camera server. Confirm that the stream is accessible and that each motor command works before starting detection.

### 2. Install Python dependencies

From the project directory, create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install ultralytics opencv-python requests
```

Ultralytics provides the YOLO package; additional installation guidance is available in the [official documentation](https://docs.ultralytics.com/). Python includes `threading` and `time` in its standard library.

### 3. Configure the tracking program

Set both addresses to your ESP32-CAM's current IP address:

```python
ESP32_IP = "http://<ESP32_IP>"
STREAM_URL = "http://<ESP32_IP>:81/stream"
```

The documented tracking configuration uses:

```python
model = YOLO("yolo11n.pt")

IMGSZ = 160
SKIP = 2
OFFSET_THRESH = 30
MIN_CMD_INTERVAL = 0.12
```

### 4. Run the tracker

Run the project's Python tracking script. Replace the example filename below with its actual filename:

```bash
python tracking_script.py
```

Move a detectable object across the camera's view and observe the response. Press `q` in the video window to exit. Confirm the motor has stopped when ending a session.

## Reducing delay

| Setting or approach | Purpose and tradeoff |
| --- | --- |
| `IMGSZ = 160` | Uses a smaller inference input to reduce processing work; small objects may become harder to detect. |
| `SKIP = 2` | Runs detection every second frame while continuing to read the stream; reduces inference work but also the detection update rate. |
| Background HTTP requests | Keeps network requests from blocking the main video loop. |
| `MIN_CMD_INTERVAL = 0.12` | Spaces command attempts by at least 120 ms, limiting traffic to roughly 8.3 attempts per second. |
| 0.05-second request timeout | Limits how long a request thread waits; slow responses may time out. |

Frame skipping reduces detection work on the laptop. The camera stream continues transmitting video. These settings are tuning choices; end-to-end latency and tracking accuracy have not been benchmarked here.

## Current limitations

- The program selects `boxes[0]` without class filtering or persistent target IDs. It can switch between objects when multiple detections are present.
- Tracking controls horizontal rotation. Vertical tracking would require another axis and corresponding control logic.
- Movement uses directional commands with a deadband. PID control is a future extension.
- Lighting and Wi-Fi conditions affect tracking. Background requests do not guarantee command delivery or arrival order.
- A stop request after a missed detection still depends on the network. A firmware watchdog that stops the motor when commands stop arriving would improve handling of connection loss.

## Future improvements

- Add PID control to vary motor speed with position error and reduce overshoot.
- Add target selection and persistent tracking to follow the same object across frames.
- Measure detection rate, command delay, and centering error to compare control settings.
- Move inference to an onboard computer, such as a Raspberry Pi, so the system can operate independently of a laptop.
- Extend the mount to support pan and tilt.

## What I learned

This project gave me hands-on experience connecting computer vision with motor control. A correct detection was only one part of the system: the camera also needed to respond while that detection was still useful. Working through that relationship helped me better understand feedback control and how communication delays affect physical behavior.
