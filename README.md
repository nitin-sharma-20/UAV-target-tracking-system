# Edge-Deployable UAV Target Tracking System

An offline, air-gapped computer vision pipeline that simulates drone-based target tracking on low-power edge hardware. 

This project trains a YOLOv8-nano model on the VisDrone aerial dataset, mathematically optimizes the weights via ONNX, and runs real-time inference strictly on CPU hardware without PyTorch or GPU dependencies.

## 🚀 Features
* **Edge-Optimized Inference:** Uses `onnxruntime` (`CPUExecutionProvider`) to run complex vision models on standard laptop or embedded CPUs.
* **Air-Gapped Execution:** Operates entirely offline. No cloud APIs or internet connection required for inference.
* **Defense-Grade Pipeline:** Simulates the hardware constraints of onboard drone flight computers.
* **High-Accuracy Classification:** Detects 10 specific aerial target classes including vehicles, pedestrians, and bicycles using the VisDrone standard.

## 🛠️ Architecture
This project is divided into two distinct environments:

1. **Cloud Training (Google Colab / GPU):** * Fetches the VisDrone dataset via Ultralytics.
   * Trains `yolov8n.pt` using an NVIDIA T4 GPU.
   * Exports the trained graph to `best.onnx` to strip heavy framework overhead.
2. **Edge Simulation (Local Windows / CPU):**
   * Ingests the ONNX model and raw video feeds.
   * Preprocesses frames using pure OpenCV and NumPy.
   * Runs inference, applies Non-Maximum Suppression (NMS), and outputs annotated tracking coordinates in real-time.

## 📦 Prerequisites
You do not need a GPU or PyTorch to run the simulation. You only need:
* Python 3.8+
* A test video of drone footage named `test_video.mp4`

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/UAV-Edge-Tracker.git](https://github.com/YOUR_USERNAME/UAV-Edge-Tracker.git)
   cd UAV-Edge-Tracker
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
## Usage

1. Ensure `inference.py`, `best.onnx`, and `test_video.mp4` are in the same directory.
2. Execute the tracking pipeline:
```bash
python inference.py
```
*(Press `q` to safely terminate the video stream).*

## Technologies Used
* **Model:** YOLOv8-nano 
* **Dataset:** VisDrone 
* **Graph Optimization:** ONNX 
* **Inference Engine:** ONNX Runtime
* **Computer Vision:** OpenCV (cv2)
* **Matrix Math:** NumPy

## License
This project is licensed under the MIT License.

