# Handwritten Digit Recognition AI

A web application for real-time handwritten digit recognition (0–9). Users draw a digit on an interactive canvas in their browser, and a trained PyTorch neural network predicts the drawn digit.

The project uses a **Multi-Layer Perceptron (MLP)** trained on the MNIST dataset, served via a **FastAPI** backend with a vanilla **HTML, CSS, and JavaScript** frontend.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [MLP Architecture](#mlp-architecture)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Installation and Setup (Windows)](#installation-and-setup-windows)
- [How to Run and Use](#how-to-run-and-use)
- [API Endpoints](#api-endpoints)
- [License](#license)

---

## Overview

This project provides an end-to-end interface for handwritten digit recognition:
1. The user draws a single digit (0 to 9) on an HTML5 canvas.
2. The frontend sends the drawing as a base64-encoded image to the FastAPI backend.
3. The backend preprocesses the drawing to match MNIST specifications:
   - Detects the bounding box of the drawn stroke.
   - Resizes the digit to fit within 20×20 pixels while preserving aspect ratio.
   - Centers the digit inside a 28×28 image based on its center of mass.
   - Normalizes pixel values from [0, 255] to [0.0, 1.0].
4. The trained PyTorch model evaluates the 28×28 tensor on CPU and returns the predicted digit, confidence percentage, and class probabilities.

---

## Features

- **Interactive Drawing Canvas**: 280×280 pixel canvas with adjustable stroke thickness (10px to 30px).
- **Digit Prediction**: Predicts digits from 0 to 9.
- **Confidence and Probabilities**: Shows the predicted digit, confidence score, and a bar chart of probabilities across all 10 digits (0–9).
- **Preprocessing Inspector**: Displays each step of the preprocessing pipeline:
  - Original canvas drawing (280×280 px)
  - Cropped bounding box of the stroke
  - Resized digit (~20×20 px)
  - Final centered 28×28 grayscale image
  - Center of mass coordinates and pixel value statistics
- **Keyboard Shortcuts**:
  - Press <kbd>Enter</kbd> to run prediction.
  - Press <kbd>C</kbd> to clear the canvas.
- **Input Validation**: Detects and rejects empty or too-small canvas drawings with a descriptive error message.

---

## Tech Stack

- **Deep Learning**: PyTorch
- **Backend**: FastAPI, Uvicorn
- **Image Processing**: Pillow (PIL), NumPy
- **Data Validation**: Pydantic
- **Frontend**: HTML5 Canvas, CSS3, JavaScript

---

## MLP Architecture

The model is a fully connected **Multi-Layer Perceptron (MLP)** implemented in PyTorch (`DigitMLP` in `backend/model.py`):

```text
Input: 28 × 28 Grayscale Image (784 pixels)
   │
   ▼
[ Flatten Layer ] ──────────────► Shape: (1, 784)
   │
   ▼
[ Linear(784, 128) + ReLU ] ────► Shape: (1, 128)
   │
   ▼
[ Linear(128, 64) + ReLU ] ─────► Shape: (1, 64)
   │
   ▼
[ Linear(64, 10) ] ─────────────► Shape: (1, 10) (Logits)
   │
   ▼
[ Softmax ] ────────────────────► 10 Class Probabilities (Digits 0–9)
```

### Layer Details

| Layer | PyTorch Module | Input Size | Output Size | Activation |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | `nn.Flatten` | 28 × 28 | 784 | None |
| **Hidden Layer 1** | `nn.Linear` | 784 | 128 | `nn.ReLU` |
| **Hidden Layer 2** | `nn.Linear` | 128 | 64 | `nn.ReLU` |
| **Output Layer** | `nn.Linear` | 64 | 10 | Softmax (applied during inference) |

---

## Model Performance

- **Dataset**: MNIST Handwritten Digit Dataset (60,000 training images, 10,000 test images)
- **Measured Test Accuracy**: **97.82%** (evaluated on the 10,000-image MNIST test dataset)
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam (learning rate = 0.001, batch size = 128, epochs = 10)
- **Model Checkpoint**: Saved as `models/best_digit_model.pth`

> **Note:** The 97.82% accuracy was measured specifically on the standardized MNIST test dataset. Real-world handwritten drawings on a canvas may vary depending on drawing style, stroke thickness, and positioning.

---

## Project Structure

```text
hand_written_digit/
├── backend/
│   ├── main.py              # FastAPI server, endpoints, and CORS configuration
│   ├── model.py             # DigitMLP architecture, preprocessor, and model loader
│   └── requirements.txt     # Python backend dependencies
├── frontend/
│   ├── index.html           # Web interface with canvas and results layout
│   ├── script.js            # Canvas drawing events, keyboard shortcuts, and API requests
│   └── style.css            # Styling for the application interface
├── models/
│   └── best_digit_model.pth # Trained PyTorch model checkpoint weights
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
```

---

## Installation and Setup (Windows)

Follow these step-by-step instructions to set up and run the application locally on Windows.

### 1. Prerequisites
- **Python 3.9+** (Python 3.10 to 3.13 supported). Verify in PowerShell:
  ```powershell
  python --version
  ```
- **Git** (optional, to clone).

### 2. Navigate to Project Directory
Open PowerShell or Command Prompt in the project root:
```powershell
cd hand_written_digit
```

### 3. Create a Virtual Environment
```powershell
python -m venv venv
```

### 4. Activate the Virtual Environment
- **Using PowerShell:**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If PowerShell displays a script execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once, then run the activate command again.)*

- **Using Command Prompt (`cmd`):**
  ```cmd
  venv\Scripts\activate.bat
  ```

### 5. Install Dependencies

You can install packages using either of the following approaches:

- **Standard Installation:**
  ```powershell
  pip install -r backend/requirements.txt
  ```

- **Lightweight CPU-Only Installation (Recommended for systems without an NVIDIA GPU):**
  To save disk space and avoid downloading unnecessary CUDA binaries, install the CPU-only PyTorch wheel first, followed by the requirements file:
  ```powershell
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -r backend/requirements.txt
  ```
  *(Because `torch>=2.0.0` is satisfied by the CPU wheel, pip will install only the remaining dependencies: `fastapi`, `uvicorn`, `numpy`, `pillow`, `python-multipart`, and `pydantic`.)*

### 6. Verify Model File
Ensure `best_digit_model.pth` exists in the `models/` directory:
```powershell
Test-Path models\best_digit_model.pth
```
*(Should return `True`)*

---

## How to Run and Use

### 1. Start the Backend Server
Run the following command from the project root:
```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

When started, the terminal will confirm:
```text
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### 2. Open in Browser
Open your web browser and navigate to:
```text
http://127.0.0.1:8000
```

### 3. Predict a Digit
1. Draw any single digit (0–9) on the black canvas.
2. Adjust the **Stroke** slider if you want a thicker or thinner brush.
3. Click **Predict Digit** (or press <kbd>Enter</kbd>).
4. View the predicted digit, confidence score, class probability bars, and preprocessing steps.
5. Click **Clear** (or press <kbd>C</kbd>) to reset the canvas.

---

## API Endpoints

The FastAPI backend exposes three main endpoints:

### 1. `GET /`
Serves the `frontend/index.html` single-page interface.

### 2. `GET /health`
Returns the status of the service and whether the PyTorch model checkpoint is loaded.

**Response Example (`200 OK`):**  
*(Illustrative example showing actual response structure)*
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_accuracy": 97.82,
  "device": "cpu"
}
```

### 3. `POST /predict`
Accepts a drawn digit image as a base64 string, runs preprocessing, and returns the model prediction.

**Request Body:**
```json
{
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAARgAAAE..."
}
```

**Response Example (`200 OK`):**  
*(Illustrative example showing actual response structure and debug fields)*
```json
{
  "success": true,
  "predicted_digit": 3,
  "confidence": 0.9942,
  "probabilities": [
    0.0001,
    0.0002,
    0.0003,
    0.9942,
    0.0001,
    0.0015,
    0.0001,
    0.0012,
    0.0018,
    0.0005
  ],
  "preprocessed_image": "iVBORw0KGgoAAAANSUhEUgAAABwAAAAc...",
  "debug": {
    "cropped_dimensions": [140, 180],
    "resized_dimensions": [16, 20],
    "final_dimensions": [28, 28],
    "original_center_of_mass": [9.85, 10.12],
    "final_center_of_mass": [13.50, 13.50],
    "horizontal_translation": 3.65,
    "vertical_translation": 3.38,
    "translation_applied": [3.65, 3.38],
    "final_bounding_box": [6, 4, 21, 24],
    "min_value": 0.0,
    "max_value": 1.0,
    "mean_value": 0.1428,
    "tensor_shape": [1, 1, 28, 28],
    "tensor_dtype": "torch.float32",
    "normalization_confirmed": true,
    "active_pixels": 112
  }
}
```

**Error Handling:**
- Returns `400 Bad Request` if the canvas is empty, the drawing is too small, the image data is corrupted, or the request JSON is malformed.
- Returns `422 Unprocessable Content` if the `image` field is missing or empty.
- Returns `503 Service Unavailable` if the model checkpoint failed to load.

---

## License

No license has been specified yet for this project.
