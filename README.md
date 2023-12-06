\# YoloV5_onnx

# YOLOv5n ONNX Face, Eye, and Mouth Detection

This project demonstrates the basic usage of YOLOv5n and ONNX for performing face, eye, and mouth detection tasks. It provides scripts to perform real-time detection using a camera (camera_onnx.py) and inference on images (detect_onnx.py).

## Prerequisites

Before running the scripts, make sure you have the following dependencies installed:

- Python 3.x
- PyTorch
- OpenCV

You can install the required Python packages by running the following command:

pip install -r requirements.txt

## Usage

### Real-time Camera Detection

To perform real-time detection using your camera, run the `camera_onnx.py` script:

python camera_onnx.py

This script will open your camera feed and display the detected faces, eyes, and mouths in real-time.

### Image Inference

To perform inference on an image, run the `detect_onnx.py` script:

python detect_onnx.py --image <path_to_image>

Replace `<path_to_image>` with the path to the image file you want to perform detection on. The script will display the image with bounding boxes around the detected faces, eyes, and mouths.

## Model and Weights

The YOLOv5n model and pre-trained weights used in this project can be found in the `weights` directory. Please ensure that the model and weights are present in the correct location before running the scripts.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
