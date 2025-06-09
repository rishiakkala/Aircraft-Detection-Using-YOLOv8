# Military Aircraft Detection using YOLOv8

This project implements an object detection system for military and civilian aircraft using the YOLOv8 model. The system is trained to identify 43 different types of aircraft with high accuracy.

## Performance Metrics

The model achieves the following performance on the validation dataset:
- **Precision**: 0.6958
- **Recall**: 0.7982
- **mAP50**: 0.8919
- **mAP50-95**: 0.7331

## Dataset

The dataset contains a diverse range of military and civilian aircraft organized into the following categories:

### 1. Fighter Jets
- F35, F22, F16, F15, F18, F14, F4, F117
- Su57, Su34
- J20
- JAS39 (Gripen)
- EF2000 (Eurofighter Typhoon)
- Rafale
- Mirage2000
- Tornado
- MiG31
- YF23

### 2. Bombers
- B1, B2, B52
- Tu95, Tu160

### 3. Transport/Cargo Aircraft
- C2, C5, C17, C130
- A400M

### 4. Special Purpose Aircraft
- E2 (Hawkeye - Early Warning)
- E7 (Wedgetail - AWACS)
- P3 (Orion - Maritime Patrol)
- U2 (Spy Plane)
- SR71 (Blackbird)
- RQ4 (Global Hawk - UAV)
- MQ9 (Reaper - UAV)

### 5. Attack Aircraft
- A10 (Thunderbolt/Warthog)
- AV8B (Harrier)

### 6. Other Aircraft
- V22 (Osprey - Tiltrotor)
- US2 (Amphibious Aircraft)
- Be200 (Amphibious Aircraft)
- AG600 (Amphibious Aircraft)
- Vulcan (Historic bomber)
- XB70 (Experimental bomber)

## Project Structure

```
├── military/
│   ├── aircraft_names.yaml    # Class names and dataset configuration
│   ├── aircraft_train.txt     # List of training images
│   ├── aircraft_val.txt       # List of validation images
│   ├── images/                # Directory containing images
│   │   ├── aircraft_train/    # Training images
│   │   └── aircraft_val/      # Validation images
│   └── labels/                # Directory containing annotation labels
│       ├── aircraft_train/    # Training labels
│       └── aircraft_val/      # Validation labels
├── models/                    # Saved model weights
├── runs/                      # Training results and visualizations
└── train_aircraft_yolov8.py   # Main training script
```

## Requirements

Install the required packages:

```bash
pip install ultralytics
pip install matplotlib
pip install numpy
pip install torch torchvision
```

For GPU acceleration (recommended):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## How to Run

1. Clone this repository
2. Install the required packages
3. Run the training script:

```bash
python train_aircraft_yolov8.py
```

### Training Process

The script will:
1. Set up GPU if available
2. Create a dataset YAML configuration file
3. Train the YOLOv8 model (default: YOLOv8s)
4. Validate the model and calculate performance metrics
5. Save the trained model to the `models/` directory
6. Generate visualization plots in the `runs/` directory

## Customization

You can modify the following parameters in the `main()` function of `train_aircraft_yolov8.py`:

- `model_size`: Model size ('n', 's', 'm', 'l', 'x')
- `epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `img_size`: Input image size

## Model Information

This project uses YOLOv8 (You Only Look Once version 8), a state-of-the-art object detection model that provides:
- Real-time inference capabilities
- High accuracy for object detection tasks
- Support for various model sizes to balance speed and accuracy
- Easy integration and deployment

## Output

After training, you'll find:
- Trained model weights in the `models/` directory
- Training metrics and visualizations in the `runs/` directory
- Validation results showing model performance

## Contributing

Feel free to contribute to this project by:
- Adding more aircraft classes
- Improving the dataset quality
- Optimizing the training parameters
- Adding inference scripts for real-time detection

