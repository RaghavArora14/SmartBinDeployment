# Automated Waste Segregating Dustbin

An IoT-enabled smart waste management system that automatically segregates waste into biodegradable and non-biodegradable categories using machine learning and real-time monitoring.

## Features

- Automated waste classification using Convolutional Neural Networks (96.13% accuracy)
- Real-time bin capacity monitoring using ultrasonic sensors
- Web-based dashboard for waste management analytics
- Automated notifications for bin capacity alerts
- Integrated servo motor system for physical waste segregation
- Cloud-based processing using AWS EC2

## System Architecture

### Hardware Components
- Raspberry Pi 3B
- IR Sensor for waste detection
- Ultrasonic Sensors (HC-SR04) for bin level monitoring
- Servo Motor (MG 996R) for waste sorting
- USB Camera (640x480 resolution)

### Software Stack
- Backend: AWS EC2 (t3.micro instance)
- Database: SQLite
- Web Framework: Flask
- Machine Learning: CNN model with 9.86M trainable parameters
- Dataset: Kaggle's "Non- and Biodegradable Waste Dataset"

## Dashboard Features

- Real-time bin capacity monitoring
- Waste distribution visualization (pie chart)
- Recent activity log (last 6 classifications)
- Weekly usage trends graph
- Automated capacity alerts

## Installation & Setup

1. Hardware Assembly
   - Mount the USB camera above the waste platform
   - Position the IR sensor on the side of the platform
   - Install ultrasonic sensors in each bin compartment
   - Connect the servo motor to the waste platform
   - Wire all components to the Raspberry Pi 3B

2. Software Setup
   ```bash
   # Clone the repository
   git clone https://github.com/projects506/SmartBin-ML-Based-Waste-Classifier
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Configure AWS credentials
   aws configure
   
   # Start the application
   python app.py
   ```

3. AWS Configuration
   - Launch t3.micro EC2 instance
   - Configure security groups
   - Deploy Flask application
   - Set up database

## Working Process

1. IR sensor detects waste placement
2. USB camera captures waste image
3. Image is sent to AWS backend for classification
4. ML model categorizes waste
5. Servo motor directs waste to appropriate bin
6. Dashboard updates with new data
7. Capacity monitoring runs continuously

## Performance Metrics

- Classification Accuracy: 96.13%
- Image Processing Time: 0.5-1 second
- Servo Motor Operation: 1-2 seconds
- Real-time Updates: Every 5 minutes

## License

This project is licensed under the Apache License 2.0
