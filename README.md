# 🐄 SmartEye: Edge AI for Early BRD Detection

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Platform: Raspberry Pi](https://img.shields.io/badge/platform-Raspberry_Pi-c51a4a.svg)

**SmartEye** is a non-invasive, passive thermal monitoring system designed to detect Bovine Respiratory Disease (BRD) days before clinical symptoms appear. Built entirely on Edge AI, it eliminates the need for expensive per-cow wearables or labor-intensive manual temperature checks.

## 🎯 The Problem
Bovine Respiratory Disease (BRD) costs the global cattle industry billions annually. Current detection methods rely on:
*   **Invasive Rectal Thermometers:** Causes stress-induced hyperthermia, skewing data and requiring intensive manual labor.
*   **Expensive Wearables/Boluses:** Linear scaling costs (1 sensor per cow) makes monitoring large herds financially unviable for many farmers.

## 💡 Our Solution
Instead of outfitting every cow with a sensor, we outfit the environment. SmartEye is an IoT camera rig mounted at a high-traffic choke point (like a water trough). 
It uses an RGB camera paired with a YOLO vision model to locate the cow's eye (the lacrimal caruncle), while an MLX90640 thermal sensor captures the precise temperature of that region. A Random Forest model then analyzes the data against environmental factors to trigger early fever alerts.

## ✨ Features
*   **Passive Monitoring:** Zero stress to the animal; no restraint required.
*   **Exponential Scalability:** One $180 device can monitor hundreds of cows.
*   **100% Edge Processing:** Runs locally on a Raspberry Pi. No cloud dependency or high-bandwidth internet required—perfect for rural farms.
*   **Scientifically Backed:** Calibrated against the Schaefer et al. (2011) veterinary benchmark for orbital temperature fluctuations.

## 🛠️ Hardware Stack
*   **Compute:** Raspberry Pi 4 (or Zero 2 W)
*   **Thermal Sensor:** MLX90640 (Long-Wave Infrared)
*   **Vision Sensor:** Standard RGB Camera Module
*   **Enclosure:** IP67 Polycarbonate Junction Box with ZnSe Laser Lens viewport.

## 💻 Software Stack
*   **Language:** Python 3.9
*   **Computer Vision:** OpenCV, YOLO (Object Detection/Eye Localization)
*   **Machine Learning:** Scikit-Learn (Random Forest for thermal classification)
*   **Data Handling:** NumPy, Pandas

## 🚀 Installation & Setup

### Prerequisites
Ensure your Raspberry Pi has I2C enabled for the MLX90640 sensor.
```bash
sudo raspi-config
# Navigate to Interfacing Options -> I2C -> Enable
