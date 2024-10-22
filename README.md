# Inclusion: The Global Multimedia Deepfake Detection

Welcome to the open-source repository for the **Inclusion: The Global Multimedia Deepfake Detection** competition on Kaggle. This project aims to advance the detection of deepfake media across various formats and platforms.

## Competition Overview

The competition consists of two phases:

### Phase 1: [Track 1: Deepfake Image Detection](https://www.kaggle.com/competitions/multi-ffdi)
- **Objective**: [In the first phase, only the training and validation sets are released, and the leaderboard is sorted by the validation set]
- **Duration**: [Jun 30 - Aug 22]
- **Evaluation Metric**: [AUC]
- **Ranking**: [38/706]
- **AUC Score**: [0.9982558829]

### Phase 2: [Phase 2: submitting test results](https://www.kaggle.com/competitions/multi-ffdi-phase2)
- **Objective**: [In phase 2, the test set is released and the test set is used as a leaderboard result for phase 2]
- **Duration**: [Aug 15 - Aug 22]
- **Evaluation Metric**: [AUC]
- **Ranking**: [46/184]
- **AUC Score**: [0.9551696556]

## Getting Started

### Prerequisites
- Python 3.x
- [PyTorch, numpy and other common deep learning libraries]
- diffusers[torch]

### How to Run the Code
1. **Prepare Your Environment**:

   Ensure that you have Python 3.x installed and that you have set up a virtual environment (optional but recommended).

3. **Clone the repository**:
   ```bash
   git clone https://github.com/gbljdgb/Kaggle-2024-Deepfake-Image-Detection.git
   cd Kaggle-2024-Deepfake-Image-Detection

4. **Prepare supplementary training set for diffusion model generation (optional)**:

    The generated diffusion model image is saved in the ./SDxl folder.
   
    ```bash
    gen_SDxl.py

5. **Train and Test**:

   The weights are stored in . /ckpts folder and output in . /outputs folder.
   
   ```bash
   python train.py
   python test.py
