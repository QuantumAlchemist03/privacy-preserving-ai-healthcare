# Privacy-Preserving AI for Healthcare

Split Federated Learning for medical imaging — a privacy-preserving deep learning framework for training diagnostic models on sensitive medical data without centralising patient information.

## Overview

This project explores **Split Federated Learning (SplitFed)**, a hybrid approach combining the strengths of Federated Learning and Split Learning, applied to medical image classification. The aim is to enable hospitals and clinical institutions to collaboratively train deep learning models without ever sharing raw patient data, addressing critical privacy concerns in healthcare AI.

This work forms the basis of an undergraduate dissertation in Artificial Intelligence at Anglia Ruskin University (2025–26).

## Motivation

Medical AI models benefit from large, diverse datasets, but data sharing in healthcare is heavily restricted by regulations such as GDPR and HIPAA. Traditional centralised training requires aggregating patient data, which is often not feasible. Split Federated Learning offers a path forward by:

- Keeping raw patient data on local devices/institutions
- Splitting the model between clients and a central server
- Sharing only intermediate activations and gradients, not data
- Reducing client-side computational load compared to standard Federated Learning

## Key Features

- Baseline centralised CNN model for breast cancer image classification
- Implementation of Split Federated Learning architecture
- Performance comparison across centralised, federated, and split-federated paradigms
- Privacy analysis and computational overhead evaluation
- Built with TensorFlow/Keras and PyTorch

## Results So Far

| Model | Accuracy | AUC |
|-------|----------|-----|
| Centralised baseline (CNN) | ~96% | ~99% |
| Split Federated Learning | *In progress* | *In progress* |

## Tech Stack

- **Languages:** Python 3.10+
- **Frameworks:** TensorFlow / Keras, PyTorch
- **Libraries:** NumPy, Pandas, Scikit-learn, Matplotlib
- **Environment:** Jupyter, VS Code
- **Version control:** Git, GitHub

## Project Structure

```
privacy-preserving-ai-healthcare/
├── data/                   # Dataset (excluded via .gitignore)
├── notebooks/              # Exploratory analysis and experiments
├── src/
│   ├── centralised/        # Baseline centralised model
│   ├── federated/          # Federated learning implementation
│   ├── split_federated/    # Split federated learning implementation
│   └── utils/              # Helper functions and data loaders
├── results/                # Trained models, metrics, plots
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/QuantumAlchemist03/privacy-preserving-ai-healthcare.git
cd privacy-preserving-ai-healthcare
```

Create a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Baseline Model

```bash
python src/centralised/train_baseline.py
```

### Running Split Federated Learning

```bash
python src/split_federated/main.py
```

## Dataset

This project uses a publicly available breast cancer imaging dataset. Due to size and licensing, the dataset is not included in this repository. Download instructions can be found in `data/README.md`.

## Roadmap

- [x] Centralised baseline CNN
- [x] Initial Split Federated Learning prototype
- [ ] Multi-client simulation with non-IID data
- [ ] Differential privacy integration
- [ ] Communication overhead analysis
- [ ] Final dissertation write-up

## Author

**Alif**
Final-year BSc Artificial Intelligence student, Anglia Ruskin University
AI Architect at VANKADEL Intelligence Platform

- GitHub: [@QuantumAlchemist03](https://github.com/QuantumAlchemist03)

## Acknowledgements

- Dissertation supervisor at Anglia Ruskin University
- Original Split Federated Learning research by Thapa et al.
- The wider open-source federated learning community

## License

This project is released for academic purposes. Please contact the author before using this code in commercial or clinical settings.

---

*This is an active research project — code, results, and documentation are being updated regularly.*
