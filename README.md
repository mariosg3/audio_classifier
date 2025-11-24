# GTZAN Audio Genre Classifier

A production-ready audio classification pipeline designed to identify music genres using the **GTZAN dataset**. This project leverages **HuggingFace Transformers (MERT)** for robust feature extraction and **Scikit-Learn (SVM)** for efficient classification.

## Key Features

* **Embeddings:** Uses [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) for extracting high-quality audio representations.
* **Smart Preprocessing:** Automatically handles audio chunking, overlapping, and padding.
* **Robust Classification:** Implements a pipeline with Scaling, PCA, and SVM (Support Vector Machine).
* **Modern Tooling:** Built with `uv` and `hatchling` for dependency management.

## Getting Started

Follow these instructions to set up your local development environment.

You can install `uv` using `pip`:
```bash
pip install uv
```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd gtzan-audio-genre-classifier
    ```

2.  **Create and activate a virtual environment:**
    Use `uv` to create a virtual environment.
    ```bash
    uv venv
    ```

3.  **Activate the environment:**
    ```bash
    # macOS/Linux
    source .venv/bin/activate
    
    # Windows
    .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Install the project and its development dependencies in editable mode.
    ```bash
    uv pip install -e .
    ```

## Project Structure

```text
.
├── data/                   # Data storage (created automatically)
├── models/                 # Saved models (created automatically)
├── src/
│   ├── svm_config.json           # Configuration for GridSearch
│   └── audio_classifier/
│       ├── dataset.py            # GTZAN loading & chunking logic
│       ├── feature_extraction.py # MERT inference script
│       ├── train.py              # SVM training & evaluation
│       └── __utils__.py          # Metrics & plotting helpers
├── pyproject.toml          # Project dependencies & script definitions
└── README.md
```
## Usage

The project includes command-line scripts for each stage of the pipeline. These scripts are defined in `pyproject.toml` and installed via `hatchling`.

### 1. Feature Extraction

This script processes the GTZAN dataset, chunks audio files, and extracts embeddings using the MERT model. The resulting features are saved to disk.

```bash
audio-extract
```

### 2. Model Training

This script loads the extracted features, performs hyperparameter tuning using `GridSearchCV` on an SVM classifier, and saves the best-performing model.

```bash
audio-train
```

*Note: You can customize the `GridSearchCV` parameters by modifying `src/svm_config.json`.*

### 3. Model Inference

Predict the genre of a specific audio file using the best saved model.

```bash
audio-inference --path "path/to/your/song.wav"
```