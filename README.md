# Income Classification using k-NN

A machine learning project implementing k-Narest Neighbors (k-NN) classifier from scratch for income prediction (>50K or ≤50K) on census data.

## Project Overview
This project explores different feature preprocessing techniques and implements k-NN classification to predict income levels based on demographic and employment data from the US Census Bureau.

## Features
- **Data Preprocessing**: Implemented three feature engineering approaches:
  - Naive binarization (all fields one-hot encoded)
  - Smart binarization (categorical fields one-hot, numerical fields preserved)
  - Smart binarization with scaling (MinMax scaling for numerical fields)
- **k-NN Implementation**: Built k-NN classifier from scratch using numpy
- **Performance Optimization**: Used vectorized operations and broadcasting for efficiency
- **Comprehensive Evaluation**: Tested Euclidean vs Manhattan distances, k values 1-99

## Key Results
- Achieved ~14-15% error rate on development set using smart binarization with scaling
- Optimized k value and distance metric for best performance
- Submitted predictions to Kaggle competition with competitive ranking
- Implemented custom k-NN that matches sklearn performance

## Technologies Used
- Python 3
- NumPy (for custom k-NN implementation)
- Scikit-learn (for preprocessing and benchmarking)
- Pandas (for data manipulation)
- Matplotlib (for visualization)

## Project Structure

### Source Code (`src/`)
- `preprocessor.py` - Data preprocessing pipelines
- `knn_classifier.py` - Custom k-NN implementation  
- `experiment_runner.py` - Main experiment script
- `utils.py` - Helper functions

### Data Files (`data/`)
- `income.train.5k.csv` - Training data (5,000 examples)
- `income.dev.csv` - Development data (1,000 examples)
- `income.test.blind.csv` - Test data (1,000 examples)

### Results (`results/`)
- `error_analysis.png` - Error rate vs k plot
- `income.test.predicted.csv` - Final predictions

## Installation & Usage
```bash
# Clone repository
git clone https://github.com/yourusername/income-classification.git
cd income-classification

# Run experiments
python src/experiment_runner.py

# Generate predictions
python src/generate_predictions.py
