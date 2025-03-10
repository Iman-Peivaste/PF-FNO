
# Fourier Neural Operator (FNO) for Grain Evolution Prediction

This repository implements a Fourier Neural Operator (FNO) model designed for predicting grain evolution dynamics in phase-field simulations. The model leverages Fourier-domain convolutional layers to efficiently capture spatial patterns, enabling accurate prediction of future grain states based on historical data.

## Project Structure
- **`data_preparation.py`**: Contains classes and functions for loading and preprocessing simulation datasets.
- **`fno_model.py`**: Defines the FNO model architecture using spectral convolution layers.
- **`trainer.py`**: Handles training the FNO model, including optimization, validation, and saving the best-performing model.
- **`tester.py`**: Provides functionalities to evaluate the trained model on test datasets, generate predictions, and visualize results.
- **`utils.py`**: Contains utility functions for loading trained models and visualizing predictions.

## Getting Started
1. **Training**: Run `python trainer.py` to train the model on prepared datasets.
2. **Testing**: Run `python tester.py` to evaluate the model and visualize predictions against ground truth.

Ensure your dataset files (`Dataset_64.npy`, `Dataset_b_64.npy`, `Dataset2_512.npy`, etc.) are correctly placed in the project directory.

---

For further information please contact iman.peivaste@list.lu
