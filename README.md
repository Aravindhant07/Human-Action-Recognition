# Human Action Recognition with Hybrid ResNet + CNN Models

A deep learning model that classifies images of people into **15 everyday actions**, such as running, eating and dancing. It combines pre-trained ResNet backbones with custom fully connected layers, in PyTorch.

MSc Data Science coursework, University of Roehampton.

## Results

| Model | Validation accuracy |
|---|---|
| ResNet18 (frozen) | 63.73% |
| ResNet50 (frozen) | 71.71% |
| ResNet50 (fine-tuned) | 79.68% |
| **Hybrid ResNet50 + custom head (final)** | **80.91%** |

The final model's best epoch (12 of 15) scored **precision 0.81, recall 0.81 and F1 0.81**.

The full write-up is in [`Human_Action_Recognition_Report.pdf`](Human_Action_Recognition_Report.pdf).

## Approach

- **Dataset:** the [Human Action Recognition (HAR) dataset](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset) on Kaggle. It has more than 12,000 labelled images in 15 roughly balanced classes.
- **Preprocessing:** images resized to 224×224 and converted to tensors, with an 80/20 train/validation split.
- **Model:** a pre-trained ResNet feature extractor feeding a custom head (Linear → BatchNorm → ReLU → Dropout layers).
- **Training:** the Adam optimiser with a StepLR learning-rate schedule. Accuracy, precision, recall and F1 were tracked every epoch, and model complexity was measured with `thop` (FLOPs and parameters).

## Running it

The notebook [`human_action_recognition.ipynb`](human_action_recognition.ipynb) was written for **Google Colab** with a GPU runtime.

1. Download the dataset from Kaggle and upload the zip to your Google Drive.
2. Open the notebook in Colab, mount your Drive, and update the zip path in the unzip cell.
3. Run all cells.

## Tech stack

Python · PyTorch · torchvision (ResNet18/50) · scikit-learn · pandas · seaborn · Google Colab
