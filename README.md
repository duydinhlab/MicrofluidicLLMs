# μ-Fluidic-LLMs: Autonomous Droplet Microfluidic Design with Large Language Models

This repository accompanies the manuscript:  
**Autonomous Droplet Microfluidic Design Framework with Large Language Models**  
*Dinh-Nguyen Nguyen, Raymond Kai-Yu Tong, Ngoc-Duy Dinh*  
Department of Biomedical Engineering, The Chinese University of Hong Kong  
Corresponding author: [ngocduydinh@cuhk.edu.hk](mailto:ngocduydinh@cuhk.edu.hk)

---
<img width="1024" height="1024" alt="image001" src="https://github.com/user-attachments/assets/fa11ee98-e1fe-470f-9fe8-251e7e2caa5b" />

## Data: [Link]([/guides/content/editing-an-existing-page](https://mycuhk-my.sharepoint.com/personal/1155187654_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1155187654%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2FAttachments%2FGithub%5FACS%5FOmega%5FUpload%2Ezip&parent=%2Fpersonal%2F1155187654%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2FAttachments&ga=1))


## Overview

Droplet microfluidic systems are widely used in biotechnology but require extensive iterative design processes. μ-Fluidic-LLMs introduces a **generalizable, language-model-driven framework** to automate performance prediction and design tasks in microfluidics. The key innovation is to **convert tabular microfluidic parameters into natural language representations**, enabling **pre-trained large language models (LLMs)** to extract contextual embeddings, which are then used for downstream prediction via standard ML models.

---

## Framework

μ-Fluidic-LLMs consists of two modules:
- **Performance Prediction:** Converts microfluidic design parameters to text and uses LLMs to predict droplet diameter, generation rate, and regime.
- **Design Automation:** Reverses the process to infer design parameters based on target outputs.

The core steps are:
1. **Text Serialization:** Convert each tabular row into a structured paragraph combining column names and values.
2. **LLM Embedding Extraction:** Feed serialized text into an open-source LLM to generate dense semantic embeddings.
3. **ML Integration:** Use these embeddings with baseline models like DNN, XGBoost, LightGBM, or SVM for prediction or reverse design.

---

## Key Results

We benchmarked μ-Fluidic-LLMs on two public datasets and compared model performance across 24 LLM–ML model pairings.

### Highlights:
- **DNN + LLAMA3.1**:  
  - Reduced MAE in droplet diameter by **~40%** vs. prior methods  
  - Improved classification accuracy of droplet regime by **~3%**
- **DNN + DEEPSEEK-R1**:  
  - Achieved the lowest RMSE for droplet generation rate
- **Tree-based models (e.g., LightGBM, XGBoost)**: Performed worse when combined with LLMs, highlighting the importance of architecture-embedding alignment.

---

## LLMs Used

| Model        | Size | License | Source     |
|--------------|------|---------|------------|
| LLAMA3.1     | 8B   | Open    | Meta       |
| DEEPSEEK-R1  | 8B   | Open    | DeepSeek   |
| GEMMA2       | 9B   | Open    | Google     |
| LLAVA        | 7B   | Open    | Microsoft  |
| MISTRAL      | 7B   | Open    | Mistral AI |

---

## Baseline Models

- Deep Neural Networks (DNN)
- XGBoost
- LightGBM
- Support Vector Machines (SVM)

---

## Tasks

**Dataset 1:**
- Predict droplet diameter (µm)
- Predict droplet generation rate (Hz)
- Classify droplet regime
- Predict capillary number (for design automation)

**Dataset 2:**
- Predict droplet diameter (µm)
- Predict droplet generation rate (Hz)

---

## Evaluation Metrics

- **Regression:** MAE, MSE, RMSE, R²  
- **Classification:** Accuracy, Precision, Recall, F1-score, ROC AUC  
- **Validation:** 10-fold cross-validation repeated 15 times with error bars

---

## Results Summary

| Task                     | Best Model         | MAE ↓ / Accuracy ↑ |
|--------------------------|--------------------|--------------------|
| Droplet Diameter (D1)    | DNN + LLAMA3.1     | **8.99 µm**        |
| Droplet Diameter (D2)    | DNN + DEEPSEEK-R1  | **4.62 µm**        |
| Generation Rate (D1)     | DNN + DEEPSEEK-R1  | **12.5 Hz**        |
| Generation Rate (D2)     | DNN + DEEPSEEK-R1  | **291.2 Hz**       |
| Droplet Regime (D1)      | DNN + LLAMA3.1     | **98.2% Accuracy** |
| Capillary Number (D1)    | DNN + GEMMA2       | **0.14 RMSE**      |

---

## Citation

Please cite our work if you find this useful:

```bibtex
@article{nguyen2025fluidicllms,
  title={Autonomous Droplet Microfluidic Design Framework with Large Language Models},
  author={Nguyen, Dinh-Nguyen and Tong, Raymond Kai-Yu and Dinh, Ngoc-Duy},
  journal={arXiv:2411.06691},
  year={2025},
  note={https://doi.org/10.48550/arXiv.2411.06691}
}
