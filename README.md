# Meme Sentiment Classification Using Images and Text

## About the project

This project compares different approaches to classifying meme sentiment using the Memotion 7K dataset.

The goal is to predict whether a meme has negative, neutral or positive sentiment and investigate whether combining image and text features produces better results.

## Dataset

The dataset is available on [Kaggle](https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k/data).

It contains 6,992 labelled memes. After removing missing or unreadable images and exact duplicates, 6,965 examples were used.

The original five sentiment categories were combined into three classes. A stratified 70/15/15 train, validation and test split was used, with random seed 42.

## Models

Six approaches were evaluated:

- Majority-class baseline
- TF-IDF with Logistic Regression
- ResNet-50 image-only classifier
- DistilBERT text-only classifier
- Early fusion of image and text embeddings
- Late fusion using weighted prediction probabilities

The neural classifiers use frozen pretrained encoders and an MLP with hidden layers of 128 and 32 units.

## Results

Macro F1 was used as the main evaluation metric because the dataset is imbalanced.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Image-only | 0.4938 | 0.3452 |
| TF-IDF + Logistic Regression | 0.4431 | 0.3443 |
| Early fusion | 0.5110 | 0.3387 |
| Text-only | 0.5129 | 0.3365 |
| Late fusion | 0.5129 | 0.3365 |
| Majority baseline | 0.5943 | 0.2485 |

Image-only achieved the highest test macro F1. Neither fusion approach consistently outperformed the individual models.

## How to run

### 1. Download the project

Download this repository using **Code → Download ZIP** and extract it.

### 2. Download the dataset

Download the [Memotion Dataset 7K from Kaggle](https://www.kaggle.com/datasets/williamscott701/memotion-dataset-7k/data).

Extract the dataset inside the `dataset` folder.

The folder structure should be:

```text
memotion-sentiment-analysis/
├── dataset/
│   ├── .gitkeep
│   └── memotion_dataset_7k/
│       ├── labels.csv
│       └── images/
├── results/
├── Memotion7K_Final_Cleaned.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

The notebook searches the dataset folder and its subfolders for `labels.csv` and meme images.

The dataset is not included in GitHub and must be downloaded separately.

### 3. Install dependencies

Python 3.11 or newer is recommended.

Open a terminal in the project folder and run:

```bash
python -m pip install -r requirements.txt
```

The default installation can run the notebook on CPU.

For NVIDIA GPU acceleration, install the matching
CUDA-enabled PyTorch build using the official
[PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).

For PyTorch 2.7.1 with CUDA 11.8:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```

The notebook automatically selects CUDA when available.


### 4. Run the notebook

Open `Memotion7K_Final_Cleaned.ipynb` in Jupyter Notebook or JupyterLab, with the project folder as the working directory.

Select **Kernel → Restart Kernel and Run All Cells**.

The notebook uses CUDA if available, otherwise CPU. Internet access is needed on the first run to download pretrained ResNet-50 and DistilBERT weights.

### 5. Expected results

After preprocessing, the notebook should retain 6,965 usable memes.

New outputs are saved in `outputs_final/`. Embeddings are cached in `outputs/cache/`.

The `results/` folder contains reference outputs from the original experiment.

The random seed is 42. Minor variations in results may occur across different hardware and software environments.

## Limitations

The negative sentiment class is underrepresented, and the models show overfitting. Sentiment labels may also be subjective, particularly for sarcastic memes.

The results are from a single dataset and should not be treated as evidence that the classifiers would generalise to other meme collections.

## Reproducibility

Random seed: 42

The same stratified split and evaluation data are used for the model comparisons. Experiment settings and library versions are recorded in the notebook and accompanying configuration files.

Some variation is possible across GPU and software environments.

## References

Dataset: Memotion 7K

Image feature extraction: ResNet-50

Text feature extraction: DistilBERT

Full methodology references and discussion are included in the project report.
