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

1. Download the Memotion 7K dataset from Kaggle.
2. Extract the dataset into a folder named `dataset` inside the project directory, keeping its original image and label files.
3. Install the packages using `pip install -r requirements.txt`.
4. Open `Memotion7K_Final.ipynb` in Jupyter.
5. Run the cells in order.

The notebook uses a GPU when CUDA is available, but can also run on CPU.

Pretrained ResNet-50 and DistilBERT weights are downloaded on the first run. Image and text embeddings are cached locally to avoid repeated feature extraction.

The generated results are saved in `outputs_final/`.

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