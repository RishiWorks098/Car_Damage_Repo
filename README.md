# Car Damage Severity Verifier

A deep learning app that classifies the severity of car damage from a photo into
one of three categories: **minor**, **moderate**, or **severe**. Built with a
fine-tuned MobileNetV2 model and a simple Streamlit web interface.

## How it works

1. Upload a photo of a damaged car.
2. The model analyzes the image and predicts a severity class.
3. You get the predicted label along with a confidence score for each class.

## Model

- **Base architecture:** MobileNetV2 (pretrained on ImageNet), used via transfer learning
- **Approach:** the base model is first frozen while a custom classification head is trained,
  then the top layers of the base model are unfrozen and fine-tuned at a low learning rate
- **Classes:** `01-minor`, `02-moderate`, `03-severe`
- **Validation accuracy:** ~74%

## Dataset

Trained on the [Car Damage Severity Dataset](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset)
(1,383 training images / 248 validation images across the three severity classes).

## Tech stack

- TensorFlow / Keras — model training and inference
- Streamlit — web interface
- Pillow / NumPy — image handling

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser. Upload an image and click **Predict Severity**.

## Project structure

```
├── app.py                                 # Streamlit web app
├── requirements.txt                       # Python dependencies
├── car_damage_model_74pct_backup.keras    # Trained model
└── README.md
```

## Limitations

- Trained on a relatively small dataset (~1,600 images total), so accuracy on
  photos very different from the training set (odd angles, poor lighting, etc.)
  may be lower.
- The "moderate" class is the hardest to classify, since it sits between two
  visually similar boundaries (minor and severe damage).
