# Car Damage Severity Verifier

A deep learning app that classifies the severity of car damage from a photo into
one of three categories: **minor**, **moderate**, or **severe**. Before running
the severity check, the app first verifies that the uploaded photo is actually a
car, using a separate car/not-car classifier. Built with fine-tuned MobileNetV2
models and a simple Streamlit web interface.

## How it works

1. Upload a photo.
2. A car classifier first checks whether the photo actually shows a car.
   - If it's not a car, the app stops here and asks for a car photo instead.
   - If it is a car, the photo is passed on to the damage severity model.
3. The damage model analyzes the image and predicts a severity class.
4. You get the predicted label along with a confidence score for each class.

## Models

This project uses two separate models, loaded independently so the damage
model never needs to be retrained:

### 1. Car classifier (`car_classifier.keras`)
- **Base architecture:** MobileNetV2 (pretrained on ImageNet), used via transfer learning
- **Approach:** the base model is first frozen while a custom classification head is trained,
then the top layers of the base model are unfrozen and fine-tuned at a low learning rate
- **Classes:** `car`, `not_car`
- **Validation accuracy:** ~99.8%

### 2. Damage severity model (`car_damage.keras`)
- **Base architecture:** MobileNetV2 (pretrained on ImageNet), used via transfer learning
- **Approach:** same two-stage freeze-then-fine-tune approach as above
- **Classes:** `01-minor`, `02-moderate`, `03-severe`
- **Validation accuracy:** ~74%

## Dataset

- **Damage severity model:** trained on the [Car Damage Severity Dataset](https://www.kaggle.com/datasets/prajwalbhamere/car-damage-severity-dataset)
(1,383 training images / 248 validation images across the three severity classes).
- **Car classifier:** trained on car images pooled from the damage severity dataset
(`car` class) and a subset of [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02)
object categories, excluding the `car_side` category (`not_car` class).

## Tech stack

- TensorFlow / Keras — model training and inference
- Streamlit — web interface
- Pillow / NumPy — image handling

## Running locally

```
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser. Upload an image and click **Predict Severity**.

## Project structure

```
├── app.py                  # Streamlit web app
├── requirements.txt        # Python dependencies
├── car_classifier.keras    # Car vs not-car gate model
├── car_damage.keras        # Damage severity model
└── README.md
```

## Limitations

- The damage severity model is trained on a relatively small dataset (~1,600 images
total), so accuracy on photos very different from the training set (odd angles,
poor lighting, etc.) may be lower.
- The "moderate" class is the hardest to classify, since it sits between two
visually similar boundaries (minor and severe damage).
- The car classifier was trained mostly on clearly distinct non-car objects
(from Caltech-101), so it may be less reliable on ambiguous edge cases like
toy cars, car-shaped objects, or extreme close-ups of car parts.
