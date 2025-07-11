# Load packages
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.utils import load_img, img_to_array
from keras.models import load_model
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries

# Set constant
Seed = 42
image_size = (224, 224)
batch_size = 32

# Load the model
model = load_model('xray_classifier_v2_87acc.h5')

#Pre-processing 
df = pd.read_csv('Chest_xray_Corona_Metadata.csv')
df.head()

# Discard stress-smoking pneumonia images
df = df[df.Label_1_Virus_category != "Stress-Smoking"]

# Create target variables
targets = [
    (df['Label'] == "Normal"),
    (df['Label'] == "Pnemonia") & (df['Label_1_Virus_category'] == "Virus"),
    (df['Label'] == "Pnemonia") & (df['Label_1_Virus_category'] == "bacteria")
]

values = ['Normal', 'Pnemonia-virus', 'Pnemonia-bacteria']
default = 'NaN'
df['target'] = np.select(targets, values, default=default)

chest_df = pd.DataFrame(df)
chest_df.shape

# Check counts of target variables
target_counts = chest_df['target'].value_counts()
print(target_counts)

# Split the dataframe
train = chest_df[chest_df['Dataset_type'] == 'TRAIN']
test = chest_df[chest_df['Dataset_type'] == 'TEST']

y_train = train['target']
y_test = test['target']


# Augment data
batch_size = 32

train['target'] = train['target'].astype(str)
test['target'] = test['target'].astype(str)


# Split the DataFrame

test_df = test
train_df, val_df= train_test_split(
    train,
    test_size=0.2,
    stratify=train['target'],
    random_state=42

)

# Create data generators
train_gen = ImageDataGenerator(
    rescale=1/255,
    zoom_range=0.1,
    rotation_range=20,
    width_shift_range=0.1,
    shear_range=0.1,
    samplewise_center=True,
    samplewise_std_normalization=True
)

val_gen = ImageDataGenerator(
    rescale=1/255,
    samplewise_center=True,
    samplewise_std_normalization=True
)


base_path = 'Coronahack-Chest-XRay-Dataset'
train_dir = os.path.join(base_path, 'train')
test_dir = os.path.join(base_path, 'test')

train_flow = train_gen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="X_ray_image_name",
    y_col='target',
    batch_size=32,
    shuffle=True,
    seed=Seed,
    class_mode='categorical',
    target_size=(224, 224)
)

val_flow = val_gen.flow_from_dataframe(
    dataframe=val_df,
    directory=train_dir,
    x_col="X_ray_image_name",
    y_col="target",
    batch_size=batch_size,
    shuffle=False,
    class_mode="categorical",
    target_size=image_size
)

test_flow = val_gen.flow_from_dataframe(
    dataframe=test,
    directory=test_dir,
    x_col="X_ray_image_name",
    y_col="target",
    batch_size=batch_size,
    shuffle=False,
    class_mode="categorical",
    target_size=image_size
)

# Performance Metrics
y_true = test_flow.classes
y_pred = model.predict(test_flow, steps=len(test_flow), verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Performance:")
print("Accuracy:", accuracy_score(y_true, y_pred_classes))
print("Micro AUC:", roc_auc_score(y_true, y_pred, average='micro', multi_class='ovr'))
print("Micro Precision:", precision_score(y_true, y_pred_classes, average='micro'))
print("Micro Recall:", recall_score(y_true, y_pred_classes, average='micro'))
print("Micro F1:", f1_score(y_true, y_pred_classes, average='micro'))

# Explainability with LIME 
def gen_sample(exp, pred_class, actual_class, ax, weight=0.005, show_positive=True, hide_background=True):
    image, mask = exp.get_image_and_mask(
        pred_class,
        positive_only=show_positive,
        num_features=6,
        hide_rest=hide_background,
        min_weight=weight
    )
    ax.imshow(mark_boundaries(image, mask))
    ax.axis("off")
    ax.set_title(f"Pred: {class_labels[pred_class]}\nActual: {actual_class}")

print("\nModel Explainability: Green increases the probability for the label, Red decreases it")

X_test = test["X_ray_image_name"].tolist()
y_true_labels = test["target"].tolist()
class_labels = list(test_flow.class_indices.keys())

fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for i in range(5):
    random_index = random.randint(0, len(X_test) - 1)
    image_path = os.path.join(test_dir, X_test[random_index])
    img = np.expand_dims(img_to_array(load_img(image_path, target_size=(224, 224))), axis=0) / 255

    pred_img = model.predict(img, verbose=0)
    actual_class = y_true_labels[random_index]

    explainer = LimeImageExplainer()
    exp = explainer.explain_instance(img[0], model.predict, top_labels=5, hide_color=0, num_samples=1000)

    gen_sample(exp, exp.top_labels[0], actual_class, axes[i], show_positive=False, hide_background=False)

plt.tight_layout()
plt.savefig("explainability.png", dpi=300)
plt.show()
