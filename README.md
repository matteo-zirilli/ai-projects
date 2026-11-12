# AI Projects

A collection of the projects I developed during my **Master's in AI Engineering**. Each project is a self-contained Jupyter notebook (built and run on Google Colab), with the code, the saved outputs (metrics, training logs, charts) and a written analysis of the results.

The projects follow the path of the course: from Python fundamentals, to classical Machine Learning with scikit-learn, to Deep Learning for computer vision with Keras/TensorFlow and PyTorch (including transfer learning on pre-trained networks), up to Reinforcement Learning.

> 🇮🇹 The original Italian versions of all notebooks are kept in the [`it/`](it/) folder. The English notebooks in the root are faithful translations: the code logic is unchanged, and only text, comments and printed messages were translated.

---

## Projects at a glance

| # | Project | Area | Main techniques | Stack | Key result |
|---|---------|------|-----------------|-------|------------|
| 1 | [Phone Book](#1-phone-book--object-oriented-python) | Python fundamentals | OOP, CRUD, JSON persistence, input validation | Python | Working command-line application |
| 2 | [Real Estate Price Forecasting](#2-real-estate-price-forecasting--regularized-linear-regression) | Regression | Ridge, Lasso, Elastic Net, log-transform, K-Fold CV | scikit-learn | R² ≈ 0.67 on test, no overfitting |
| 3 | [Exotic Fruit Classification](#3-exotic-fruit-classification--k-nearest-neighbors) | Classification | KNN, Yeo-Johnson transform, K tuning, GridSearchCV | scikit-learn | 92% test accuracy |
| 4 | [Animal Recognition for Self-Driving Cars](#4-animal-recognition-for-self-driving-cars--cnn-on-cifar-10) | Computer vision | Custom CNN, class regrouping, early stopping, error analysis | Keras / TensorFlow | 92% accuracy, 0.90 macro-F1 |
| 5 | [Food Image Classification](#5-food-image-classification--cnn-from-scratch-vs-transfer-learning) | Computer vision | CNN from scratch, data augmentation, VGG16 transfer learning | PyTorch | 77% accuracy on 14 classes (transfer) |
| 6 | [Flower Recognition for an AgriTech Company](#6-flower-recognition-for-an-agritech-company--transfer-learning-with-resnet-18) | Computer vision | ResNet-18 transfer learning, metric-driven early stopping | PyTorch / timm | 0.89 macro-F1 |
| 7 | [Cybersecurity Defense with Reinforcement Learning](#7-cybersecurity-defense-with-reinforcement-learning--sarsa-vs-double-dqn) | Reinforcement learning | SARSA (tabular), Double DQN, state design, custom Gym wrapper | PyTorch / gym-idsgame | 91.5% successful defenses vs the strongest attacker |

---

## 1. Phone Book — Object-Oriented Python

📓 [`Phone_Book_Python_OOP.ipynb`](Phone_Book_Python_OOP.ipynb) · [Open in Colab](https://colab.research.google.com/github/matteo-zirilli/ai-projects/blob/main/Phone_Book_Python_OOP.ipynb)

**Goal.** Build an interactive phone book application in pure Python, applying object-oriented programming.

**What it shows.**
- A `Contatto` (Contact) class with validation of numeric fields, equality (`__eq__`) and alphabetical sorting by last/first name (`__lt__`).
- A `Rubrica` (Phone book) class with the full set of operations: add, search (by name or by number), view, edit and delete contacts. The rule that no two identical contacts can exist is enforced everywhere.
- Persistence on a JSON file: the phone book is loaded at startup and saved at exit. Contacts are serialized dynamically using `inspect.signature`.
- A menu built dynamically from the class methods via introspection (`dir()`, `getattr()`), plus robust handling of invalid input (`try/except`, `assert`).

---

## 2. Real Estate Price Forecasting — Regularized Linear Regression

📓 [`Real_Estate_Price_Forecasting.ipynb`](Real_Estate_Price_Forecasting.ipynb) · [Open in Colab](https://colab.research.google.com/github/matteo-zirilli/ai-projects/blob/main/Real_Estate_Price_Forecasting.ipynb)

**Scenario.** *RealEstateAI Solutions* wants more accurate and stable house price estimates, reducing overfitting through regularization.

**Data.** Housing dataset with 545 properties and 13 columns (area, bedrooms, bathrooms, air conditioning, furnishing status, …).

**Approach.**
- Exploratory analysis: missing values, correlations, distribution of the target.
- One-Hot Encoding of `furnishingstatus`. A **log transformation of the price** reduced the skewness from 1.21 to 0.14.
- Standardization and comparison of **Ridge, Lasso and Elastic Net** over a range of `alpha` values (0.001 → 10), tracking R², MSE and the number of active features.
- Validation with **5-fold cross-validation**. The notebook includes charts of R² vs alpha, the coefficient paths and the residuals analysis.

**Results.**
- Ridge is the most stable model: **R² ≈ 0.70 on train and ≈ 0.67 on test** (K-Fold average). Lasso and Elastic Net collapse for `alpha ≥ 0.1` as they zero out the features.
- Train and test scores are close, so there is no overfitting. The limit is structural underfitting: the relationship between features and price is not purely linear.
- The most influential features are **area, air conditioning and number of bathrooms**.

---

## 3. Exotic Fruit Classification — K-Nearest Neighbors

📓 [`Exotic_Fruit_Classification_KNN.ipynb`](Exotic_Fruit_Classification_KNN.ipynb) · [Open in Colab](https://colab.research.google.com/github/matteo-zirilli/ai-projects/blob/main/Exotic_Fruit_Classification_KNN.ipynb)

**Goal.** Classify 5 types of fruit (apple, banana, kiwi, orange, grapes) from their physical characteristics.

**Data.** 500 samples with 5 numeric features: weight, average diameter, average length, peel hardness and sweetness. The column and class names are in Italian, as in the original dataset; the notebook includes a glossary.

**Approach.**
- Analysis of the feature distributions and pair plots to spot which features separate the classes.
- **Yeo-Johnson power transformation** to make the features more normal and the feature space more isotropic, which matters for a distance-based model.
- Choice of K by comparing train/test accuracy for K = 1…20, then **5-fold cross-validation** and **GridSearchCV**.
- Confusion matrices and per-class performance charts.

**Results.**
- With **K = 15**, the model reaches **92% test accuracy** (94% train). Cross-validation gives a mean of 91.4% with a standard deviation of 2.4%.
- Banana and grapes are always classified correctly. Most errors are between orange, apple and kiwi.
- GridSearchCV selects K = 6 (best CV score 0.92). The notebook discusses why this differs from the K chosen on the test set.

---

## 4. Animal Recognition for Self-Driving Cars — CNN on CIFAR-10

📓 [`Animal_Recognition_for_Self_Driving_Cars.ipynb`](Animal_Recognition_for_Self_Driving_Cars.ipynb) · [Open in Colab](https://colab.research.google.com/github/matteo-zirilli/ai-projects/blob/main/Animal_Recognition_for_Self_Driving_Cars.ipynb)

**Scenario.** An image recognition system for road safety, able to distinguish animals from vehicles.

**Data.** **CIFAR-10** (60,000 color images, 32×32 px). The 10 original classes are regrouped into 3: **animals**, **road vehicles** (automobile, truck) and **non-road vehicles** (airplane, ship). The resulting dataset is imbalanced, with a 3:1:1 ratio.

**Approach.**
- A documented architectural decision: keep 3 classes instead of a binary problem, so that no data is thrown away and the system learns to ignore objects that are not relevant.
- A compact **Keras CNN** (2 Conv2D + MaxPooling blocks, GlobalAveragePooling, dense softmax head, about 84k parameters).
- Early stopping on the validation loss and an experiment with data augmentation, which was discarded with motivation.
- Evaluation focused on **macro-F1 and per-class recall** because of the imbalance, plus confusion matrices, learning curves and a visual analysis of misclassified images.

**Results.**
- **92% test accuracy and 0.90 macro-F1**. Every class is above 0.80 F1 (animals 0.95, road vehicles 0.90, non-road vehicles 0.84).
- The error analysis shows that the model is sometimes driven by the background. For example, airplanes in the sky and ships near vegetation get classified as animals.

---

## 5. Food Image Classification — CNN from Scratch vs Transfer Learning

📓 [`Food_Image_Classification_CNN_vs_Transfer_Learning.ipynb`](Food_Image_Classification_CNN_vs_Transfer_Learning.ipynb) · [Open in Colab](https://colab.research.google.com/github/matteo-zirilli/ai-projects/blob/main/Food_Image_Classification_CNN_vs_Transfer_Learning.ipynb)

**Scenario.** Optimize the performance of a neural network that classifies dishes for a food-tech company (*GourmetAI*).

**Data.** 14 balanced food categories (sushi, tacos, donuts, cheesecake, …): 8,960 training images and 2,800 test images.

**Approach.** Three PyTorch models compared under the same conditions:
1. **Baseline**: a 4-stage CNN built from scratch (about 527k parameters).
2. **Augmented**: the same CNN trained with data augmentation (rotation, horizontal flip).
3. **Transfer learning**: a pre-trained **VGG16** with the backbone frozen and a new 14-class output layer (about 57k trainable parameters).

The notebook also includes:
- learning-rate tuning
- a custom training loop with early stopping and best-weights restore
- saving and loading of weights and training histories on Google Drive
- evaluation with classification reports, learning curves and confusion matrices

**Results.**

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Baseline CNN | 0.62 | 0.62 |
| CNN + augmentation | 0.60 | 0.60 |
| **VGG16 transfer learning** | **0.77** | **0.77** |

Transfer learning gains about 15 points while training about 9× fewer parameters. The remaining errors are between visually similar dishes, for example taco vs taquito and apple pie vs cheesecake.

---

## 6. Flower Recognition for an AgriTech Company — Transfer Learning with ResNet-18

📓 [`Flower_Recognition_for_an_AgriTech_Company.ipynb`](Flower_Recognition_for_an_AgriTech_Company.ipynb) · [Open in Colab](https://colab.research.google.com/github/matteo-zirilli/ai-projects/blob/main/Flower_Recognition_for_an_AgriTech_Company.ipynb)

**Scenario.** A prototype for an AgriTech company that tells **daisies** from **dandelions** as reliably as possible.

**Data.** A small dataset: 1,275 training, 364 validation and 182 test images, slightly imbalanced.

**Approach.**
- A **ResNet-18 pre-trained on ImageNet** (via `timm`), with the backbone frozen and a new 2-class head.
- Augmentation designed for field conditions: rotation, flips, and brightness/contrast jitter.
- A reusable training framework: an `EarlyStopper` class with `min`/`max` mode and a training loop driven by a configurable metric. Here the metric is **macro-F1**, chosen instead of accuracy because of the imbalance.
- A manual data-quality review: one likely mislabeled image was found, and images with insects on the flowers were kept on purpose.

**Results.**
- **Macro-F1 ≈ 0.89 on the test set** (daisy 0.88, dandelion 0.91). Errors are almost symmetric between the two classes.
- The model already reached about 0.86 macro-F1 after the first epoch, a sign of how well ImageNet features transfer to this task.
- The notebook ends with a critical discussion of the limits (same-source test set, label noise) and next steps such as fine-tuning the backbone.

## 7. Cybersecurity Defense with Reinforcement Learning — SARSA vs Double DQN

📓 [`Cybersecurity_Defense_with_Reinforcement_Learning.ipynb`](Cybersecurity_Defense_with_Reinforcement_Learning.ipynb) · [Open in Colab](https://colab.research.google.com/github/matteo-zirilli/ai-projects/blob/main/Cybersecurity_Defense_with_Reinforcement_Learning.ipynb)

**Scenario.** Train an agent to defend a network under attack, using **gym-idsgame**: a two-player intrusion-detection game in which an attacker tries to compromise a data node and the defender reinforces the nodes and tries to detect the attack. The agent plays the defender against two built-in attackers: a **random** attacker and a **maximal** attacker that targets the weakest node.

**Approach.**
- **Making a legacy environment work.** `gym-idsgame` was written for old versions of Gym and does not run on a modern Colab environment as it is. The notebook patches a library bug and writes a custom single-agent wrapper that hides the multi-agent interface and exposes only the defender's point of view.
- **Environment analysis.** Reading the environment code clarified how episodes end and how rewards are assigned. That led to choosing the **successful-defense rate** as the main metric, and to setting random-defender baselines.
- **SARSA (tabular, on-policy)** with two state representations:
  - **Approach A:** the raw 20-value observation, which produced about 17,800 distinct states, half of them visited only once.
  - **Approach B:** 3 hand-crafted features, which reduced the states to 132 with a median of 55 visits.
  - The epsilon-decay schedule was tuned over several runs.
- **Double DQN (deep RL, off-policy)** in PyTorch, with replay buffer and target network. It was trained on both attack scenarios and the seed variability was checked.

**Results.**

| Scenario | Agent | Successful defenses |
|----------|-------|---------------------|
| Random attack | Random baseline | 51.8% |
| Random attack | SARSA, raw state | 51.4% |
| Random attack | SARSA, feature-based state | 73.2% (not reproducible across runs) |
| Random attack | Double DQN | 70.0% |
| Maximal attack | Random baseline | 35.4% |
| Maximal attack | **Double DQN** | **91.5%** |

- The clearest result is the Double DQN against the maximal attacker: from 35% to **91.5% of games won**, with a regular learning curve.
- Against the random attacker both methods can beat chance, but not reliably from run to run.
- The notebook explains why: the defender's observation contains no information about the attacker, so a greedy policy always plays the same fixed sequence.
- The conclusions discuss limits (single runs, untuned DDQN hyperparameters) and next steps.

---

## Tech stack

- **Languages & environment:** Python, Jupyter, Google Colab (GPU)
- **Data & ML:** pandas, NumPy, scikit-learn, SciPy
- **Deep Learning:** PyTorch, torchvision, timm, TensorFlow / Keras
- **Reinforcement Learning:** gym-idsgame (Gym), tabular SARSA, Double DQN
- **Visualization:** Matplotlib, seaborn

## Repository structure

```
ai-projects/
├── README.md
├── Phone_Book_Python_OOP.ipynb
├── Real_Estate_Price_Forecasting.ipynb
├── Exotic_Fruit_Classification_KNN.ipynb
├── Animal_Recognition_for_Self_Driving_Cars.ipynb
├── Food_Image_Classification_CNN_vs_Transfer_Learning.ipynb
├── Flower_Recognition_for_an_AgriTech_Company.ipynb
├── Cybersecurity_Defense_with_Reinforcement_Learning.ipynb
└── it/                      # original Italian versions of the notebooks
```

## How to run

Each notebook can be opened directly in Google Colab with the links above. The datasets are downloaded from public URLs or from `torchvision`/`keras.datasets`. Two notebooks expect files on Google Drive:
- the food classification dataset
- the phone book JSON file

To run them, adjust the Drive paths at the top of the notebook. Deep learning notebooks should be run on a GPU runtime.

A few notes on the translation:
- The saved outputs come from the original runs, so the text inside the charts (titles, axis labels) is still in Italian until a notebook is re-run.
- Some identifiers, dataset column names and class labels are kept in Italian where changing them would alter the program's behavior. Where this happens, the notebook includes a short glossary.

## License

Released under the [MIT License](LICENSE).
