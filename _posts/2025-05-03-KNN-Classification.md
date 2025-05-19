**1. The Business Event:**

**What Happened?** 
- ABC Grocery wants to know if we can predict who will signup to the "Delivery Club" based on recent activity.

**Who Was Involved?** 
 - Marketing department, "Delivery Club" management, and customers.

**2. The Core Challenge/Question:**
 
 **The Uncertainty/Problem:** 
 - We want to build a smart system that can predict which customers are most likely to sign up for the loyalty program. This code is the blueprint for creating and testing that prediction machine.

**Summary:**

We've taken raw customer data, cleaned it, transformed it into a language our model understands, intelligently selected the most predictive pieces of information, and then trained and fine-tuned a K-Nearest Neighbors model.

The end result is a **data-driven system capable of predicting which customers are likely to sign up**, along with a clear understanding (from the metrics and confusion matrix) of how accurate and reliable these predictions are. The feature importance insights also tell us _what_ characteristics are most indicative of a customer who will sign up, which can be incredibly valuable for marketing and business strategy! This is a comprehensive and robust approach to building a valuable classification model.


**Phase 1: Gathering Our Ingredients & Specialized Tools (The Setup)**

```python
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import RFECV
```

**What's happening?** We're essentially opening our data science toolkit. Each `import` line brings in a specialized set of tools:
- `pandas`: Our super-powered spreadsheet. It helps us organize and manipulate customer data.
- `pickle`: A way to save our work (like prepared data or even the finished model) so we can easily load it later.
- `matplotlib.pyplot` & `numpy`: Our drawing kit for charts/graphs (`matplotlib`) and our heavy-lifter for numerical calculations (`numpy`).
- `sklearn` (scikit-learn): This is the star! It's our main machine learning library, packed with tools for:
    - `KNeighborsClassifier`: The specific type of prediction model we'll be building. (More on this superstar later!)
    - `shuffle`: For mixing up our data to ensure fairness.
    - `train_test_split`: To wisely divide our data for training and testing our model.
    - Various `metrics`: These are our "measuring tapes" and "scorecards" to see how good our model is.
    - `OneHotEncoder`, `MinMaxScaler`: Tools to get our data into the perfect shape and format for the model.
    - `RFECV`: A clever tool to help us pick only the most important pieces of information from our data.

**Phase 2: Getting the Raw Ingredients & Initial Prep (Data Loading & Basic Cleaning)**

```python
# import data
data_for_model = pd.read_pickle("data/abc_classification_modelling.p")
# drop customer_id - don't need it.
data_for_model.drop("customer_id", axis = 1, inplace = True)
#shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)
```

**What's happening?**
    1. `pd.read_pickle(...)`: We're loading our main ingredient – the customer data. It's stored in a special `pickle` file, which means it's likely already been pre-processed to some extent.
    2. `data_for_model.drop("customer_id", ...)`: We're removing the "customer_id".
        - **Why?** While a customer ID is unique, it doesn't tell us _why_ a customer might sign up. It's like knowing a student's ID number – it doesn't predict their exam score. We need descriptive information.
    3. `data_for_model = shuffle(...)`: We're randomly mixing up all the customer records.
        - **Why?** Imagine if your data was sorted by sign-up date. The model might accidentally learn a pattern based on this order, which isn't a real predictive factor. Shuffling is like shuffling a deck of cards before dealing – it ensures fairness and helps the model learn genuine patterns. `random_state = 42` is like a specific shuffling technique that we can repeat exactly if needed.

**Phase 3: Quality Control for Our Ingredients (Data Health Checks)**

```python
# check class balance
data_for_model["signup_flag"].value_counts(normalize = True)

# deal with missing values
data_for_model.isna().sum()
data_for_model.dropna(how = "any", inplace = True)
```

**What's happening?**
- data_for_model["signup_flag"].value_counts(normalize = True)`: We're peeking at our target: "signup_flag" (did they sign up or not?). This line counts the proportion of customers who signed up versus those who didn't.
	- **Why?** If, say, 99% of customers _didn't_ sign up, a lazy model could just predict "no signup" for everyone and look 99% accurate! But that wouldn't be useful for finding those rare signups. Knowing this "class balance" helps us evaluate our model fairly later.
- `data_for_model.isna().sum()` & `data_for_model.dropna(...)`: We're hunting for any missing pieces of information (like blank cells in our spreadsheet). The comment "because so few missing values we will drop the rows" tells us that since only a tiny amount of data is incomplete, we're simply removing those few customers with missing details.
	- **Why?** Models don't like missing information. For a small number of missing entries, removal is often the simplest and cleanest solution.

**Phase 4: Handling Extreme Cases (Outlier Treatment)**

```python
# deal with outliers
outlier_investigation = data_for_model.describe()
outlier_columns = ["distance_from_store", "total_sales", "total_items"]
# Boxplot approach
for column in outlier_columns:
    # ... (IQR calculation code) ...
    data_for_model.drop(outliers, inplace=True)
```

**What's happening?** We're looking for "outliers" – data points that are wildly different from the rest. Imagine you're looking at customer spending, and most spend between $10-$200, but one customer somehow spent $5,000,000. That's an outlier!
    
1. We identify specific columns (`outlier_columns`) where these extremes might occur.
2. The loop uses a common statistical method (the "Interquartile Range" or IQR, often visualized with a boxplot) to define what's "normal."
      - It finds the range where the middle 50% of the data lies (from the 25th percentile to the 75th percentile).
      - It then sets up "fences" quite a bit wider than this middle range (here, `iqr * 2`). Any data point falling outside these extended fences is considered an outlier.
 3. `data_for_model.drop(outliers, ...)`: We remove these identified outlier customers from our dataset for the specified columns. 
       - **Why?** Extreme outliers can skew our model's understanding of typical customer behavior, like one very tall person making the average height of a group seem unusually high. By removing them, we aim for a model that learns from more representative patterns. The `iqr * 2` makes the "fences" quite generous, so we're only removing very extreme values.

**Phase 5: Structuring for Learning (Splitting Data)**

```python
# split input and output variables
X = data_for_model.drop(["signup_flag"], axis = 1)
y = data_for_model["signup_flag"]

# split training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
```

**What's happening?** This is fundamentally important!
  1. `X = ...` and `y = ...`: We separate our data. `X` contains all the information _about_ the customers that we'll use to make predictions (their characteristics, like age, gender, purchase history – these are our "features"). `y` contains the "signup_flag" – the outcome we want to predict.
  2. `train_test_split(...)`: We divide our data into two crucial sets:
        - **Training Set (`X_train`, `y_train`):** This is the larger portion (80% here, because `test_size = 0.2`). Our model will _learn_ from this data. It's like giving a student textbooks and practice problems.
        - **Test Set (`X_test`, `y_test`):** This smaller portion (20%) is kept completely separate and hidden from the model during training. We use it at the very end to evaluate how well the model learned and if it can make accurate predictions on _new, unseen customers_. It's like giving the student a final exam based on the concepts, but with questions they haven't seen before.
        - `stratify = y`: This is a smart touch! It ensures that both our training and test sets have a similar proportion of customers who signed up versus those who didn't. So, if 10% of all customers signed up, both our training and test sets will have roughly 10% signups. This makes our model's "exam" much fairer and more realistic.

**Phase 6: Making Data Model-Friendly (Transformations)**

Models understand numbers best. We need to convert some of our data.

```python
# deal with categorical values
categorical_vars = ["gender"]
# ... (OneHotEncoder code) ...
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis =1, inplace = True)
# ... (Similar for X_test) ...
```

**What's happening?** We're handling "categorical variables" like "gender."
    
  1. `OneHotEncoder`: This tool transforms text categories (e.g., 'Male', 'Female') into numbers. For 'gender', it might create a new column like 'gender_Male'. If a customer is male, this column gets a 1; otherwise, a 0. `drop="first"` is a technical step to avoid redundancy (if not male, and there are only two options, one must be female).
  2. The `fit_transform` is done on `X_train` (learn the categories and transform), and then `transform` is done on `X_test` (apply the _same_ learned transformation). This prevents "data leakage" – peeking at the test set answers during study time!
  3. The `pd.concat` and `drop` lines are about neatly merging these new numerical columns back into our main data tables and removing the original text-based 'gender' column. 

**Why?** This allows the model to mathematically use information like gender in its predictions.


```python
# feature scaling
scale_norm = MinMaxScaler()
X_train = pd.DataFrame(scale_norm.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(scale_norm.transform(X_test), columns = X_test.columns)
```

**What's happening?** We're performing "feature scaling" using `MinMaxScaler`.
- Imagine you have customer age (e.g., 20-80 years) and total sales (e.g., $10 - $5000). The sales numbers are much larger. Some models can be unfairly influenced by features with larger numerical values.
- `MinMaxScaler` rescales all numerical features to a common range, typically 0 to 1.
- Again, we `fit_transform` on `X_train` (learn the min/max for each feature from training data and scale) and then `transform` on `X_test` (apply the _same_ scaling using the training data's min/max).
- **Why?** This ensures all features contribute more equally to the model's learning process, especially for distance-based models like the K-Nearest Neighbors we'll use.

**Phase 7: Focusing on the "Vital Few" – Intelligent Feature Selection**

Not all information is equally helpful. Some data might even be noise!

```python
# feature selection
from sklearn.ensemble import RandomForestClassifier # Using a different model type for selection
clf_rf = RandomForestClassifier(random_state=42) # Renamed for clarity
feature_selector = RFECV(clf_rf) # Using the renamed Random Forest
fit = feature_selector.fit(X_train, y_train)
optimal_feature_count = feature_selector.n_features_
# ... (update X_train and X_test to keep only selected features) ...
# ... (plotting code for feature selection) ...
```

**What's happening?** We're using a sophisticated technique called "Recursive Feature Elimination with Cross-Validation" (`RFECV`).
    
1. We temporarily use another powerful model, `RandomForestClassifier` (think of it as a panel of experts), because it's very good at judging how important each feature is.
2. `RFECV` works like this:
    - It starts with all available features.
    - It builds a model and evaluates it.
    - Then, it "recursively" removes the least important feature.
    - It rebuilds the model and re-evaluates.
    - It repeats this process, eliminating features one by one.
    - "Cross-Validation" (`CV`) means it does this multiple times on different slices of the training data to get a very robust estimate of feature importance and model performance.
3. The goal is to find the smallest set of features (`optimal_feature_count`) that still gives us excellent predictive power.
4. We then trim down our `X_train` and `X_test` to include _only_ these champion features.
5. The plot visualizes this process, showing how model performance changes as features are removed, helping us confirm the optimal number. 
    
**Why?** Using fewer, more impactful features can lead to simpler, faster, and often more accurate and robust models. It's like decluttering – keeping only what truly sparks joy (or in this case, predictive power!).

**Phase 8: Building and Teaching Our Prediction Machine (The K-Nearest Neighbors Model)**

```python
# Model training
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
```

**What's happening?**
1. `clf = KNeighborsClassifier()`: Now we bring in our chosen model for the final prediction: the **K-Nearest Neighbors (KNN)** classifier.
    - **How KNN works (the brilliant intuition!):** To predict if a _new_ customer will sign up, KNN looks at the 'K' most similar customers (its "nearest neighbors") from the training data it has already seen. If the majority of these 'K' neighbors signed up, then KNN predicts that the new customer will also sign up. It's like saying, "Show me your 'K' closest friends (in terms of data characteristics), and I'll tell you who you are (or what you'll do)." 'K' is a number we'll fine-tune later.
2. `clf.fit(X_train, y_train)`: This is the "teaching" or "training" phase. The KNN model meticulously studies the (now highly prepared, scaled, and feature-selected) `X_train` data and the corresponding `y_train` signup flags. It's essentially memorizing the characteristics of the training customers to find neighbors later.

**Phase 9: The Grand Reveal – How Well Did Our Model Learn? (Model Assessment)**

```python
# Model Assessment
y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:, 1] # Probability of signup

# confusion matrix
# ... (code to create and plot confusion matrix) ...

# Accuracy, Precision, Recall, F1 Score calculations
accuracy_score(y_test, y_pred_class)
precision_score(y_test, y_pred_class)
recall_score(y_test, y_pred_class)
f1_score(y_test, y_pred_class)
```

**What's happening?** It's exam time for our model!
1. `y_pred_class = clf.predict(X_test)`: We ask our trained KNN model to predict the "signup_flag" for every customer in the `X_test` set (the data it has never seen).
2. `y_pred_prob = ...`: This line also gets predictions, but as probabilities – how confident is the model that a customer will sign up (e.g., 75% chance).
3. **Confusion Matrix:** This is a fantastic visual report card. It's a table that shows:
    - **True Positives (TP):** Customers who signed up, AND our model correctly predicted they would. (Awesome!)
    - **True Negatives (TN):** Customers who didn't sign up, AND our model correctly predicted they wouldn't. (Great!)
    - **False Positives (FP):** Customers who didn't sign up, BUT our model _incorrectly_ said they would. (A "false alarm" – Oops!)
    - **False Negatives (FN):** Customers who _did_ sign up, BUT our model _incorrectly_ said they wouldn't. (A "missed opportunity" – Double oops!) The plotted matrix helps us instantly see where the model is strong and where it's making errors.
4. **Key Performance Metrics:**
    - `accuracy_score`: "Overall, what percentage of predictions did the model get right (both signups and no signups)?"
    - `precision_score`: "When our model predicts a customer WILL sign up, how often is it correct?" (Important if the cost of a false alarm is high, e.g., sending an expensive welcome gift).
    - `recall_score`: "Of all the customers who ACTUALLY signed up, how many did our model successfully identify?" (Important if missing a potential signup is very costly).
    - `f1_score`: "This is a brilliant, balanced score that considers both Precision and Recall. It's often the go-to metric when you care about minimizing both false alarms and missed opportunities."

<img width="551" alt="Pasted image 20250517144834" src="https://github.com/user-attachments/assets/35bfa281-2992-4eff-980b-ca37431cbb4f" />


**Phase 10: Optimizing Our Machine – Finding the Perfect 'K' (Hyperparameter Tuning)**

Our KNN model's performance depends on that 'K' (number of neighbors). Is it better to look at 3 neighbors? 5? 10?

```python
# finding altimate value of K
k_list = list(range(2,25))
accuracy_scores = [] # Will store F1 scores
for k in k_list:
    # ... (train KNN with current k, predict, calculate F1 score) ...
    accuracy_scores.append(accuracy) # 'accuracy' here is actually F1 score
# ... (find k with max F1 score) ...
# ... (plot F1 score vs. K) ...
```

**What's happening?** We're systematically searching for the best value for 'K'.
    
1. `k_list = list(range(2,25))`: We create a list of 'K' values to try out (from 2 neighbors up to 24).
2. The `for k in k_list:` loop: For each of these 'K' values:
    - We build a _new_ KNN model using that specific 'K'.
    - We train it on `X_train`.
    - We make predictions on `X_test`.
    - We calculate the `f1_score` (our chosen metric for "goodness") for that 'K'.
    - We store this F1 score.
3. `max_accuracy = max(accuracy_scores) ... optimal_k_value = ...`: After trying all the 'K's, we find which 'K' value resulted in the highest F1 score. This `optimal_k_value` is the best setting for 'K' for our specific problem and data.
4. The final plot shows the F1 score for each 'K' value tested, with a red 'X' highlighting the champion 'K'. This visual helps confirm our choice and understand how sensitive the model is to 'K'. 
    
**Why?** This "hyperparameter tuning" step is like fine-tuning an engine. It ensures our chosen model (KNN) is working at its absolute best by finding its optimal internal settings.

<img width="914" alt="Pasted image 20250517145050" src="https://github.com/user-attachments/assets/150cf45a-9680-4efa-9c97-7c9b252d6fe7" />

**The Grand Conclusion of This Code's Journey:**

After all these meticulous steps, what have we achieved?

We've taken raw customer data, cleaned it, transformed it into a language our model understands, intelligently selected the most predictive pieces of information, and then trained and fine-tuned a K-Nearest Neighbors model.

The end result is a **data-driven system capable of predicting which customers are likely to sign up**, along with a clear understanding (from the metrics and confusion matrix) of how accurate and reliable these predictions are. The feature importance insights also tell us _what_ characteristics are most indicative of a customer who will sign up, which can be incredibly valuable for marketing and business strategy! This is a comprehensive and robust approach to building a valuable classification model.
