**Our Goal:** To build a reliable predictive model that can identify potential customers who will sign up.

**1. The Business Event:**

**What Happened?** 
- ABC Grocery recently concluded a promotional campaign for its "Delivery Club."

**Who Was Involved?** 
 - Marketing department, "Delivery Club" management, and customers.

**What Were the Initial Outcomes/Observations?** 
- The promotion successfully increased club memberships. However, there's internal discussion regarding the cost-effectiveness of the mailer quality.  ABC Grocery divided it's customers into 3 groups.  Group1 received a basic mailer, Group2 a nicer more expensive  mailer and Group3 received neither (control group). 

**2. The Core Challenge/Question:**
 
 **The Uncertainty/Problem:** 
 - Was the investment in the current mailer quality justifiedl? Specifically, did the higher-quality (and more expensive) mailer (mailer 2) generate a significantly higher number of sign-ups, or conversely, did the lower-quality (and cheaper) mailer (mailer 1) achieve similar results?

**Why Does It Matter?** 
- Understanding this relationship is crucial for optimizing future marketing spend, maximizing return on investment (ROI), and ensuring sustainable membership growth.

**The Data-Driven Goal:**
- To determine the relationship between mailer quality/cost and membership conversion rates to inform future campaign budget allocation for promotional materials.

**3. Methodology Used:


Here's how we get there:

**Phase 1: Setting Up Our Workshop and Getting the Raw Materials (Your Data)**


Python

```
# import data
data_for_model = pd.read_pickle("data/abc_classification_modelling.p")
# data_for_model = pickle.load(open("data/abc_classification_modelling.p", "rb"))

# drop customer_id - don't need it.
data_for_model.drop("customer_id", axis = 1, inplace = True)

#shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)
```

- **What it is:**
    1. `pd.read_pickle(...)`: We're loading your customer data, which has been previously processed and saved in a file named "abc_classification_modelling.p".
    2. `data_for_model.drop("customer_id", ...)`: We remove the "customer_id" column.
        - **Why?** While customer ID is unique to each customer, it doesn't actually tell us anything about _why_ they might sign up. It's like knowing a student's ID number – it doesn't predict their test score. We need information that describes their characteristics or behaviors.
    3. `data_for_model = shuffle(...)`: We randomly mix up the rows of your customer data.
        - **Why?** Imagine your data was sorted by when customers joined. If we don't shuffle, our model might accidentally learn something based on this order, which isn't a real pattern. Shuffling ensures fairness and helps the model learn more general patterns. `random_state = 42` is just a way to ensure that if we run this code again, the shuffle happens in the exact same way, making our results reproducible.

**Phase 2: Cleaning and Preparing Your Data (Quality Control)**

Good predictions require good quality data. This phase is all about making sure the data is ready for the model.

Python

```
# check class balance
data_for_model["signup_flag"].value_counts(normalize = True)
```

- **What it is:** We're looking at the "signup_flag" column (which tells us if a customer signed up or not) and counting how many customers did sign up versus how many didn't. `normalize = True` shows this as percentages.
    - **Why?** It's important to know if we have a balanced dataset. For example, if 95% of customers didn't sign up and only 5% did, a lazy model could just predict "no signup" for everyone and be 95% accurate, but it wouldn't be useful for finding those valuable 5%. Knowing this balance helps us choose the right ways to evaluate our model later.

<!-- end list -->

Python

```
# deal with missing values
data_for_model.isna().sum()
# because so few missing values we will drop the rows.
data_for_model.dropna(how = "any", inplace = True)
```

- **What it is:**
    1. `data_for_model.isna().sum()`: This counts any missing pieces of information (blank cells) in each column of your data.
    2. `data_for_model.dropna(...)`: If we find any rows (customers) with missing information, we remove them.
        - **Why?** Predictive models don't like missing information. Since the comment says "so few missing values," dropping them is a simple and effective solution here. If there were many, we'd use more complex methods to fill them in.

<!-- end list -->

Python

```
# deal with outliers
outlier_investigation = data_for_model.describe()

outlier_columns = ["distance_from_store", "total_sales", "total_items"]

# Boxplot approach
for column in outlier_columns:
    
    lower_quartile = data_for_model[column].quantile(0.25)
    upper_quartile = data_for_model[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2 # Using a multiplier of 2 instead of 1.5 for a wider range
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = data_for_model[(data_for_model[column] < min_border) | (data_for_model[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    data_for_model.drop(outliers, inplace=True)
```

- **What it is:** This section handles "outliers" – data points that are unusually high or low compared to the rest.
    1. `data_for_model.describe()`: Gives us a statistical summary (like average, min, max) for numerical columns, helping us spot potential outliers.
    2. `outlier_columns = [...]`: We've identified specific columns where we want to look for extreme values (e.g., a customer extremely far from a store, or someone with exceptionally high sales).
    3. The `for` loop: For each of these columns, we're using a common statistical method (based on "quartiles" and the "interquartile range" or IQR) to define a "normal" range.
        - Think of quartiles like dividing your data into four equal parts. The IQR is the range covering the middle 50% of your data.
        - We then extend this range (here, by multiplying the IQR by 2). Any data point falling outside these wider "fences" (`min_border`, `max_border`) is considered an outlier.
    4. `data_for_model.drop(outliers, ...)`: We remove these identified outlier customers from our dataset.
        - **Why?** Extreme values can sometimes skew the model's learning process, making it less accurate for the typical customer. By removing them, we aim for a model that better represents the majority. The multiplier of 2 (instead of a more common 1.5) means we're being a bit more lenient and only removing more extreme outliers.

**Phase 3: Structuring Data for the Model**

Python

```
# split input and output variables
X = data_for_model.drop(["signup_flag"], axis = 1)
y = data_for_model["signup_flag"]
```

- **What it is:** We're splitting our data into two parts:
    - `X`: This contains all the information _about_ the customers that we'll use to make predictions (e.g., their gender, distance from store, total sales). These are our "input features" or "predictors."
    - `y`: This contains only the "signup_flag" – the outcome we want to predict. This is our "target variable."
    - **Why?** The model learns by looking at `X` and trying to figure out the pattern that leads to the outcome in `y`.

<!-- end list -->

Python

```
# split training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
```

- **What it is:** This is a critical step. We divide our data (`X` and `y`) into two sets:
    - **Training Set (`X_train`, `y_train`):** This is usually the larger portion (here, 80% because `test_size = 0.2` means 20% for testing). Our model will learn from this data. It's like giving a student textbooks and examples to study.
    - **Test Set (`X_test`, `y_test`):** This smaller portion (20%) is kept separate. The model never sees this data during its learning phase. We use it at the end to evaluate how well the model learned to predict on new, unseen data. It's like giving the student a final exam.
    - **Why?** This prevents "cheating." If the model saw all the data, it might just memorize it and perform well, but then fail on new customers. The test set gives us a true measure of its predictive power.
    - `stratify = y`: This is an important detail. It ensures that both the training and test sets have roughly the same percentage of customers who signed up and who didn't, just like in the original dataset. This makes our test more realistic.

**Phase 4: Further Data Refinement (Making it Model-Friendly)**

Models often have specific requirements for the data they consume.

Python

```
# deal with categorical values
categorical_vars = ["gender"]
one_hot_encoder = OneHotEncoder(sparse_output = False, drop = "first")

# ... (encoding steps) ...

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis =1, inplace = True)

# (similar steps for X_test)
X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis =1, inplace = True)
```

- **What it is:** Some data isn't numerical (e.g., "gender" might be 'Male', 'Female'). Models usually prefer numbers.
    
    1. `categorical_vars = ["gender"]`: We identify the column "gender" as needing transformation.
    2. `OneHotEncoder(...)`: This tool converts categorical text data into numbers. For example, if "gender" has 'Male' and 'Female':
        - It might create a new column, say "gender_Male". A customer who is male would have a '1' in this column, and a '0' if female.
        - `drop = "first"` is a technical detail to avoid redundant information (if a customer isn't 'Male', and there are only two options, they must be 'Female').
    3. `fit_transform(X_train[categorical_vars])`: The encoder _learns_ the categories from the **training data** and transforms it.
    4. `transform(X_test[categorical_vars])`: The _same_ learned transformation is applied to the **test data**.
    5. The rest of the lines in this block are about neatly adding these new numerical columns back into our `X_train` and `X_test` tables and removing the original text column ("gender"). <!-- end list -->
    
    - **Why?** This allows the model to use information like gender in its calculations. It's crucial to learn (fit) on the training data only and then apply (transform) that learning to the test data to simulate real-world scenarios where new data comes in.

<!-- end list -->

Python

```
# feature scaling
scale_norm = MinMaxScaler()
X_train = pd.DataFrame(scale_norm.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(scale_norm.transform(X_test), columns = X_test.columns)
```

- **What it is:** This step, "feature scaling," adjusts the range of our numerical data.
    
    1. `MinMaxScaler()`: This specific scaler transforms each feature (like 'total_sales' or 'age') so that all its values fall between 0 and 1.
    2. `fit_transform(X_train)`: It learns the minimum and maximum values for each column from the **training data** and then scales it.
    3. `transform(X_test)`: It applies the _same_ scaling (using the min/max from the training data) to the **test data**. <!-- end list -->
    
    - **Why?** Some models, including the `KNeighborsClassifier` we're using, can be sensitive to the scale of the data. If one feature has very large numbers (e.g., sales in thousands) and another has small numbers (e.g., number of items bought, 1-10), the feature with larger numbers might unfairly dominate the model's calculations. Scaling puts all features on a level playing field.

**Phase 5: Selecting the Most Important Information (Focusing the Model)**

Python

```
# feature selection
from sklearn.ensemble import RandomForestClassifier # Using a different model type for selection

clf_rf = RandomForestClassifier(random_state=42) # Renamed to avoid conflict with KNN clf
feature_selector = RFECV(clf_rf) # Using the renamed RandomForestClassifier

fit = feature_selector.fit(X_train, y_train)

optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features: {optimal_feature_count}")

X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]

# ... (plotting code) ...
plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker="o")
# ... (rest of plotting) ...
plt.show()
```

- **What it is:** We might have a lot of information (features) about our customers, but not all of it might be equally important for predicting signups. This step helps us select only the most relevant features.
    
    1. `RandomForestClassifier(...)`: Interestingly, we're temporarily using a different type of model (Random Forest) here. This model is quite good at figuring out which features are most influential.
    2. `RFECV(clf_rf)`: This stands for "Recursive Feature Elimination with Cross-Validation." It's a smart technique:
        - It starts with all your features.
        - It builds a model (our temporary Random Forest) and sees how well it performs.
        - Then, it removes the least important feature and builds a new model.
        - It repeats this, removing features one by one, and uses a robust testing method ("cross-validation" on the training data) to see how performance changes.
        - The goal is to find the smallest set of features that still gives good predictive accuracy.
    3. `fit = feature_selector.fit(X_train, y_train)`: The RFECV process is run on our training data.
    4. `optimal_feature_count`: This tells us the ideal number of features identified by RFECV.
    5. `X_train = X_train.loc[:, feature_selector.get_support()]` and `X_test = ...`: We update our training and test datasets to keep _only_ these most important features.
    6. The `plt.plot(...)` code generates a graph. This graph shows how the model's score (how good it is) changes as we use different numbers of features. We're looking for the point where using more features doesn't really improve the score much. <!-- end list -->
    
    - **Why?** Using fewer, more relevant features can lead to simpler, faster, and sometimes even more accurate models. It's like decluttering – keeping only what's truly useful.

**Phase 6: Building and Testing Our Predictive Model**

Now we finally build and evaluate the model we set out to create.

Python

```
# Model training
clf = KNeighborsClassifier() # This is our chosen model
clf.fit(X_train, y_train)
```

- **What it is:**
    1. `clf = KNeighborsClassifier()`: We create an instance of our chosen model, the "K-Nearest Neighbors" classifier.
        - **How it works (simply):** To predict if a _new_ customer will sign up, this model looks at the 'K' most similar customers it has seen in the _training data_ (its "neighbors"). If the majority of those neighbors signed up, it predicts the new customer will also sign up. The 'K' is a number we can tune.
    2. `clf.fit(X_train, y_train)`: This is the "training" step. The model learns the patterns from our prepared training data (`X_train` which now has scaled, encoded, and selected features) and the corresponding signup outcomes (`y_train`).

<!-- end list -->

Python

```
# Model Assessment
y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:, 1] # Probability of signup
```

- **What it is:**
    1. `y_pred_class = clf.predict(X_test)`: Now we use our trained model (`clf`) to make predictions on the **test set** (`X_test`) – the data it has never seen before. `y_pred_class` will contain the model's predictions (e.g., 1 for signup, 0 for no signup) for each customer in the test set.
    2. `y_pred_prob = clf.predict_proba(X_test)[:, 1]`: This gives us more detail – not just the prediction, but the _probability_ or confidence the model has that a customer will sign up. For example, it might be 85% sure one customer will sign up, and 60% sure for another. We're grabbing the probability for the "signup" class.

<!-- end list -->

Python

```
# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
# ... (plotting code for confusion matrix) ...
plt.show()
```

- **What it is:**
    
    1. `confusion_matrix(...)`: This creates a table that gives us a detailed breakdown of our model's performance by comparing its predictions (`y_pred_class`) to the actual outcomes (`y_test`). It shows:
        - **True Positives (TP):** Customers who actually signed up, and the model correctly predicted they would. (Good!)
        - **True Negatives (TN):** Customers who didn't sign up, and the model correctly predicted they wouldn't. (Good!)
        - **False Positives (FP):** Customers who didn't sign up, but the model _incorrectly_ predicted they would. (A "false alarm.")
        - **False Negatives (FN):** Customers who actually signed up, but the model _incorrectly_ predicted they wouldn't. (A "missed opportunity.")
    2. The plotting code then visualizes this matrix as a colored grid, making it easy to see these four categories. <!-- end list -->
    
    - **Why?** This matrix is crucial for understanding _how_ our model is performing, not just if it's "accurate." Depending1 on your business goals, some errors (like missing a potential signup) might be more costly than others (like wrongly predicting a signup).

<!-- end list -->

Python

```
# Accuracy
accuracy_score(y_test, y_pred_class)
# precision
precision_score(y_test, y_pred_class)
# Recall
recall_score(y_test, y_pred_class)
# F1 Score
f1_score(y_test, y_pred_class)
```

- **What it is:** These lines calculate several standard metrics to evaluate the model:
    - **Accuracy:** What percentage of predictions did the model get right overall (both signups and no-signups)?
        - `(TP + TN) / Total`
    - **Precision:** When the model predicted a customer _would_ sign up, how often was it correct?
        - `TP / (TP + FP)`
        - **Why it matters:** High precision is important if the cost of a false positive is high (e.g., if you offer an expensive incentive to predicted signups, you don't want to waste it on those who won't actually sign up).
    - **Recall (or Sensitivity):** Of all the customers who _actually_ signed up, how many did our model correctly identify?
        - `TP / (TP + FN)`
        - **Why it matters:** High recall is important if the cost of missing a potential signup (a false negative) is high. You want to capture as many true signups as possible.
    - **F1 Score:** This is a balanced measure that considers both Precision and Recall. It's the harmonic mean of the two.
        - **Why it matters:** It's often useful when you need a good balance between not having too many false alarms (Precision) and not missing too many actual opportunities (Recall).

**Phase 7: Fine-Tuning the Model (Finding the Best 'K')**

Our K-Nearest Neighbors model depends on a parameter 'K' (the number of neighbors to consider). We need to find the best value for 'K'.

Python

```
# finding ultimate value of K
k_list = list(range(2,25)) # K values to try
accuracy_scores = [] # To store F1 scores for each K

for k in k_list:
    
    clf_k = KNeighborsClassifier(n_neighbors=k) # Create model with current K, renamed to avoid conflict
    clf_k.fit(X_train, y_train) # Train it
    y_pred = clf_k.predict(X_test) # Predict
    current_accuracy = f1_score(y_test, y_pred) # Calculate F1 score (variable renamed for clarity)
    accuracy_scores.append(current_accuracy)

max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_k_value = k_list[max_accuracy_idx]

# ... (plotting code for K values) ...
plt.plot(k_list, accuracy_scores)
# ... (rest of plotting) ...
plt.show()
```

- **What it is:**
    
    1. `k_list = list(range(2,25))`: We create a list of 'K' values we want to test (from 2 neighbors up to 24 neighbors).
    2. The `for k in k_list:` loop: For each value of 'K' in our list:
        - We create a _new_ `KNeighborsClassifier` model using that specific 'K'.
        - We train this model on our `X_train` and `y_train` data.
        - We make predictions on the `X_test` data.
        - We calculate the `f1_score` for these predictions (the code calls this `accuracy`, but it's using `f1_score` which is good for balance).
        - We store this F1 score.
    3. `max_accuracy = max(accuracy_scores) ... optimal_k_value = ...`: After trying all the 'K' values, we find which 'K' gave the highest F1 score. This is our "optimal K."
    4. The `plt.plot(...)` code then generates a graph showing the F1 score for each 'K' value. The red 'x' on the graph will mark the 'K' that performed best. <!-- end list -->
    
    - **Why?** The choice of 'K' can significantly impact the model's performance. Too small a 'K' can make the model sensitive to noise; too large a 'K' can make it overlook local patterns. This process helps us empirically find the sweet spot for 'K' for your specific data.

**In Summary:**

This code takes your customer data through a comprehensive pipeline:

1. **Loading and Initial Cleanup:** Getting the data in, removing irrelevant IDs, and shuffling.
2. **Data Quality Checks:** Looking at class balance, handling missing information, and removing extreme outliers.
3. **Structuring for Modeling:** Separating inputs from the target, and splitting data for reliable training and testing.
4. **Data Transformation:** Converting text data (like gender) to numbers and scaling all numerical features to a common range.
5. **Feature Selection:** Intelligently picking out the most important pieces of customer information for making predictions.
6. **Model Training & Evaluation:** Building the K-Nearest Neighbors model, testing it on unseen data, and looking at various metrics (like accuracy, precision, recall, and the confusion matrix) to understand its strengths and weaknesses.
7. **Model Optimization:** Fine-tuning the model by finding the best number of 'neighbors' (K) to consider.

The end result is a K-Nearest Neighbors model that has been carefully prepared, trained, and optimized to predict customer signups based on the patterns learned from your historical data. The charts and metrics produced help us understand how well it's likely to perform.
