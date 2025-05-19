---
layout: post
title: Predicting Customer Loyalty Scores
image: "/posts/grocery_shopping.png"
tags: [ABC, loyalty]
---


# Predicting Missing Loyalty Scores

**1. The Business Event:**

**What Happened?** 
- Our client, ABC Grocery, manages a customer loyalty program reliant on a comprehensive customer database.

**Who Was Involved?** 
 - Marketing department, loyalty program management, and customers.

**What Were the Initial Outcomes/Observations?** 
- Some customer records are missing loyalty scores making it impossible for ABC to personalize engagement with the customer. 

**2. The Core Challenge/Question:**
 
 **The Uncertainty/Problem:** 
 - A historical inconsistency in data capture by agents has resulted in a significant portion of customer records lacking loyalty scores. This data gap critically undermines the ability to accurately segment customers, personalize engagement, and equitably distribute rewards, thereby limiting the overall effectiveness and ROI of the loyalty program. 

**Why Does It Matter?** 
- Numerous customers lacking assigned loyalty scores, directly impairs the program's ability to function as designed. Without these scores, targeted marketing, personalized offers, and effective loyalty tier management are compromised.

**The Data-Driven Goal:**
- To create a system (a predictive model) that can estimate a customer's loyalty score based on their other characteristics. We also want to understand which customer details are most important for this prediction.

**3. Methodology Used:**
- Random Forest Regressor Model. This is our main prediction tool. Because we're predicting a numerical score (like a loyalty score from 1 to 100), we use a "Regressor." Think of it as a committee of many expert decision-makers (decision trees) who collaborate to come up with a precise numerical estimate.
- Cross Validation using KFold. 
- R - Square and Adjusted R Square scoring.
- Feature Importance of Random Forest: determining what features are important.

**In Summary:**

- **The first 5 phases** is all about the detailed process of building a predictive model: preparing data, training the Random Forest Regressor, rigorously evaluating how well it predicts customer loyalty scores, understanding which customer features are most important for this prediction, and finally, saving the trained model and the data preparation tools (like the encoder).
- **The 6th phase** takes that saved model and applies it to a completely new set of customers. It then generates loyalty score predictions for these new customers.

These phases together demonstrate a common workflow in data science: first, you build and validate a model, and then you deploy it to make predictions on new, incoming data.


**Phase 1: Setting Up Tools and Getting the Data**

** We're gathering our toolkit:**
- `pandas`, `pickle`, `matplotlib.pyplot`: Our standard tools for handling data in tables, saving/loading our work, and drawing charts.
- `RandomForestRegressor` (from `sklearn.ensemble`): This is our main prediction tool. Because we're predicting a numerical score (like a loyalty score from 1 to 100), we use a "Regressor." Think of it as a committee of many expert decision-makers (decision trees) who collaborate to come up with a precise numerical estimate.
- `shuffle`, `train_test_split`: For mixing and splitting our data for reliable model building.
- `cross_val_score`, `KFold`: For more thorough testing of our model's performance.
- `r2_score`: A specific way to measure how good our model is at predicting these numerical scores. It's called "R-squared."
- `OneHotEncoder`: To convert non-numerical data (like 'gender') into a format the model can understand.
- `permutation_importance`: An advanced technique to figure out which pieces of customer information are most influential in predicting their loyalty score.

```python
# import data
data_for_model = pickle.load(open("data/abc_regression_modelling.p", "rb"))
# drop customer_id - don't need it.
data_for_model.drop("customer_id", axis = 1, inplace = True)
#shuffle data
data_for_model = shuffle(data_for_model, random_state = 42)
```

**What it is:**
    1. `pickle.load(...)`: We're loading customer data that was previously prepared and saved in a file named "abc_regression_modelling.p".
    2. `data_for_model.drop("customer_id", ...)`: We remove the "customer_id" column because, while it identifies customers, it doesn't help predict their loyalty score.
    3. `data_for_model = shuffle(...)`: We randomly mix up the order of customer records to ensure fairness when we train our model.


**Phase 2: Data Cleaning and Preparation**

```python
# deal with missing values
data_for_model.isna().sum()
# because so few missing values we will drop the rows.
data_for_model.dropna(how = "any", inplace = True)
```

**What it is:** We check for any missing information (blank cells) for any customer. Since there are "so few missing values," we simply remove the records of those few customers with incomplete data.

```python
# split input and output variables
X = data_for_model.drop(["customer_loyalty_score"], axis = 1)
y = data_for_model["customer_loyalty_score"]
```

**What it is:** We separate our data into two parts:
   - `X`: This contains all the information _about_ the customers that we'll use to make predictions (e.g., their gender, purchase history, etc.). These are our "input features."
   - `y`: This contains only the "customer_loyalty_score" – the actual numerical score we want our model to learn to predict. This is our "target variable."

```python
# split training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```

**What it is:** We divide our data into a training set (80%) and a test set (20%).
   - The model will _learn_ from the training data (`X_train`, `y_train`).
   - The test data (`X_test`, `y_test`) is kept hidden from the model during learning. We'll use it later to see how accurately the model can predict loyalty scores for customers it hasn't seen before.

```python
# deal with categorical values
# ... (OneHotEncoder code as explained in previous examples) ...
```

**What it is:** This section converts non-numerical data, like the "gender" column, into a numerical format that the `RandomForestRegressor` can understand. It creates new columns (e.g., "gender_Male") with 1s and 0s. This process is applied consistently to both the training and test sets, learning the categories from the training data first.

**Phase 3: Building and Evaluating the Loyalty Score Predictor**

```python
# Model training
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)
```

**What it is:**
    1. `regressor = RandomForestRegressor(...)`: We create our Random Forest Regressor model.
    2. `regressor.fit(X_train, y_train)`: This is the "training" step. The model carefully analyzes the customer information in `X_train` and their corresponding actual loyalty scores in `y_train`. It learns the complex patterns and relationships between customer characteristics and how loyal they are.

```python
# Model Assessment
# predict on test set
y_pred = regressor.predict(X_test)
```

**What it is:** Now we use our trained `regressor` to make predictions on the `X_test` data – the customers the model has never seen before. `y_pred` will contain the model's estimated loyalty scores for these test customers.

```python
# calculate R-Squared
r_squared = r2_score(y_test, y_pred)
print(r_squared)
```

**What it is:** We calculate the "R-squared" value.
 - **Why?** R-squared tells us how much of the variation in actual customer loyalty scores our model can explain with its predictions. It's a score between 0 and 1. An R-squared of 0.70, for example, means our model can explain 70% of why loyalty scores differ among customers, based on the information we gave it. A higher R-squared is generally better, indicating a closer fit between predictions and actual scores.

```python
# cross Validation
cv = KFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring="r2")
cv_scores.mean()
```

**What it is:** This is a more robust way to check our model's performance, called "cross-validation."
- We take our _training data_ and split it further into several smaller pieces (here, 4 "folds" using `KFold`).
- The model is trained on 3 of these pieces and tested on the 1 remaining piece. This is repeated 4 times, with each piece getting a turn to be the test piece.
- We get an R-squared score for each of these 4 rounds. `cv_scores.mean()` gives us the average R-squared across these tests.
- **Why?** This gives us a more reliable estimate of how well our model is likely to perform on completely new, unseen data, as it reduces the chance that our initial good (or bad) `r_squared` on the single test set was just due to luck in how that particular test set was chosen.

```python
# calculated adjusted r-square store
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)
```

**What it is:** We calculate the "Adjusted R-squared."
   - **Why?** Regular R-squared can sometimes be misleading if you add many input variables (features) to your model – even if those new variables aren't very helpful, R-squared might still go up slightly. Adjusted R-squared accounts for the number of input variables used. It "penalizes" the score for adding variables that don't meaningfully improve the model's predictive power. This often gives a more honest assessment of the model's performance, especially when comparing models with different numbers of features.

**Phase 4: Understanding What Drives Loyalty (Feature Importance)**

It's not enough to predict; we also want to know _why_ the model makes certain predictions. Which customer characteristics are most important for determining loyalty?

```python
# feature importance
# ... (code to get feature_importances_ from regressor) ...
plt.barh(...) # Bar chart of feature importances
plt.show()
```

**What it is:** Our `RandomForestRegressor` can directly tell us which input features (like age, purchase frequency, gender, etc.) it found most influential when making its predictions. We then create a horizontal bar chart to visualize these importances, making it easy to see which factors have the biggest impact on predicting loyalty scores.


<img width="425" alt="Pasted image 20250516164449" src="https://github.com/user-attachments/assets/e58c0b2c-cf18-495c-8737-b2f73595bd1b" />


```
# Permutation Importance
# ... (code using permutation_importance) ...
plt.barh(...) # Bar chart of permutation importances
plt.show()
```

**What it is:** This is another, often more reliable, method to assess feature importance.
    
1. We take our trained model and our test data.
2. For each input feature (e.g., 'age'), we randomly shuffle _just that feature's values_ in the test data. This breaks any real relationship that feature had with the loyalty scores.
3. We then see how much worse our model's R-squared score gets with this shuffled data.
4. If the score drops a lot, that feature was very important. If the score barely changes, the feature wasn't critical.
5. This process is repeated several times (`n_repeats=10`) for each feature to get a stable average.
    
We then plot these "permutation importances" in another bar chart.
- **Why two methods?** They look at importance from slightly different angles. The first method (`feature_importances_`) comes directly from how the Random Forest was built, while permutation importance is based on how performance changes when a feature is "neutralized." Seeing both can give you more confidence in which features truly matter.


<img width="426" alt="Pasted image 20250516162955" src="https://github.com/user-attachments/assets/bac4c66f-efa5-4512-9b1e-ae29cfd77d96" />


**Phase 5: A Peek Under the Hood of Random Forest (Optional Detail)**

```python
# predictions under the hood
# ... (code iterating through regressor.estimators_) ...
```

**What it is:** This section is a demonstration of how the Random Forest makes its final prediction.
 - A Random Forest is made up of many individual "decision trees."
- This code takes the first customer from our test set.
- It then asks each individual tree in the forest to make its own prediction for that customer's loyalty score.
- Finally, it shows that the Random Forest's overall prediction (which we got earlier with `y_pred[0]`) is essentially the _average_ of the predictions from all those individual trees.
- **Why show this?** It helps to demystify the Random Forest a bit, showing it's a combination of many simpler predictions, making its final output more robust and generally more accurate than any single tree's prediction.


---

**Using the Model to Predict Loyalty for New Customers**

**Our Goal:** Now that we have a trained model, we want to use it to predict the customer loyalty scores for a _new_ batch of customers.

**Phase 6: Making the Predictions!**

```python
# make our predictions
loyalty_predictions = regressor.predict(to_be_scored)
```

**What it is:** This is the payoff!
  - We take our loaded and trained `regressor` model.
  - We feed it the newly prepared `to_be_scored` customer data.
  - The `regressor.predict(...)` function then calculates the predicted customer loyalty score for each customer in our new dataset.
  - The variable `loyalty_predictions` will now hold a list of these predicted scores.

**In Summary:**

- **The first 5 phases** is all about the detailed process of building a predictive model: preparing data, training the Random Forest Regressor, rigorously evaluating how well it predicts customer loyalty scores, understanding which customer features are most important for this prediction, and finally, saving the trained model and the data preparation tools (like the encoder).
- **The 6th phase** takes that saved model and applies it to a completely new set of customers. It then generates loyalty score predictions for these new customers.

These phases together demonstrate a common workflow in data science: first, you build and validate a model, and then you deploy it to make predictions on new, incoming data.
