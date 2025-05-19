---
layout: post
title: Hypothesis Testing - Promoting Delivery Club
image: "/posts/ABTesting.png"
tags: [ABTesting, Hypothesis]
---

# Mailer Quality vs. Membership Growth. Optimizing ABC Grocery's "Delivery Club" Promotion Spend

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
- To determine if there's a statistically significant relationship between the `mailer_type` sent to customers and their `signup_flag` (whether they signed up or not). In simpler terms: did one mailer type work better than another at getting signups, or are any differences we see just due to random chance?

**Phase 1: Setting Up and Getting the Data**

```python
import pandas as pd
from scipy.stats import chi2_contingency, chi2
```

**What it is:** We're importing our necessary tools:
- `pandas as pd`: This is our trusty library for working with data in tables (like spreadsheets, but in code).
- `chi2_contingency`, `chi2` (from `scipy.stats`): These are specialized statistical functions. `chi2_contingency` will be the main engine for our test, and `chi2` helps us interpret the results.

```python
campaign_data = pd.read_excel("grocery_database.xlsx", sheet_name="campaign_data")
```

**What it is:** We're loading data from an Excel file named "grocery_database.xlsx," specifically from a sheet (tab) called "campaign_data." This table likely contains information about the campaign, such as which customer received which mailer and whether they signed up.

**Phase 2: Focusing the Analysis and Summarizing Observations**

```python
campaign_data = campaign_data.loc[campaign_data["mailer_type"] != "Control"]
```

**What it is:** We're filtering our data. It looks like there might have been a "Control" group in the campaign (perhaps customers who received no mailer or a standard one). This line removes those "Control" group records from our analysis.
- **Why?** For this specific test, we're interested in comparing the effectiveness of the _different active mailer types_ against each other (e.g., comparing Mailer Type A directly with Mailer Type B), rather than comparing them to a no-mailer baseline.

```python
observed_values = pd.crosstab(campaign_data["mailer_type"], campaign_data["signup_flag"]).values
```

**What it is:** This is a very important step. We're creating a summary table, often called a "contingency table" or "cross-tabulation."
- Imagine a table where the rows are the different `mailer_type` (e.g., "Mailer 1", "Mailer 2").
- The columns are the `signup_flag` (e.g., "Signed Up", "Did Not Sign Up").
- The numbers _inside_ this table are the actual counts of how many customers fall into each combination. For example, it would show how many people got "Mailer 1" AND "Signed Up", how many got "Mailer 1" AND "Did Not Sign Up", and so on for "Mailer 2".
- `.values` just extracts these counts into a format our statistical test can use. These are our "observed" results from the campaign.

**Phase 3: Getting a Quick Look at Signup Rates (Descriptive Statistics)**

```python
mailer1_signup_rate = 123 / (252 + 123)
mailer2_signup_rate = 127 / (209 + 127)
print(mailer1_signup_rate, mailer2_signup_rate)
```

**What it is:** Here, we're calculating the raw signup rates for two specific mailer types. The numbers (123, 252, 127, 209) would have come from the `observed_values` table we just created.
- For "Mailer 1": 123 people signed up, and 252 did not. So, the total who received Mailer 1 was (252 + 123). The signup rate is (number who signed up) / (total who received it).
- The same calculation is done for "Mailer 2."
- **Why?** This gives us a straightforward look at how each mailer performed. For instance, if Mailer 1 has a rate of 0.32 (32%) and Mailer 2 has a rate of 0.37 (37%), we see Mailer 2 did slightly better in terms of raw numbers. However, this doesn't tell us if this difference is "real" (statistically significant) or if it could have just happened by random chance. That's what the next steps are for.

**Phase 4: Setting Up the Formal Statistical Test (Hypothesis Testing)**

This is where we use a method called "hypothesis testing" to make a data-driven decision.

```python
null_hypothesis = "There is no relationship between mailer_type and signup_flag"
alternate_hypothesis = "There is a relationship between mailer_type and signup_flag"
acceptance_criteria = 0.05
```

**What it is:**
- `null_hypothesis`: This is our starting assumption, like a "devil's advocate" position. It states that the type of mailer has _no effect_ on whether someone signs up. Any differences we see in the signup rates are purely due to random variation in our sample of customers.
- `alternate_hypothesis`: This is what we're trying to find evidence _for_. It states that there _is_ a real relationship – the type of mailer sent _does_ influence whether people sign up.
- `acceptance_criteria = 0.05`: This is our "significance level" (often called alpha). It's a threshold we set _before_ doing the test. 0.05 (or 5%) is a common choice. It means we're willing to accept a 5% chance of making a mistake – specifically, wrongly concluding there _is_ a relationship when there isn't one (this is called a "Type I error" or a "false positive").

**Phase 5: Performing the Chi-Squared Test**

```python
chi2_statistic, p_value, dof, expected_values = chi2_contingency(observed_values, correction=False)
print(chi2_statistic, p_value)
```

**What it is:** This is where the core statistical calculation happens using the `chi2_contingency` function.
- It takes our `observed_values` (the actual counts from our campaign).
- It internally calculates what the counts _would theoretically look like if the null hypothesis were true_ (i.e., if mailer type had no effect on signups). These are the `expected_values`.
- `chi2_statistic`: This is a single number that measures how far apart our `observed_values` are from these `expected_values`. A larger Chi-Squared statistic means there's a bigger difference between what we actually saw and what we'd expect if there were no relationship.
- `p_value`: This is a very important probability. The p-value tells us: **If the null hypothesis (no relationship) is actually true, what is the chance of seeing a `chi2_statistic` as large as (or even larger than) the one we just calculated from our data?** A small p-value suggests that our observed data is quite unlikely if there's truly no relationship.
- `dof`: Stands for "degrees of freedom." It's a technical parameter related to the size of our summary table (number of mailer types and signup categories).
- `correction=False`: This is a technical detail about the calculation, often set to `False` when sample sizes are reasonably large.

**Phase 6: Making a Decision Based on the Test Results**

There are two common ways to use the Chi-Squared test results to make a decision:

**Method 1: Comparing the Chi-Squared Statistic to a Critical Value**

```python
critical_value = chi2.ppf(1 - acceptance_criteria, dof)
print(critical_value)

if chi2_statistic >= critical_value:
    print(f"As our chi-square statistic of {chi2_statistic} is higher than our critical value of {critical_value} - we reject the null hyposthesis, and conclude that: {alternate_hypothesis}")
else:
    print(f"As our chi-square statistic of {chi2_statistic} is lower than our critical value of {critical_value} - we retain the null hyposthesis, and conclude that: {null_hypothesis}")
```

**What it is:**
- `critical_value = chi2.ppf(...)`: Based on our `acceptance_criteria` (0.05) and the `dof`, we calculate a "critical value." Think of this as a threshold or a line in the sand for our `chi2_statistic`.
- The `if` statement then compares our calculated `chi2_statistic` to this `critical_value`:
    - If our `chi2_statistic` is **higher than or equal to** the `critical_value`, it means our observed results are "extreme" enough (far enough from what's expected if there's no relationship) to cast serious doubt on the null hypothesis. So, we **reject the null hypothesis**. The conclusion printed is that there _is_ a relationship between mailer type and signup.
    - If our `chi2_statistic` is **lower than** the `critical_value`, our results aren't extreme enough. We don't have enough evidence to say the null hypothesis is wrong. So, we **retain the null hypothesis** (we don't say we "accept" it, just that we don't have enough evidence to reject it). The conclusion printed is that there's no evidence of a relationship.

**Method 2: Comparing the p-value to the Acceptance Criteria (Significance Level)**

```python
if p_value <= acceptance_criteria:
    print(f"As our p_value statistic of {p_value} is lower than our acceptance criteria of {acceptance_criteria} - we retain the null hyposthesis, and conclude that: {alternate_hypothesis}")
else:
    print(f"As our p_value statistic of {p_value} is higher than our acceptance criteria of {acceptance_criteria} - we retain the null hyposthesis, and conclude that: {null_hypothesis}")
```

**What it is:** This method directly uses the `p_value`.
- The `if` statement compares the `p_value` to our `acceptance_criteria` (0.05).
- **Standard Interpretation:**
    - If the `p_value` is **less than or equal to** our `acceptance_criteria` (e.g., p-value ≤ 0.05), it means that the observed data1 is unlikely to have occurred if the null hypothesis were true. Therefore, we **reject the null hypothesis** and conclude that there _is_ a statistically significant relationship (supporting the `alternate_hypothesis`).
    - If the `p_value` is **greater than** our `acceptance_criteria` (e.g., p-value > 0.05), it means that the observed data is reasonably likely to have occurred even if the null hypothesis were true. Therefore, we **fail to reject (or "retain") the null hypothesis**, meaning we don't have enough evidence to say there's a relationship.
- **Regarding the code's print statements:** Looking at the specific print statements in _this_ block of code, there seems to be a mix-up. For instance, when `p_value <= acceptance_criteria`, the code prints that it _retains_ the null hypothesis but concludes the _alternate_ hypothesis. The standard statistical practice is:
    - If `p_value <= acceptance_criteria`: **Reject Null Hypothesis**, conclude Alternate Hypothesis.
    - If `p_value > acceptance_criteria`: **Retain Null Hypothesis**, conclude Null Hypothesis (or rather, lack of evidence for the alternate). You'll want to interpret the `p_value` result based on this standard statistical logic, rather than relying solely on the exact wording of these specific print statements if they deviate.

**In a Nutshell:**

This code performs a Chi-Squared test to see if the type of mailer sent significantly impacted whether customers signed up. It compares what actually happened (`observed_values`) to what would be expected if the mailers made no difference.

- If the `chi2_statistic` is high enough (above the `critical_value`), OR
- If the `p_value` is low enough (below or equal to the `acceptance_criteria` of 0.05),

...then we conclude that the differences in signup rates between the mailer types are likely real and not just due to random chance. This would mean at least one mailer type was genuinely more (or less) effective than another. Otherwise, we conclude that we don't have enough statistical evidence to say the mailers performed differently.

In this scenario both tests confirmed:

The null hypothesis is retained and there is no relationship between the mailer type and sign up flag, the features are independent.  ABC can potentially save some money and go with the cheaper mailer.  We must stress this does not mean, absolutely, money will be saved, but you need to keep this in mind when making your decision.   
