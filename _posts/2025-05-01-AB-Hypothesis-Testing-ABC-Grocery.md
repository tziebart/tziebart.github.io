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
- To determine the relationship between mailer quality/cost and membership conversion rates to inform future campaign budget allocation for promotional materials.

**3. Methodology Used:
- A/B Tests: Randomized experiment containing two groups, A&B.
- Hypothesis Testing: Paired T-Test. 
- Assessing Campaign Performance Using Chi-Square Test For Independence

The campaign data was gathered onto an excel spreadsheet.  
```python
campaign_data = pd.read_excel("grocery_database.xlsx", sheet_name="campaign_data")
```
The control group data was ignored as the question is whether or not there was a significant difference between sign-up rates for mailer 1 (cheap) and mailer 2 (expensive)?

```python
campaign_data = campaign_data.loc[campaign_data["mailer_type"] != "Control"]
```

The crosstab pandas method was then used to gather data on mailer 1 and mailer 2 to help determine the signup rates. 

```python
observed_values = pd.crosstab(campaign_data["mailer_type"], campaign_data["signup_flag"]).values
```

The results showed not only how many customers received each mailer but whether or not they signed up.   From there the signup rate for each mailer could be calculated.

|     Mailer Type      | No. Customers received Mailer | No. Customers Signed Up for Club | Rate |
| :------------------ | :-------------: | :------------------: | :----------------:|
|   Mailer 1 (cheap)   | 252           | 123                | 0.328
| Mailer 2 (expensive) | 209           | 127                |0.380

Now comes the question as to whether or not the values are significant.  For that we turn to the hypothesis test.  Before running the test the null hypothesis, alternative hypothesis and acceptance criteria are determined.  

null hypothesis = "There is no relationship between mailer type and signup flag"
alternate hypothesis = "There is a relationship between mailer type and signup flag"
acceptance criteria = 0.05

The Chi-Square test is applied.

```python
chi2_statistic, p_value, dof, expected_values = chi2_contingency(observed_values, correction=False)
```

Where chi2_statistic is the calculated test statistic. In simple terms a low value suggests the null hypothesis to be correct. 

The p_value is a probability. if the p_value is low than the null hypothesis is rejected. 

The dof (degrees of freedom) is used with the Chi2 statistic to calculate the p_value.  

To make the final determination we turn to calculating the critical_value.  From this we can see if the null hypothesis is accepted or rejected.

```python
critical_value = chi2.ppf(1 - acceptance_criteria, dof)
```

 The final calculation using the Chi2 Statistic and Critical Value.

```python
if chi2_statistic >= critical_value:
    print(f"As our chi-square statistic of {chi2_statistic} is higher than our critical value of {critical_value} - we reject the null hypothesis, and conclude that: {alternate_hypothesis}")
else:
    print(f"As our chi-square statistic of {chi2_statistic} is lower than our critical value of {critical_value} - we retain the null hyposthesis, and conclude that: {null_hypothesis}")
```

And for good measure to see if the results are strong we can compare the p_value against the acceptance criteria.

```python
if p_value <= acceptance_criteria:
    print(f"As our p_value statistic of {p_value} is lower than our acceptance criteria of {acceptance_criteria} - we reject the null hyposthesis, and conclude that: {alternate_hypothesis}")
else:
    print(f"As our p_value statistic of {p_value} is higher than our acceptance criteria of {acceptance_criteria} - we retain the null hyposthesis, and conclude that: {null_hypothesis}")

```

In this scenario both tests confirmed:

The null hypothesis is retained and there is no relationship between the mailer type and sign up flag are independent.  So ABC can potentially save some money a go with the cheaper mailer.  We must stress this does not mean absolutely, go with the cheaper mailer, but you need to keep this in mind when making your decision.   