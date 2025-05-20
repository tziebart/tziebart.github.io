---
layout: post
title: Customer Classification - For Marketing
image: "/posts/customers_outside_store_entrance.png"
tags: [K-Means, Classification]
---

**Our Goal:** To identify distinct groups of customers who show similar purchasing patterns in different grocery product areas. This can help you understand your customer base better and tailor marketing or product strategies to these groups.


**In Summary:**

This code takes your grocery transaction data and:

1. **Loads and Combines:** Gets transaction details and links them to product area names.
2. **Cleans:** Focuses on food items by removing "Non-Food" purchases.
3. **Transforms for Insight:** Calculates each customer's spending per product area and then converts this into percentages of their total spend to understand _relative preferences_.
4. **Prepares for Clustering:** Scales the percentage data so all product areas are treated equally by the algorithm.
5. **Determines Optimal Group Count:** Uses the "Elbow Method" (by plotting WCSS scores) to help decide on a sensible number of customer segments to create.
6. **Assigns Customers to Segments:** Runs the K-Means algorithm to group customers based on their scaled spending percentages.
7. **Profiles Segments:** Calculates the average spending profile for each segment to understand their distinct characteristics (e.g., "Dairy enthusiasts," "Bulk buyers in staples," etc.).

The ultimate output is a set of customer segments, each with a distinct purchasing profile, which can be invaluable for targeted business strategies.

Here's the journey this code takes:

**Phase 1: Gathering Tools and Raw Materials (Your Grocery Data)**

Python

```
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
```

**What it is:** Just like before, we're importing our necessary toolkits:
- `KMeans` (from `sklearn.cluster`): This is the specific algorithm we'll use to find the customer groups (clusters). It tries to group customers so that those within the same group are as similar as possible in their shopping habits, and customers in different groups are as different as possible.
- `MinMaxScaler` (from `sklearn.preprocessing`): A tool to put all our numerical data on a similar scale, which helps the `KMeans` algorithm work effectively.
- `pandas as pd`: Our go-to tool for handling data in tables (like a super-powered spreadsheet).
- `matplotlib.pyplot as plt`: Our drawing kit for creating graphs to help us understand the results.

Python

```
# Create the data

# import tables
transactions = pd.read_excel("data/grocery_database.xlsx", sheet_name= "transactions")
product_areas = pd.read_excel("data/grocery_database.xlsx", sheet_name= "product_areas")
```

**What it is:** We're loading data from an Excel file named "grocery_database.xlsx". This file has two sheets (tabs) that we're interested in:
- `"transactions"`: This likely contains records of individual customer purchases, including what they bought and how much it cost.
- `"product_areas"`: This probably lists the different sections in your grocery store, like "Dairy," "Fruit," "Meat," etc., possibly with an ID for each area.

**Phase 2: Combining and Refining the Data**

Python

```
# merage data on product_area name
transactions = pd.merge(transactions, product_areas, how = "inner", on = "product_area_id")
```

**What it is:** We're combining the `transactions` data with the `product_areas` data.
- **Why?** The `transactions` table has a `product_area_id` (like a code for the product section), but not the actual name (e.g., "Dairy"). The `product_areas` table links these IDs to their names. This `pd.merge` step matches them up using the `product_area_id` so that each transaction record now also knows the _name_ of the product area it belongs to.
- `how = "inner"` means we only keep transactions that have a matching product area in the `product_areas` table.

Python

```
# drop the non-food category
transactions.drop(transactions[transactions["product_area_name"] == "Non-Food"].index, inplace = True)
```

**What it is:** We're removing all transaction records where the `product_area_name` is "Non-Food."
- **Why?** For this particular analysis, we want to focus specifically on how customers shop for _food_ items. "Non-Food" items might have very different purchasing patterns and could obscure the grocery-specific segments we're trying to find.

**Phase 3: Reshaping Data to Focus on Customer Spending Patterns**

This is a crucial part where we transform the raw transaction data into a format suitable for finding customer groups.

Python

```
# aggregate sales a customer level (by product area)
transaction_summary = transactions.groupby(["customer_id", "product_area_name"])["sales_cost"].sum().reset_index()
```

**What it is:** We're summarizing the data. For each unique `customer_id` and for each `product_area_name` they shopped in, we're calculating the total amount they spent (`sales_cost`).
- **Example:** After this step, we'd have data like:
    - Customer 101, Dairy, $25.50
    - Customer 101, Fruit, $15.00
    - Customer 205, Meat, $40.00
    - ...and so on.

Python

```
# pivot data to place product areas as columns
transaction_summary_pivot = transactions.pivot_table(index = "customer_id",
                                                     columns = "product_area_name",
                                                     values = "sales_cost",
                                                     aggfunc = "sum",
                                                     fill_value = 0,
                                                     margins = True,
                                                     margins_name = "Total").rename_axis(None, axis = 1)
```

**What it is:** This "pivots" or reshapes our `transaction_summary` data. Imagine taking the list from the previous step and turning it into a table where:
- Each row is a unique `customer_id`.
- Each column represents a different `product_area_name` (e.g., "Dairy", "Fruit", "Meat").
- The values inside the table are the total `sales_cost` for that customer in that product area.
- `fill_value = 0`: If a customer didn't buy anything from a particular product area, we put a '0' in that cell.
- `margins = True, margins_name = "Total"`: This cleverly adds an extra column named "Total" at the end of each row, showing the total spending for that customer across all product areas. It also adds a "Total" row at the bottom (though we don't use the row total later).

**Why pivot?** This format is perfect for clustering. We now have a profile for each customer showing their spending across all relevant product categories in one row.

Python

```
# turn into % of sales
transaction_summary_pivot = transaction_summary_pivot.div(transaction_summary_pivot["Total"], axis = 0)
```

**What it is:** For each customer, we're now converting their spending in each product area into a _percentage_ of their total spending.
- **Example:** If Customer 101 spent $25 in Dairy and their total spending was $100, their "Dairy" column would now show 0.25 (or 25%).
- **Why?** This is a very important step for this kind of analysis! We want to group customers based on their _relative shopping preferences_, not just by who spends the most overall. A customer who spends $500 with 20% in "Meat" has a similar meat preference to a customer who spends $50 with 20% in "Meat". Using percentages allows us to compare their shopping _patterns_ fairly.

Python

```
# drop total column
data_for_clustering = transaction_summary_pivot.drop(["Total"], axis = 1)
```

**What it is:** We remove the "Total" spending column.
- **Why?** We only needed it temporarily to calculate the percentages. Now that we have the percentage breakdown for each product area, the absolute total isn't needed for the clustering itself (as we're focusing on the _proportion_ of spend).

**Phase 4: Preparing Data for the Clustering Algorithm**

Python

```
# Data Preparation and Cleaning

# check for missing values
data_for_clustering.isna().sum()
```

**What it is:** A quick check to see if any missing values (empty cells) have appeared during our transformations. It's good practice to ensure data quality.

Python

```
# normalise data
scale_norm = MinMaxScaler()
data_for_clustering_scaled = pd.DataFrame(scale_norm.fit_transform(data_for_clustering), columns= data_for_clustering.columns)
```

**What it is:** We use the `MinMaxScaler` again. This time, it takes all the percentage values for each product area (which are already between 0 and 1 if a customer bought something, or 0 if they didn't, but could still have different distributions) and scales them. While percentages are already somewhat normalized, this ensures all features (product areas) are on an identical scale (0 to 1 range based on the min/max observed for _that feature across all customers_).

 **Why?** The K-Means algorithm calculates "distances" between customers to see how similar they are. If different product areas (our features) have vastly different ranges or spreads even as percentages, some might unfairly influence the distance calculation. Scaling puts them all on a truly level playing field.

**Phase 5: Finding the Right Number of Customer Groups (Clusters)**

The K-Means algorithm needs us to tell it _how many_ groups (K) to look for. We don't usually know this in advance.

Python

```
# Use BCSS to find a good value for k (Note: Code uses WCSS, not BCSS directly)
k_values = list(range(1,10))
wcss_list = [] # WCSS = Within Cluster Sum of Squares

for k in k_values:
    kmeans = KMeans(n_clusters = k, random_state= 42, n_init='auto') # Added n_init
    kmeans.fit(data_for_clustering_scaled)
    wcss_list.append(kmeans.inertia_)
```

**What it is:**
1. `k_values = list(range(1,10))`: We decide to test out creating 1 group, then 2 groups, all the way up to 9 groups.
2. `wcss_list = []`: We create an empty list to store a score for each 'k' we try. `WCSS` stands for "Within-Cluster Sum of Squares." It measures how compact the clusters are – a lower WCSS means customers within each cluster are very similar to each other (in terms of their spending percentages across product areas).
3. The `for k in k_values:` loop: For each number of clusters 'k' we're testing:
    - `kmeans = KMeans(n_clusters = k, ...)`: We set up the K-Means algorithm to find 'k' clusters. `random_state=42` ensures that if we run this again, we get the same starting point, making results reproducible. `n_init='auto'` is a good default to ensure robustness.
    - `kmeans.fit(data_for_clustering_scaled)`: We run the K-Means algorithm on our scaled customer spending data.
    - `wcss_list.append(kmeans.inertia_)`: The `inertia_` attribute gives us the WCSS for the clusters found with this 'k'. We save this score.

Python

```
plt.plot(k_values, wcss_list)
plt.title("Within Cluster Sum of Squares - by k")
plt.xlabel("k")
plt.ylabel("WCSS Score")
plt.tight_layout()
plt.show()
```

**What it is:** This code generates a line graph.
- The horizontal axis (x-axis) shows the number of clusters (k) we tried.
- The vertical axis (y-axis) shows the WCSS score for that number of clusters.
- **Why? (The "Elbow Method")**: We're looking for an "elbow point" in this graph. As you increase 'k', the WCSS will generally decrease (more groups mean they can be tighter). However, there's usually a point where adding more clusters doesn't lead to a _significant_ improvement in WCSS. This point, which looks like an elbow, is often a good indicator of a natural and meaningful number of clusters in the data.

<img width="914" alt="Pasted image 20250517151051" src="https://github.com/user-attachments/assets/5098481f-2de4-4a08-b7d5-0fc6379041c9" />


**Phase 6: Creating and Examining the Customer Segments**

Based on the elbow plot, you would decide on the best 'k'. The code then proceeds, and implicitly, the `kmeans` object from the _last_ iteration of the loop (where k was 9) is used. **Ideally, you would choose 'k' from the plot and re-run `KMeans` with that specific `n_clusters` value before this next part.**

Python

```
# Instatiate and fit model
# (Code for re-fitting with chosen K would ideally go here, e.g.:)
# chosen_k = 3 # Example: if elbow plot suggested 3 clusters
# kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init='auto')
# kmeans.fit(data_for_clustering_scaled)

# Use cluster information

# add cluster labels to our data
data_for_clustering["cluster"] = kmeans.labels_ # Uses kmeans from the last loop iteration (k=9)
```

**What it is:**
- The `kmeans.labels_` attribute contains the cluster assignment for each customer (e.g., customer 1 is in cluster 0, customer 2 is in cluster 1, etc.).
- We add these cluster labels as a new column named "cluster" to our `data_for_clustering` table (which has the percentage spending data). Now, each customer row not only shows their spending pattern but also which group they belong to.

Python

```
# check cluster sizes
data_for_clustering["cluster"].value_counts()
```

**What it is:** This counts how many customers fall into each cluster.
- **Why?** It helps us see if the clusters are reasonably balanced. If one cluster has 90% of the customers and the others are tiny, the segmentation might not be very insightful.

Python

```
# Profile our clusters
cluster_summary = data_for_clustering.groupby("cluster")[["Dairy", "Fruit", "Meat", "Vegetables"]].mean().reset_index()
```

**What it is:** This is where we try to understand the "personality" of each cluster.
- For each cluster, we group all the customers belonging to it.
- Then, for each of these groups (clusters), we calculate the _average percentage of spending_ in key product areas like "Dairy," "Fruit," "Meat," and "Vegetables."
- **Why?** This helps us describe each segment. For example:
    - Cluster 0 might have a high average spend in "Fruit" and "Vegetables" – "Health-Conscious Shoppers."
    - Cluster 1 might have a high average spend in "Meat" – "Meat Lovers."
    - Cluster 2 might have fairly even spending across categories – "Balanced Shoppers."
- This profiling is what makes the clusters actionable. You can then think about how to target marketing, promotions, or store layouts for these different customer types.

**In Summary:**

This code takes your grocery transaction data and:

1. **Loads and Combines:** Gets transaction details and links them to product area names.
2. **Cleans:** Focuses on food items by removing "Non-Food" purchases.
3. **Transforms for Insight:** Calculates each customer's spending per product area and then converts this into percentages of their total spend to understand _relative preferences_.
4. **Prepares for Clustering:** Scales the percentage data so all product areas are treated equally by the algorithm.
5. **Determines Optimal Group Count:** Uses the "Elbow Method" (by plotting WCSS scores) to help decide on a sensible number of customer segments to create.
6. **Assigns Customers to Segments:** Runs the K-Means algorithm to group customers based on their scaled spending percentages.
7. **Profiles Segments:** Calculates the average spending profile for each segment to understand their distinct characteristics (e.g., "Dairy enthusiasts," "Bulk buyers in staples," etc.).

The ultimate output is a set of customer segments, each with a distinct purchasing profile, which can be invaluable for targeted business strategies.
