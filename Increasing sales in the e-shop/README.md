# Task description

I needed to increase the number of additional sales in the online store. To do this, I decided to classify products by popularity in order to give out in the recommendations those products that will most likely be purchased.

# Data

Events:

- timestamp — event time;
- visitorid — user ID;
- event — event type;
- itemid — item ID;
- transactionid — transaction ID, if it took place;
- category_tree — a file with a tree of categories (the tree can be restored).

Category_tree:

- category_id — category ID;
- parent_id — parent category ID.

Item_properties:

- timestamp — the moment the property value was written;
- item_id — object identifier;
- property — property, it seems that all of them, except for the category, are hashed;
- value — property value.

# Data preprocessing

I looked at the main variables of the event dataset and determined:
- there are no gaps in the dataset, except for _transactionid_ - there are gaps in cases where the purchase did not take place,
- I make an assumption that _transactionid_ is the amount of the purchase,
- the cost is known only for 12025 goods out of 235061, so I abandoned the attribute with the prices of goods,
- there are three possible event types - view, add to cart and buy,
- the number of unique users - 1407580, on average, one user performs 2 actions within the provided data,
- I processed the timestamp feature and extracted useful information from it regarding the time the user performed the action: time of day, day, day of the week and month.

I looked at the main dataset variables with product properties and determined:
- 1104 unique properties are presented in the dataset,
- of these, only _categoryid_ and _available_ are known for products, the remaining 1102 properties are hashed,
- after analyzing the distribution of properties from the top 10, I left only the properties _categoryid_ and _790_, both of which are categorical.
- I grouped data with product properties not by _timestamp_, but by _itemid_, while selecting for each _itemid_ the properties that were added last in the dataset chronology.

I analyzed _category_tree_ and came to the following conclusions:
- _category_tree_ is a tree / graph whose nodes are category ids, in other words _parentid_ is a parent node in the _category_id_ hierarchy,
- _parent_id_ has only 25 gaps, I will not delete them, but I will look at the number of gaps after assembling the common dataframe.

# Target

As a target variable, I took _event_ - the event type. I consider _transaction_ a successful purchase, and _view_ and _addtocart_ as an incomplete purchase.

# Feature Engineering

**Receipt** is a new feature, which is a sequence, one value of which combines actions performed within the same moment of time for the same user.

**Purchase qty** - I counted the number of purchases in the current dataset for each item. Thus, I got a relative indicator of the popularity of goods. I will feed it into the model in a normalized form - a numerical sequence from 0 to 1.

# Data imbalance

The data turned out to be unbalanced with a large skew:
* class 0 - 1428518
* class 1 - 21226

To eliminate this, I used a combined approach: first, I deleted a data slice from the dominant class in a random sequence, and then generated new data with the desired target using the SMOTE technique - this is a duplicate sample of the minority class, although these examples do not add any new information to the model and are synthesized from existing records.

# Encoding

Encoding is the transformation of features into a numerical representation, this is necessary so that the machine learning model can process information within the dataset, emphasize the relationship between data and issue a correct prediction.

One Hot Encoding - I converted categorical features into a matrix representation, the number of unique classes of which was small (<50). I encoded the rest of the categorical features using the Frequency Encoding method - I replaced the values for each class with their frequency in the dataset. Thus, I made it clear to the model which classes are more significant and which are less.
  
# Validation

As a validation, I used the classic splitting of the dataset into training and validation samples (70% by 30%). Our model will be trained on the first sample, and it will be tested on the second sample - all data will be fed into the model, except for the target, after which the model will predict this target itself.

I compare the targets predicted by the model with real targets and evaluate them using the precision metric.
Precision can be interpreted as the proportion of goods classified by the classifier as successful purchases and, at the same time, are actually successful purchases.

# Machine learning

As a machine learning model for the classification problem, I chose XGBoost, one of the most popular and effective implementations of the gradient boosting algorithm on decision trees.
With the basic settings, XGBoost showed a prediction accuracy of 95% for the precision metric, and after adjusting the hyperparameters using Optuna, the result improved to 96%!

Optuna is a framework for automated search for optimal hyperparameters for machine learning models. She selects these parameters by trial and error.

Optuna picked up hyperparameters from ranges that I set manually. It went through all the ranges for each hyperparameter and eventually returned the best combination of such hyperparameters (in a cycle of 30 “passes”).

For example:
max_depth=7 - maximum depth of each decision tree,
learning_rate=0.456 - a parameter that controls the step size at which the algorithm updates the model weights.

# API

In order for the user to communicate with the model, I wrote a simple API service using the FastAPI framework.

FastAPI is a framework for building concise and fairly fast HTTP API servers with built-in validation, serialization, and asynchrony. The logic of actions is the following:
- API loads model from file.
- The API applies the model to data received from the outside (from the user).
- The API will be deployed in a Docker container, so it will be launched after the container is launched. More on this below.

# Docker

In order to create a docker image on the local machine, you need to go to the terminal in the app folder and run the following commands (with all the dots and hyphens!):

1. **docker build -t ds_app:1.0 .**

This command creates a docker image for our API on the local machine. At the same time, an “environment” based on the Docker file is installed - the operating system and all the libraries that will be needed to start the service are registered there.

2. **sudo docker run --rm -p 50000:51234 ds_app:1.0 uvicorn app:app --host 0.0.0.0 --port 51234**

This command starts our service in a docker image with the environment installed.

3. Next, you need to follow the link **http://127.0.0.1:50000/docs** and get into the Swagger UI - it is with interactive interaction, it calls and tests the API directly from the browser.

4. This will open the Swagger UI. Click on the POST method and in the expanded tab, click on the right button “Try it out”.

5. Then, in the “Request body” field, insert the converted product data (list of lists). Each product must have 54 float properties in the list. After filling in the data in this window, you need to click the “Execute” button.

6. Then you need to scroll down the page to _predictions_ and there will be a prediction of the model. Here you can also download the forecast in json format.

# Conclusion

The machine learning model is now fault-tolerant and understandable to the end user.

Let's return to the origins of the task - it was necessary to increase additional sales of the online store by 20%. As a result, I rolled out a system that classifies goods into those that are most likely to be bought or not.

I suggest optimizing the model for business as follows:

- Analyze the shopping cart of the buyer's goods and fix the categories of goods that are in the cart,
- Then, by these categories, filter the products that are still in the online store, taking into account current balances,
- The resulting pool of goods is fed into the model, and then we sort and take the top 3 goods that are most likely to be purchased according to the results of the model's forecast.
