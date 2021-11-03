# Harvard CS50's Introduction to Artificial Intelligence with Python 2021 course

### Project 4a - Shopping

* An AI to predict whether visitors of an e-shop will end up making a purchase or not.

##### Implementation
* We build a `nearest-neighbor classifier` to solve the problem.
* We evaluate the classifier measuring two values:
    * `Sensitivity`: proportion of positive examples that were correctly identified
    * `Specificity`: proportion of negative examples that were correctly identified
* We use the `scikit-learn` package (installing with `pip3 install scikit-learn`) to train the model and get the predictions
* We are given the dataset at `shopping.csv` of around 12000 datapoints
* Each datapoint has some `input` columns and one `output` column `(revenue)`, which is the `label` value that we want our classifier to be able to predict 
* The goal was to implement the functions `load_data`, `train_model` and `evaluate`

- - -
* Developer: Giannis Athanasiou
* Github Username: John-Atha
* Email: giannisj3@gmail.com