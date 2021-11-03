import csv
import sys
import calendar
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    

def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # a simple if-else for columns that require special handling
    def translate_special_index(index, val):

        def month_to_int(month):
            if month=="June": month="Jun"
            return list(calendar.month_abbr).index(month)

        def visitor_type_to_int(visitor_type):
            return 1 if visitor_type == "Returning_Visitor" else 0

        def weekend_to_int(val):
            return 1 if val == "TRUE" else 0

        if index == 10:
            return month_to_int(val)
        elif index == 15:
            return visitor_type_to_int(val)
        elif index == 16:
            return weekend_to_int(val)

    # categorizes the indexes in 3 sets to use them later
    def categorize_indexes(header_row):

        int_titles = set(["Administrative", "Informational", "ProductRelated", "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend"])
        float_titles = set(["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"])

        int_indexes = set()
        float_indexes = set()

        month_index = 10
        visitor_type_index = 15
        weekend_index = 16
        
        special_indexes = set([month_index, visitor_type_index, weekend_index])

        for i in range(len(header_row)):
            if header_row[i] in int_titles:
                int_indexes.add(i)
            elif header_row[i] in float_titles:
                float_indexes.add(i)
        
        return (int_indexes, float_indexes ,special_indexes)

    # start handling the csv
    with open(filename) as f:
        reader = csv.reader(f)
        
        # categorize indexes
        header_row = next(reader)
        int_indexes, float_indexes, special_indexes = categorize_indexes(header_row)
        
        # fill the lists
        evidence = []
        labels = []
        for row in reader:
            row_evidence = []
            # handle each cell accordingly
            for index in range(len(row)-1):
                if index in special_indexes:
                    row_evidence.append(translate_special_index(index, row[index]))
                elif index in int_indexes:
                    row_evidence.append(int(row[index]))
                else: # index in float_indexes
                    row_evidence.append(float(row[index]))
            # append evidence row and label
            evidence.append(row_evidence)
            labels.append(1 if row[17] == "TRUE" else 0)
      
    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors = 1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    sensitivity, specificity, total_pos, total_neg = 0, 0, 0, 0

    for i in range(len(labels)):
        if labels[i]:
            total_pos += 1
        else:
            total_neg += 1

        if labels[i] == predictions[i] == 1:
            sensitivity += 1
        elif labels[i] == predictions[i] == 0:
            specificity += 1
   
    sensitivity /= total_pos
    specificity /= total_neg
    
    return (sensitivity, specificity) 


if __name__ == "__main__":
    main()
