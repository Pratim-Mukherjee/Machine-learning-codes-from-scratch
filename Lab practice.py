#!/usr/bin/env python
# coding: utf-8

# # 1. Simple Linear Regression
# **Without OLS**

# In[1]:


# Function to parse a comma-separated string and convert it to a list of integers
def parse_input(input_string):
    # Split the input string by comma, strip whitespace, and convert each value to an integer
    return [int(value.strip()) for value in input_string.split(",")]

# Accepting inputs from the user
x_input = input("Enter the x values (comma separated): ")
y_input = input("Enter the y values (comma separated): ")

# Parsing the inputs
x = parse_input(x_input)
y = parse_input(y_input)

# Number of data points
n = len(x)

# Ensure the lengths of x and y match
if len(y) != n:
    raise ValueError("The number of x and y values must match.")

# Calculate means of x and y
mean_x = sum(x) / n
mean_y = sum(y) / n

# Calculate slope (m)
numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
m = numerator / denominator

# Calculate y-intercept (b)
b = mean_y - m * mean_x

# Display the regression line equation
print(f"The regression line equation is: y = {m:.2f}x + {b:.2f}")


# # 2. Simple Linear Regression
# **With OLS**

# In[22]:


def parse_input(input_string):
    # Split the input string by comma, strip whitespace, and convert each value to an integer
    return [int(value.strip()) for value in input_string.split(",")]

# Accepting inputs from the user
x_input = input("Enter the x values (comma separated): ")
y_input = input("Enter the y values (comma separated): ")

# Parsing the inputs
x = parse_input(x_input)
y = parse_input(y_input)

# Number of data points
n = len(x)

# Ensure the lengths of x and y match
if len(y) != n:
    raise ValueError("The number of x and y values must match.")

# Calculate the sum of x, y, xy, and x^2
sum_x = sum(x)
sum_y = sum(y)
sum_xy = sum(x[i] * y[i] for i in range(n))
sum_x2 = sum(x[i] ** 2 for i in range(n))

# Calculate the slope (m) and y-intercept (b)
denominator = n * sum_x2 - sum_x ** 2
m = (n * sum_xy - sum_x * sum_y) / denominator
b = (sum_y * sum_x2 - sum_x * sum_xy) / denominator

# Display the regression line equation
print(f"The regression line equation is: y = {m:.2f}x + {b:.2f}")


# # 3. Multiple Linear Regression
# using ridge and lasso

# In[11]:


def ridge_regression(X, y, alpha=1.0):

    X = np.column_stack((np.ones(len(y)), X))
    n_features = X.shape[1]
    I = np.eye(n_features)
    X_transpose = X.T
    theta = np.linalg.inv(X_transpose.dot(X) + alpha * I).dot(X_transpose).dot(y)
    return theta


n = int(input("Enter the number of independent variables: "))


y_input = input("y: ")
y = np.array(list(map(float, y_input.replace(',', ' ').split())))


independent_vars = []
for i in range(n):
    x_input = input(f"x{i+1}: ")
    x_i = np.array(list(map(float, x_input.replace(',', ' ').split())))
    independent_vars.append(x_i)

# Train the model using Ridge Regression with regularization parameter alpha
alpha = 1.0  # You can adjust the value of alpha as needed
theta = ridge_regression(np.array(independent_vars).T, y, alpha)


print("Optimal coefficients (intercept and feature coefficients):\n", theta)


def display_regression_equation(theta):
    intercept = theta[0]
    terms = [f"{theta[i]:.2f} * x{i}" for i in range(1, len(theta))]
    equation = " + ".join(terms)
    print(f"The regression equation is: y = {intercept:.2f} + {equation}")

display_regression_equation(theta)


# # 4. Support Confidence and Lift

# In[14]:


def calculate_metrics(transactions, antecedent, consequent):
    antecedent_set = set(antecedent)
    consequent_set = set(consequent)
    
    
    num_transactions = len(transactions)
    
    support_count = 0
    antecedent_count = 0
    consequent_count = 0
    
    for transaction in transactions:
        transaction_set = set(transaction)
        
        if antecedent_set.issubset(transaction_set) and consequent_set.issubset(transaction_set):
            support_count += 1
        
        if antecedent_set.issubset(transaction_set):
            antecedent_count += 1
        
        if consequent_set.issubset(transaction_set):
            consequent_count += 1
    
    support = support_count / num_transactions
    confidence = support_count / antecedent_count if antecedent_count > 0 else 0
    lift = support / ((consequent_count / num_transactions) * (antecedent_count / num_transactions)) if consequent_count > 0 else 0
    
    return support, confidence, lift

def main():

    transactions_input = input("Enter the list of transactions (separated by semicolons, items in each transaction separated by spaces): ").strip()
    transactions = [transaction.split(',') for transaction in transactions_input.split(' ') if transaction]
    
    rules_input = input("Enter the association rules (separated by semicolons, each rule in the form 'antecedent => consequent'): ").strip()
    rules = [rule.split('=>') for rule in rules_input.split(';') if rule.strip()]
    
    for rule in rules:
        antecedent = rule[0].strip().split(',')
        consequent = rule[1].strip().split(',')
        support, confidence, lift = calculate_metrics(transactions, antecedent, consequent)
        print(f"Rule: {antecedent} => {consequent}")
        print(f"Support: {support:.2f}, Confidence: {confidence:.2f}, Lift: {lift:.2f}\n")

main()


# # 5. Return the equation of hyperplane for SVM

# In[20]:


def calculate_hyperplane(sv1, sv2):
 
    # Calculate the weight vector (w) as the difference between the support vectors
    w = [(sv2[i] - sv1[i]) for i in range(len(sv1))]

    # Calculate the dot product of weight vector and the average of the support vectors
    dot_product = sum(w[i] * ((sv1[i] + sv2[i]) / 2) for i in range(len(w)))

    # Bias term (b) is the negative of the dot product
    b = -dot_product

    return w, b

def print_hyperplane_equation(w, b):
  
    # Formulate the equation of the hyperplane
    equation = f"{w[0]:.2f} * x1 + {w[1]:.2f} * x2 + {b:.2f} = 0"
    
    # Print the equation of the hyperplane
    print("Equation of the hyperplane:", equation)

def main():
    # Define support vectors for Problem 1
    sv1_problem1 = [1, 0]  # Class I
    sv2_problem1_1 = [3, 1]  # Class II
    sv2_problem1_2 = [3, -1]  # Class II

    # Calculate and print the hyperplane for Problem 1 using the first support vector of Class II
    w1, b1 = calculate_hyperplane(sv1_problem1, sv2_problem1_1)
    print("Problem 1 using the first support vector of Class II:")
    print_hyperplane_equation(w1, b1)

    # Calculate and print the hyperplane for Problem 1 using the second support vector of Class II
    w1, b1 = calculate_hyperplane(sv1_problem1, sv2_problem1_2)
    print("\nProblem 1 using the second support vector of Class II:")
    print_hyperplane_equation(w1, b1)

    # Define support vectors for Problem 2
    sv1_problem2 = [1, 1]  # Class I
    sv2_problem2 = [2, 2]  # Class II

    # Calculate and print the hyperplane for Problem 2
    w2, b2 = calculate_hyperplane(sv1_problem2, sv2_problem2)
    print("\nProblem 2:")
    print_hyperplane_equation(w2, b2)

# Run the main function
main()


# # 6. PDF plot using Triangular Kernel Function

# In[24]:


import numpy as np
import matplotlib.pyplot as plt

def triangular_kernel(x,data,h):
    n=len(data)
    pdf=np.zeros_like(x)
    for xi in data:
        pdf += 1/(n*h)*np.maximum(0,1-np.abs(x-xi)/h)
    return pdf

def plot_triangular_kernel(data,h):
    x=np.linspace(min(data)-1,max(data)+1,1000)
    pdf = triangular_kernel(x,data,h)
    plt.plot(x,pdf,label='Tk pdf')
    plt.xlabel('data')
    plt.ylabel('density')
    plt.title("triangular kernel density")
    plt.show()
h=2
data=[2,5,6,8,10]
plot_triangular_kernel(data,h)
print(f" The PDF is:",triangular_kernel(2.5,data,h))


# # 7. KNN-Classifier

# In[25]:


import numpy as np
from collections import Counter

# Function to design a classifier using the KNN estimator
def knn_classifier(data, k, math, cs):
    # Calculate the Euclidean distances from the new data point to each data point in the dataset
    distances = []
    for point in data:
        math_score, cs_score, result = point
        distance = np.sqrt((math - math_score)**2 + (cs - cs_score)**2)
        distances.append((distance, result))
    
    # Sort the distances in ascending order
    distances.sort(key=lambda x: x[0])
    
    # Get the k-nearest neighbors
    nearest_neighbors = distances[:k]
    
    # Calculate the majority class among the k-nearest neighbors
    labels = [neighbor[1] for neighbor in nearest_neighbors]
    majority_class = Counter(labels).most_common(1)[0][0]
    
    # Return the classification of the new data point
    return majority_class

# Define the dataset as a list of tuples (Math score, CS score, Result)
data = [(4, 3, 'Fail'), (6, 7, 'Pass'), (7, 8, 'Pass'), (5, 5, 'Fail'), (8, 8, 'Pass')]

# Define the number of nearest neighbors to consider
k = 3

# New data point to classify
new_math_score = 6
new_cs_score = 8

# Classify the new data point using the KNN classifier
classification = knn_classifier(data, k, new_math_score, new_cs_score)

print(f"The classification of the student with Mathematics = {new_math_score} and Computer Science = {new_cs_score} is: {classification}")


# # 8. PDF plot using a histogram 

# In[26]:


import numpy as np
import matplotlib.pyplot as plt

def plot_histogram_pdf(data, bins=10):

    # Calculate the histogram
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    
    # Calculate the width of each bin
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Plot the histogram as a bar plot
    plt.bar(bin_edges[:-1], counts, width=bin_width, edgecolor='black', alpha=0.7, label='PDF')
    
    # Add labels and title
    plt.xlabel('Data')
    plt.ylabel('Probability Density')
    plt.title('Probability Density Function using Histogram')
    
    # Show the plot
    plt.legend()
    plt.show()

# Define the data set
data = [2, 3, 5, 6, 8, 12, 13, 17, 19, 21, 22, 23, 24, 25, 26, 24.5, 27, 31, 31.5, 32, 33.5, 34, 35, 36.5, 36.7, 37, 38, 41, 42, 43, 44.5, 45.5, 46, 49, 51, 51.5, 52, 52.5, 53, 54, 56, 57, 58, 61, 65, 71, 72, 75, 77, 92]

# Plot the probability density function using a histogram
plot_histogram_pdf(data, bins=10)


# # 9. Return frequent itemsets (APRIORI)

# In[33]:


data = {'t1': (1, 3, 4), 't2': (2, 3, 5), 't3': (1, 2, 3, 5), 't4': (2, 5), 't5': (1, 3, 5)}

# Function to create subsets from the list of original items
def create_subset(original_items, output, i, final_subsets):
    if i >= len(original_items):
        # Append the current output list as a copy to the final_subsets
        final_subsets.append(output.copy())
        return
    
    # Recursively create subsets without including the current item
    create_subset(original_items, output, i + 1, final_subsets)
    
    # Include the current item in the output list and recursively create subsets
    output.append(original_items[i])
    create_subset(original_items, output, i + 1, final_subsets)
    
    # Backtrack by removing the last item from the output list
    output.pop()

# Create a list of all unique items in the data
unique_items = set()
for transaction in data.values():
    unique_items.update(transaction)

# Convert the set of unique items to a list
unique_items = list(unique_items)

# Create an empty list to store the final subsets
final_subsets = []

# Call the create_subset function to generate all possible subsets
create_subset(unique_items, [], 0, final_subsets)

# Define the minimum support threshold
min_support = 2

# Create a list to store the frequent itemsets
frequent_itemsets = []

# Iterate through each subset in the final_subsets list
for subset in final_subsets:
    # Count the occurrences of the subset in the transactions
    count = 0
    for transaction in data.values():
        if set(subset).issubset(transaction):
            count += 1
    
    # If the count meets or exceeds the minimum support, add the subset to the frequent_itemsets list
    if count >= min_support:
        frequent_itemsets.append(subset)

# Find the maximum length of frequent itemsets
max_length = max(len(itemset) for itemset in frequent_itemsets)

# Filter the frequent itemsets to include only those with the maximum length
max_length_itemsets = [itemset for itemset in frequent_itemsets if len(itemset) == max_length]

# Print the frequent itemsets with the maximum length
print(max_length_itemsets)


# # 10. Python function to return the values of support, confidence, and lift. Create a CSV file for 20 items and 50 transactions. Upload the CSV file and use the function to find the same for five self-written rules.
# 
# 

# In[36]:


import csv
import random
import pandas as pd

items=[f'item{i}' for i in range(1,21)]
transactions=[[random.choice(items) for _ in range(6)] for _ in range(50)]
csv_file='transactions.csv'
with open(csv_file, 'w', newline='') as file:
    writer=csv.writer(file)
    writer.writerow(['Transactions'])
    writer.writerows([','.join(transaction)] for transaction in transactions)
print("CSV created successfully")

def calculate_metrics(transactions, antecedent, consequent):
    def calculate_support(itemset):
        count=sum(1 for transaction in transactions if set(itemset).issubset(transaction))
        support=count/len(transactions)
        return support
    antecedent_support=calculate_support(antecedent)
    rule_support=calculate_support(antecedent+consequent)
    
    if antecedent_support == 0:
        confidence = 0
    else:
        confidence = rule_support/ antecedent_support
    consequent_support=calculate_support(consequent)
    if antecedent_support ==0 or consequent_support == 0:
        lift=0
    else:
        lift=rule_support/ (antecedent_support * consequent_support)
    return {
        'Support':rule_support,
        'Confidence': confidence,
        'Lift': lift
    }
df =pd.read_csv('transactions.csv')
transactions= df['Transactions'].apply(lambda x: x.split(',')).tolist()

rules=[
    (['item1', 'item3'], ['item5']),
    (['item2'], ['item4']),
    (['item6'], ['item8']),
    (['item9'], ['item11']),
    (['item10'], ['item12'])
]

results = []
for rule in rules:
    antecedent, consequent = rule
    metrics = calculate_metrics(transactions, antecedent, consequent)
    results.append({
        'Rule':f"{antecedent} => {consequent}",
        'Support': metrics['Support'],
        'Confidence': metrics['Confidence'],
        'Lift': metrics['Lift']
    })
print("Analysis of Rules:")
for result in results:
    print(f"Rule: {result['Rule']}")
    print(f"Support: {result['Support']:.2%}")
    print(f"Confidence: {result['Confidence']:.2%}")
    print(f"Lift: {result['Lift']:.2f}")
    print()


# In[ ]:




