### USEFUL LIBRARIES
- Pandas: opensource framework to manipulate data
- Scikit
- Matplotlib
- NumPy

## SELECTING AN ESTIMATOR
https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

## TO READ
1. https://www.coursera.org/learn/python-machine-learning/supplement/EoTru/rules-of-machine-learning-best-practices-for-ml-engineering-optional
2. 

### USEFUL DATA STRUCTURES
- **[Pandas] Series**: is a one-dimensional labeled array capable of holding data of any type (integer, string, float, python objects, etc.). The axis labels are collectively called index. Pandas Series is nothing but a column in an excel sheet.
Labels need not be unique but must be a hashable type. The object supports both integer and label-based indexing and provides a host of methods for performing operations involving the index.


- **[Pandas] DataFrame**: TODO DESCRIPTION
	- Select all cases where the first name is not missing and nationality is USA ``` df[df['first_name'].notnull() & (df['nationality'] == "USA")] ```   
	- Create variable with TRUE if age is greater than 50 ```elderly = df['age'] > 50``` 
	- df[df.City.str.contains('ville',case=False)] # select with condition on string
	- ```data = data.drop(['compliance','compliance_detail'], axis = 1)``` #drop columns
	- ```X_train_reduced = X_train_reduced.apply(pd.to_numeric, errors = 'coerce')``` # convert all columns of DataFrame to numeric

 - **[Pandas] List**: TODO DESCRIPTION
	- ```my_list = [1, 2, 3]``` 
	- ```my_list = [1, "Hello", 3.4]``` # list with mixed data types
	- ```my_list = ["mouse", [8, 4, 6], ['a']]``` # nested list

- **[Numpy] Array**: TODO DESCRIPTION
	- ```a = np.array([2,3,4])```

## TECHNIQUES
- **Boolean masking**:  create a boolean matrix over your data

- **Handle NaN**  
	```isnan = data().values```  
	```data[isnan] = 0```

- **Normalization**:  
	```from sklearn.preprocessing import MinMaxScaler```  
	```scaler = MinMaxScaler()```  
	```X_train_scaled = scaler.fit_transform(X_train)```  
	```X_test_scaled = scaler.transform(X_test)```

## MISCELLANEOUS

- get type
type(my_data)

- select columns with copy
data = full_data.loc[:, ["CLASS", "COUNTY", "geometry"]].copy()

- select rows if column value equals a value in a list
wild_lands = data.loc[data.CLASS.isin(['WILD FOREST', 'WILDERNESS'])].copy()

- get enums count within a column
data.CLASS.value_counts()

- Train/Test Split:
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

- Timer:
	timeit

- Merge datasets:
-- Use merge() function
-- Can use specific columns (as in SQL) or both table indices

- Pandas Idioms (make your code "pandorable"):
    - Index chaining (to be avoided)
    - Method chaining: include all code into a bracket to improve readability
    - Apply function to run the same function on all rows

- GroupBy:
-- can use Agg() function to compute variables on the aggregated object
-- Dispatching: generate additional key to split data to be treated by different jobs
  
- Merge:
-- Used to join two dataframes. For example: 
	m= pd.merge(top15,energy, how='inner', left_on='Country', right_on = 'Country' )

-- Select
	df.loc[:, ['foo','bar','dat']] # select columns foo, bar and dat

-Apply:
	def money_to_float(money_str):
    	return float(money_str.replace("$","").replace(",",""))
	df['SAL-RATE'].apply(money_to_float)

- Scales:
-- Ratio
-- Interval
-- Ordinal
-- Nominal

- Interpolation
from scipy.interpolate import interp1d
f1 = interp1d(precision, recall, kind='nearest')
value = f1(newx)

- Hypothesis testing:
-- Hp = a statement we can test
-- Critical Value (alpha) = a threshold one is willing to accept



- Model Evaluation and Selection for Supervised Algorithms
-- Generic golden rule: 
1. train set -> build model
2. validation set (among train set) -> select model
3. test set -> final evaluation
-- Accuracy = correctly_labelled/sample_size . This is not always a good metric. Acc = (TP + TN)/(TP + FP + TN + FN)
-- Dummy classifier can be used for sanity check. One can use the following streategies: most_frequent, stratified, uniform, constant
-- Confusion matrix [TN FP; FN TP]
from sklearn.metrics import confusion_matrix
# Negative class (0) is most frequent
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
y_majority_predicted = dummy_majority.predict(X_test)
confusion = confusion_matrix(y_test, y_majority_predicted)
-- True Positive Rate (AKA Recall, Sensitivity, Probability of detection)= TP/{TP + FN}
-- Precision = TP / {TP + FP}
-- False Positive Rate (aka Specificity) = FP / {TN + FP}
-- Recall oriented application: tumor detection, 
-- Precision oriented application: document classification, search engine ranking
-- FI-score = 2 * precision * recall / {precision + recall} = 2 * TP / {2 * TP + FN + FP}
-- How to establish the decision threshold? Recall vs Precision tradeoff
-- ROC curve: True Positive Rate vs False Positive Rate
import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


-- Multi-class evaluation:
--- macro average: each class has the same weight
--- micro average: each instance has equal weight 
 
- Evaluation for Regression Algorithms
-- Dummy regressors can be used for sanity check. Mean, median, quantile, constant
-- Matrics: r2_score (good), mean_absolute_error, mean_squared_error, median_absolute_error

## COLUMN MANIPULATION
- Convert column from string to datetime:
	qd.columns = pd.to_datetime(qd.columns)
- Sort dataframe by column:
	result = df.sort(['A', 'B'], ascending=[1, 0])
- Rename single columns:
	data.rename(columns={'gdp':'log(gdp)'}, inplace=True)
- Set column to true if other column is NaN:
	df.loc[df.Col1.isnull(), 'newCol'] = 1
- Dataframe dimensions:
	data.shape
	
## DEFINITIONS:
- Stochastic variable: variable whose values depend on the outcome of a non-deterministic event. A random variable has a probability distribution which specifies the probability of its values.

- Expected value: mean value if an infinite number of samples were drawn from the distribution

- Skewness: measurement of the asymmetry of a distribution

- R^2 = evaluation metric AKA "coefficient of determination"


## Plotting

- DATAVIZ (Alberto Cairo)
-- Abstraction vs Figuration
-- Functionality vs Decoration
-- Density vs Lightness
-- Multi-dimension vs Unidimension
-- Originality vs Familiarity
-- Novelty vs Redundancy

- REM: Data-ink ratio (Tufte)

- Truthful Art
-- Be aware that your actions are not misleading:
--- Yourself
--- The audience
-- Functionality
--- Beauty
--- Insightful
--- Enlightening

- Scatter matrix
	cmap = cm.get_cmap('gnuplot')
	scatter = pd.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

- 3D plot
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s=100)
	ax.set_xlabel('width')
	ax.set_ylabel('height')
	ax.set_zlabel('color_score')
	plt.show()

##
- Grid Search
dataset = load_digits()
X, y = dataset.data, dataset.target == 1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = SVC(kernel='rbf')
grid_values = {'gamma': [0.0001, 0.001, 0.01, 0.05, 0.1, 1, 10, 100]}

# default metric to optimize over grid parameters: accuracy
grid_clf_acc = GridSearchCV(clf, param_grid = grid_values)
grid_clf_acc.fit(X_train, y_train)
y_decision_fn_scores_acc = grid_clf_acc.decision_function(X_test) 

print('Grid best parameter (max. accuracy): ', grid_clf_acc.best_params_)
print('Grid best score (accuracy): ', grid_clf_acc.best_score_)


## Algorithms
- KNN:
-- Parameters: 
1. Distance metric
2. K
3. Optional weighting function
4. Aggregation method

- Least Squares (LS) Regression: y = w*x + b
-- Ordinary LS Regression: w and b found by mininiming the mean square error
-- Ridge Reg.: w and b found by minimising the mean square error with regularization (squared). It requires feature normalization (!!!)
-- Lasso Reg.: w and b found by minimising the mean square error with regularization (abs value). It requires feature normalization (!!!)
-- LS Polynomial Reg: add multiplicative combinations of the features and apply regression. 
	from sklearn.linear_model import LinearRegression
	from sklearn.preprocessing import PolynomialFeatures
	poly = PolynomialFeatures(degree=2)
	X_F1_poly = poly.fit_transform(X_F1)
	X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1,random_state = 0)
	linreg = LinearRegression().fit(X_train, y_train)

- Linear Regression: 
-- Used for binary classification
-- Computes a real value output based on a linear compbination of the input x. y = logistic ( w*x + bar)
-- logistic function is a non-linear s-shape function. 

- Decision Tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
dt = DecisionTreeClassifier()
dt = dt.fit(X_train, y_train)
tree.plot_tree(clf.fit(iris.data, iris.target)) 

- Naive Bayes Classification
-- Assumes no correlation between features for instances of the same class
-- Gaussian [uses mean and std dev of features]
from sklearn.naive_bayes import GaussianNB
nbclf = GaussianNB().fit(X_train, y_train)
plot_class_regions_for_classifier(nbclf, X_train, y_train, X_test, y_test,
                                 'Gaussian Naive Bayes classifier: Dataset 2')

- Random Forests: Ensemble method combining several decision trees.
-- Steps: 
1/ Bootstrapped randomized copies 
2/ Randomized feature split
3/ Ensemble prediction
-- good prediction performances
-- easy to parallelize
-- doesn't perform well on high dimensional spaces
-- hard to interpret for humans
-- parameters: n_estimators (how many trees), max_features (how many features to consider in each tree), max_depth, n_jobs (number of cores to use)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_features = 8, random_state = 0)
clf.fit(X_train, y_train)

- Gradient Boosted Decision Trees
-- series of weak decision trees 
-- parameters: n_estimators = # of small decision trees to use in the ensemble, learning_rate = controls emphasis on fixing errors from the previous tree
-- often performs well off-the-shelf
-- hard to interpret
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier().fit(X_train, y_train)

- Support Vector Machines
-- Linear binary classifier f(x) = sign (Wx + b)
-- Classifier margin: maximum width the the decision boundary can be increased before hitting a data point
--  SVM is the linear classifier with maximum classifier margin
-- Regularization: C parameter. greater C, fit training data as well as possible. 
-- Example:  
    from sklearn.svm import SVC
    clf = SVC(kernel='rbf', random_state=0, C=1)
    clf = clf.fit(X_train2, y_train2)
