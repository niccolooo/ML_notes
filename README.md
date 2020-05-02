## DEFINITIONS

- **Correlation**: *any statistical relationship, whether causal or not, between two random variables or bivariate data. In the broadest sense correlation is any statistical association, though it commonly refers to the degree to which a pair of variables are linearly related.*

- **Expected value**: *mean value if an infinite number of samples were drawn from the distribution*. 

- **R^2** = evaluation metric AKA "coefficient of determination" TODO

- **Skewness**: *measure of the asymmetry of the probability distribution of a real-valued random variable about its mean. The skewness value can be positive or negative, or undefined.*

- **Stochastic variable**: *variable whose values depend on the outcome of a non-deterministic event. A random variable has a probability distribution which specifies the probability of its values.*  

- **Student T**: TODO

- **ChiSq**: TODO

## USEFUL LIBRARIES
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
	- ```df.loc[:, ['foo','bar','dat']]``` # select columns foo, bar and dat
	- select first X and last Y columns 
	```df.iloc[:, list(range(9)) + [-1]]

	- Create variable with TRUE if age is greater than 50 ```elderly = df['age'] > 50``` 
	- df[df.City.str.contains('ville',case=False)] # select with condition on string
	- ```data = data.drop(['compliance','compliance_detail'], axis = 1)``` #drop columns
	- ```X_train_reduced = X_train_reduced.apply(pd.to_numeric, errors = 'coerce')``` # convert all columns of DataFrame to numeric
	- ```data = full_data.loc[:, ["CLASS", "COUNTY", "geometry"]].copy()``` # select columns with copy
	- ```wild_lands = data.loc[data.CLASS.isin(['WILD FOREST', 'WILDERNESS'])].copy()``` # select rows if column value equals a value in a list
	- ```data.CLASS.value_counts()``` # get enums count within a column
	- Convert column from string to datetime:
		```qd.columns = pd.to_datetime(qd.columns)```  
	- Sort dataframe by column:
		``````result = df.sort(['A', 'B'], ascending=[1, 0])```  
	- Rename single columns:
		```data.rename(columns={'gdp':'log(gdp)'}, inplace=True)```  
	- Set column to true if other column is NaN:
		```df.loc[df.Col1.isnull(), 'newCol'] = 1```  
	- Dataframe dimensions:
		```data.shape'```  
	- Select all null 
		df[df['Col2'].isnull()]
	- Group by multiple columns
		worlddata.columns = ['Confirmed','Fatalities']
		worlddata = worlddata.reset_index()
		worlddata = worlddata.groupby(['Date', 'Country_Region']).agg({'ConfirmedCases': ['sum'],'Fatalities': ['sum']})
	- Moving average
	    d['rolling_fatalities'] = d.Fatalities.rolling(window=3).mean()

		
 - **[Pandas] List**: TODO DESCRIPTION
	- ```my_list = [1, 2, 3]``` 
	- ```my_list = [1, "Hello", 3.4]``` # list with mixed data types
	- ```my_list = ["mouse", [8, 4, 6], ['a']]``` # nested list

- **[Numpy] Array**: TODO DESCRIPTION
	- ```a = np.array([2,3,4])```

## TECHNIQUES

- **Feature selection**: 
	1. **Filter Method**: filter and take only the subset of the relevant features.
		```
		from sklearn.datasets import load_boston
		import pandas as pd
		import numpy as np
		import matplotlib
		import matplotlib.pyplot as plt
		import seaborn as sns
		import statsmodels.api as sm
		%matplotlib inline
		from sklearn.model_selection import train_test_split
		from sklearn.linear_model import LinearRegression
		from sklearn.feature_selection import RFE
		from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
		#Loading the dataset
		x = load_boston()
		df = pd.DataFrame(x.data, columns = x.feature_names)
		df["MEDV"] = x.target
		X = df.drop("MEDV",1)   #Feature Matrix
		y = df["MEDV"]          #Target Variable
		df.head()
		#Using Pearson Correlation
		plt.figure(figsize=(12,10))
		cor = df.corr()
		sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
		plt.show()
		#Correlation with output variable
		cor_target = abs(cor["MEDV"])
		#Selecting highly correlated features
		relevant_features = cor_target[cor_target>0.5]
		# Check correlation between selected features
		print(df[["LSTAT","PTRATIO"]].corr())
		print(df[["RM","LSTAT"]].corr())

	2. **Wrapper Method**
		- **Backward elimination**: As the name suggest, we feed all the possible features to the model at first. We check the performance of the model and then iteratively remove the worst performing features one by one till the overall performance of the model comes in acceptable range. The performance metric used here to evaluate feature performance is pvalue. If the pvalue is above 0.05 then we remove the feature, else we keep it.
		- **RFE (Recursive Feature Elimination)**: The Recursive Feature Elimination (RFE) method works by recursively removing attributes and building a model on those attributes that remain. It uses accuracy metric to rank the feature according to their importance.

	3. **Embedded Method**: Embedded methods are iterative in a sense that takes care of each iteration of the model training process and carefully extract those features which contribute the most to the training for a particular iteration. Regularization methods are the most commonly used embedded methods which penalize a feature given a coefficient threshold.

- **Boolean masking**:  create a boolean matrix over your data

- **Handle NaN**  
	```isnan = data().values```  
	```data[isnan] = 0```

- **Normalization**:  
	```from sklearn.preprocessing import MinMaxScaler```  
	```scaler = MinMaxScaler()```  
	```X_train_scaled = scaler.fit_transform(X_train)```  
	```X_test_scaled = scaler.transform(X_test)```

## EXPLORATION
- Get columns whose data type is object i.e. string
	```filteredColumns = empDfObj.dtypes[empDfObj.dtypes == np.object]```

- Get column info
	```empDfObj.info()```
	
## MISCELLANEOUS
- Grid Search
	```
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

- get type: ```type(my_data)```

- Train/Test Split: ```X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)```

- Timer:
	timeit

- Merge datasets:
	- Use merge() function
	- Can use specific columns (as in SQL) or both table indices
	- ```m= pd.merge(top15,energy, how='inner', left_on='Country', right_on = 'Country' )```

- Pandas Idioms (make your code "pandorable"):
    - Index chaining (to be avoided)
    - Method chaining: include all code into a bracket to improve readability
    - Apply function to run the same function on all rows

- GroupBy:
-- can use Agg() function to compute variables on the aggregated object
-- Dispatching: generate additional key to split data to be treated by different jobs
	df.groupby(['Animal']).mean()
	# or
	df.groupby(['col1','col2']).mean()
	
-Apply:
	```def money_to_float(money_str):```  
    	```return float(money_str.replace("$","").replace(",",""))```  
	```df['SAL-RATE'].apply(money_to_float)```

- Scales:
	- Ratio
	- Interval
	- Ordinal
	- Nominal

- Interpolation:
	```
	from scipy.interpolate import interp1d  
	f1 = interp1d(precision, recall, kind='nearest  
	value = f1(newx  

- Hypothesis testing:
	- Hp = a statement we can test
	- Critical Value (alpha) = a threshold one is willing to accept

## MODEL EVALUATION AND SELECTION (SUPERVISED)
- Generic golden rule: 
	1. train set -> build model
	2. validation set (among train set) -> select model
	3. test set -> final evaluation
- **Accuracy** = *correctly_labelled/sample_size* . This is not always a good metric. ```Acc = (TP + TN)/(TP + FP + TN + FN)```
- Confusion matrix [TN FP; FN TP] ```from sklearn.metrics import confusion_matrix
- **True Positive Rate (AKA Recall, Sensitivity, Probability of detection)**= TP/{TP + FN}
- **Precision** = TP / {TP + FP}
- **False Positive Rate (aka Specificity)** = FP / {TN + FP}
- Recall oriented application: tumor detection, 
- Precision oriented application: document classification, search engine ranking
- **FI-score** = 2 * precision * recall / {precision + recall} = 2 * TP / {2 * TP + FN + FP}
- How to establish the decision threshold? Recall vs Precision tradeoff
- **ROC curve**: True Positive Rate vs False Positive Rate
	```
	import sklearn.metrics as metrics  
	# calculate the fpr and tpr for all thresholds of the classification  
	probs = model.predict_proba(X_test)  
	preds = probs[:,1]
	fpr, tpr, threshold = metrics.roc_curve(y_test, preds)  
	roc_auc = metrics.auc(fpr, tpr)
	
- **Dummy classifier** can be used for sanity check. One can use the following streategies: most_frequent, stratified, uniform, constant
	
	```
	# Negative class (0) is most frequent  
	dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
	y_majority_predicted = dummy_majority.predict(X_test)
	confusion = confusion_matrix(y_test, y_majority_predicted)
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

- **Multi-class evaluation**
	- macro average: each class has the same weight
	- micro average: each instance has equal weight 
 
- **Evaluation for Regression Algorithms
	- Dummy regressors can be used for sanity check. Mean, median, quantile, constant
	- Matrics: r2_score (good), mean_absolute_error, mean_squared_error, median_absolute_error

## ALGORITHMS
### KNN:
- Parameters
	1. Distance metric
	2. K
	3. Optional weighting function
	4. Aggregation method

### Least Squares (LS) Regression: y = w*x + b
- Ordinary LS Regression: w and b found by mininiming the mean square error
- Ridge Reg.: w and b found by minimising the mean square error with regularization (squared). It requires feature normalization (!!!)
- Lasso Reg.: w and b found by minimising the mean square error with regularization (abs value). It requires feature normalization (!!!)
- LS Polynomial Reg: add multiplicative combinations of the features and apply regression. 
	```
	from sklearn.linear_model import LinearRegression
	from sklearn.preprocessing import PolynomialFeatures
	poly = PolynomialFeatures(degree=2)
	X_F1_poly = poly.fit_transform(X_F1)
	X_train, X_test, y_train, y_test = train_test_split(X_F1_poly, y_F1,random_state = 0)
	linreg = LinearRegression().fit(X_train, y_train)

### Linear Regression
- Used for binary classification
- Computes a real value output based on a linear compbination of the input x. y = logistic ( w*x + bar)
- logistic function is a non-linear s-shape function. 

### Decision Tree TODO DESCRIPTION
	```
	from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
	dt = DecisionTreeClassifier()
	dt = dt.fit(X_train, y_train)
	tree.plot_tree(clf.fit(iris.data, iris.target)) 

### Naive Bayes Classification
- Assumes no correlation between features for instances of the same class
- Gaussian [uses mean and std dev of features]
	```
	from sklearn.naive_bayes import GaussianNB
	nbclf = GaussianNB().fit(X_train, y_train)
	plot_class_regions_for_classifier(nbclf, X_train, y_train, X_test, y_test,'Gaussian Naive Bayes classifier: Dataset 2')

### Random Forests: Ensemble method combining several decision trees.
- Steps: 
	1. Bootstrapped randomized copies 
	2. Randomized feature split
	3. Ensemble prediction
- good prediction performances
- easy to parallelize
- doesn't perform well on high dimensional spaces
- hard to interpret for humans
- parameters: n_estimators (how many trees), max_features (how many features to consider in each tree), max_depth, n_jobs (number of cores to use)
	```
	from sklearn.ensemble import RandomForestClassifier
	clf = RandomForestClassifier(max_features = 8, random_state = 0)
	clf.fit(X_train, y_train)

### Gradient Boosted Decision Trees
- series of weak decision trees 
- parameters: n_estimators = # of small decision trees to use in the ensemble, learning_rate = controls emphasis on fixing errors from the previous tree
- often performs well off-the-shelf
- hard to interpret
	```
	from sklearn.ensemble import GradientBoostingClassifier
	clf = GradientBoostingClassifier().fit(X_train, y_train)

### Support Vector Machines
- Linear binary classifier f(x) = sign (Wx + b)
- Classifier margin: maximum width the the decision boundary can be increased before hitting a data point
-  SVM is the linear classifier with maximum classifier margin
- Regularization: C parameter. greater C, fit training data as well as possible. 
- Example:  
	```
	from sklearn.svm import SVC
    	clf = SVC(kernel='rbf', random_state=0, C=1)
    	clf = clf.fit(X_train2, y_train2)

## PLOTTING

- Dataviz (Alberto Cairo): TODO DESCRIPTION
	- Abstraction vs Figuration
	- Functionality vs Decoration
	- Density vs Lightness
	- Multi-dimension vs Unidimension
	- Originality vs Familiarity
	- Novelty vs Redundancy
	- Data-ink ration (Tufte)

- Truthful Art - Be aware that your actions are not misleading:
	- Yourself
	- The audience
	- Functionality
	- Beauty
	- Insightful
	- Enlightening

- Scatter matrix
	```
	cmap = cm.get_cmap('gnuplot')
	scatter = pd.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

- 3D plot
	```
	from mpl_toolkits.mplot3d import Axes3D  
	fig = plt.figure()  
	ax = fig.add_subplot(111, projection = '3d')  
	ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s=100)  
	ax.set_xlabel('width')  
	ax.set_ylabel('height')  
	ax.set_zlabel('color_score')  
	plt.show()  

- roatated labels on axis
	import matplotlib
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	ax.set_xticklabels(labels,rotation=45)

- Plot several lines 
	from datetime import datetime
	import matplotlib
	import matplotlib.pyplot as plt

	provinces = latest_prov.province[:(int)(n_provinces_to_plot/2)]
	fig2, ax2 = plt.subplots()

	for p in provinces:
		
		provdata = prov[prov.province==p]
		plt.plot(provdata.date, provdata.total_cases, '-xb', label=p, c=np.random.rand(3,)) 
		plt.legend(loc = 'upper left')
		plt.plot([0, 1], [0, 1],'r--')  
		#plt.ylabel('True Positive Rate')
		#plt.xlabel('False Positive Rate')
		ax2.set_xticklabels(list(provdata.date),rotation=70)

	plt.show()
	
- Make room for x axis
	plt.tight_layout()
	
- Format plot in matplotlib
	fmt = '[marker][line][color]'
	
- Set ticks
	major_ticks = np.arange(0, 101, 20)
	minor_ticks = np.arange(0, 101, 5)

	ax.set_xticks(major_ticks)
	ax.set_xticks(minor_ticks, minor=True)
	ax.set_yticks(major_ticks)
	ax.set_yticks(minor_ticks, minor=True)
	
- Rotate axis values
	plt.xticks(rotation=90)
	
## FILTERING
- Wavelet filter
- Average smooting
	def average_smoothing(signal, kernel_size=3, stride=1):
		sample = []
		start = 0
		end = kernel_size
		while end <= len(signal):
			start = start + stride
			end = end + stride
			sample.extend(np.ones(end - start)*np.mean(signal[start:end]))
		return np.array(sample)

