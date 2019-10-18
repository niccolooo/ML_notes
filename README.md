### USEFUL LIBRARIES
- Pandas: opensource framework to manipulate data
- Scikit
- Matplotlib
- NumPy

## PANDAS USEFUL DATA STRUCTURES:
- Series
- Data Frame

## TECHNIQUES
- Boolean masking: create a boolean matrix over your data

- Normalization:
		from sklearn.preprocessing import MinMaxScaler
		scaler = MinMaxScaler()
		X_train_scaled = scaler.fit_transform(X_train)
		X_test_scaled = scaler.transform(X_test)
		
- Cross validation
-- Run multiple tran-test splits 
-- K-fold CV: split the dataset into K folds and run K trainings modifying the test each time.
-- Stratified CV:
-- Leave-one-out CV: each fold has one single element

## MISCELLANEOUS
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

- Hypothesis testing:
-- Hp = a statement we can test
-- Critical Value (alpha) = a threshold one is willing to accept


## COLUMN MANIPULATION
- Convert column from string to datetime:
	qd.columns = pd.to_datetime(qd.columns)
- Sort dataframe by column:
	result = df.sort(['A', 'B'], ascending=[1, 0])
- Rename single columns:
	data.rename(columns={'gdp':'log(gdp)'}, inplace=True)
- Set column to true if other column is NaN:
	df.loc[df.Col1.isnull(), 'newCol'] = 1
- Filter rows by date:
	
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

- Support Vector Machines
-- Classifier margin: maximum width the the decision boundary can be increased before hitting a data point
--  SVM is the linear classifier with maximum classifier margin

- Kernelized SVM
-- kernel = similary measure between data points

- Decision Trees
-- 
