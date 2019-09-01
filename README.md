## USEFUL LIBRARIES
#Pandas: opensource framework to manipulate data
#Scikit
#Matplotlib
#NumPy

PANDAS USEFUL DATA STRUCTURES:
	- Series
	- Data frames

TECHNIQUES
  - Boolean masking: create a boolean matrix over your data

MISCELLANEOUS

  *Timer:
    - timeit

  *Merge datasets:
    - Use merge() function
    - Can use specific columns (as in SQL) or both table indices

  *Pandas Idioms (make your code "pandorable"):
    - Index chaining (to be avoided)
    - Method chaining: include all code into a bracket to improve readability
    - Apply function to run the same function on all rows

  *GroupBy:
    -  can use Agg() function to compute variables on the aggregated object
    - Dispatching: generate additional key to split data to be treated by different jobs
  
  *Merge: 
    - Used to join two dataframes. For example: m= pd.merge(top15,energy, how='inner', left_on='Country', right_on = 'Country' )

Select
# select specific columns with a list
# select columns foo, bar and dat
df.loc[:, ['foo','bar','dat']]


Apply:
def money_to_float(money_str):
    return float(money_str.replace("$","").replace(",",""))
df['SAL-RATE'].apply(money_to_float)
> 

  Scales:
    1. Ratio
    2. Interval
    3. Ordinal
    4. Nominal

Hypothesis testing:
	- Hp = a statement we can test
	- Critical Value (alpha) = a threshold one is willing to accept


COLUMN MANIPULATION
	- Convert column from string to datetime:
  qd.columns = pd.to_datetime(qd.columns)
	- Sort dataframe by column:
	result = df.sort(['A', 'B'], ascending=[1, 0])
	- Rename single columns:
	data.rename(columns={'gdp':'log(gdp)'}, inplace=True)
	- Set column to true if other column is NaN:
	df.loc[df.Col1.isnull(), 'newCol'] = 1
	- Filter rows by date:
	
DEFINITIONS:

	- Stochastic variable: variable whose values depend on the outcome of a non-deterministic event. A random variable has a probability distribution which specifies the probability of its values.

	- Expected value: mean value if an infinite number of samples were drawn from the distribution

	- Skewness: measurement of the asymmetry of a distribution

--
Plotting

DATAVIZ (Alberto Cairo)
	- Abstraction vs Figuration
	- Functionality vs Decoration
	- Density vs Lightness
	- Multi-dimension vs Unidimension
	- Originality vs Familiarity
	- Novelty vs Redundancy

Data-ink ratio (Tufte)

Truthful Art
	1. Be aware that your actions are not misleading:
		a. Yourself
		b. The audience
	2. Functionality
	3. Beauty
	4. Insightful
	5. Enlightening

Scatter matrix
cmap = cm.get_cmap('gnuplot')
scatter = pd.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

3D plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c = y_train, marker = 'o', s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
plt.show()

Algorithms
KNN

