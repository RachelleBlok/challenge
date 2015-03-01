import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression, LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, BaggingClassifier, BaggingRegressor, RandomForestClassifier, AdaBoostRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from scipy import stats
import operator
import copy
import time

class MyAutoML:
	''' Rough sketch of a class that "solves" the AutoML problem. We illustrate various type of data that will be encountered in the challenge can be handled.
		 Also, we make sure that the model regularly outputs predictions on validation and test data, such that, if the execution of the program is interrupted (timeout)
		 there are still results provided by the program. The baseline methods chosen are not optimized and do not provide particularly good results.
		 In particular, no special effort was put into dealing with missing values and categorical variables.
		 
		 The constructor selects a model based on the data information passed as argument. This is a form of model selection "filter".
		 We anticipate that the participants may compute a wider range of statistics to perform filter model selection.
		 We also anticipate that the participants will conduct cross-validation experiments to further select amoung various models
		 and hyper-parameters of the model. They might walk trough "model space" systematically (e.g. with grid search), heuristically (e.g. with greedy strategies),
		 or stochastically (random walks). This example does not bother doing that. We simply use a growing ensemble of models to improve predictions over time.
		 
		 We use ensemble methods that vote on an increasing number of classifiers. For efficiency, we use WARM self.start that re-uses
		 already trained base predictors, when available.
		 
		IMPORTANT: This is just a "toy" example:
			- if was checked only on the phase 0 data at the time of release
			- not all cases are considered
			- this could easily break on datasets from further phases
			- this is very inefficient (most ensembles have no "warm self.start" option, hence we do a lot of unnecessary calculations)
			- there is no preprocessing
		 '''
		
		 
	def __init__(self, info, X, Y, time_budget, time_spent, verbose=True, debug_mode=False):
		self.start = time.time()
		time_budget = time_budget - time_spent
		print "TIME BUDGET"
		print time_budget
		time_spent = 0
	
		self.classifiers = {}	#dictionary to save all models
		self.scoring = {}			#dictionary to save the scores of all models
		
		# get data from info file 	
		self.train_num = info['train_num']
		self.feat_num = info['feat_num']
		self.label_num=info['label_num']
		self.target_num=info['target_num']
		self.task = info['task']
		self.metric = info['metric']
		self.postprocessor = None
		#self.postprocessor = MultiLabelEnsemble(LogisticRegression(), balance=True) # To calibrate proba
		self.postprocessor = MultiLabelEnsemble(LogisticRegression(), balance=False) # To calibrate proba
	   
		####----------- TASK IS REGRESSION ------------####
		if info['task']=='regression':
			name0 = "LinearRegression"
			name1 = "Ridge"
			name2 = "GradientBoostingRegressor"
			name3 = "RandomForestRegressor"
			model0 = LinearRegression(fit_intercept = True, normalize = True, copy_X = True) 
			model1 = Ridge(alpha=1.0, fit_intercept = True, normalize = True, copy_X = True) 
			model2 = GradientBoostingRegressor(n_estimators=1, verbose=verbose, warm_start = True)
			model3 = RandomForestRegressor()
			
			self.classifiers.update({name0:model0, name1:model1, name2:model2, name3:model3})
	
		####----------- TASK IS REGRESSION ------------####	
		
		####----------- TASK IS MULTICLASS ------------####	
		
		elif info['task'] == 'multiclass.classification':
			print "Bagging Classifier"
			name0 = "BaggingNBClassifier"
			model0 = BaggingClassifier(base_estimator=BernoulliNB(), n_estimators=1, verbose=verbose)
			
			self.classifiers.update({name0:model0})
			self.predict_method = model0.predict_proba 
		
		####----------- TASK IS MULTICLASS ------------####	
		
		print "CLASSIFIERS"
		print self.classifiers
		
		for key in self.classifiers:	#fit all models
			self.name = key
			self.model = self.classifiers[key]
			print "HUIDIGE MODEL"
			print self.model
				
			if self.name == "GradientBoostingRegressor":	#increase number of estimators
				cycle = 5
				max_cycle = 6
				time_spent = time.time() - self.start
				while (time_spent <= time_budget/8 and cycle <= max_cycle):
					self.model.n_estimators = int(np.exp2(cycle))
					cycle += 1
					self.fit(X, Y)
					time_spent = time.time() - self.start
			
			if self.name == "RandomForestRegressor":	#increase number of estimators
				cycle = 3
				max_cycle = 5
				time_spent = time.time() - self.start
			
				while (time_spent <= time_budget/1.6 and cycle <= max_cycle):
					self.model.n_estimators = int(np.exp2(cycle))
					cycle += 1
					self.fit(X, Y)
					time_spent = time.time() - self.start		
				
			elif self.name == "Ridge": #randomized grid search (cross validation)
				param_grid = {'alpha': stats.uniform(), 'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']}
				n_iter = 100
				self.randomsearch(param_grid, n_iter, X, Y)
			
				
			elif self.name == "LinearRegression": #randomized grid search (cross validation)
				param_grid = {'fit_intercept': [True, False], 'normalize':[True, False]}
				n_iter = 4
				self.randomsearch(param_grid, n_iter, X, Y)
			
			else:	#k-fold cross validation	
				self.fit(X, Y)
		
		print "SCORES"
		print self.scoring
		
		self.bestmodel(X, Y)	#fit the model with highest score and predict target values			  

	def __repr__(self):
		return "MyAutoML : " + self.name

	def __str__(self):
		return "MyAutoML : \n" + str(self.model) 
		
	def randomsearch(self, parameters, number_iterations, X, Y):
		rsearch = RandomizedSearchCV(estimator = self.model, param_distributions = parameters, n_iter = number_iterations, cv=10)
		rsearch.fit(X, Y)
		print "BESTE SCORE"
		print(rsearch.best_score_)
		print "BESTE PARAMETERS"
		print(rsearch.best_estimator_)
		self.scoring.update({self.name:rsearch.best_score_})	# update dictionary with best score  
		self.classifiers.update({self.name:rsearch.best_estimator_})	
			
	
	def bestmodel(self, X, Y):	#get the model with the highest score and fit & predict the model
		maxelement = max(self.scoring.iteritems(), key=operator.itemgetter(1))[0]

		for key in self.classifiers: #set the model with highest score
			if key == maxelement:
				self.model = self.classifiers[maxelement]
				self.name = key
		
		print "BEST MODEL"
		print self.name
		print self.model
		
		print "FITTEN"
		time_spent = time.time() - self.start
		print "TIME SPENT"
		print time_spent 
		self.model.fit(X, Y)
		
		if self.task == "multiclass.classification":
			self.predict_method = self.model.predict_proba 
		else:
			self.predict_method = self.model.predict #predict values


	def fit(self, X, Y):
		self.model.fit(X,Y)			
		if self.task == 'multiclass.classification':
			kf = cross_validation.KFold(len(X.toarray()), n_folds=10, indices = True)
		else:
			kf = cross_validation.KFold(len(X), n_folds=10)
		
		score = cross_val_score(self.model, X, Y, cv=kf, n_jobs=-1).mean()
		self.scoring.update({self.name:score})
		
		# Train a calibration model postprocessor
		if self.task != 'regression' and self.postprocessor!=None:
			Yhat = self.predict_method(X)
			if len(Yhat.shape)==1: # IG modif Feb3 2015
				Yhat = np.reshape(Yhat,(-1,1))			 
			self.postprocessor.fit(Yhat, Y)
		return self
		
	def predict(self, X):
		prediction = self.predict_method(X)
		# Calibrate proba
		if self.task != 'regression' and self.postprocessor!=None:			
			prediction = self.postprocessor.predict_proba(prediction)
		# Keep only 2nd column because the second one is 1-first	
		if self.target_num==1 and len(prediction.shape)>1 and prediction.shape[1]>1:
			prediction = prediction[:,1]
		# Make sure the normalization is correct
		if self.task=='multiclass.classification':
			eps = 1e-15
			norma = np.sum(prediction, axis=1)
			for k in range(prediction.shape[0]):
				prediction[k,:] /= sp.maximum(norma[k], eps)  
		return prediction

class MultiLabelEnsemble:
	''' MultiLabelEnsemble(predictorInstance, balance=False)
		Like OneVsRestClassifier: Wrapping class to train multiple models when 
		several objectives are given as target values. Its predictor may be an ensemble.
		This class can be used to create a one-vs-rest classifier from multiple 0/1 labels
		to treat a multi-label problem or to create a one-vs-rest classifier from
		a categorical target variable.
		Arguments:
			predictorInstance -- A predictor instance is passed as argument (be careful, you must instantiate
		the predictor class before passing the argument, i.e. end with (), 
		e.g. LogisticRegression().
			balance -- True/False. If True, attempts to re-balance classes in training data
			by including a random sample (without replacement) s.t. the largest class has at most 2 times
		the number of elements of the smallest one.
		Example Usage: mymodel =  MultiLabelEnsemble (GradientBoostingClassifier(), True)'''
	
	def __init__(self, predictorInstance, balance=False):
		self.predictors = [predictorInstance]
		self.n_label = 1
		self.n_target = 1
		self.n_estimators =	 1 # for predictors that are ensembles of estimators
		self.balance=balance
		
	def __repr__(self):
		return "MultiLabelEnsemble"

	def __str__(self):
		return "MultiLabelEnsemble : \n" + "\tn_label={}\n".format(self.n_label) + "\tn_target={}\n".format(self.n_target) + "\tn_estimators={}\n".format(self.n_estimators) + str(self.predictors[0])
	
	def fit(self, X, Y):
		if len(Y.shape)==1: 
			Y = np.array([Y]).transpose() # Transform vector into column matrix
			# This is NOT what we want: Y = Y.reshape( -1, 1 ), because Y.shape[1] out of range
		self.n_target = Y.shape[1]				   # Num target values = num col of Y
		self.n_label = len(set(Y.ravel()))		   # Num labels = num classes (categories of categorical var if n_target=1 or n_target if labels are binary )
		# Create the right number of copies of the predictor instance
		if len(self.predictors)!=self.n_target:
			predictorInstance = self.predictors[0]
			self.predictors = [predictorInstance]
			for i in range(1,self.n_target):
				self.predictors.append(copy.copy(predictorInstance))
		# Fit all predictors
		for i in range(self.n_target):
			# Update the number of desired prodictos
			if hasattr(self.predictors[i], 'n_estimators'):
				self.predictors[i].n_estimators=self.n_estimators
			# Subsample if desired
			if self.balance:
				pos = Y[:,i]>0
				neg = Y[:,i]<=0
				if sum(pos)<sum(neg): 
					chosen = pos
					not_chosen = neg
				else: 
					chosen = neg
					not_chosen = pos
				num = sum(chosen)
				idx=filter(lambda(x): x[1]==True, enumerate(not_chosen))
				idx=np.array(zip(*idx)[0])
				np.random.shuffle(idx)
				chosen[idx[0:min(num, len(idx))]]=True
				# Train with chosen samples			   
				self.predictors[i].fit(X[chosen,:],Y[chosen,i])
			else:
				self.predictors[i].fit(X,Y[:,i])
		return
		
	def predict_proba(self, X):
		if len(X.shape)==1: # IG modif Feb3 2015
			X = np.reshape(X,(-1,1))   
		prediction = self.predictors[0].predict_proba(X)
		if self.n_label==2:					# Keep only 1 prediction, 1st column = (1 - 2nd column)
			prediction = prediction[:,1]
		for i in range(1,self.n_target): # More than 1 target, we assume that labels are binary
			new_prediction = self.predictors[i].predict_proba(X)[:,1]
			prediction = np.column_stack((prediction, new_prediction))
		return prediction
		
class RandomPredictor:
	''' Make random predictions.'''
	
	def __init__(self, target_num):
		self.target_num=target_num
		return
		
	def __repr__(self):
		return "RandomPredictor"

	def __str__(self):
		return "RandomPredictor"
	
	def fit(self, X, Y):
		if len(Y.shape)>1:
			assert(self.target_num==Y.shape[1])
		return self
		
	def predict_proba(self, X):
		prediction = np.random.rand(X.shape[0],self.target_num)
		return prediction			