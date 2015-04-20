import numpy as np
import scipy as sp
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression, LinearRegression, SGDClassifier, Lasso, ElasticNet
from sklearn.svm import SVR, SVC, NuSVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, BaggingClassifier, BaggingRegressor, RandomForestClassifier, AdaBoostRegressor, RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier 
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
import random
import data_converter
import data_manager
import data_io
import os

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
		
		 
	def __init__(self, info, Xtrain, Ytrain, Xtest, Xvalid, time_budget, time_spent, verbose=True, debug_mode=False):
		self.start = time.time()
		time_budget = time_budget - time_spent
		print "TIME BUDGET"
		print time_budget
		#time_spent = 0
	
		self.classifiers = {}	#dictionary to save all models
		self.scoring = {}			#dictionary to save the scores of all models
		
		# get data from info file 
		self.dataset = info['name']
		self.score = info['metric'][0:-7] #voor de scoring parameter in cv goed te zetten
		self.has_categorical = info['has_categorical']
		self.has_missing = info['has_missing'] 	
		self.sparse = info['is_sparse']
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
			model3 = RandomForestRegressor(n_estimators=1, verbose = verbose)
		
			self.classifiers.update({name0:model0, name1:model1, name2:model2, name3:model3})
			

			for key in self.classifiers:
				self.model = self.classifiers[key]	
				self.predict_method = self.model.predict 
	
	
		####----------- TASK IS REGRESSION ------------####	
		
		####----------- TASK IS MULTICLASS ------------####	
		
		elif info['task'] == 'multiclass.classification':
			
			if self.sparse == 1:
				print "SGD"
				#name0 = "SGD"
				#model0 = SGDClassifier(loss='log', n_iter = np.ceil(10**6 / self.train_num), alpha=0.001, penalty='elasticnet', l1_ratio = 0.001, shuffle = True, verbose=verbose, warm_start = True)
		
				name0  = "LogisticRegression"
				model0 = LogisticRegression() 
			else:
				print "SVC"
				name0 = "SVC"
				model0 = SVC(C=10, verbose = 2, probability = True)
			
			
			self.classifiers.update({name0:model0})
			
			for key in self.classifiers:
				self.model = self.classifiers[key]
				self.predict_method = self.model.predict_proba  
			
		
		####----------- TASK IS MULTICLASS ------------####	
		
		####----------- TASK IS MULTILABEL ------------####	
		
		elif info['task']=='multilabel.classification':
			print "MultiLabel"
			name0 = "MultiLabelRandomForest"
			model1 = RandomForestClassifier(n_estimators=30, verbose=verbose)
			model0 =  OneVsRestClassifier (model1)
			
			self.classifiers.update({name0:model0})
			for key in self.classifiers:
				self.model = self.classifiers[key]
				self.predict_method = self.model.predict_proba

		####----------- TASK IS MULTILABEL ------------####	
		
		####----------- TASK IS BINARY ------------####	
		
		elif info['task']=='binary.classification': 
			name0 = "LogisticRegression"
			#name1 = "LinearSVC" --> SVC van maken 
			name2 = "SGDClassifier"
			model0 = LogisticRegression() 
			#model1 = LinearSVC(verbose = verbose)
			model2 = SGDClassifier(n_iter = np.ceil(10**6 / self.train_num),loss = 'log', verbose = verbose)
			
			self.classifiers.update({ name0:model0})
				
			for key in self.classifiers:
				self.model = self.classifiers[key]	
				self.predict_method = self.model.predict_proba
		
		####----------- TASK IS BINARY ------------####	
		
		self.trymodels(Xtrain, Ytrain, time_budget)
	
		self.bestmodel(Xtrain, Ytrain)	#fit the model with highest score and predict target values	
		
		time_spent = time.time() - self.start
		print "TIME SPENT"
		print time_spent
		
		self.model.fit(Xtrain, Ytrain)		  

	def __repr__(self):
		return "MyAutoML : " + self.name

	def __str__(self):
		return "MyAutoML : \n" + str(self.model) 


	def balanced_subsample(self,x,y,subsample_size):
		xs = []
		ys = []
		
		min_elems = self.train_num 
		
		print self.dataset
	
		if subsample_size < 1:
			use_elems = int(min_elems*subsample_size)
		else:
			use_elems = min_elems
		
		xs = x
		ys = y	
		

		#while xs.shape[0] > use_elems:
		'''
			print xs.shape[0]
			rands = random.randint(0, self.train_num-1)
			print rands
			try:
				xs = np.delete(xs, rands)
				ys = np.delete(ys,rands)
				print "gelukt"
			except:
				pass	
		'''
				
		with open('subsamplex', 'w') as f:
			f.write(str(xs))
		f.close() # you can omit in most cases as the destructor will call if
		
		with open('subsampley', 'w') as f:
			f.write(str(ys))
		f.close() # you can omit in most cases as the destructor will call if

		#DATA OMZETTEN MET DATA IO: data_func = {'dense':data_io.data, 'sparse':data_io.data_sparse, 'sparse_binary':data_io.data_binary_sparse}
		
		
		return xs, ys


	def randomsearch(self, parameters, number_iterations, Xtrain, Ytrain):
		rsearch = RandomizedSearchCV(estimator = self.model, param_distributions = parameters, n_iter = number_iterations, cv=10)
		print "VOOR RANDOM FITTEN"
		if self.train_num > 10000:
			hulpje = []
			hulpje = self.balanced_subsample(Xtrain, Ytrain, 0.6)
			Xsub = hulpje[0]
			Ysub = hulpje[1]
			rsearch.fit(Xsub, Ysub)
		else:
			rsearch.fit(Xtrain, Ytrain)
		print "BESTE SCORE"
		print(rsearch.best_score_)
		print "BESTE PARAMETERS"
		print(rsearch.best_estimator_)
		self.scoring.update({self.name:rsearch.best_score_})	# update dictionary with best score  
		self.classifiers.update({self.name:rsearch.best_estimator_})	
		self.model.fit(Xtrain, Ytrain)
		
	def trymodels(self, Xtrain, Ytrain, time_budget):
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
					self.crossvalidation(Xtrain, Ytrain)
					time_spent = time.time() - self.start
			
			elif self.name == "RandomForestRegressor":	#increase number of estimators
				cycle = 3
				max_cycle = 5
				time_spent = time.time() - self.start
			
				while (time_spent <= time_budget/1.6 and cycle <= max_cycle):
					self.model.n_estimators = int(np.exp2(cycle))
					cycle += 1
					self.crossvalidation(Xtrain, Ytrain)
					time_spent = time.time() - self.start		
				
			elif self.name == "Ridge": #randomized grid search (cross validation)
				param_grid = {'alpha': stats.uniform(), 'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']}
				n_iter = 100
				self.randomsearch(param_grid, n_iter, Xtrain, Ytrain)	
				
			elif self.name == "LinearRegression": #randomized grid search (cross validation)
				param_grid = {'fit_intercept': [True, False], 'normalize':[True, False]}
				n_iter = 4
				self.randomsearch(param_grid, n_iter, Xtrain, Ytrain)
			
			elif self.name == "LogisticRegression" or self.name == "LinearSVC": #randomized grid search (cross validation)
				param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
				n_iter = 7
				self.randomsearch(param_grid, n_iter, Xtrain, Ytrain)
		
			elif self.name == "SGDClassifier": #randomized grid search (cross validation)
				param_grid = {'alpha': stats.uniform(), 'l1_ratio': stats.uniform(),  'loss':[ 'log', 'modified_huber'], 'penalty':['l1', 'l2', 'elasticnet']}
				n_iter = 10
				self.randomsearch(param_grid, n_iter, Xtrain, Ytrain)
			
			else:	#k-fold cross validation	
				self.crossvalidation(Xtrain, Ytrain)
		
		print "SCORES"
		print self.scoring	
			
	
	def bestmodel(self, Xtrain, Ytrain):	#get the model with the highest score and fit & predict the model
		maxelement = max(self.scoring.iteritems(), key=operator.itemgetter(1))[0]

		for key in self.classifiers: #set the model with highest score
			if key == maxelement:
				self.model = self.classifiers[maxelement]
				self.name = key
		
		print "BEST MODEL"
		print self.name
		print self.model
		
		
	def crossvalidation(self, Xtrain, Ytrain):	
		if self.sparse == 1:
			print "CV sparse"
			
			if self.train_num > 10000:
				hulpje = []
				hulpje = self.balanced_subsample(Xtrain, Ytrain, 0.6)
				Xsub = hulpje[0]
				Ysub = hulpje[1]
				kf = cross_validation.KFold(Xsub.shape[0], n_folds=10)	
			else:
				kf = cross_validation.KFold(Xtrain.shape[0], n_folds=10)	
		else:
			print "CV normaal"
			if self.train_num > 10000:
				hulpje = []
				hulpje = self.balanced_subsample(Xtrain, Ytrain, 0.6)
				Xsub = hulpje[0]
				Ysub = hulpje[1]
				kf = cross_validation.KFold(len(Xsub), n_folds=10)	
			else:
				kf = cross_validation.KFold(len(Xtrain), n_folds=10)
		
		if self.name == "MultiLabelRandomForest":
			if self.train_num > 10000:
				hulpje = []
				hulpje = self.balanced_subsample(Xtrain, Ytrain, 0.6)
				Xsub = hulpje[0]
				Ysub = hulpje[1]
				score = cross_val_score(self.model, Xsub, Ysub, cv=kf, scoring="f1", n_jobs=-1).mean()
			else:
				score = cross_val_score(self.model, Xtrain, Ytrain, cv=kf, scoring="f1", n_jobs=-1).mean()
		else:
			if self.train_num > 10000:
				hulpje = []
				hulpje = self.balanced_subsample(Xtrain, Ytrain, 0.6)
				Xsub = hulpje[0]
				Ysub = hulpje[1]
				score = cross_val_score(self.model, Xsub, Ysub, cv=kf, scoring="f1", n_jobs=-1).mean()
			else:
				score = cross_val_score(self.model, Xtrain, Ytrain, cv=kf, n_jobs=-1).mean()
		
		print "SCORE"
		print score
		self.scoring.update({self.name:score})


	def fit(self, Xtrain, Ytrain):
		self.model.fit(Xtrain, Ytrain)			
		
		# Train a calibration model postprocessor
		
		if self.task != 'regression'  and self.postprocessor!=None:
			Yhat = self.predict_method(Xtrain)
			if len(Yhat.shape)==1: # IG modif Feb3 2015
				Yhat = np.reshape(Yhat,(-1,1))			 
			self.postprocessor.fit(Yhat, Ytrain)
		
		return self
		
	def predict(self, X):
		print "predict(self, X)"
			
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
	
	def fit(self, Xtrain, Ytrain):
		if len(Ytrain.shape)==1: 
			Ytrain = np.array([Ytrain]).transpose() # Transform vector into column matrix
			# This is NOT what we want: Y = Y.reshape( -1, 1 ), because Y.shape[1] out of range
		self.n_target = Ytrain.shape[1]				   # Num target values = num col of Y
		self.n_label = len(set(Ytrain.ravel()))		   # Num labels = num classes (categories of categorical var if n_target=1 or n_target if labels are binary )
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
				pos = Ytrain[:,i]>0
				neg = Ytrain[:,i]<=0
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
				self.predictors[i].fit(Xtrain[chosen,:],Ytrain[chosen,i])
			else:
				self.predictors[i].fit(Xtrain,Ytrain[:,i])
		return
		
	def predict_proba(self, Xtrain):
		if len(Xtrain.shape)==1: # IG modif Feb3 2015
			X = np.reshape(Xtrain,(-1,1))   
		prediction = self.predictors[0].predict_proba(Xtrain)
		if self.n_label==2:					# Keep only 1 prediction, 1st column = (1 - 2nd column)
			prediction = prediction[:,1]
		for i in range(1,self.n_target): # More than 1 target, we assume that labels are binary
			new_prediction = self.predictors[i].predict_proba(Xtrain)[:,1]
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
	
	def fit(self, Xtrain, Ytrain):
		if len(Ytrain.shape)>1:
			assert(self.target_num==Ytrain.shape[1])
		return self
		
	def predict_proba(self, Xtrain):
		prediction = np.random.rand(Xtrain.shape[0],self.target_num)
		return prediction			