import numpy as np
import scipy as sp
from data_manager import DataManager
import data_converter
from data_io import vprint
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression, LinearRegression, SGDClassifier, Lasso, ElasticNet
from sklearn.svm import SVR, SVC, NuSVC, LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, BaggingClassifier, BaggingRegressor, RandomForestClassifier, AdaBoostRegressor, RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier 
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from scipy import stats
import operator
import copy
import time
import random

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
		
		 
	def __init__(self, info, Xtrain, Ytrain, Xtest, Xvalid, time_budget, time_spent, subsample, Xsub, Ysub, verbose=True, debug_mode=False):
		self.start = time.time()
		time_budget = time_budget - time_spent
		print "TIME BUDGET"
		print time_budget
		#time_spent = 0
	
		self.classifiers = {}	#dictionary to save all models
		self.scoring = {}			#dictionary to save the scores of all models
		
		self.Xsub = Xsub			#subsample x values
		self.Ysub = Ysub			#subsample y values
		
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
		self.feat_type = info['feat_type']
		self.task = info['task']
		self.metric = info['metric']
		self.postprocessor = None
		#self.postprocessor = MultiLabelEnsemble(LogisticRegression(), balance=True) # To calibrate proba
		self.postprocessor = MultiLabelEnsemble(LogisticRegression(), balance=False) # To calibrate proba
		
		if (subsample == True):
			print "SUBSAMPLE PREPROCESSING"
			self.preprocessing_subsample()
		else:
			print "GEEN SUBSAMPLE PREPROCESSING"
		
		self.Xtrain = Xtrain
		self.Ytrain = Ytrain
		self.Xtest = Xtest
		self.Xvalid = Xvalid
		
		print "PREPROCESSING"
		self.preprocessing(verbose=verbose)
	
		#try models 
		if info['task']=='regression':
			self.regression(verbose)
		elif info['task'] == 'multiclass.classification':
			self.multiclass(verbose)
		elif info['task']=='multilabel.classification':
			self.multilabel(verbose)
		elif info['task']=='binary.classification': 
			self.binary(verbose)
		
		self.trymodels(time_budget)
	
		self.bestmodel()	#fit the model with highest score and predict target values	
		
		time_spent = time.time() - self.start
		print "TIME SPENT"
		print time_spent
		
		self.model.fit(self.Xtrain, self.Ytrain)		  

	def __repr__(self):
		return "MyAutoML : " + self.name

	def __str__(self):
		return "MyAutoML : \n" + str(self.model) 
	
	def regression(self, verbose = True):
		print "Regression"
		name0 = "LinearRegression"
		model0 = LinearRegression(fit_intercept = True, normalize = True, copy_X = True) 
		
		name1 = "Ridge"
		model1 = Ridge(alpha=1.0, fit_intercept = True, normalize = True, copy_X = True)
		
		name2 = "GradientBoostingRegressor"
		model2 = GradientBoostingRegressor(n_estimators=1, verbose=verbose, warm_start = True)
		
		name3 = "RandomForestRegressor"
		model3 = RandomForestRegressor(n_estimators=1, verbose = verbose)
		
		self.classifiers.update({name0:model0, name1:model1, name2:model2, name3:model3})

		for key in self.classifiers:
			self.model = self.classifiers[key]	
			self.predict_method = self.model.predict 
	
	def multiclass(self, verbose = True):
		print "Multiclass"
		if self.sparse == 1:
			print "SGD"
			name1 = "SGD"
			model1 = SGDClassifier(loss='log', n_iter = np.ceil(10**6 / self.train_num), alpha=0.001, penalty='elasticnet', l1_ratio = 0.001, shuffle = True, verbose=verbose, warm_start = True)
		
			print "LogisticRegression"
			name0  = "LogisticRegression"
			model0 = LogisticRegression(C = 0.01) 
			
			self.classifiers.update({name0:model0})
		else:
			print "SVC"
			name0 = "SVC"
			model0 = SVC(C=10, verbose = 2, probability = True)
				
			self.classifiers.update({name0:model0})
			
		for key in self.classifiers:
			self.model = self.classifiers[key]
			self.predict_method = self.model.predict_proba  
	
	def multilabel(self, verbose = True):
		print "MultiLabel"
		name0 = "MultiLabelRandomForest"
		model1 = RandomForestClassifier(n_estimators=30, verbose=verbose)
		model0 =  OneVsRestClassifier (model1)
			
		self.classifiers.update({name0:model0})
		for key in self.classifiers:
			self.model = self.classifiers[key]
			self.predict_method = self.model.predict_proba
	
	def binary(self, verbose = True):
		print "Binary"
		name0 = "LogisticRegression"
		model0 = LogisticRegression() 
		
		name1 = "SVC" 
		model1 = SVC(probability = True, verbose = verbose)
		
		name2 = "SGDClassifier"
		model2 = SGDClassifier(n_iter = np.ceil(10**6 / self.train_num),loss = 'log', verbose = verbose)
			
		name3 = "RandomForestClassifier"
		model3 = RandomForestClassifier(n_estimators = 30, verbose = False)
			
		name4 = "KNeighborsClassifier"
		model4 = KNeighborsClassifier()
			
		name5 = "BaggingClassifier"
		model5 = BaggingClassifier(n_estimators = 30, verbose = verbose)
		
		name6 = "GradientBoostingClassifier"
		model6 = GradientBoostingClassifier(n_estimators = 30, verbose = verbose, warm_start = True)
			
		self.classifiers.update({name6:model6})
				
		for key in self.classifiers:
			self.name = key
			self.model = self.classifiers[key]	
			
		self.predict_method = self.model.predict_proba		
	
	def preprocessing_subsample(self, verbose=True):
		vprint (verbose, "preprocessing subsample ")		
		if self.has_missing== 1:
			vprint (verbose, "MISSING ")
			self.Xsub = np.nan_to_num(self.Xsub)
		if self.has_categorical == 1:
			vprint (verbose, "OHE ")
			cat_features = np.where(self.feat_type == 'Categorical')[0]
			ohe = preprocessing.OneHotEncoder(categorical_features = cat_features)
			ohe.fit(self.Xsub)
			self.Xsub = ohe.transform(self.Xsub)
		if self.feat_type == 'Numerical' or self.feat_type == 'Binary':
			vprint (verbose, "Normalize ")
			preprocessing.normalize(self.Xsub)
		if self.sparse == 0 and self.feat_num >= 1400 and self.feat_type == 'Numerical':
			vprint (verbose, "PCA ")
			pca = decomposition.PCA(n_components = int(self.feat_num*0.04), whiten = False)
			pca.fit(self.Xsub)
			self.Xsub = pca.transform(self.Xsub)
		if self.sparse == 1 and self.feat_num >= 1400:
			vprint (verbose, "feature selection")
			idx=[]
			fn = min(self.Xsub.shape[1], 1000)       
			idx = data_converter.tp_filter(self.Xsub, self.Ysub, feat_num=fn, verbose=verbose)
			self.Xsub = self.Xsub[:,idx]
			self.feat_idx = np.array(idx).ravel()
		if self.sparse == 0 and self.feat_type == 'Numerical':
			vprint (verbose, "Min Max Scaler ")
			mms = preprocessing.MinMaxScaler()
			mms.fit(self.Xsub)
			self.Xsub = mms.transform(self.Xsub)
		if self.sparse == 1 and (self.feat_type == 'Numerical' or self.feat_type == 'Binary' ):
			vprint (verbose, "Standard Scaler ")
			ss = preprocessing.StandardScaler(with_mean = False, copy = False)
			ss.fit(self.Xsub)
			self.Xsub = ss.transform(self.Xsub)
			
	def preprocessing(self, verbose=True): 
		vprint (verbose, "preprocessing ")		
		if self.has_missing== 1:
			vprint (verbose, "MISSING ")
			self.Xtrain = np.nan_to_num(self.Xtrain)
			self.Xtest = np.nan_to_num(self.Xtest)
			self.Xvalid = np.nan_to_num(self.Xvalid)
		if self.has_categorical == 1:
			vprint (verbose, "OHE ")
			cat_features = np.where(self.feat_type == 'Categorical')[0]
			ohe = preprocessing.OneHotEncoder(categorical_features = cat_features)
			ohe.fit(self.Xtrain)
			self.Xtrain = ohe.transform(self.Xtrain)
			self.Xtest = ohe.transform(self.Xtest)
			self.Xvalid = ohe.transform(self.Xvalid)
		if self.feat_type == 'Numerical' or self.feat_type == 'Binary':
			vprint (verbose, "Normalize ")
			preprocessing.normalize(self.Xtrain)
			preprocessing.normalize(self.Xtest)
			preprocessing.normalize(self.Xvalid)
		if self.sparse == 0 and self.feat_num >= 1400 and self.feat_type == 'Numerical':
			vprint (verbose, "PCA ")
			pca = decomposition.PCA(n_components = int(self.feat_num*0.04), whiten = False)
			pca.fit(self.Xtrain)
			self.Xtrain = pca.transform(self.Xtrain)
			self.Xtest = pca.transform(self.Xtest)
			self.Xvalid = pca.transform(self.Xvalid)
		if self.sparse == 1 and self.feat_num >= 1400:
			vprint (verbose, "feature selection")
			idx=[]
			fn = min(self.Xtrain.shape[1], 1000)       
			idx = data_converter.tp_filter(self.Xtrain, self.Ytrain, feat_num=fn, verbose=verbose)
			self.Xtrain = self.Xtrain[:,idx]
			self.Xvalid = self.Xvalid[:,idx]
			self.Xtest = self.Xtest[:,idx]  
			self.feat_idx = np.array(idx).ravel()
		if self.sparse == 0 and self.feat_type == 'Numerical':
			vprint (verbose, "Min Max Scaler ")
			mms = preprocessing.MinMaxScaler()
			mms.fit(self.Xtrain)
			self.Xtrain = mms.transform(self.Xtrain)
			self.Xtest = mms.transform(self.Xtest)
			self.Xvalid = mms.transform(self.Xvalid)
		if self.sparse == 1 and (self.feat_type == 'Numerical' or self.feat_type == 'Binary' ):
			vprint (verbose, "Standard Scaler ")
			ss = preprocessing.StandardScaler(with_mean = False, copy = False)
			ss.fit(self.Xtrain)
			self.Xtrain = ss.transform(self.Xtrain)
			self.Xtest = ss.transform(self.Xtest)
			self.Xvalid = ss.transform(self.Xvalid)		

	def randomsearch(self, parameters, number_iterations):
		rsearch = RandomizedSearchCV(estimator = self.model, param_distributions = parameters, n_iter = number_iterations, cv=10)
		print "VOOR RANDOM FITTEN"
		if self.train_num > 10000 or (self.train_num < 10000 and self.feat_num > 50000): #train on subsample of data 
			rsearch.fit(self.Xsub, self.Ysub)
		else:
			rsearch.fit(self.Xtrain, self.Ytrain)
		print "BESTE SCORE"
		print(rsearch.best_score_)
		print "BESTE PARAMETERS"
		print(rsearch.best_estimator_)
		self.scoring.update({self.name:rsearch.best_score_})	# update dictionary with best score  
		self.classifiers.update({self.name:rsearch.best_estimator_})	
		self.model.fit(self.Xtrain, self.Ytrain)
		
	def trymodels(self, time_budget):
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
					self.crossvalidation()
					time_spent = time.time() - self.start
			
			elif self.name == "RandomForestRegressor":	#increase number of estimators
				cycle = 3
				max_cycle = 5
				time_spent = time.time() - self.start
			
				while (time_spent <= time_budget/1.6 and cycle <= max_cycle):
					self.model.n_estimators = int(np.exp2(cycle))
					cycle += 1
					self.crossvalidation()
					time_spent = time.time() - self.start	
			
			elif self.name == "RandomForestClassifier" or self.name == "BaggingClassifier" or self.name == "GradientBoostingClassifier":	#increase number of estimators
				if self.sparse == 1:
					self.Xtrain = self.Xtrain.toarray()
					self.Xsub = self.Xsub.toarray()
				self.crossvalidation()
			
			elif self.name == "Ridge": #randomized grid search (cross validation)
				param_grid = {'alpha': stats.uniform(), 'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']}
				n_iter = 100
				self.randomsearch(param_grid, n_iter)	
				
			elif self.name == "LinearRegression": #randomized grid search (cross validation)
				param_grid = {'fit_intercept': [True, False], 'normalize':[True, False]}
				n_iter = 4
				self.randomsearch(param_grid, n_iter)
			
			elif self.name == "LogisticRegression": #randomized grid search (cross validation)
				param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
				n_iter = 7
				self.randomsearch(param_grid, n_iter)
			
			elif self.name == "KNeighborsClassifier": #randomized grid search (cross validation)
				param_grid = {'n_neighbors': [1, 2, 3, 4, 5]}
				n_iter = 5
				self.randomsearch(param_grid, n_iter)
		
			elif self.name == "SGDClassifier": #randomized grid search (cross validation)
				param_grid = {'alpha': stats.uniform(), 'l1_ratio': stats.uniform(),  'loss':[ 'log', 'modified_huber'], 'penalty':['l1', 'l2', 'elasticnet']}
				n_iter = 10
				self.randomsearch(param_grid, n_iter)
			
			else:	#k-fold cross validation	
				self.crossvalidation()
		
		print "SCORES"
		print self.scoring	
			
	
	def bestmodel(self):	#get the model with the highest score and fit & predict the model
		maxelement = max(self.scoring.iteritems(), key=operator.itemgetter(1))[0]

		for key in self.classifiers: #set the model with highest score
			if key == maxelement:
				self.model = self.classifiers[maxelement]
				self.name = key
		
		print "BEST MODEL"
		print self.name
		print self.model
		
		
	def crossvalidation(self):	
		if self.sparse == 1:
			print "CV sparse"
			if self.train_num > 10000 or (self.train_num < 10000 and self.feat_num > 50000):	#use subsample of the data
				kf = cross_validation.KFold(self.Xsub.shape[0], n_folds=10)	
			else:
				kf = cross_validation.KFold(self.Xtrain.shape[0], n_folds=10)	
		else:
			print "CV normaal"
			if self.train_num > 10000 or (self.train_num < 10000 and self.feat_num > 50000):	#use subsample of the data
				kf = cross_validation.KFold(len(self.Xsub), n_folds=10)
			else:
				kf = cross_validation.KFold(len(self.Xtrain), n_folds=10)
		
		if self.name == "MultiLabelRandomForest":
			if self.train_num > 10000 or (self.train_num < 10000 and self.feat_num > 50000):	#use subsample of the data
				score = cross_val_score(self.model, self.Xsub, self.Ysub, cv=kf, scoring="f1", n_jobs=-1).mean()
			else:
				score = cross_val_score(self.model, self.Xtrain, self.Ytrain, cv=kf, scoring="f1", n_jobs=-1).mean()
		else:
			if self.train_num > 10000 or (self.train_num < 10000 and self.feat_num > 50000):	#use subsample of the data
				score = cross_val_score(self.model, self.Xsub, self.Ysub, cv=kf, n_jobs=-1).mean()
			else:
				score = cross_val_score(self.model, self.Xtrain, self.Ytrain, cv=kf, n_jobs=-1).mean()
		
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
		
	def predict(self, X, isTrain):
		print "predict(self, X)"
		if (self.name == "RandomForestClassifier" or self.name == "BaggingClassifier" or self.name == "GradientBoostingClassifier") and self.sparse == 1 and isTrain == False:
			X = X.toarray()
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