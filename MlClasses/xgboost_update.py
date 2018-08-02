import xgboost
import numpy as np

class XGBClassifier(xgboost.XGBClassifier):
	

	separation_facet = 0.5 # default value
	# the value of separator should be changed: XGBClassifier.separation_facet = 0.7 ...

	def predict(self, data, output_margin=False, ntree_limit=0):
		"""
		Predict with `data`. 
		NOTE: This function is not thread safe.
		      For each booster object, predict can only be called from one thread.
		      If you want to run prediction using multiple thread, call xgb.copy() to make copies
		      of model object and then call predict
		Parameters
		----------
		data : DMatrix
		    The dmatrix storing the input.
		output_margin : bool
		    Whether to output the raw untransformed margin value.
		ntree_limit : int
		    Limit number of trees in the prediction; defaults to 0 (use all trees).
		Returns
		-------
		prediction : numpy array
		"""

		# Same changes were added to seperate signal with probabulity > self.separation_facet
		# print '____updated xgboost method______'
		test_dmatrix = xgboost.DMatrix(data, missing=self.missing, nthread=self.n_jobs)
		class_probs = self.get_booster().predict(test_dmatrix,
		                                         output_margin=output_margin,
		                                         ntree_limit=ntree_limit)
		if len(class_probs.shape) > 1:
			column_indexes = np.argmax(class_probs, axis=1)
		else:
			column_indexes = np.repeat(0, class_probs.shape[0])
			column_indexes[class_probs > self.separation_facet] = 1
		return self._le.inverse_transform(column_indexes)
	
