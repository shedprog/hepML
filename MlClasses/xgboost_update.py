import xgboost
import numpy as np

class XGBClassifier(xgboost.XGBClassifier):

    def __init__(self, separation_facet = 0.5,max_depth=3, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 objective="binary:logistic", booster='gbtree',
                 n_jobs=1, nthread=None, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, random_state=0, seed=None, missing=None, **kwargs):


        super(XGBClassifier, self).__init__(max_depth, learning_rate,
                                            n_estimators, silent, objective, booster,
                                            n_jobs, nthread, gamma, min_child_weight,
                                            max_delta_step, subsample,
                                            colsample_bytree, colsample_bylevel,
                                            reg_alpha, reg_lambda,
                                            scale_pos_weight, base_score,
                                            random_state, seed, missing, **kwargs)
        self.separation_facet = separation_facet # Add internal variable for classifier facet

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
	
