def LinModReg(dados, target):
	from numpy import mean, std, logspace, array, sqrt, max, min, argmin
	import matplotlib.pyplot as plt

	#Usar regressao linear para prever o valor do monthlyIncome a partir das outras variaveis
	from sklearn.linear_model import Lasso
	from sklearn.model_selection import KFold
	from sklearn.model_selection import cross_val_score

	valorSemNa = dados.dropna()
	from sklearn import linear_model

	X = valorSemNa.drop([target], axis=1)
	y = valorSemNa[target]

	alphas = logspace(-4,10,10)
	print(alphas)

	lasso = Lasso(random_state=0)

	scores = list()
	scores_std = list()

	n_folds = 3

	for alpha in alphas:
		lasso.alpha = alpha
		this_scores = cross_val_score(lasso, X, y, cv=n_folds, n_jobs=1, scoring = 'neg_mean_squared_error')
		scores.append(mean(this_scores))
		scores_std.append(std(this_scores))

	scores, scores_std = array(scores), array(scores_std)

	plt.figure().set_size_inches(8, 6)
	plt.plot(alphas, scores)

	# plot error lines showing +/- std. errors of the scores
	std_error = scores_std / sqrt(n_folds)

	plt.plot(alphas, scores + std_error, 'b--')
	plt.plot(alphas, scores - std_error, 'b--')

	# alpha=0.2 controls the translucency of the fill color
	plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

	plt.ylabel('CV score +/- std error')
	plt.xlabel('alpha')
	plt.axhline(max(scores), linestyle='--', color='.5')
	plt.xlim([alphas[0], alphas[-1]])
	
	print 'alpha escolhido: ' + str(alphas[argmin(scores)])

