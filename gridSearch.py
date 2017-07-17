def gridSearch(estimator ,grid, X_train, y_train, X_test, y_test, random_size = None):
    from sklearn.model_selection import ParameterGrid
    from sklearn.metrics import f1_score
    import pprint 
    if random_size:
        sizes = np.random.choice(np.arange(1, X_train.shape[1]), 
                                 random_size).tolist()
    else:
        sizes = [1]
    output = {}
    best_score = 0.0
    ticker = 0
    col_names = list(X_train)
    for size in sizes:
        select = np.random.choice(list(X_train), size, replace = False).tolist()
        X_train_slct = X_train[select]
        X_test_slct = X_test[select]
        for g in ParameterGrid(grid):
            ticker = ticker + 1
            clf = estimator
            clf.set_params(**g)
            clf.fit(X_train_slct, y_train)
            pred = clf.predict(X_test_slct)
            score = f1_score(y_test, pred)
            acc_score = clf.score(X_test_slct, y_test)
            output['run_' + str(ticker)] = {'columns': list(X_train_slct),
                        'params' : g,
                        'pred': pred, 
                        'score': score,
                        'acc': acc_score}
    return(output)