features = ['bathrooms', 'bedrooms', 'longitude', 'latitude', 'price']
dfr = df[features]
X_train, y_train = dfr.drop('price',axis=1), dfr['price']
X_train['random'] = np.random.random(size=len(X_train))
rf = RandomForestRegressor(
         n_estimators=100,
         min_samples_leaf=1,
         n_jobs=-1,
         oob_score=True)
rf.fit(X_train, y_train)

###

features = ['bathrooms', 'bedrooms', 'price', 'longitude', 'latitude', 'interest_level']
dfc = df[features]
X_train, y_train = dfc.drop('interest_level',axis=1), dfc['interest_level']
X_train['random'] = np.random.random(size=len(X_train))
rf = RandomForestClassifier(
         n_estimators=100,
         # better generality with 5
         min_samples_leaf=5,
         n_jobs=-1,
         oob_score=True)
rf.fit(X_train, y_train)

###

def permutation_importances(rf, X_train, y_train, metric):
    baseline = metric(rf, X_train, y_train)
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = metric(rf, X_train, y_train)
        X_train[col] = save
        imp.append(baseline - m)
    return np.array(imp)

##

rf = RandomForestRegressor(...)
rf.fit(X_train, y_train) # rf must be pre-trained
imp = permutation_importances(rf, X_train, y_train,


rf = RandomForestClassifier(...)
imp = permutation_importances(rf, X_train, y_train,
                              oob_classifier_accuracy)


####
def dropcol_importances(rf, X_train, y_train):
    rf_ = clone(rf)
    rf_.random_state = 999
    rf_.fit(X_train, y_train)
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        rf_ = clone(rf)
        rf_.random_state = 999
        rf_.fit(X, y_train)
        o = rf_.oob_score_
        imp.append(baseline - o)
    imp = np.array(imp)
    I = pd.DataFrame(data={'Feature': X_train.columns, 'Importance': imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    return I

### rfpimp

from rfpimp import *
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
%time I = oob_importances(rf, X_train, y_train)

%time I = importances(rf, X_valid, y_valid, n_samples=-1)
%time I = cv_importances(rf, X_train, y_train, k=5)
%time I = dropcol_importances(rf, X_train, y_train)


from eli5.sklearn import PermutationImportance
%time perm = PermutationImportance(rf).fit(X_test, y_test)
