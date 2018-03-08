from sklearn.linear_model import LogisticRegression

from sklearn import metrics


# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(cv, label)

# check the accuracy on the training set
print model.score(cv, label)


