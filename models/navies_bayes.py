import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.prior = {}

        for c in self.classes:
            X_c = X[y == c]

            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.prior[c] = X_c.shape[0] / X.shape[0]

    def gaussian(self, x, mean, var):
        eps = 1e-6
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
        exponent = np.exp(- (x - mean) ** 2 / (2 * var + eps))
        return coeff * exponent

    def predict(self, X):
        y_pred = []

        for x in X:
            scores = {}

            for c in self.classes:
                log_prior = np.log(self.prior[c])
                log_likelihood = np.sum(
                    np.log(self.gaussian(x, self.mean[c], self.var[c]))
                )

                scores[c] = log_prior + log_likelihood

            y_pred.append(max(scores, key=scores.get))

        return np.array(y_pred)