import numpy as np
import plotly
import plotly.graph_objects as go
import sklearn.metrics as m

COLORS = ["#8D99AE", "#D90429", "#2B2D42", "#EDF2F4"] # hex codes for the colors of the visuals

class ModelSummary:

    """ Model Summary Class

    An object for providing visualization of metrics for regression
    but only binary classification.

    Parameters
    -----------
    ytest : array-like, shape(n_observations,)
        The target variable/binary class values.

    ypred : array-like, shape(n_observations,)
        The predicted value or class values.

    Attributes
    -----------
    _len : int
        The length of ytest

    """

    def __init__(self, ytest, ypred):

        self.ytest = ytest
        self.ypred = ypred
        self._length = len(self.ytest)
        self._acc = m.accuracy_score(self.ytest, self.ypred)
        self._precision = m.precision_score(self.ytest, self.ypred)
        self._recall = m.recall_score(self.ytest, self.ypred)
        # self._auc = m.auc(self.ytest, self.ypred)
        self._matrix = m.confusion_matrix(self.ytest, self.ypred)
        self._tn, self._fp, self._fn, self._tp = self._matrix.ravel()

        if len(self.ytest) != len(self.ypred): # checks if ytest and ypred are the same length

            raise ValueError("ytest and ypred need to be the same length.") # raise a ValueError

    @property
    def length(self):
        return self._length

    def acc(self):
        return self._acc

    def precision(self):
        return self._precision

    def recall(self):
        return self._recall

    def auc(self):
        return self._auc

    def matrix(self):
        return self._matrix

    def tn(self):
        return self._tn

    def fp(self):
        return self._fp

    def fn(self):
        return self._fn

    def tp(self):
        return self._tp

    @staticmethod
    def reg(self):
        """
        This method is what I like to call a helper method
        because it is a method that helps a class. It helps
        the show method if it is a regression model by calculating
        the points of the line of best fit.

        Returns
        --------
        The points of the line of best fit
        """
        # instatiates the x axis ticks and the y axis ticks
        x, y = [i for i in range()], self.ytest

        # instatiates slope of the line of the best fit
        m = (np.mean(x) * np.mean(y) - np.mean(x * y)) / (np.mean(x) ** 2 - np.mean(x ** 2))

        # insatiates the intercept (bias)
        c = np.mean(y) - m * np.mean(x)

        # instatiates the points of the line of best fit
        best_line = [m * x[i] + c for i in range(len(x))]

        # returns the line of best fit
        return best_line

    def show_summary(self, type_pred='classify', **kargs):
        """
        A visualization method to print out some sort of
        visual whether it be classifying or predicting.

        Parameters
        -----------
        None.

        **kwargs
        type_pred : bool, optional, default='classify'

        Returns
        --------
        Nothing. Prints out a roc curve if type_pred equals
        'classify' but if 'reg' then prints out a residuals
        plot.
        """
        if type_pred == "classify": # checks whether to plot ROC curve

            fpr, tpr, thresh = m.roc_curve(self.ytest, self.ypred) # instantiating false pos rate and true pos rate
            acc = m.accuracy_score(self.ytest, self.ypred) # instatiating accuracy score

            # instantiating the meta data for the roc curve
            # first go.Scatter creates the ROC line
            # second go.Scatter creates the traditional hypotenuse of the ROC curve
            data = [
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    name="ROC Curve",
                    line=dict(dash="solid", color=COLORS[1], width=4),
                ),
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    line=dict(dash="dash", color=COLORS[-1], width=2),
                ),
            ]

            #instatiating the style for the ROC curve
            layout = go.Layout(
                title=f"ROC Curve {acc}",
                plot_bgcolor=COLORS[0],
                paper_bgcolor=COLORS[-1],
                xaxis=dict(title="False Positive Rate"),
                yaxis=dict(title="True Positive Rate"),
            )

            fig = go.Figure(data=data, layout=layout) # instatiates the figure
            fig.show() # renders the plot
            print("AUC Score -------> ", m.roc_auc_score(self.ytest, self.ypred)) # explanatory

        else:
            best_line = reg() # instatiates with the class static method to find the best line of fit of ytest
            residuals = self.ytest - self.ypred # instatiates errors

            # 1. go.Scatter plots the line of best fit
            # 2. go.Scatter plots the errors
            # 3. go.Scatter plots ytest
            data = [
                go.Scatter(
                    x=_len,
                    y=best_line,
                    mode="line",
                    name="Best Fit Line of Class Values",
                ),
                go.Scatter(x=_len, y=residuals, mode="marker", name="Predicted Values"),
                go.Scatter(x=_len, y=self.ytest, mode="marker", name="Real Values"),
            ]

            # go.Layout styles the residual plot
            layout = go.Layout(
                title="Residual Plot",
                xaxis=dict(title="# of Observations", showticklabels=False),
                yaxis=dict(title="Predicted & Real Values"),
            )

            fig = go.Figure(data=data, layout=layout) # instatiates the figure
            fig.show() # renders the plot

        return None

    def confusion_visual(self):
        """

        A function to visualize the values in a confusion
        matrix providing by the sklearn.metrics method 
        confusion_matrix().

        Parameters
        -----------
        ytest : array-like, shape(n_observations,)
            The target variable/binary class values.

        ypred : array-like, shape(n_observations,)
            The predicted value or class values.

        Returns
        --------
        Doesn't technically return anything, but it shows
        bar plot of the true negatives and positives and 
        the false negatives and positives.

        """

        # instantiates the values from the confusion matrix    
        tn, fn, fp, tp = m.confusion_matrix(self.ytest, self.ypred).ravel()

        # creates the bar plots for each value from the confusion matrix
        data = go.Bar(x=[tn, fn, fp, tp], 
                      y=['True Negs', 'False Negs', 'False Pos', 'True Pos'],
                      orientation='h')

        # creates the meta data and styling for the plot
        layout = go.Layout(title="Confusion Bar Not Confusion Matrix")

        # creates the figure for the plot
        fig = go.Figure(data=data, layout=layout)
        fig.show() # renders the plot

        return None
    


    def report_summary(self, type_pred=None, loss=None):
        """
        A method to print out a model report composed of model evaluation metrics
        from sklearn of accuracy, confusion matrix, the classification report, loss
        funtion metric, and a r2 score.

        Returns
        --------
        Nothing. Prints multiple model evaluation metrics
        """

        if type_pred == "classify": # prints the classification model report

            # chain of print statements of the classification metrics
            print("_________________")
            print("Accuracy: ", m.accuracy_score(self.ytest, self.ypred))
            print("-----------------")
            print("Confusion Matrix: \n", m.confusion_matrix(self.ytest, self.ypred))
            print("-----------------")
            print(
                "Classification Report: \n",
                m.classification_report(self.ytest, self.ypred),
            )
            print("-----------------")
            print("MAE Score: ", m.mean_absolute_error(self.ytest, self.ypred))
            print("_________________")

        else:

            # chain of print statements of the regression metrics and loss functions
            print("_________________")
            print("R" + "\xb2" + "Score: ", m.r2_score(self.ytest, self.ypred))
            print("-----------------")

            if loss == "mse": # prints MSE score
                print(
                    "Loss Function: Mean Squared Error\n",
                    m.mean_squared_error(self.ytest, self.ypred),
                )
                print("_________________")

            elif loss == "mae": # prints MAE score
                print(
                    "Loss Function: Mean Absolute Error\n",
                    m.mean_absolute_error(self.ytest, self.ypred),
                )
                print("_________________")

            else: # prints average residual error
                avg_err = sum(self.ytest - self.ypred) / len(self.ytest)
                print("Loss Function: Average Error\n", avg_err)

        return None

if __name__ == '__main__':
    ytest = [1,0,0,0,1,1,0,1]
    ypred = [1,0,1,1,1,1,0,1]
    sum = Model_Summary(ytest, ypred)
    sum.show(type_pred='classify')
