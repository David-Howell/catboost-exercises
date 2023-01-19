import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor, XGBClassifier, XGBRFClassifier


from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score, \
precision_recall_curve, recall_score, precision_score, accuracy_score, ConfusionMatrixDisplay
from scipy import stats

from formating import bold, underline, strike, Percent, col_cap

# class models()

tnfp = np.array([['TN', 'FP'],['FN', 'TP']])

def prep_X_sets(X_train: pd.DataFrame, X_validate: pd.DataFrame, X_test: pd.DataFrame):
    '''
    Prepares the X_train, X_validate, and X_test sets for modeling
    -----------

    Parameters    :
    -----------

    X_train       :    The training X set of variables, in a DataFrame
    X_validate    :    The validate X set of variables, in a DataFrame
    X_test        :    The  testing X set of variables, in a DataFrame

    Returns       :
    -----------

    X_train, X_validate, X_test : with the objectionable columns removed

    *returned as a list: 
    
    it will automatically unpack if you have 
    
    three_variables, seperated, by_commas = prep_X_sets(X_train, X_validate, X_test)
    '''

    X = [X_train, X_validate, X_test]
    
    drop_these = ['object_number',
    'accessionyear',
    'constituent_id',
    'artist_role',
    'artist_alpha_sort',
    'artist_nationality',
    'artist_begin_date',
    'artist_end_date',
    'artist_gender',
    'object_begin_date',
    'country',
    'classification',
     'object_wikidata_url',
    'tags',
    #  'gallery_number',
    #  'department',
    #  'object_name',
    #  'culture',
    #  'credit_line',
    #  'medium',
    'title',
    'portfolio',
    'cluster_strong_yes',
    'cluster_strong_no',
    ]  

    for x in X:
        x.drop(columns= drop_these, inplace=True)

    return X

class make_model:
    '''
    Welcome to the Make_Model class!
----------------------------------------
    
    Initialize  :
    -------------
    X           :       The X_train dataset
    y           :       The y_train target data
    X_val       :       The X_validate dataset
    y_val       :       The y_validate target data
    model_name  :       
    
    from {'decision_tree': DecisionTreeClassifier(),
          'xgbreg': XGBRegressor(),
          'xgbclass': XGBClassifier(),
          'xgbrf': XGBRFClassifier
        }
    scoring_method :    = 'recall' # You can change this to any sklearn scorer
    maximum_depth  :    = None # Set to your hearts' desire
    learning_rate  :    = 1 # will change the learning rate in the Gradient Descent

    '''
    def __init__(self, X, y, X_val, y_val, model_name, scoring_method= 'recall', maximum_depth= None, learning_rate = 1):
        
        # Map the False and True values to 0 and 1 respectively. 
        # This way the model doesn't spit out that future warning
        y = y.map({False: 0, True: 1})
        y_val = y_val.map({False: 0, True: 1})
        
        self.train = [X, y]
        self.validate = [X_val, y_val]
        self.name = model_name
        self.scoring = scoring_method
        self.max_depth = maximum_depth
# 

# Need to rewrite this so that we have a dictionary of model_names with models
#   So we can call the models and adjust them if we want 
#     ~~Additionally, if it's called from the dict it can be called and refit for the cross trees~~

        if self.name.lower() == 'decision_tree':
            model = DecisionTreeClassifier(max_depth= maximum_depth, random_state=123)
        
        elif model_name.lower() == 'xgbreg':
            model = XGBRegressor(objective= 'reg:logistic', random_state=123)

        elif model_name.lower() == 'xgbclass':
            model = XGBClassifier(random_state=123, max_depth= maximum_depth, eval_metric= 'logloss', use_label_encoder= False)

        elif model_name.lower() == 'xgbrf':
            model = XGBRFClassifier(learning_rate= learning_rate, random_state=123)

        else:
            print('You did not select a model properly, please check the documentation')
            
            
        # Save the model as .model_ : This is not fit, and can be used to test fit other data
        self.model_ = model
        # Save the fit model as .model
        self.model = model.fit(X,y)

        self.train_preds = self.model.predict(X).round().astype('int')
        
        self.val_preds = self.model.predict(X_val).round().astype('int')

        self.recall = {'train': recall_score(y, self.train_preds),
                        'validate': recall_score(y_val, self.val_preds)}

        self.precision = {'train': precision_score(y, self.train_preds),
                          'validate': precision_score(y_val, self.val_preds)}

        self.accuracy = {'train': accuracy_score(y, self.train_preds),
                         'validate': accuracy_score(y_val, self.val_preds)}


        self.confusion_ = {'train': confusion_matrix(y, self.train_preds),
                          'validate': confusion_matrix(y_val, self.val_preds)}

        self.report_ = {'train': classification_report(y, self.train_preds),
                        'validate': classification_report(y_val, self.val_preds)}

    def confusion(self):

        for x in self.confusion_:
            disp = ConfusionMatrixDisplay(confusion_matrix= self.confusion_[x], display_labels= self.model.classes_)
            target_title = ' '.join([x.capitalize() for x in self.train[1].name.split('_')])
            disp.plot()
            plt.title(f'Confusion Matrix for {col_cap(self.train[1].name)} in the {x.capitalize()} Dataset', y= 1.1, fontsize= 20)
            plt.show()
    
    def report(self):
        #  Init the dictionary of confusion dictionaries
        conf_dict = {}

# Each of Train and Validate in the confusion_ dictionary
        for tvt_set in self.confusion_:
        
            # Build the confusion dictionary for each dataset, Train, Validate(, Test)
            conf_each = {}
            # 2 for loops iterate through the 2x2 confusion matrix np.array
            for i in range(2):
                for j in range(2):
                    # Saving the results of each into the confusion dictionary
                    # using the TN, FP, FN, TP keys in: 
                    # `tnfp` the 2X2 np.array saved above
                    conf_each[tnfp[i][j]] = self.confusion_[tvt_set][i][j]
            # Save the resultant confusion dictionary into 
            # the dictionary of such dictionaries...
            conf_dict[tvt_set] = conf_each

        # Grab the name of the target from the y_train array
        target = self.train[1].name
        
        for tvt_set in conf_dict:

            confusion(tvt_set, 
                      TN= conf_dict[tvt_set]['TN'],
                      FP= conf_dict[tvt_set]['FP'],
                      FN= conf_dict[tvt_set]['FN'],
                      TP= conf_dict[tvt_set]['TP'],
                      target= target
            )


    def cross_trees(self, cv=5, tests= 20, long_run= False):

        if (self.name == 'decision_tree') | (long_run == True):

            results = []

            for x in range(1,(tests + 1)):
                tree = self.model_.set_params(max_depth=x)
                the_cross = cross_validate(tree, self.train[0], self.train[1], cv=cv, scoring= self.scoring)
                score = the_cross['test_score'].mean()
                results.append([x, score])
                
            pd.DataFrame(results, columns = ['max_depth', self.scoring])\
            .set_index('max_depth').plot(xticks=range(1,21))
            plt.title(f'{self.scoring.capitalize()} vs Max Depth in Decision Tree')
            plt.show()

        else:
            print('''
                Only DecisionTreeClissifier models can give you a cross_tree() graph... :/
                ... Unless you set long_run = True ... at your own risk of time
                ''')

    def test(self, X_test, y_test):
        y_test = y_test.map({False: 0, True: 1})

        self.test_preds = self.model.predict(X_test).round().astype('int')

        self.recall['test'] = recall_score(y_test, self.test_preds)

        self.precision['test'] = precision_score(y_test, self.test_preds)

        self.accuracy['test'] = accuracy_score(y_test, self.test_preds)

        self.confusion_['test'] = confusion_matrix(y_test, self.test_preds)

        self.report_['test'] = classification_report(y_test, self.test_preds)


def confusion(tvt, TN, TP, FN, FP, target=''):
    acc = (TP+TN)/(TP+TN+FP+FN)
    pre = (TP/(TP+FP))
    NPV = (TN/(TN+FN))
    rec = (TP/(TP+FN))
    spe = (TN/(TN+FP))
    f1s = stats.hmean([(TP/(TP+FP)),(TP/(TP+FN))])
    print(
    f'''
    {tvt.capitalize()} Dataset : Evaluation Report — Target : {col_cap(target)}
    _______________________________________________________________________________________
    
    True Positive = {TP} ---- False Positive = {FP}
    True Negative = {TN} ---- False Negative = {FN}
    
    Correct predictions = {TP+TN} (True Pos + True Neg)
      Total predictions = {TP+FN+FP+TN} predictions
    
    REAL POSITIVE = (TP + FN) = {TP+FN} ---- PREDICTED POSITIVE = (TP + FP) = {TP+FP}
    
    REAL NEGATIVE = (TN + FP) = {TN+FP} ---- PREDICTED NEGATIVE = (TN + FN) = {TN+FN}
     
        Accuracy = {acc:.2%} -->> Correct Predictions / Total Predictions
       Precision = {pre:.2%} -->> True Positive / Predicted Positive
             NPV = {NPV:.2%} -->> True Negative / Predicted Negative
          Recall = {rec:.2%} -->> True Positive / Real Positive
     Specificity = {spe:.2%} -->> True Negative / Real Negative
        f1-score = {f1s:.2%} -->> Harmonic Mean of Precision and Recall
    _______________________________________________________________________________________
    
    
    
    '''
    )