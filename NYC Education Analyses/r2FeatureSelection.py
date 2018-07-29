# This function uses the cross validation technique from sci-kit learn to implement a forward stepwise
# regression. It requires the inputs to be in array form, with the target and feature names as
# separate array and list objects respectively. The function returns a pandas data frame listing each
# features individual r2 score as well as the r2 score using a given feature and all features before
# it. 

def r2FeatureSelection(data,target,featureNames):
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score

    ## Create the individual feature r2 scores and rank
    lm = LinearRegression()
    output = pd.DataFrame(data=featureNames,columns= ['features'])
    inScores = [cross_val_score(estimator = lm,
                                X = data[:,x].reshape(-1, 1),
                                y = target,
                                cv=5,
                                scoring="r2").mean() for x in range(len(featureNames))]
    output['individualScores'] = inScores
    
    ## sort the scores from highest to lowest
    #output['delta'] = output.individualScores.apply(lambda x: 1- x)
    output.sort_values(by=['individualScores'],inplace = True)
    output.drop("delta",axis = 1,inplace = True)
    #featureIndex = output.index
    
    ## Create output list of features starting with best ranked from earlier,
    ## then successively add each next best feature 
    addScores = [output.iloc[0,1]]
    f = [featureIndex[0]]
    for i in range(1,len(featureNames)):
        f.append(featureIndex[i])
        addScores.append(cross_val_score(estimator = lm,
                                X = data[:,f],
                                y = target,
                                cv=5,
                                scoring="r2").mean())
    output['addedScores'] = addScores
    
    ## return feature selection data 
    return(output)