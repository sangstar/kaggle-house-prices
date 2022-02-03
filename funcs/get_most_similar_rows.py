import pandas as pd
import numpy as np    
from numpy import dot
from numpy.linalg import norm
import json
from scipy.stats import normaltest

alpha = 1e-3


def cos_sim(a,b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

def get_most_similar_rows(cos_sim_df, train):
    try:
        with open("files/most_sim_rows"+"_"+train+".json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        results = {}
        i = 0
        for index, row in cos_sim_df.iterrows():
            #print(i,'/',len(cos_sim_df))
            #print(index, row['Id'])
            print(index)
            if any([pd.isnull(cos_sim_df.loc[index, col]) for col in cos_sim_df.columns]):
                #vector_with_nan = cos_sim_df.iloc[index]
                vector_with_nan = [cos_sim_df.loc[index, col] for col in cos_sim_df.columns]
                pos_of_nan = np.where([pd.isnull(x) for x in vector_with_nan])[0]
                col_of_nan = [cos_sim_df.columns[x] for x in pos_of_nan]
                print(col_of_nan)
                #print(col_of_nan)
                #if index == 197:
                #    print('important: ',col_of_nan)
                #nan_vec = vector_with_nan[pos_of_nan]
                #nan_vec = [cos_sim_df.loc[index, cos_sim_df.columns[x]] for x in pos_of_nan]
                #print('vec with nan: ', vector_with_nan)
                #print('new nan vec',nan_vec)
                if not all([pd.isnull(cos_sim_df.loc[index, x]) for x in col_of_nan]):
                    #print(nan_vec)
                    raise ValueError('One of the values is not NaN')
                permitted_indices = list(range(len(vector_with_nan))) 
                #if len(permitted_indices) != len(vector_with_nan):
                #    raise ValueError('Length of permitted indices and full vec need to be the same.')
                permitted_indices = [x for x in permitted_indices if x not in pos_of_nan]
                new_vector = pd.Series(vector_with_nan)[permitted_indices][1:] # First index is Id
                #print('vector of interest looks like ',new_vector[:10])
                cos_sim_scores = []
                for ind in cos_sim_df.index:
                    if ind != index:
                        to_compare = cos_sim_df.iloc[ind][permitted_indices][1:]
                        #print('to compare: ', to_compare[:10])
                        result = cos_sim(new_vector,to_compare)
                        if not pd.isnull(result):
                            cos_sim_scores.append((ind, result, col_of_nan))
                results.update({index : sorted(cos_sim_scores, key = lambda x: x[1], reverse = True)})
            i += 1
        with  open("files/most_sim_rows"+"_"+train+".json", "w") as f:
            json.dump(results, f)
        return results


def smart_fillna(df, cutoff, results, all_numerical):
    for indices in df.index:
        #print(indices in list(results.keys()))
        #print(results[indices])
        #try:
        print(indices in list(results.keys()))
        #print(results_indices)
        if str(indices) in list(results.keys()):
            most_similar = [x[0] for x in results[str(indices)] if x[1] >= cutoff]
            columns_to_fill = [x[2] for x in results[str(indices)] if x[1] >= cutoff]
            subset = df.iloc[most_similar]
            #print(columns_to_fill)
            columns_to_fill = list(set(tuple(row) for row in columns_to_fill))
            #print('columns to fill:', columns_to_fill)
            for col_list in columns_to_fill: # Don't want to use 'mean' on categorical data
                for col in col_list:
                    print(indices, col)
                    print('before at index',indices,'and column', col,'sample is ',df.loc[indices, col])
                    if not pd.isnull(df.loc[indices, col]):
                        raise ValueError(df.loc[indices, col], 'should be null')
                    subset_test = subset[col]
                    if col in all_numerical:
                        if len(subset_test) >= 8: # normaltest requires at least 8 samples
                            normal_test_results = normaltest(subset_test)
                            #print(normal_test_results)
                            if normal_test_results[1] > alpha: #null hypothesis cannot be rejected that distribution is normal. Take mean of gaussian
                                df.loc[indices, col] = np.nanmean(subset_test)
                                print('mean')
                            else:
                                df.loc[indices, col] = np.nanmedian(subset_test)
                                print('med')
                        else:
                            print('less than 8 samples')
                            mean = np.nanmean(subset_test)
                            median = np.nanmedian(subset_test)
                            print(mean, median)
                            if np.abs((mean-median)/mean)*100 < 10: # Mean and median are close together -- can pick either, will pick mean
                                df.loc[indices, col] = mean
                            else: # Pick median -- not close together possibly due to outliers
                                df.loc[indices, col] = median
                    else:
                        print('not in all numerical')
                        print('set: ', set(subset_test))
                        df.loc[indices, col] = np.nanmedian(subset_test)
                    print('after at index',indices,'and column', col,'sample is ',df.loc[indices, col])
    return df
