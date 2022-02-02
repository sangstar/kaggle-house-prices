import pandas as pd
import numpy as np    
from numpy import dot
from numpy.linalg import norm
import json

def cos_sim(a,b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim

def get_most_similar_rows(cos_sim_df):
    try:
        with open("files/most_sim_rows.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        results = {}
        for index, row in cos_sim_df.iterrows():
            if cos_sim_df.iloc[index].isnull().any():
                vector_with_nan = cos_sim_df.iloc[index]
                pos_of_nan = np.where([pd.isnull(x) for x in cos_sim_df.iloc[index]])[0]
                nan_vec = vector_with_nan[pos_of_nan]
                print(pos_of_nan)
                if not all([pd.isnull(cos_sim_df.loc[index, cos_sim_df.columns[x]]) for x in pos_of_nan]):
                    raise ValueError('One of the values is not NaN')
                permitted_indices = list(range(len(vector_with_nan)))
                if len(permitted_indices) != len(vector_with_nan):
                    raise ValueError('Length of permitted indices and full vec need to be the same.')
                permitted_indices = [x for x in permitted_indices if x not in pos_of_nan]
                new_vector = vector_with_nan[permitted_indices]
                cos_sim_scores = []
                for ind in cos_sim_df.index:
                    if ind != index:
                        to_compare = cos_sim_df.iloc[ind][permitted_indices]
                        result = cos_sim(new_vector,to_compare)
                        if not pd.isnull(result):
                            cos_sim_scores.append((ind, result, pos_of_nan.tolist()))
            results.update({row['Id'] : sorted(cos_sim_scores, key = lambda x: x[1], reverse = True)})
        #with open('files/most_sim_rows.json', 'w') as f:
        #    json.dump(results, f)
        return results