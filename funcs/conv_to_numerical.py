import pandas as pd
import numpy as np

def conv_to_numerical(df, ordered_categories):
    #print(ordered_categories.keys())
    for col in ordered_categories.keys():
        #print(col)
        values = ordered_categories[col]
        ordering = list(reversed(range(len(values))))
        mappings = dict(list(zip(values, ordering)))
        #print(mappings)
        #print(df[col])
        for index, row in df.iterrows():
            x = row[col]
            if not isinstance(x, float):
                df.loc[index, col] = mappings[x]
        #print(df[col])
    return df