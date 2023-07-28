from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
from mlxtend.frequent_patterns import apriori

data = [['milk', 'onion', 'nutmeg', 'kidney beans', 'eggs', 'yogurt'],
        ['dill', 'onion', 'nutmeg', 'kidney beans', 'eggs', 'yogurt'],
        ['milk', 'apple', 'kidney beans', 'eggs'],
        ['milk', 'unicorn', 'corn', 'kidney beans', 'yogurt'],
        ['corn', 'onion', 'onion', 'kidney beans', 'ice cream', 'eggs']]

trans = TransactionEncoder()
trans_array = trans.fit(data).transform(data)
print(trans_array)
print(trans.columns_)
df = pd.DataFrame(trans_array, columns=trans.columns_)
print(df)

frequent_data = apriori(df, min_support=0.6, use_colnames=trans.columns_)
frequent_data['length'] = frequent_data['itemsets'].apply(lambda x: len(x))
print(frequent_data)

print(frequent_data[(frequent_data['support'] > 0.6) \
                    & (frequent_data['length'] == 2)])

