import pickle

# Path to adjacency file
adj_path = '/Users/ark/.cache/kagglehub/datasets/annnnguyen/metr-la-dataset/versions/4/adj_METR-LA.pkl'

import pickle

# Path to adjacency file
adj_path = '/Users/ark/.cache/kagglehub/datasets/annnnguyen/metr-la-dataset/versions/4/adj_METR-LA.pkl'

with open(adj_path, 'rb') as f:
    adj_data = pickle.load(f, encoding='latin1')
    print(type(adj_data))
    if isinstance(adj_data, dict):
        for k in adj_data:
            print(f"Key: {k}, type: {type(adj_data[k])}")
            if hasattr(adj_data[k], 'shape'):
                print(f"  shape: {adj_data[k].shape}")
    else:
        print(adj_data)
