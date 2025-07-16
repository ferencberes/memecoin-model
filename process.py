import json, sys, os
import numpy as np
import pandas as pd

def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    
    with open(data_name) as f:
        s = next(f)
        print(s)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])
            
            
            
            ts = float(e[2])
            label = int(e[3])
            
            feat = np.array([float(x) for x in e[4:]])
            
            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            
            feat_l.append(feat)
    return pd.DataFrame({'u': u_list, 
                         'i':i_list, 
                         'ts':ts_list, 
                         'label':label_list, 
                         'idx':idx_list}), np.array(feat_l)



def reindex(df):
    assert(df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert(df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    # shift user indices if needed
    if df.u.min() > 0:
        df.u = df.u - df.u.min()
    # shift item indices if needed
    if df.i.min() > 0:
        df.i = df.i - df.i.min()
    # shift timestamps if needed
    if df.ts.min() > 0:
        df.ts = df.ts - df.ts.min()
    
    upper_u = df.u.max() + 1
    new_i = df.i + upper_u
    
    new_df = df.copy()
    print('Max user:', new_df.u.max())
    print('Max item:', new_df.i.max())

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

    print('Max user updated:', new_df.u.max())
    print('Max item updated:', new_df.i.max())

    return new_df


def run(data_name):
    PATH = './processed/{}.csv'.format(data_name)
    if os.path.exists(PATH):
        OUT_DF = './processed/ml_{}.csv'.format(data_name)
        OUT_FEAT = './processed/ml_{}.npy'.format(data_name)
        OUT_NODE_FEAT = './processed/ml_{}_node.npy'.format(data_name)
        
        df, feat = preprocess(PATH)
        new_df = reindex(df)

        print('Edge features shape:', feat.shape)
        empty = np.zeros(feat.shape[1])[np.newaxis, :]
        feat = np.vstack([empty, feat])
        print('Edge features updated shape:', feat.shape)

        max_idx = max(new_df.u.max(), new_df.i.max())
        rand_feat = np.zeros((max_idx + 1, feat.shape[1]))
        print('Node features shape:', rand_feat.shape)
        
        new_df.to_csv(OUT_DF)
        # all provided features will be edge features
        np.save(OUT_FEAT, feat)
        # random features are used to identify nodes
        np.save(OUT_NODE_FEAT, rand_feat)
        return True
    else:
        return False
    
for action in ['buy', 'sell']:
    for prefix in ['', '_no_features']:
        for i in [20000, 50000]:
            data_id = f"{action}{prefix}_{i}"
            success = run(data_id)
            if success:
                print(f"Processed {data_id}")