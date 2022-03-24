import random

def make_fold_indices(this_fold, fold, indices, cumsum_list):
    if this_fold == 0 :
        return indices[cumsum_list[0]:], indices[:cumsum_list[0]]

    elif this_fold == (fold-1) :
        return indices[: cumsum_list[-2]], indices[cumsum_list[-2]:]
    
    else :
        train_indices = []
        train_indices.extend(indices[:cumsum_list[this_fold-1]])
        train_indices.extend(indices[cumsum_list[this_fold]:])
        valid_indices = indices[cumsum_list[this_fold-1]: cumsum_list[this_fold]]
        return train_indices, valid_indices
        
def make_cumsum(lis):
    count = 0
    new_lis = [0] * len(lis)
    for i, l in enumerate(lis) :
        count += l
        new_lis[i] = count
    return new_lis

