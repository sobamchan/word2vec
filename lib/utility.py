def make_batch_set(dataset, ids, sampler, negative_num, window_size):
    xb, yb, tb = [], [], []
    for i in ids:
        xid = dataset[i]
        for ind in range(1, window_size):
            p = i - ind
            if p >= 0:
                xb.append(xid)
                yid = dataset[p]
                yb.append(yid)
                tb.append(1) # positive sample
                for nid in sampler.sample(negative_num):
                    xb.append(yid)
                    yb.append(nid)
                    tb.append(0)
            p = i + ind
            if p < len(dataset):
                xb.append(xid)
                yid = dataset[p]
                yb.append(yid)
                tb.append(1)
                for nid in sampler.sample(negative_num):
                    xb.append(yid)
                    yb.append(nid)
                    tb.append(0)
    
    return xb, yb, tb
