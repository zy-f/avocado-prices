import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

def default_train(model, dataset, loss_func, optimizer, epochs=10, pred_y_func=None):#, show_plot=False):
    print("===Train===")
    model.train()
    outputs = []
    outputs2 = []
    for epoch in range(epochs):
        loss_size = 0.0
        iters = 0
        preds = np.zeros((len(dataset),)+model.out_shape)
        ys = []
        for i,batch in enumerate(dataset.get_batches()):
            inp = batch['inputs']
            lbl = batch['labels']
            out = model(inp)
            preds[i*dataset.bsz:(i+1)*dataset.bsz] = out.detach().numpy()
            ys += lbl.detach().numpy().tolist()
            loss = loss_func(out, lbl)
            loss.backward()
            optimizer.step()
            loss_size += loss.item()
            iters += 1
        loss_size /= iters
        ys = np.array(ys)    
        # acc = sum(np.equal(preds,train_y[:len(preds)]))/len(preds)*100
        if epochs <= 50 or epoch%(epochs//50) == 0:
            if pred_y_func is not None:
                comparator_out = pred_y_func(preds, ys)
            loss_str=f"{loss_size} [{('-.---') if len(outputs)==0 else ('%+.3f' % (loss_size-outputs[-1]))}]"
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_str}, {comparator_out}")#, Acc: {acc}%")
        outputs.append(loss_size)
    # if show_plot:
    plt.plot(outputs)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    return model


def default_test(model, dataset, pred_y_func=None, print_output_head=0):
    print("===Test===")
    model.eval()
    preds = np.zeros((len(dataset),)+model.out_shape)
    ys = []
    for i,batch in enumerate(dataset.get_batches()):
        inp = batch['inputs']
        lbl = batch['labels']
        out = model(inp)
        preds[i*dataset.bsz:(i+1)*dataset.bsz] = out.detach().numpy()
        ys += lbl.detach().numpy().tolist()
    ys = np.array(ys)
    if print_output_head != 0:
        for k in range(print_output_head if (print_output_head > 0 and len(dataset) > print_output_head) else len(dataset)):
            print(f"PRED = {preds[k]}\nTRUE = {ys[k]}\n")
    if pred_y_func is not None:
        print(pred_y_func(preds, ys))
    df_idx = dataset.df.index[:len(dataset)]
    preds = pd.Series(preds.squeeze(), index=df_idx)
    return preds

class Dataset(object):
    def __init__(self, df, inp_cols, out_cols, batch_size=64):
        self.bsz = batch_size
        self.df = df
        self.inp_cols = inp_cols
        self.out_cols = out_cols
    
    def get_batches(self, seed=None):
        self.df = self.df.sample(frac=1, random_state=seed)
        inp_data = self.df[self.inp_cols]
        out_data = self.df[self.out_cols]
        for k in range(len(self)//self.bsz):
            inputs = torch.from_numpy(np.array(inp_data[k*self.bsz:(k+1)*self.bsz], dtype=np.float32))
            labels = torch.from_numpy(np.array(out_data[k*self.bsz:(k+1)*self.bsz], dtype=np.float32))
            yield {'inputs': inputs, 'labels':labels}
        return
    
    def __getitem__(self, i):
        return self.df.iloc[i]
    
    def __len__(self):
        return len(self.df.index) - len(self.df.index)%self.bsz

def normalize_data(df, cols):
    df = df.copy(deep=True)
    for column in cols:
        initial = df.loc[:,column]
        max_val = max(initial)
        min_val = min(initial)
        normalized = (initial-min_val)/(max_val-min_val)
        df.loc[:,column] = normalized
    return df

def standardize_data(df, cols):
    df = df.copy(deep=True)
    for column in cols:
        initial = df.loc[:,column]
        stdev = initial.std()
        mean = initial.mean()
        standardized = (initial-mean)/stdev
        df.loc[:,column] = standardized
    return df

def int_category_encode(df, cols):
    int_encoded = []
    n_categories = []
    for col in cols:
        label_names = sorted(list(set(df[col])))
        col_series = pd.Series(data=np.zeros(len(df)), name=col)
        for k, val in enumerate(df[col]):
            col_series.iloc[k] = label_names.index(val)
        int_encoded.append(col_series)
        n_categories.append(len(label_names))
    return int_encoded, n_categories