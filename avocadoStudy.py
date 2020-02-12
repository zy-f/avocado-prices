import pandas as pd
import numpy as np
import torch
from avocadoNet import AvocadoNet
import matplotlib.pyplot as plt
from nnet_gen_utils import *
from datetime import datetime

def compute_MSE_error(preds, ys):
    total_err = sum((ys-preds)**2)
    avg_err = total_err/len(ys)
    return f"Average Error: {avg_err}"

def compute_residual_error(preds, ys):
    total_err = sum(np.absolute(ys-preds))
    avg_err = total_err/len(ys)
    return f"Average Residual Error: {avg_err}"

def main(pretrained_path=None):
    pd.set_option('display.max_rows', 100)
    dataset = pd.read_csv("avocado.csv")
    dataset = dataset.iloc[:,1:]
    dataset = dataset[dataset['region']=='TotalUS']
    print(dataset)
    input('>paused')
    valid_cols = []
    valid_cols = ['AveragePrice', '4046', '4225', '4770', 'Small Bags', 'Large Bags', 'XLarge Bags']
    valid_dataset = dataset[valid_cols]
    valid_dataset['Date'] = [datetime.strptime(date,"%Y-%m-%d").timetuple().tm_yday for date in dataset['Date']]
    valid_dataset['isOrganic'] = (np.array(dataset['type']) == 'organic').astype(np.float64)*2-1
    valid_dataset = valid_dataset.sample(frac=1, random_state=1).reset_index(drop=True)
    cleaned_dataset = standardize_data(valid_dataset,list(valid_dataset.columns)[1:])
    print(cleaned_dataset)
    input('>paused')

    useable_cols = list(cleaned_dataset.columns)
    out_cols = [useable_cols.pop(0)]
    inp_cols = useable_cols

    if pretrained_path is not None:
        avocado_model = torch.load(pretrained_path)
        test_set = Dataset(cleaned_dataset, inp_cols, out_cols, batch_size=16)
        split_idx = 0
    else:
        split_idx = int(len(dataset)*.8)
        train_data = cleaned_dataset.iloc[:split_idx]
        test_data = cleaned_dataset.iloc[split_idx:]
        
        """
        hyperparams for current best trainable network: 
        -train batch size=32
        -hidden_dims=50
        -lr=3e-5
        -weight_decay=0
        -epochs=2000
        """
        train_set = Dataset(train_data, inp_cols, out_cols, batch_size=32)
        test_set = Dataset(test_data, inp_cols, out_cols, batch_size=4)
        avocado_model = AvocadoNet(in_params=len(inp_cols), hidden_dims=50)
        print(avocado_model)
        
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(avocado_model.parameters(), lr=3e-5, weight_decay=0)
        
        # train the model
        avocado_model = default_train(avocado_model, train_set, loss_func, optimizer, epochs=2000, pred_y_func=compute_MSE_error)

    # evaluate the model
    preds = default_test(avocado_model, test_set, pred_y_func=compute_residual_error, print_output_head=0)

    # display model results
    test_data_raw = valid_dataset.iloc[split_idx:]
    test_data_raw['PredAvgPrice'] = preds.round(2)
    test_data_raw['PriceError'] = test_data_raw['PredAvgPrice'] - test_data_raw['AveragePrice']
    test_data_raw = test_data_raw[ list(test_data_raw.columns)[-2:]+list(test_data_raw.columns)[:-2] ]
    test_data_raw = test_data_raw[test_data_raw['PredAvgPrice'].notnull()]
    test_data_raw = test_data_raw.sort_values('AveragePrice')
    print(test_data_raw.head())
    fig = plt.figure(1)
    ax1 = plt.subplot(111)
    ax1.scatter(range(len(test_data_raw)), test_data_raw['PredAvgPrice'])
    ax1.scatter(range(len(test_data_raw)), test_data_raw['AveragePrice'])
    plt.title('Predicted VS Actual Average Price')
    plt.show()
    plt.close()

    if pretrained_path is None:
        filename = input('Save as? ')
        if filename != 'n' and 'skip' not in filename:
            torch.save(avocado_model, f"{filename}.pth")


def large_test(pretrained_path):
    pd.set_option('display.max_rows', 100)
    dataset = pd.read_csv("2019-plu-total-hab-data.csv", usecols=['Geography','Current Year Week Ending','Type','ASP Current Year','4046 Units','4225 Units',"4770 Units","SmlBagged Units","LrgBagged Units","X-LrgBagged Units"])
    dataset.columns = ['region','Date','type','AveragePrice', '4046', '4225', '4770', 'Small Bags', 'Large Bags', 'XLarge Bags']
    dataset = dataset[dataset['region']=='Total U.S.']
    
    valid_cols = ['AveragePrice', '4046', '4225', '4770', 'Small Bags', 'Large Bags', 'XLarge Bags']
    valid_dataset = dataset[valid_cols]
    valid_dataset['Date'] = [datetime.strptime(date,"%Y-%m-%d %H:%M:%S").timetuple().tm_yday for date in dataset['Date']]
    valid_dataset['isOrganic'] = (np.array(dataset['type']) == 'Organic').astype(np.float64)*2-1
    
    cleaned_dataset = standardize_data(valid_dataset,list(valid_dataset.columns)[1:])
    
    useable_cols = list(cleaned_dataset.columns)
    out_cols = [useable_cols.pop(0)]
    inp_cols = useable_cols

    avocado_model = torch.load(pretrained_path)
    test_set = Dataset(cleaned_dataset, inp_cols, out_cols, batch_size=16)

    # evaluate the model
    preds = default_test(avocado_model, test_set, pred_y_func=compute_residual_error, print_output_head=0)

    # show off the model
    test_data_raw = valid_dataset
    test_data_raw['PredAvgPrice'] = preds.round(2)
    test_data_raw['PriceError'] = test_data_raw['PredAvgPrice'] - test_data_raw['AveragePrice']
    test_data_raw = test_data_raw[ list(test_data_raw.columns)[-2:]+list(test_data_raw.columns)[:-2] ]
    test_data_raw = test_data_raw[test_data_raw['PredAvgPrice'].notnull()]
    test_data_raw = test_data_raw.sort_values('AveragePrice')
    print(test_data_raw.head())
    fig = plt.figure(1)
    ax1 = plt.subplot(111)
    ax1.scatter(range(len(test_data_raw)), test_data_raw['PredAvgPrice'])
    ax1.scatter(range(len(test_data_raw)), test_data_raw['AveragePrice'])
    plt.title('Predicted VS Actual Average Price')
    plt.show()
    plt.close()



if __name__ == '__main__':
    # Network Training 
    # set show_pretrained to True to show the best NN's performance on all training data
    main(pretrained_path=None)
    
    # Network Testing on Recent Hass Avocado Board Data
    large_test(pretrained_path="test.pth")