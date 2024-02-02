def split_data(data, split_ratio = 0.8):
    
    '''
	ToDo: Split the dataset into train and test data using the split_ratio.
	Input:
		data: dataframe containing the dataset. 
		split_ratio: desired ratio of the train and test splits.
		
	Output:
		train: train split of the data
		test: test split of the data
	'''
    split_n = int(data.shape[0]*split_ratio)
    train = data.iloc[0:split_n]
    test = data.iloc[split_n:]

    return train, test
