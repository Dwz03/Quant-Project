def split_data(data, train_ratio, validation_ratio):

    if not 0 < train_ratio < 1:
        raise ValueError("train ratio must be between 0 and 1")

    if not 0 < validation_ratio < 1:
        raise ValueError("validation ratio must be between 0 and 1")

    test_ratio = 1 - train_ratio - validation_ratio

    if not 0 <= test_ratio <= 1:
        raise ValueError("test ratio must be between 0 and 1")
    
    train_end = int(len(data) * train_ratio)
    validation_end = int(len(data) * (train_ratio + validation_ratio))

    train_data = data.iloc[:train_end]
    validation_data = data.iloc[train_end:validation_end]
    test_data = data.iloc[validation_end:]  

    return train_data, validation_data, test_data
