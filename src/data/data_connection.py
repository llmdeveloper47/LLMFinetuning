import datasets
from datasets import load_dataset
from datasets import DatasetDict
from get_logging import logger_object
logger = logger_object()

class DataConnector:
 
    @staticmethod
    def get_data(path):
        # TODO: Implement the DataConnector.get_data method 
        # of the data_connection.py file. Use the load_dataset 
        # function from the datasets package and select split = "train" 
        # to make sure we only train on the training data.

        assert  path == 'ybisk/piqa'
        dataset = load_dataset(path, trust_remote_code=True)
        assert type(dataset) == datasets.dataset_dict.DatasetDict
        logger.debug(f'loaded dataset : {path}')
        return dataset
