from model.model_connection import Model
from training.train_supervised import SupervisedTrainer
from data.data_connection import DataConnector
from data.data_processing import DataProcessor
from get_logging import logger_object
logger = logger_object()
import wandb
def run(training_type):
    

    # TODO: implement this function.
    model_class = Model()
    model_id = 'openai-community/gpt2'
    model, tokenizer = model_class.get_model_for_LM(model_id=model_id)
    logger.debug(f'loaded model and tokenizer for model_id : {model_id}')
    supervise_trainer = SupervisedTrainer(model, tokenizer)
    logger.debug(f'initialized SupervisedTrainer')

    dataset_id = 'ybisk/piqa'
    dataset_connector = DataConnector()
    dataset = dataset_connector.get_data(path = dataset_id)
    logger.debug(f'loaded dataset')

    dataset_processor = DataProcessor(tokenizer=tokenizer)
    logger.debug(f'initialized DataProcessor')
    processed_data = dataset_processor.prepare_for_supervised_training(dataset=dataset)
    logger.debug(f'proeprocessed dataset')

    if training_type == 'supervised':
        logger.debug(f'training started')
        supervise_trainer.train(processed_data)
        logger.debug(f'training completed')

    elif training_type == 'reward':
        raise NotImplementedError

    elif training_type == 'ppo':
        raise NotImplementedError

    elif training_type == 'dpo':
        raise NotImplementedError

    elif training_type == 'orpo':
        raise NotImplementedError

    else: 
        raise NotImplemented
        
if __name__ == '__main__':
    wandb.init(project="huggingface", name="gpt-model-training-run")
    run('supervised')
    wandb.finish()