from data.data_connection import DataConnector
from data.data_processing import DataProcessor
from model.model_connection import Model
from training.train_supervised import SupervisedTrainer
from get_logging import logger_object
import wandb
logger = logger_object()


def run(training_type):

    # get model artifacts
    logger.debug('loading model tokenizer artifacts')
    model_id = 'openai-community/gpt2'
    model_artifacts_object = Model()
    model, tokenizer = model_artifacts_object.get_model_for_LM(model_id)
    
    # initialize trainer
    logger.debug('initializing SupervisedTrainer')
    model_trainer = SupervisedTrainer(model, tokenizer)
    
    # get raw dataset
    logger.debug('loading raw dataset')
    path = 'ybisk/piqa'
    data_object = DataConnector()
    dataset = data_object.get_data(path)
    
    # get processed dataset
    logger.debug('processing raw dataset')
    processed_data_object = DataProcessor(tokenizer)
    processed_dataset = processed_data_object.prepare_for_supervised_training(dataset)
    
    # start supervised training
    if training_type == 'supervised':
        logger.debug('starting training...')
        model_trainer.train(processed_dataset)
        
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
        
    trainer.train(processed_data)


if __name__ == '__main__':
    wandb.init(project="huggingface", name="gpt-model-training-run")
    run('supervised')
    wandb.finish()
