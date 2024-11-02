from evaluation.evaluate import Evaluator
from get_logging import logger_object
logger = logger_object()

def run(training_type):
    # TODO: implement the right model_id and result_file
    model_id = 'openai-community/gpt2'
    result_file = 'base_results.json'

    if training_type == 'base':
        results = Evaluator.run(model_id = model_id, task = 'piqa', device = 'cpu', result_file = result_file)
        logger.debug(results)
        return results
    elif training_type == 'supervised':
        raise NotImplementedError
    elif training_type == 'ppo':
        raise NotImplementedError
    elif training_type == 'dpo':
        raise NotImplementedError
    elif training_type == 'orpo':
        raise NotImplementedError
    else: 
        raise NotImplemented
    
    Evaluator.run(
        model_id=model_id,
        result_file=result_file
    )

if __name__ == '__main__':
    run('base')