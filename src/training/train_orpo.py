from training.base_train import BaseTrainer


class MyORPOTrainer(BaseTrainer):

    def __init__(
            self, 
            model, 
            tokenizer, 
            num_epoch=3, 
            batch_size=4, 
            output_dir='HW2-dpo',
            result_file='dpo_results.json'
        ):
        super().__init__(model, tokenizer, num_epoch, batch_size, output_dir, result_file)
        
        # TODO: Set the training arguments up with the ORPOConfig
        self.args = None

    def train(self, dataset):
        # TODO:  Set the training up with the ORPOTrainer.  
        # Call the train method of the ORPOTrainer class, 
        # and don't forget to push the model to the model hub
        trainer = None