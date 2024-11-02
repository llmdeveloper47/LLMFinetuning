from training.base_train import BaseTrainer, login, HUGGINGFACE_TOKEN
  


class RewardModelTrainer(BaseTrainer):

    def __init__(
            self, 
            model, 
            tokenizer, 
            num_epoch=3, 
            batch_size=4, 
            output_dir='HW2-reward',
        ):
        super().__init__(model, tokenizer, num_epoch, batch_size, output_dir)

        # TODO: set up the training arguments
        self.args = None

    def train(self, tokenized_data):
        # TODO: Use the RewardTrainer to set up the training. 
        # Call the train method of the RewardTrainer class, 
        # and don't forget to push the model to the model hub.

        raise NotImplementedError
    

class RLHFTrainer(BaseTrainer):

    def __init__(
            self, 
            model, 
            tokenizer, 
            num_epoch=3, 
            batch_size=8, 
            output_dir='HW2-ppo',
            result_file='ppo_results.json'
        ):
        super().__init__(model, tokenizer, num_epoch, batch_size, output_dir, result_file)
        
        # TODO: implement the training arguments with the PPOConfig. 
        self.args = None

    def _get_collator(self, data): 
        return dict((key, [d[key] for d in data]) for key in data[0])

    def train(self, tokenized_data):
        # TODO: implement the trainer with the PPOTrainer
        trainer = None
        # TODO: implement the generation_kwargs that will be used in the PPOTrainer.generate method 
        generation_kwargs = None
        # TODO: Implement the reward_pipeline by using your reward model and pipeline function
        reward_pipeline = None

        for epoch in range(self.num_epoch):
            for batch in trainer.dataloader: 
                query_tensors = batch["input_ids"]    
                
                #### Get response from SFTModel
                # TODO: Generate the response_tensors from the query_tensors
                response_tensors = None
                # TODO: Decode the response_tensors by using the tokenizer
                batch["response"] = None
            
                #### Compute reward score
                # TODO: Create the input text for the reward_pipeline by using batch["goal"] and batch["response"]
                texts = None
                # TODO: Pass the input text to the reward_pipeline and extract the score output.
                rewards = None
            
                #### Run PPO step 

                # TODO: Update the PPO model by using the query_tensors, response_tensors,  and rewards.

    def save(self, trainer):
        login(token=HUGGINGFACE_TOKEN)
        trainer.push_to_hub(self.output_dir)