from training.base_train import BaseTrainer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from get_logging import logger_object
logger = logger_object()

class SupervisedTrainer(BaseTrainer):

    def __init__(
            self, 
            model, 
            tokenizer, 
            num_epoch=3, 
            batch_size=16, 
            output_dir='HW2-supervised',
            result_file='supervised_results.json'
        ):
        super().__init__(model, tokenizer, num_epoch, batch_size, output_dir, result_file)
        
        # TODO: set up the training arguments
        self.args = SFTConfig(
            do_train=True,
            do_eval=True,
            num_train_epochs=num_epoch,                   # Set number of training epochs
            per_device_train_batch_size=batch_size,        # Set training batch size per device
            per_device_eval_batch_size=batch_size,         # Set evaluation batch size per device
            logging_steps=10,                     # Log every 10 steps
            save_steps=10,                       # Save checkpoint every 500 steps
            eval_strategy="epoch",          # Evaluate at the end of every epoch
            save_strategy="epoch",
            learning_rate=2e-5,                    # Fine-tuning learning rate
            output_dir="/tmp",
            load_best_model_at_end = True,
        )
        # TODO: set up the data collator to prepare the data for training. 
        # I suggest using the DataCollatorForCompletionOnlyLM data collator
        response_template = " ### Answer:"
        self.collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

        

    def train(self, dataset):
        # TODO: Use the SFTTrainer to set up the training. 
        # Call the train method of the SFTTrainer class, 
        # and don't forget to push the model to the model hub.

        def formatting_prompts_func(example):
            output_texts = []
            for i in range(len(example['goal'])):
                text = f"### Question: {example['goal'][i]}\n ### Answer: {example['labels'][i]}"
                output_texts.append(text)
            return output_texts

        trainer = SFTTrainer(
            model = self.model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            args=self.args,
            formatting_func = formatting_prompts_func,
            data_collator=self.collator,
        )
        
        trainer.train()
        trainer.push_to_hub()


