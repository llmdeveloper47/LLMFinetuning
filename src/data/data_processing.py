
import datasets
class DataProcessor:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def prepare_for_supervised_training(self, dataset):
        # TODO: implement the method. 
        # The method should return one column with the text organized as such:
        # "Question: {goal}\nAnswer: {chosen_solution}"
        # This is because it is the way the instructions get presented 
        # to the models in the PIQA test of the Evaluation Harness. 
        # The goal represents the text of the original ''goal" column,
        #  and the chosen_solution is the solution that corresponds to the "label" column. 
        # The returned dataset doesn't need to be tokenized, 
        # but it should be in a DatasetDict format. 
        # You could decide to return a validation dataset as well 
        # to get better visibility during your training.

        assert type(dataset) == datasets.dataset_dict.DatasetDict

        def format_qa_column(dataset):
            def create_qa_text(row):
                chosen_solution = row['sol1'] if row['label'] == 0 else row['sol2']
                return f"Question: {row['goal']}\nAnswer: {chosen_solution}"
            dataset = dataset.map(lambda row: {"labels": create_qa_text(row)})
            return dataset

        def preprocess_function(example):
            inputs = self.tokenizer(example["labels"], padding="max_length", truncation=True, return_tensors="pt")
            inputs["labels"] = inputs["input_ids"].clone()
            inputs["labels"][inputs["labels"] == self.tokenizer.pad_token_id] = -100
            return inputs

        processed_dataset = format_qa_column(dataset)
        processed_dataset = processed_dataset.map(preprocess_function, batched=True)
        processed_dataset = processed_dataset.remove_columns(["goal", "sol1", "sol2", "label"])


    def _tokenize_for_reward_training(self, examples):
        # TODO: The RewardTrainer expects the data 
        # to be tokenized with very specific column names:
        # - "input_ids_chosen"
        # - "attention_mask_chosen"
        # - "input_ids_rejected"
        # - "attention_mask_rejected"
        # Implement the method to tokenize the data using the expected format for the RewardTrainer.
        new_examples = None
        return new_examples
    
    def prepare_for_reward_training(self, dataset):
        # TODO: Prepare the data: Now that we have a function that 
        # tokenizes the data, we need to prepare that data. 
        # The prepare_for_reward_training method should do the following things:
        # - create two new columns, "chosen" and "rejected," from the original data.
        # - tokenize those columns and return the tokenized data
        # The "chosen" column should have the following format:
        # "Question: {goal}\nAnswer: {chosen_solution}"
        # and the "rejected" column should have the following format:
        # "Question: {goal}\nAnswer: {rejected_solution}"
        tokenized_data = None
        return tokenized_data
    
    def prepare_for_ppo_training(self, dataset):
        # TODO:  Implement the method. 
        # We just need to add the indicators "Question: {goal}\nAnswer: " 
        # and tokenize the resulting text.
        tokenized_data = None
        return tokenized_data
    
    def prepare_for_dpo_training(self, dataset):
        # TODO: implement the metho. The HuggingFace DPOTrainer 
        # expects a very specific format of the input data. 
        # We don't need to input data to be tokenized, 
        # but we need three columns with those names:
        # - 'prompt': in our case, the 'goal' column.
        # - 'chosen': the correct response
        # - 'rejected': the wrong response
        processed_dataset = None
        return processed_dataset
    
    def prepare_for_orpo_training(self, dataset):
        # TODO: implement the metho. The HuggingFace ORPOTrainer 
        # expects a very specific format of the input data. 
        # We don't need to input data to be tokenized, 
        # but we need three columns with those names:
        # - 'prompt': in our case, the 'goal' column.
        # - 'chosen': the correct response
        # - 'rejected': the wrong response
        processed_dataset = None
        return processed_dataset