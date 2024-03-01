import os
import datasets
import pandas as pd

from transformers.integrations import WandbCallback

def decode_predictions(tokenizer, predictions, samples):
    labels = [[l for l in label if l != -100] for label in predictions.label_ids]
    labels = tokenizer.batch_decode(labels,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
    inputs = tokenizer.batch_decode(samples['input_ids'],
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
    logits = predictions.predictions.argmax(axis=-1)
    prediction_text = tokenizer.batch_decode(logits, 
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
    return {"inputs": inputs, "labels": labels, "predictions": prediction_text}

class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each 
    logging step during training. It allows to visualize the 
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset 
          for generating predictions.
        num_samples (int, optional): Number of samples to select from 
          the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, trainer, tokenizer, val_dataset,
                 num_samples=100, freq=2):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated 
              with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from 
              the validation dataset for generating predictions.
              Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # control the frequency of logging by logging the predictions
        # every `freq` epochs
        if int(state.epoch) % self.freq == 0:
            # generate predictions
            predictions = self.trainer.predict(self.sample_dataset)
            # decode predictions and labels
            predictions = decode_predictions(self.tokenizer, predictions, self.sample_dataset)
            # add predictions to a wandb.Table
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            
            if args.local_rank == 0:
                # log the table to wandb
                self._wandb.log({"sample_predictions": records_table})

    # def on_train_end(self, args, state, control, **kwargs):
    #   super().on_train_end(args, state, control, **kwargs)
    #   if args.local_rank == 0:
    #     log_path = os.path.join(args.output_dir, 'exp.log')
        
    #     log_reader = self._wandb.Artifact("log", type="log")
    #     log_reader.add_file(log_path)
    #     self._wandb.run.log_artifact(log_reader)
