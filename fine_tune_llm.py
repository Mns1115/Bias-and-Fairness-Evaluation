from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments
import datasets

# Load data
data = datasets.load_dataset("csv", data_files={"train": "../data/processed/stereoset_cleaned.csv"})

# Load model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define training arguments
training_args = TrainingArguments(output_dir="../models/fine_tuned/fine_tuned_bert", num_train_epochs=3)

# Initialize Trainer
trainer = Trainer(model=model, args=training_args, train_dataset=data["train"])

# Train model
trainer.train()

# Save fine-tuned model
model.save_pretrained("../models/fine_tuned/fine_tuned_bert/")
tokenizer.save_pretrained("../models/fine_tuned/fine_tuned_bert/")
