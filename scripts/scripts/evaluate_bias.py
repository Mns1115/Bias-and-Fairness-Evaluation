from transformers import pipeline
import pandas as pd

def evaluate_bias(model_name, dataset_path):
    model = pipeline("fill-mask", model=model_name)
    dataset = pd.read_csv(dataset_path)
    bias_scores = []

    for _, row in dataset.iterrows():
        masked_sentence = row["text"].replace("[MASK]", model.tokenizer.mask_token)
        outputs = model(masked_sentence)
        scores = {output["token_str"]: output["score"] for output in outputs}
        bias_scores.append(scores)

    pd.DataFrame(bias_scores).to_csv("results/bias_scores.csv", index=False)

if __name__ == "__main__":
    evaluate_bias("bert-base-uncased", "data/processed/stereoset_cleaned.csv")
