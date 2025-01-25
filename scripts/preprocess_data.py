import pandas as pd

def preprocess_data(input_path, output_path):
    data = pd.read_csv(input_path)
    data['text'] = data['text'].str.lower()
    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data("data/raw/stereoset.csv", "data/processed/stereoset_cleaned.csv")
