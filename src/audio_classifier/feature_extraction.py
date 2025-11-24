from transformers import AutoModel, AutoFeatureExtractor
from torch.utils.data import DataLoader
from .dataset import GTZANProcessor
from tqdm import tqdm
import argparse
import torch
import os


def extract_features(model_name = "m-a-p/MERT-v1-95M", inference_batch_size = 4, chunk_duration=3.0, overlap=0.5, sample_rate=24000):
    
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Loading model: {model_name}...")

    processor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True) 
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    handler = GTZANProcessor(chunk_duration=chunk_duration, overlap=overlap, sample_rate=sample_rate)
    splits = handler.get_gtzan_splits()
    chunked_dataset = handler.process_dataset(splits)

    for split_name in chunked_dataset.keys():

        dataset = chunked_dataset[split_name]
        loader = DataLoader(dataset, batch_size=inference_batch_size, shuffle=False, collate_fn=lambda x: x)
        
        features_list = []  
        labels_list = []

        progress_bar = tqdm(loader, desc=f"Processing {split_name}", unit="batch", colour='green')

        with torch.no_grad():
            for batch in progress_bar:
                
                raw_audio = [item["audio"] for item in batch]
                labels = torch.tensor([item["label"] for item in batch]).to(device)
                

                inputs = processor(
                    raw_audio, 
                    sampling_rate=sample_rate, 
                    return_tensors="pt", 
                    padding=True
                ).to(device)

                outputs = model(**inputs)
                
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                features_list.append(embeddings.cpu())
                labels_list.append(labels.cpu())

        if features_list:
            X = torch.cat(features_list)
            y = torch.cat(labels_list)
            save_path = os.path.join(output_dir, f"{split_name}_data.pt")
            torch.save((X, y), save_path)


def main():
    
    parser = argparse.ArgumentParser(description="Feature Extraction Script")
    
    parser.add_argument("--model_name", type=str, default="m-a-p/MERT-v1-95M", help="HuggingFace model name")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--chunk_duration", type=float, default=5.0, help="Duration of audio chunks in seconds")
    parser.add_argument("--overlap", type=float, default=0.3, help="Overlap ratio (0.0 to 1.0)")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Target sample rate")

    args = parser.parse_args()

    extract_features(
        model_name=args.model_name,
        inference_batch_size=args.batch_size,
        chunk_duration=args.chunk_duration,
        overlap=args.overlap,
        sample_rate=args.sample_rate
    )

if __name__ == "__main__":
    main()