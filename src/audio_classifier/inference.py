import argparse
import os
import torch
import librosa
import numpy as np
import joblib
from transformers import AutoModel, AutoFeatureExtractor
from collections import Counter

MODEL_NAME = "m-a-p/MERT-v1-95M"
SAMPLE_RATE = 24000
CHUNK_DURATION = 5.0
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
CLASSES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def load_and_chunk_audio(path):


    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    num_samples = len(audio)
    chunks = []

    if num_samples < CHUNK_SAMPLES:
        padding = CHUNK_SAMPLES - num_samples
        chunk = np.pad(audio, (0, padding), mode='constant')
        chunks.append(chunk)
    else:
        for start in range(0, num_samples - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
            chunk = audio[start : start + CHUNK_SAMPLES]
            chunks.append(chunk)
    
    return chunks

def main():

    parser = argparse.ArgumentParser(description="Audio Genre Inference")
    parser.add_argument("--path", type=str, required=True, help="Path to the audio file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    chunks = load_and_chunk_audio(args.path)
    
    print(f"Generated {len(chunks)} chunks of {CHUNK_DURATION}s.")

    processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for chunk in chunks:
            inputs = processor(
                chunk, 
                sampling_rate=SAMPLE_RATE, 
                return_tensors="pt", 
                padding=True
            ).to(device)

            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(embedding)

    X = np.vstack(embeddings)

    model_path = os.path.join("models", "best_svm_model.pkl")
    
    clf = joblib.load(model_path)
    
    predictions = clf.predict(X)
    probs = clf.predict_proba(X)

    vote_counts = Counter(predictions)
    most_common_idx, count = vote_counts.most_common(1)[0]
    
    if isinstance(most_common_idx, (int, np.integer)):
        most_common_genre = CLASSES[most_common_idx]
    else:
        most_common_genre = str(most_common_idx)


    avg_probs = np.mean(probs, axis=0)
    
    if hasattr(clf, "classes_"):
        class_labels = clf.classes_
        if np.issubdtype(class_labels.dtype, np.integer):
             class_names = [CLASSES[i] for i in class_labels]
        else:
             class_names = class_labels
    else:
        class_names = CLASSES

    print("\n" + "="*30)
    print(f"PREDICTED GENRE: {most_common_genre.upper()}")
    print("="*30)
    print("Class Probabilities (Average across chunks):")
    
    prob_dict = {cls: p for cls, p in zip(class_names, avg_probs)}
    for genre in CLASSES:
            p = prob_dict.get(genre, 0.0)
            print(f"  {genre:<10}: {p:.4f}")

if __name__ == "__main__":
    main()