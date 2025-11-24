from datasets import load_dataset, Audio, DatasetDict
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class GTZANProcessor:

    def __init__(self, 
                 dataset_name="marsyas/gtzan", 
                 sample_rate=16000, 
                 chunk_duration=3.0,
                 overlap=0.5):
        
        self.dataset_name = dataset_name
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.stride_samples = int(self.chunk_samples * (1 - overlap))

    def get_gtzan_splits(self):

        dataset = load_dataset(self.dataset_name, "all", split="train", trust_remote_code=True)
        
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sample_rate))

        labels = dataset["genre"]
        indices = np.arange(len(dataset))

        train_idx, temp_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=labels
        )

        temp_labels = [labels[i] for i in temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42, stratify=temp_labels
        )

        splits = DatasetDict({
            "train": dataset.select(train_idx),
            "validation": dataset.select(val_idx),
            "test": dataset.select(test_idx)
        })

        return splits

    def _chunk_function(self, batch):

        new_audio = []
        new_labels = []

        for i in range(len(batch["audio"])):
            
            audio_array = batch["audio"][i]["array"]
            label = batch["genre"][i]
            
            num_samples = len(audio_array)
            
            if num_samples < self.chunk_samples:

                padding = self.chunk_samples - num_samples
                chunk = np.pad(audio_array, (0, padding), mode='constant')
                new_audio.append(chunk)
                new_labels.append(label)
            
            else:
                for start in range(0, num_samples - self.chunk_samples + 1, self.stride_samples):
                    chunk = audio_array[start : start + self.chunk_samples]
                    new_audio.append(chunk)
                    new_labels.append(label)

        return {"audio": new_audio, "label": new_labels}

    def process_dataset(self, raw_splits):
        
        cols_to_remove = raw_splits["train"].column_names
        if "label" in cols_to_remove: cols_to_remove.remove("label")
        
        processed_splits = raw_splits.map(
            self._chunk_function,
            batched=True,
            batch_size=4,
            remove_columns=["file", "audio", "genre"],
            num_proc=1
        )
        
        print("Processing complete!")
        for split in processed_splits:
            print(f"  {split}: {len(processed_splits[split])} chunks")
            
        return processed_splits