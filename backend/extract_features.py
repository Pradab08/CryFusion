import librosa
import numpy as np
import os
from pathlib import Path
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MFCCExtractor:
    """
    Proper MFCC feature extraction for CNN+LSTM architecture
    This extracts (time_steps, 13) MFCC features instead of spectrograms
    """

    def __init__(self, config: dict):
        self.config = config
        self.sample_rate = config.get('sample_rate', 16000)
        self.n_mfcc = config.get('n_mfcc', 13)
        self.n_fft = config.get('n_fft', 2048)
        self.hop_length = config.get('hop_length', 512)
        self.max_audio_length = config.get('max_audio_length', 5.0)  # seconds
        self.target_length = int(self.max_audio_length * self.sample_rate / self.hop_length)

    def load_audio(self, file_path: str):
        """
        Load and preprocess audio file
        """
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)

            # Remove silence
            audio, _ = librosa.effects.trim(audio, top_db=20)

            # Normalize
            if len(audio) > 0:
                audio = librosa.util.normalize(audio)

            return audio

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def extract_mfcc(self, audio: np.ndarray):
        """
        Extract MFCC features with proper temporal structure
        """
        try:
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=True
            )

            # Transpose to (time_steps, n_mfcc) format
            mfcc = mfcc.T

            # Standardize length
            if len(mfcc) > self.target_length:
                # Truncate
                mfcc = mfcc[:self.target_length]
            elif len(mfcc) < self.target_length:
                # Pad with zeros
                padding = self.target_length - len(mfcc)
                mfcc = np.pad(mfcc, ((0, padding), (0, 0)), mode='constant')

            return mfcc

        except Exception as e:
            print(f"Error extracting MFCC: {e}")
            return None

    def extract_from_directory(self, data_dir: str):
        """
        Extract MFCC features from organized directory structure
        """
        print(f"🎵 EXTRACTING MFCC FEATURES FROM: {data_dir}")

        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        features = []
        labels = []
        file_info = []

        # Supported audio formats
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.aac']

        # Process each class directory
        class_dirs = [d for d in data_path.iterdir() if d.is_dir()]

        if not class_dirs:
            raise ValueError(f"No class directories found in {data_dir}")

        print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

        total_files = 0
        for class_dir in class_dirs:
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(class_dir.glob(f"*{ext}"))
            total_files += len(audio_files)

        print(f"Total audio files to process: {total_files}")

        with tqdm(total=total_files, desc="Extracting MFCC") as pbar:
            for class_dir in sorted(class_dirs):
                class_name = class_dir.name
                print(f"\nProcessing class: {class_name}")

                # Find all audio files in this class
                audio_files = []
                for ext in audio_extensions:
                    audio_files.extend(class_dir.glob(f"*{ext}"))

                class_count = 0
                for audio_file in sorted(audio_files):
                    # Load audio
                    audio = self.load_audio(str(audio_file))
                    if audio is None:
                        pbar.update(1)
                        continue

                    # Extract MFCC
                    mfcc = self.extract_mfcc(audio)
                    if mfcc is None:
                        pbar.update(1)
                        continue

                    # Store results
                    features.append(mfcc)
                    labels.append(class_name)
                    file_info.append({
                        'file_path': str(audio_file),
                        'class': class_name,
                        'shape': mfcc.shape,
                        'audio_length': len(audio) / self.sample_rate
                    })

                    class_count += 1
                    pbar.update(1)

                print(f"  Processed {class_count} files for class '{class_name}'")

        if not features:
            raise ValueError("No features extracted! Check audio files and paths.")

        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels)

        print(f"\n✅ MFCC EXTRACTION COMPLETE!")
        print(f"Features shape: {features.shape}")
        print(f"Expected shape: (samples, {self.target_length}, {self.n_mfcc})")
        print(f"Labels shape: {labels.shape}")

        # Show class distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\nClass distribution:")
        for label, count in zip(unique, counts):
            print(f"  {label:12s}: {count:3d} samples")

        return features, labels, file_info

    def save_features(self, features: np.ndarray, labels: np.ndarray, 
                      file_info: list, output_path: str):
        """
        Save extracted features to file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as compressed numpy file
        np.savez_compressed(
            output_path,
            features=features,
            labels=labels,
            file_info=file_info,
            config=self.config
        )

        print(f"💾 Features saved to: {output_path}")

        # Also save extraction info as JSON
        info_path = output_path.with_suffix('.json')
        extraction_info = {
            'feature_shape': features.shape,
            'label_count': len(labels),
            'config': self.config,
            'classes': list(np.unique(labels)),
            'class_distribution': {
                label: int(count) for label, count in 
                zip(*np.unique(labels, return_counts=True))
            }
        }

        with open(info_path, 'w') as f:
            json.dump(extraction_info, f, indent=2)

        print(f"📋 Extraction info saved to: {info_path}")


def main():
    """
    Extract MFCC features for CNN+LSTM architecture
    """
    print("🎵 MFCC FEATURE EXTRACTION FOR CNN+LSTM")
    print("="*50)

    # Configuration for MFCC extraction
    config = {
        'sample_rate': 16000,
        'n_mfcc': 13,              # Standard number of MFCC coefficients
        'n_fft': 2048,
        'hop_length': 512,
        'max_audio_length': 5.0,   # Standardize to 5 seconds
    }

    # Initialize extractor
    extractor = MFCCExtractor(config)

    # Define paths
    data_dir = r"E:\Capstone Project\Capstone Project - CryFusion\data\Baby Cry Sence Dataset"  # Your audio files organized by class
    output_path = r"features/mfcc_features.npz"

    # Check if data directory exists
    if not Path(data_dir).exists():
        print(f"❌ Data directory not found: {data_dir}")
        print(f"\nExpected structure:")
        print(f"{data_dir}/")
        print(f"├── hungry/")
        print(f"│   ├── hungry_001.wav")
        print(f"│   └── hungry_002.wav")
        print(f"├── tired/")
        print(f"│   ├── tired_001.wav")
        print(f"│   └── tired_002.wav")
        print(f"└── ... (other classes)")
        return

    try:
        # Extract features
        features, labels, file_info = extractor.extract_from_directory(data_dir)

        # Save features
        extractor.save_features(features, labels, file_info, output_path)

        print(f"\n🎯 READY FOR CNN+LSTM TRAINING!")
        print(f"Use these MFCC features with the original CNN+LSTM model.")
        print(f"Features shape: {features.shape} (samples, time_steps, mfcc_coeffs)")

        # Verify the shape is correct for CNN+LSTM
        expected_time_steps = int(5.0 * 16000 / 512)  # ~156 time steps
        if features.shape[1] == expected_time_steps and features.shape[2] == 13:
            print(f"✅ Perfect! Shape {features.shape} is ideal for CNN+LSTM")
        else:
            print(f"⚠️  Unusual shape. Expected (~{expected_time_steps}, 13)")

    except Exception as e:
        print(f"\n❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()