"""
Model inference module for extracting predictions, embeddings, and attention weights
"""

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time

from .utils import AnalysisLogger, DataManager


class ProteinDataset(Dataset):
    """Dataset for protein sequences"""

    def __init__(self, sequences: List[str], tokenizer, max_length: int = 512):
        """
        Initialize dataset

        Args:
            sequences: List of protein sequences
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = str(self.sequences[idx])

        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


class ProteinAnalyzer:
    """Main class for model inference and feature extraction"""

    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        logger: Optional[AnalysisLogger] = None,
        use_multi_gpu: bool = True
    ):
        """
        Initialize analyzer with model

        Args:
            model_path: Path to saved model directory
            device: Device to use ('auto', 'cuda', 'cpu')
            logger: Logger instance
            use_multi_gpu: Whether to use DataParallel for multi-GPU
        """
        self.model_path = model_path
        self.logger = logger if logger else AnalysisLogger()
        self.device = self._setup_device(device)
        self.use_multi_gpu = use_multi_gpu

        # Load model and tokenizer
        self.logger.info(f"Loading model from: {model_path}")
        self.tokenizer, self.model = self._load_model()
        self.logger.info(f"Model loaded successfully on {self.device}")

    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Log GPU information
        if device == 'cuda' and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            self.logger.info(f"Found {num_gpus} GPU(s) available")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                self.logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        return device

    def _load_model(self) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
        """Load model and tokenizer"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                output_hidden_states=True,  # Enable hidden states extraction
                output_attentions=True      # Enable attention extraction
            )

            # Move to device
            model.to(self.device)

            # Use DataParallel for multi-GPU if available and enabled
            if self.use_multi_gpu and self.device == 'cuda' and torch.cuda.device_count() > 1:
                num_gpus = torch.cuda.device_count()
                self.logger.info(f"Using DataParallel on {num_gpus} GPUs")
                model = torch.nn.DataParallel(model)
                self.multi_gpu = True
            else:
                self.multi_gpu = False
                if self.device == 'cuda' and torch.cuda.device_count() > 1 and not self.use_multi_gpu:
                    self.logger.info(f"Multi-GPU disabled by configuration (found {torch.cuda.device_count()} GPUs)")

            model.eval()
            return tokenizer, model

        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def analyze_proteins(
        self,
        sequences: List[str],
        labels: List[str],
        batch_size: int = 16,
        extract_embeddings: bool = True,
        extract_attentions: bool = True
    ) -> Dict[str, any]:
        """
        Analyze proteins: predictions, embeddings, and attentions

        Args:
            sequences: List of protein sequences
            labels: List of protein labels/IDs
            batch_size: Batch size for inference
            extract_embeddings: Whether to extract embeddings
            extract_attentions: Whether to extract attention weights

        Returns:
            Dictionary with predictions, embeddings, and attentions
        """
        # Adjust batch size for multi-GPU
        effective_batch_size = batch_size
        if self.multi_gpu:
            num_gpus = torch.cuda.device_count()
            effective_batch_size = batch_size * num_gpus
            self.logger.info(f"Multi-GPU: increasing batch size from {batch_size} to {effective_batch_size} ({num_gpus} GPUs)")

        self.logger.info(f"Analyzing {len(sequences)} protein sequences...")
        self.logger.info(f"Batch size: {effective_batch_size}")
        start_time = time.time()

        # Create dataset and dataloader
        dataset = ProteinDataset(sequences, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=False)

        # Storage for results
        all_predictions = []
        all_probabilities = []
        all_embeddings = [] if extract_embeddings else None
        all_attentions = [] if extract_attentions else None
        all_attention_masks = [] if extract_attentions else None

        # Process batches
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing batches"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=extract_embeddings,
                    output_attentions=extract_attentions
                )

                # Extract predictions
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                # Extract embeddings from last hidden state
                if extract_embeddings and outputs.hidden_states is not None:
                    # Get last hidden state: (batch_size, seq_len, hidden_dim)
                    last_hidden_state = outputs.hidden_states[-1]

                    # Mean pooling over sequence length (considering attention mask)
                    # Shape: (batch_size, hidden_dim)
                    embeddings = self._mean_pooling(last_hidden_state, attention_mask)
                    all_embeddings.extend(embeddings.cpu().numpy())

                # Extract attention weights
                if extract_attentions and outputs.attentions is not None:
                    # outputs.attentions is a tuple of (num_layers,)
                    # Each element: (batch_size, num_heads, seq_len, seq_len)
                    # We'll take the last layer's attention
                    last_layer_attention = outputs.attentions[-1]  # (batch, heads, seq, seq)

                    all_attentions.extend(last_layer_attention.cpu().numpy())
                    all_attention_masks.extend(attention_mask.cpu().numpy())

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)

        elapsed_time = time.time() - start_time
        self.logger.info(f"Analysis completed in {elapsed_time:.2f}s")

        # Prepare results dictionary
        results = {
            'labels': np.array(labels),
            'sequences': np.array(sequences),
            'predictions': predictions,
            'probabilities': probabilities,
        }

        if extract_embeddings and all_embeddings:
            results['embeddings'] = np.array(all_embeddings)
            self.logger.info(f"Extracted embeddings: {results['embeddings'].shape}")

        if extract_attentions and all_attentions:
            results['attentions'] = np.array(all_attentions)
            results['attention_masks'] = np.array(all_attention_masks)
            self.logger.info(f"Extracted attentions: {results['attentions'].shape}")

        # Log statistics
        self._log_statistics(predictions, probabilities)

        return results

    def _mean_pooling(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling over sequence length, considering attention mask

        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)
            attention_mask: (batch_size, seq_len)

        Returns:
            Pooled embeddings: (batch_size, hidden_dim)
        """
        # Expand attention mask to match hidden_states dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

        # Sum over sequence length
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)

        # Count valid tokens
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        # Average
        return sum_embeddings / sum_mask

    def _log_statistics(self, predictions: np.ndarray, probabilities: np.ndarray):
        """Log prediction statistics"""
        total = len(predictions)
        num_effectors = np.sum(predictions == 1)
        num_non_effectors = np.sum(predictions == 0)
        mean_confidence = np.mean(np.max(probabilities, axis=1))

        self.logger.info(f"\n=== Prediction Statistics ===")
        self.logger.info(f"Total proteins: {total}")
        self.logger.info(f"Predicted as Effector: {num_effectors} ({100*num_effectors/total:.1f}%)")
        self.logger.info(f"Predicted as Non-effector: {num_non_effectors} ({100*num_non_effectors/total:.1f}%)")
        self.logger.info(f"Mean confidence: {mean_confidence:.3f}")

    def create_predictions_dataframe(
        self,
        results: Dict[str, np.ndarray],
        additional_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Create a comprehensive predictions DataFrame

        Args:
            results: Results from analyze_proteins
            additional_data: Additional metadata to merge

        Returns:
            DataFrame with all prediction results
        """
        class_names = {0: "Non_effector", 1: "Effector"}

        df = pd.DataFrame({
            'label': results['labels'],
            'sequence': results['sequences'],
            'prediction': [class_names[pred] for pred in results['predictions']],
            'prediction_numeric': results['predictions'],
            'probability_non_effector': results['probabilities'][:, 0],
            'probability_effector': results['probabilities'][:, 1],
            'confidence': np.max(results['probabilities'], axis=1)
        })

        # Merge with additional data if provided
        if additional_data is not None:
            # Exclude 'sequence' from additional_data to avoid duplicates (sequence_x, sequence_y)
            cols_to_merge = [col for col in additional_data.columns if col not in ['sequence']]
            df = df.merge(additional_data[cols_to_merge], on='label', how='left')

        return df


def run_full_analysis(
    data_path: str,
    model_path: str,
    output_dir: str = "data/processed",
    batch_size: int = 16,
    device: str = 'auto',
    extract_embeddings: bool = True,
    extract_attentions: bool = True,
    master_csv_path: Optional[str] = None,
    use_multi_gpu: bool = True
) -> Dict[str, any]:
    """
    Run full analysis pipeline

    Args:
        data_path: Path to input CSV file (full proteome)
        model_path: Path to model directory
        output_dir: Output directory for results
        batch_size: Batch size for inference
        device: Device to use
        extract_embeddings: Whether to extract embeddings
        extract_attentions: Whether to extract attention weights
        use_multi_gpu: Whether to use DataParallel for multi-GPU
        master_csv_path: Path to master CSV with true labels (optional)

    Returns:
        Dictionary with all results and file paths
    """
    # Initialize logger and data manager
    logger = AnalysisLogger(log_dir=f"{output_dir}/logs")
    data_manager = DataManager(base_dir=output_dir)

    logger.info("="*60)
    logger.info("STARTING PROTEIN ANALYSIS PIPELINE")
    logger.info("="*60)

    # Load input data
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} proteins from proteome")

    # Load master data with true labels if provided
    master_df = None
    if master_csv_path:
        logger.info(f"Loading master data with true labels from: {master_csv_path}")
        master_df = pd.read_csv(master_csv_path)
        logger.info(f"Loaded {len(master_df)} proteins with true labels")

        # Merge to get true labels for proteins that are in master
        df = df.merge(
            master_df[['label', 'is_effector', 'is_housekeeping']],
            on='label',
            how='left',
            suffixes=('', '_master')
        )

        # Count how many proteins have true labels
        proteins_with_labels = df['is_effector'].notna().sum()
        logger.info(f"Found true labels for {proteins_with_labels} proteins ({100*proteins_with_labels/len(df):.1f}%)")

    sequences = df['sequence'].tolist()
    labels = df['label'].tolist()

    # Initialize analyzer
    analyzer = ProteinAnalyzer(
        model_path=model_path,
        device=device,
        logger=logger,
        use_multi_gpu=use_multi_gpu
    )

    # Run analysis
    results = analyzer.analyze_proteins(
        sequences=sequences,
        labels=labels,
        batch_size=batch_size,
        extract_embeddings=extract_embeddings,
        extract_attentions=extract_attentions
    )

    # Create predictions DataFrame
    predictions_df = analyzer.create_predictions_dataframe(results, df)

    # Save results
    logger.info("\n=== Saving Results ===")

    saved_files = {}

    # Save predictions
    pred_file = data_manager.save_predictions(predictions_df, "predictions.csv")
    logger.info(f"Predictions saved: {pred_file}")
    saved_files['predictions'] = pred_file

    # Save embeddings
    if 'embeddings' in results:
        metadata = {
            'localisation': df['localisation'].values if 'localisation' in df.columns else None,
            'localisation_score': df['localisation_score'].values if 'localisation_score' in df.columns else None,
            'is_effector': df['is_effector'].values if 'is_effector' in df.columns else None,
            'is_housekeeping': df['is_housekeeping'].values if 'is_housekeeping' in df.columns else None
        }
        # Remove None values
        metadata = {k: v for k, v in metadata.items() if v is not None}

        emb_file = data_manager.save_embeddings(
            embeddings=results['embeddings'],
            labels=results['labels'],
            sequences=results['sequences'],
            metadata=metadata,
            filename="embeddings.npz"
        )
        logger.info(f"Embeddings saved: {emb_file}")
        saved_files['embeddings'] = emb_file

    # Save attentions
    if 'attentions' in results:
        att_file = data_manager.save_attentions(
            attentions=results['attentions'],
            labels=results['labels'],
            layer_info={
                'num_heads': results['attentions'].shape[1],
                'sequence_length': results['attentions'].shape[2]
            },
            filename="attentions.npz"
        )
        logger.info(f"Attentions saved: {att_file}")
        saved_files['attentions'] = att_file

        # Also save attention masks
        if 'attention_masks' in results:
            np.savez_compressed(
                f"{output_dir}/attention_masks.npz",
                attention_masks=results['attention_masks'],
                labels=results['labels']
            )
            logger.info(f"Attention masks saved: {output_dir}/attention_masks.npz")

    logger.info("\n" + "="*60)
    logger.info("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*60)

    return {
        'results': results,
        'predictions_df': predictions_df,
        'saved_files': saved_files
    }
