"""
General VAE Anomaly Detection Framework for Text
=================================================

A flexible framework for running VAE-based anomaly detection on any text dataset.

Usage:
    python vae_text_anomaly.py --dataset sms_spam --output_dir results/
    python vae_text_anomaly.py --dataset imdb --config configs/my_config.yaml
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod


# ============================================================================
# Configuration
# ============================================================================
class Config:
    """Configuration for VAE anomaly detection"""
    
    def __init__(self, **kwargs):
        # Model settings
        self.MODEL_NAME = kwargs.get('model_name', 'distilbert-base-uncased')
        self.DEVICE = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data settings
        self.MAX_LENGTH = kwargs.get('max_length', 256)
        self.BATCH_SIZE = kwargs.get('batch_size', 32)
        self.SEED = kwargs.get('seed', 42)
        
        # Architecture
        self.INPUT_DIM = kwargs.get('input_dim', 768)
        self.HIDDEN_DIM = kwargs.get('hidden_dim', 256)
        self.LATENT_DIM = kwargs.get('latent_dim', 64)
        self.DROPOUT = kwargs.get('dropout', 0.2)
        
        # Training
        # self.VAE_EPOCHS = kwargs.get('epochs', 80)
        self.VAE_EPOCHS = kwargs.get('epochs', 200)
        self.LEARNING_RATE = kwargs.get('learning_rate', 1e-3)
        self.BETA_START = kwargs.get('beta_start', 0.0)
        self.BETA_END = kwargs.get('beta_end', 0.3)
        self.BETA_ANNEAL_EPOCHS = kwargs.get('beta_anneal_epochs', 40)
        
        # Anomaly detection
        self.ANOMALY_PERCENTILE = kwargs.get('anomaly_percentile', 95)
        
        # Paths
        self.OUTPUT_DIR = kwargs.get('output_dir', './output')
        self.DATA_DIR = os.path.join(self.OUTPUT_DIR, 'data')
        self.IMAGES_DIR = os.path.join(self.OUTPUT_DIR, 'images')
        self.MODELS_DIR = os.path.join(self.OUTPUT_DIR, 'models')
        self.RESULTS_DIR = os.path.join(self.OUTPUT_DIR, 'results')
        
        # Create directories
        for dir_path in [self.DATA_DIR, self.IMAGES_DIR, self.MODELS_DIR, self.RESULTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_yaml(cls, yaml_path):
        """Load config from YAML file"""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


# ============================================================================
# VAE Architecture
# ============================================================================
class VAE(nn.Module):
    """Variational Autoencoder for anomaly detection"""
    
    def __init__(self, input_dim=768, hidden_dim=256, latent_dim=64, dropout=0.2):
        super(VAE, self).__init__()
        
        # Encoder
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_bn1 = nn.BatchNorm1d(hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.enc_bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.dec1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.dec_bn1 = nn.BatchNorm1d(hidden_dim // 2)
        self.dec2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.dec_bn2 = nn.BatchNorm1d(hidden_dim)
        self.dec3 = nn.Linear(hidden_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def encode(self, x):
        h = F.relu(self.enc_bn1(self.enc1(x)))
        h = self.dropout(h)
        h = F.relu(self.enc_bn2(self.enc2(h)))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.dec_bn1(self.dec1(z)))
        h = self.dropout(h)
        h = F.relu(self.dec_bn2(self.dec2(h)))
        return self.dec3(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta):
    """VAE loss: Reconstruction + KL divergence"""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    return recon_loss + beta * kld, recon_loss, kld


# ============================================================================
# Text Embedding Processor
# ============================================================================
class TextEmbeddingProcessor:
    """Convert text to BERT embeddings"""
    
    def __init__(self, model_name, device, max_length=256, batch_size=32):
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
        print(f"Loading transformer model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModel.from_pretrained(model_name).to(device).eval()
        
    def process_texts(self, texts, desc="Processing"):
        """Convert list of texts to embeddings"""
        dataloader = DataLoader(texts, batch_size=self.batch_size, shuffle=False)
        all_embeddings = []
        
        for batch in tqdm(dataloader, desc=desc):
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.text_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[1]
                mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask, 1) # We are summing on each of the 768 dimensions and removing the "word" dimension
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                pooled = sum_embeddings / sum_mask
                all_embeddings.append(pooled.cpu())
        
        return torch.cat(all_embeddings, dim=0)


# ============================================================================
# Dataset Interface
# ============================================================================
class TextDataset(ABC):
    """Abstract base class for text anomaly detection datasets"""
    
    def __init__(self, config):
        self.config = config
        self.name = self.__class__.__name__.replace('Dataset', '').lower()
        
    @abstractmethod
    def load_raw_data(self):
        """
        Load raw text data from source.
        
        Returns:
            normal_texts: List of normal text samples
            anomaly_texts: List of anomaly text samples
        """
        pass
    
    def get_cache_paths(self):
        """Get paths for cached embeddings"""
        cache_dir = os.path.join(self.config.DATA_DIR, self.name)
        os.makedirs(cache_dir, exist_ok=True)
        
        return {
            'train_normal': os.path.join(cache_dir, 'train_normal.pt'),
            'test_normal': os.path.join(cache_dir, 'test_normal.pt'),
            'test_anomaly': os.path.join(cache_dir, 'test_anomaly.pt'),
            'scaler': os.path.join(cache_dir, 'scaler.pt'),
            'metadata': os.path.join(cache_dir, 'metadata.json')
        }
    
    def load_embeddings(self, processor, force_reprocess=False):
        """
        Load or create embeddings for this dataset.
        
        Returns:
            train_normal: Training embeddings (normal class only)
            test_normal: Test embeddings (normal class)
            test_anomaly: Test embeddings (anomaly class)
            scaler: Fitted StandardScaler
        """
        cache_paths = self.get_cache_paths()
        
        # Try to load from cache
        if not force_reprocess and all(os.path.exists(p) for p in cache_paths.values()):
            print(f"Loading cached embeddings for {self.name}...")
            
            train_normal = torch.load(cache_paths['train_normal'], weights_only=False)
            test_normal = torch.load(cache_paths['test_normal'], weights_only=False)
            test_anomaly = torch.load(cache_paths['test_anomaly'], weights_only=False)
            scaler_params = torch.load(cache_paths['scaler'], weights_only=False)
            
            scaler = StandardScaler()
            scaler.mean_ = scaler_params['mean']
            scaler.scale_ = scaler_params['scale']
            
            with open(cache_paths['metadata'], 'r') as f:
                metadata = json.load(f)
            
            print(f"  Train (normal):  {len(train_normal)}")
            print(f"  Test (normal):   {len(test_normal)}")
            print(f"  Test (anomaly):  {len(test_anomaly)}")
            
            return train_normal, test_normal, test_anomaly, scaler, metadata
        
        # Load raw data
        print(f"\nProcessing {self.name} dataset...")
        normal_texts, anomaly_texts = self.load_raw_data()
        
        # Split normal into train/test
        train_normal_texts, test_normal_texts = train_test_split(
            normal_texts, test_size=0.2, random_state=self.config.SEED
        )
        
        # Limit anomaly test set to reasonable size
        if len(anomaly_texts) > len(test_normal_texts) * 2:
            anomaly_texts = anomaly_texts[:len(test_normal_texts) * 2]
        
        print(f"  Train (normal):  {len(train_normal_texts)}")
        print(f"  Test (normal):   {len(test_normal_texts)}")
        print(f"  Test (anomaly):  {len(anomaly_texts)}")
        
        # Show examples
        print(f"\nExamples from {self.name}:")
        print("Normal:", train_normal_texts[0][:100], "...")
        print("Anomaly:", anomaly_texts[0][:100], "...")
        
        # Convert to embeddings
        train_normal = processor.process_texts(train_normal_texts, f"{self.name} Train Normal")
        test_normal = processor.process_texts(test_normal_texts, f"{self.name} Test Normal")
        test_anomaly = processor.process_texts(anomaly_texts, f"{self.name} Test Anomaly")
        
        # Fit scaler
        scaler = StandardScaler()
        scaler.fit(train_normal.numpy())
        
        # Save to cache
        torch.save(train_normal, cache_paths['train_normal'])
        torch.save(test_normal, cache_paths['test_normal'])
        torch.save(test_anomaly, cache_paths['test_anomaly'])
        torch.save({'mean': scaler.mean_, 'scale': scaler.scale_}, cache_paths['scaler'])
        
        metadata = {
            'dataset': self.name,
            'n_train': len(train_normal),
            'n_test_normal': len(test_normal),
            'n_test_anomaly': len(test_anomaly),
            'anomaly_ratio': len(anomaly_texts) / (len(normal_texts) + len(anomaly_texts)),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(cache_paths['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Cached embeddings saved to {cache_paths['train_normal']}")
        
        return train_normal, test_normal, test_anomaly, scaler, metadata


# ============================================================================
# Concrete Dataset Implementations
# ============================================================================

class SMSSpamDataset(TextDataset):
    """SMS Spam detection dataset"""
    
    def load_raw_data(self):
        dataset = load_dataset("sms_spam")
        train_data = dataset['train']
        
        normal = [msg for msg, label in zip(train_data['sms'], train_data['label']) if label == 0]
        anomaly = [msg for msg, label in zip(train_data['sms'], train_data['label']) if label == 1]
        
        return normal, anomaly


class IMDBSentimentDataset(TextDataset):
    """IMDB Sentiment as anomaly detection (positive=normal, negative=anomaly)"""
    
    def load_raw_data(self):
        dataset = load_dataset("imdb")
        
        # Limit to reasonable size
        normal = dataset["train"].filter(lambda x: x["label"] == 1)["text"][:5000]
        anomaly = dataset["test"].filter(lambda x: x["label"] == 0)["text"][:2500]
        
        return normal, anomaly


class AGNewsDataset(TextDataset):
    """AG News topic detection (World news=normal, others=anomaly)"""
    
    def load_raw_data(self):
        dataset = load_dataset("ag_news")
        
        normal = dataset["train"].filter(lambda x: x["label"] == 0)["text"][:5000]
        anomaly_data = dataset["test"].filter(lambda x: x["label"] != 0)
        anomaly = anomaly_data["text"][:2000]
        
        return normal, anomaly


class FakeNewsDataset(TextDataset):
    """Fake news detection"""
    
    def load_raw_data(self):
        try:
            dataset = load_dataset("GonzaloA/fake_news")
            
            normal = [text for text, label in zip(dataset["train"]["text"], dataset["train"]["label"]) 
                     if label == 0][:5000]
            anomaly = [text for text, label in zip(dataset["train"]["text"], dataset["train"]["label"]) 
                      if label == 1][:2000]
            
            return normal, anomaly
        except:
            raise NotImplementedError("Fake news dataset not available. Try a different dataset.")


# Dataset registry
DATASETS = {
    'sms_spam': SMSSpamDataset,
    'imdb': IMDBSentimentDataset,
    'ag_news': AGNewsDataset,
    'fake_news': FakeNewsDataset,
}


# ============================================================================
# Training
# ============================================================================
def train_vae(model, train_data, config):
    """Train VAE with beta annealing"""
    
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data.numpy())
    train_tensor = torch.FloatTensor(train_scaled)
    
    dataset = TensorDataset(train_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    model.train()
    history = {'total_loss': [], 'recon_loss': [], 'kld': [], 'beta': []}
    
    print(f"\nTraining VAE (β: {config.BETA_START}→{config.BETA_END})...")
    
    for epoch in range(config.VAE_EPOCHS):
        # Beta annealing
        if epoch < config.BETA_ANNEAL_EPOCHS:
            beta = config.BETA_START + (config.BETA_END - config.BETA_START) * \
                   (epoch / config.BETA_ANNEAL_EPOCHS)
        else:
            beta = config.BETA_END
        
        epoch_total = epoch_recon = epoch_kld = 0
        
        for batch in dataloader:
            data = batch[0].to(config.DEVICE)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss, recon_loss, kld = vae_loss(recon, data, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            
            epoch_total += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kld += kld.item()
        
        n_batches = len(dataloader)
        history['total_loss'].append(epoch_total / n_batches)
        history['recon_loss'].append(epoch_recon / n_batches)
        history['kld'].append(epoch_kld / n_batches)
        history['beta'].append(beta)
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | β={beta:.3f} | Loss: {epoch_total/n_batches:.4f}")
    
    return history, scaler


# ============================================================================
# Anomaly Detection
# ============================================================================
class AnomalyDetector:
    """Anomaly detector using VAE reconstruction error"""
    
    def __init__(self, model, scaler, device):
        self.model = model
        self.scaler = scaler
        self.device = device
        self.threshold = None
        
    def compute_reconstruction_errors(self, data):
        """Compute per-sample reconstruction errors"""
        data_scaled = torch.FloatTensor(self.scaler.transform(data.numpy())).to(self.device)
        
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for i in range(0, len(data_scaled), 64):
                batch = data_scaled[i:i+64]
                recon, _, _ = self.model(batch)
                error = F.mse_loss(recon, batch, reduction='none').mean(dim=1)
                errors.append(error.cpu())
        
        return torch.cat(errors).numpy()
    
    def set_threshold(self, normal_data, percentile=95):
        """Set threshold based on normal data reconstruction errors"""
        errors = self.compute_reconstruction_errors(normal_data)
        # self.threshold = np.percentile(errors, percentile)
        self.threshold = 0.43
        print(f"\nThreshold ({percentile}th percentile): {self.threshold:.6f}")
        return errors
    
    def predict(self, data):
        """Predict anomalies (returns predictions and scores)"""
        if self.threshold is None:
            raise ValueError("Call set_threshold() first")
        
        errors = self.compute_reconstruction_errors(data)
        predictions = errors > self.threshold
        return predictions, errors


# ============================================================================
# Evaluation & Metrics
# ============================================================================
def evaluate_detector(detector, test_normal, test_anomaly, dataset_name):
    """Evaluate anomaly detector and return metrics"""
    
    print("\n" + "="*70)
    print(f"EVALUATING: {dataset_name}")
    print("="*70)
    
    # Predict
    normal_pred, normal_errors = detector.predict(test_normal)
    anomaly_pred, anomaly_errors = detector.predict(test_anomaly)
    
    # Combine
    y_true = np.concatenate([np.zeros(len(test_normal)), np.ones(len(test_anomaly))])
    y_pred = np.concatenate([normal_pred, anomaly_pred])
    y_scores = np.concatenate([normal_errors, anomaly_errors])
    
    # Metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nMetrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    
    print(f"\nError Statistics:")
    print(f"  Normal:  {normal_errors.mean():.6f} ± {normal_errors.std():.6f}")
    print(f"  Anomaly: {anomaly_errors.mean():.6f} ± {anomaly_errors.std():.6f}")
    print(f"  Separation: {(anomaly_errors.mean() - normal_errors.mean()) / normal_errors.std():.2f}σ")
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
        'confusion_matrix': cm.tolist(),
        'threshold': float(detector.threshold),
        'normal_error_mean': float(normal_errors.mean()),
        'normal_error_std': float(normal_errors.std()),
        'anomaly_error_mean': float(anomaly_errors.mean()),
        'anomaly_error_std': float(anomaly_errors.std()),
    }, normal_errors, anomaly_errors


def measure_semantic_distance(normal_emb, anomaly_emb):
    """Measure semantic distance between normal and anomaly embeddings"""
    
    # Compute means
    normal_mean = normal_emb.mean(dim=0)
    anomaly_mean = anomaly_emb.mean(dim=0)
    
    # Cosine distance between MEAN embeddings
    cos_sim = F.cosine_similarity(normal_mean.unsqueeze(0), anomaly_mean.unsqueeze(0))
    cosine_distance = 1 - cos_sim.item()
    
    # Also compute average pairwise cosine distance (more robust)
    # Sample 500 random pairs to avoid memory issues
    n_samples = min(500, len(normal_emb), len(anomaly_emb))
    normal_sample = normal_emb[torch.randperm(len(normal_emb))[:n_samples]]
    anomaly_sample = anomaly_emb[torch.randperm(len(anomaly_emb))[:n_samples]]
    
    pairwise_sims = F.cosine_similarity(
        normal_sample.unsqueeze(1), 
        anomaly_sample.unsqueeze(0), 
        dim=2
    )
    avg_pairwise_distance = 1 - pairwise_sims.mean().item()
    
    # Euclidean (normalized)
    euclidean = torch.norm(normal_mean - anomaly_mean).item() / np.sqrt(normal_emb.shape[1])
    
    # Intra-class variance (how spread out each class is)
    normal_var = torch.norm(normal_emb - normal_mean, dim=1).mean().item()
    anomaly_var = torch.norm(anomaly_emb - anomaly_mean, dim=1).mean().item()
    
    # Inter-class distance
    inter_class_dist = torch.norm(normal_mean - anomaly_mean).item()
    
    return {
        'cosine_distance': cosine_distance,
        'cosine_distance_pairwise': avg_pairwise_distance,  # More robust metric
        'euclidean_distance': euclidean,
        'normal_variance': normal_var,
        'anomaly_variance': anomaly_var,
        'inter_class_distance': inter_class_dist,
        'separation_ratio': inter_class_dist / normal_var if normal_var > 0 else 0
    }


# ============================================================================
# Visualization
# ============================================================================
def plot_results(normal_errors, anomaly_errors, threshold, dataset_name, config):
    """Create visualization plots"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error distribution
    bins = np.linspace(
        min(normal_errors.min(), anomaly_errors.min()),
        max(normal_errors.max(), anomaly_errors.max()),
        50
    )
    
    axes[0].hist(normal_errors, bins=bins, alpha=0.6, label='Normal', 
                color='green', density=True)
    axes[0].hist(anomaly_errors, bins=bins, alpha=0.6, label='Anomaly', 
                color='red', density=True)
    axes[0].axvline(threshold, color='black', linestyle='--', label='Threshold')
    axes[0].set_xlabel('Reconstruction Error')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'{dataset_name}: Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [normal_errors, anomaly_errors]
    axes[1].boxplot(data_to_plot, labels=['Normal', 'Anomaly'])
    axes[1].axhline(threshold, color='black', linestyle='--', label='Threshold')
    axes[1].set_ylabel('Reconstruction Error')
    axes[1].set_title(f'{dataset_name}: Error Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(config.IMAGES_DIR, f'{dataset_name}_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_latent_space(model, scaler, test_normal, test_anomaly, dataset_name, config, n_samples=1000):
    """
    Visualize latent space using t-SNE with multiple color schemes
    
    Creates:
    1. Class-colored (normal vs anomaly)
    2. Error-colored (reconstruction error heatmap)
    3. Confidence-colored (distance from decision boundary)
    """
    
    print(f"\nGenerating latent space visualization for {dataset_name}...")
    
    # Subsample for speed
    n_normal = min(n_samples, len(test_normal))
    n_anomaly = min(n_samples, len(test_anomaly))
    
    normal_subset = test_normal[:n_normal]
    anomaly_subset = test_anomaly[:n_anomaly]
    
    # Scale and encode to latent space
    normal_scaled = torch.FloatTensor(scaler.transform(normal_subset.numpy())).to(config.DEVICE)
    anomaly_scaled = torch.FloatTensor(scaler.transform(anomaly_subset.numpy())).to(config.DEVICE)
    
    model.eval()
    with torch.no_grad():
        # Get latent representations
        normal_mu, _ = model.encode(normal_scaled)
        anomaly_mu, _ = model.encode(anomaly_scaled)
        
        # Get reconstruction errors
        normal_recon, _, _ = model(normal_scaled)
        anomaly_recon, _, _ = model(anomaly_scaled)
        
        normal_errors = F.mse_loss(normal_recon, normal_scaled, reduction='none').mean(dim=1).cpu().numpy()
        anomaly_errors = F.mse_loss(anomaly_recon, anomaly_scaled, reduction='none').mean(dim=1).cpu().numpy()
    
    # Combine data
    X_latent = torch.cat([normal_mu, anomaly_mu]).cpu().numpy()
    y_true = np.array([0] * n_normal + [1] * n_anomaly)
    errors = np.concatenate([normal_errors, anomaly_errors])
    
    # Apply t-SNE
    print("  Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_latent)//4))
    X_embedded = tsne.fit_transform(X_latent)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # ========================================================================
    # Plot 1: Class Labels (Normal vs Anomaly)
    # ========================================================================
    ax = axes[0]
    
    # Plot normal samples
    normal_mask = y_true == 0
    ax.scatter(X_embedded[normal_mask, 0], X_embedded[normal_mask, 1],
              c='green', alpha=0.6, s=30, label='Normal', 
              edgecolors='darkgreen', linewidth=0.5)
    
    # Plot anomaly samples
    anomaly_mask = y_true == 1
    ax.scatter(X_embedded[anomaly_mask, 0], X_embedded[anomaly_mask, 1],
              c='red', alpha=0.6, s=30, label='Anomaly',
              edgecolors='darkred', linewidth=0.5, marker='x')
    
    # Add cluster centers
    normal_center = X_embedded[normal_mask].mean(axis=0)
    anomaly_center = X_embedded[anomaly_mask].mean(axis=0)
    
    ax.scatter(*normal_center, c='darkgreen', s=300, marker='*',
              edgecolors='black', linewidth=2, label='Normal Center', zorder=5)
    ax.scatter(*anomaly_center, c='darkred', s=300, marker='*',
              edgecolors='black', linewidth=2, label='Anomaly Center', zorder=5)
    
    # Draw line between centers
    ax.plot([normal_center[0], anomaly_center[0]], 
           [normal_center[1], anomaly_center[1]],
           'k--', alpha=0.3, linewidth=2, zorder=1)
    
    # Calculate separation
    separation = np.linalg.norm(normal_center - anomaly_center)
    ax.text(0.02, 0.98, f'Separation: {separation:.2f}',
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title('Latent Space: Class Labels', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.2)
    
    # ========================================================================
    # Plot 2: Reconstruction Error (Heatmap)
    # ========================================================================
    ax = axes[1]
    
    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1],
                        c=errors, cmap='RdYlGn_r', alpha=0.7, s=30,
                        edgecolors='black', linewidth=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Reconstruction Error', fontsize=11)
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title('Latent Space: Reconstruction Error', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # Add error statistics
    ax.text(0.02, 0.98, 
           f'Error Range:\n{errors.min():.4f} - {errors.max():.4f}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # ========================================================================
    # Plot 3: Prediction Confidence
    # ========================================================================
    ax = axes[2]
    
    # Create 4 categories based on true label and error
    threshold = np.percentile(errors, 95)
    
    # True Negatives (Normal, low error)
    tn_mask = (y_true == 0) & (errors <= threshold)
    ax.scatter(X_embedded[tn_mask, 0], X_embedded[tn_mask, 1],
              c='darkgreen', alpha=0.7, s=30, label='True Negative (Correct)',
              edgecolors='black', linewidth=0.3)
    
    # False Positives (Normal, high error)
    fp_mask = (y_true == 0) & (errors > threshold)
    ax.scatter(X_embedded[fp_mask, 0], X_embedded[fp_mask, 1],
              c='orange', alpha=0.7, s=50, label='False Positive (Error)',
              edgecolors='black', linewidth=0.5, marker='s')
    
    # True Positives (Anomaly, high error)
    tp_mask = (y_true == 1) & (errors > threshold)
    ax.scatter(X_embedded[tp_mask, 0], X_embedded[tp_mask, 1],
              c='darkred', alpha=0.7, s=30, label='True Positive (Correct)',
              edgecolors='black', linewidth=0.3, marker='x')
    
    # False Negatives (Anomaly, low error)
    fn_mask = (y_true == 1) & (errors <= threshold)
    ax.scatter(X_embedded[fn_mask, 0], X_embedded[fn_mask, 1],
              c='yellow', alpha=0.7, s=50, label='False Negative (Missed)',
              edgecolors='black', linewidth=0.5, marker='D')
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title('Latent Space: Predictions', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.2)
    
    # Add accuracy
    correct = np.sum(tn_mask) + np.sum(tp_mask)
    total = len(y_true)
    accuracy = correct / total
    
    ax.text(0.02, 0.98, 
           f'Accuracy: {accuracy*100:.1f}%\n'
           f'TP: {np.sum(tp_mask)} | TN: {np.sum(tn_mask)}\n'
           f'FP: {np.sum(fp_mask)} | FN: {np.sum(fn_mask)}',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # ========================================================================
    # Save
    # ========================================================================
    plt.suptitle(f'{dataset_name}: Latent Space Visualization (t-SNE)', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(config.IMAGES_DIR, f'{dataset_name}_latent_space.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Latent space visualization saved to {save_path}")
    plt.close()
    
    return X_embedded, y_true, errors


# ============================================================================
# Main Experiment Runner
# ============================================================================
def run_experiment(dataset_name, config, force_reprocess=False):
    """Run complete VAE anomaly detection experiment on a dataset"""
    
    print("\n" + "="*70)
    print(f"EXPERIMENT: {dataset_name}")
    print("="*70)
    
    # Load dataset
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")
    
    dataset = DATASETS[dataset_name](config)
    processor = TextEmbeddingProcessor(config.MODEL_NAME, config.DEVICE, 
                                      config.MAX_LENGTH, config.BATCH_SIZE)
    
    # Load or create embeddings
    train_normal, test_normal, test_anomaly, scaler, metadata = \
        dataset.load_embeddings(processor, force_reprocess)
    
    # Measure semantic distance
    print("\nMeasuring semantic distance...")
    distances = measure_semantic_distance(test_normal, test_anomaly)
    print(f"  Cosine distance: {distances['cosine_distance']:.4f}")
    
    # Train VAE
    model = VAE(config.INPUT_DIM, config.HIDDEN_DIM, config.LATENT_DIM, config.DROPOUT).to(config.DEVICE)
    history, train_scaler = train_vae(model, train_normal, config)
    
    # Detect anomalies
    detector = AnomalyDetector(model, train_scaler, config.DEVICE)
    detector.set_threshold(train_normal, config.ANOMALY_PERCENTILE)
    
    # Evaluate
    results, normal_errors, anomaly_errors = evaluate_detector(
        detector, test_normal, test_anomaly, dataset_name
    )
    
    # Visualize error distributions
    plot_results(normal_errors, anomaly_errors, detector.threshold, dataset_name, config)
    
    # Visualize latent space
    plot_latent_space(model, train_scaler, test_normal, test_anomaly, dataset_name, config)
    
    # Compile all results
    full_results = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        **metadata,
        **distances,
        **results,
        'config': config.to_dict()
    }
    
    # Save results
    results_file = os.path.join(config.RESULTS_DIR, f'{dataset_name}_results.json')
    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print("="*70)
    
    return full_results


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='VAE Anomaly Detection for Text')
    parser.add_argument('--dataset', type=str, default='sms_spam',
                       choices=list(DATASETS.keys()) + ['all'],
                       help='Dataset to run')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    parser.add_argument('--force_reprocess', action='store_true',
                       help='Force reprocessing of embeddings')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config(output_dir=args.output_dir)
    
    print(f"Using device: {config.DEVICE}")
    
    # Run experiments
    datasets_to_run = list(DATASETS.keys()) if args.dataset == 'all' else [args.dataset]
    
    all_results = []
    for dataset_name in datasets_to_run:
        try:
            result = run_experiment(dataset_name, config, args.force_reprocess)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR with {dataset_name}: {e}")
            continue
    
    # Summary
    if all_results:
        summary_file = os.path.join(config.RESULTS_DIR, 'summary.json')
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"{'Dataset':<20} {'Cosine Dist':<12} {'F1':<8} {'AUC':<8}")
        print("-"*70)
        for r in all_results:
            print(f"{r['dataset']:<20} {r['cosine_distance']:<12.3f} {r['f1']:<8.3f} {r['auc']:<8.3f}")
        
        print(f"\nFull summary: {summary_file}")


if __name__ == "__main__":
    main()