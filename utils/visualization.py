import json
import os
from pathlib import Path


class Visualization(object):
    """
    Lightweight visualization class that logs metrics to JSON and console.
    Replaces tensorboardX to avoid protobuf dependency conflicts.
    """
    def __init__(self):
        self.metrics = {}
        self.log_dir = None
        self.model_type = None

    def create_summary(self, model_type='U_Net'):
        """Initialize visualization for a model"""
        self.model_type = model_type
        self.log_dir = Path('./runs') / str(model_type)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
        print(f"✓ Visualization initialized for {model_type}")
        print(f"  Metrics will be saved to: {self.log_dir}")

    def add_scalar(self, epoch, value, params='loss'):
        """Log a scalar metric value"""
        if params not in self.metrics:
            self.metrics[params] = {}
        
        self.metrics[params][epoch] = float(value)
        
        # Log to console for immediate feedback
        print(f"  [{params}] Epoch {epoch}: {value:.6f}")
        
        # Periodically save to JSON
        if epoch % 5 == 0 or epoch == 1:  # Save every 5 epochs or first epoch
            self._save_metrics()

    def add_graph(self, model):
        """Placeholder for model graph visualization"""
        pass

    def add_iamge(self, epoch, tag, image, dataformats='HWC'):
        """Placeholder for image visualization (Note: typo preserved from original)"""
        pass

    def close_summary(self):
        """Save final metrics and close"""
        self._save_metrics()
        print("✓ Metrics saved to JSON")

    def _save_metrics(self):
        """Save metrics to JSON file for later analysis"""
        if self.log_dir is None:
            return
        
        metrics_file = self.log_dir / 'metrics.json'
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            print(f"⚠ Warning: Could not save metrics: {e}")
