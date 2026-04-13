import mlflow
import os
from typing import Dict, Any

class MLFlowLogger:
    """Wrapper around MLFlow for unified metric and parameter tracking."""
    def __init__(self, tracking_uri: str, experiment_name: str):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        # Ensure directory exists for local file tracking
        if tracking_uri.startswith("file://"):
            os.makedirs(tracking_uri.replace("file://", ""), exist_ok=True)
            
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        self.active_run = None
        
    def start_run(self, run_name: str = None):
        self.active_run = mlflow.start_run(run_name=run_name)
        return self.active_run

    def resume_run(self, run_id: str):
        """Rejoin an existing MLflow run by ID.

        Called by the notebook after a Colab session restart so all epochs of
        a training run land in the same MLflow entry.  If the run is already
        FINISHED (normal completion) MLflow will raise; we catch and fall back
        to opening a fresh run so re-running a completed notebook is still safe.
        """
        try:
            self.active_run = mlflow.start_run(run_id=run_id)
        except mlflow.exceptions.MlflowException:
            # Run was FINISHED or the id is stale — start fresh
            self.active_run = mlflow.start_run()
        return self.active_run
        
    def log_params(self, params: Dict[str, Any] | Any):
        if self.active_run is None:
            raise RuntimeError("Must start a run before logging")
            
        # Convert namespace if needed
        if hasattr(params, '__dict__'):
            params = params.__dict__
            
        flat_params = self._flatten_dict(params)
        
        try:
            mlflow.log_params(flat_params)
        except mlflow.exceptions.MlflowException as e:
            # Handle the case where we resume a run but some params (like epochs) changed.
            # MLflow doesn't allow changing params mid-run. We log a warning and continue.
            print(f"MLflow Warning: Could not log all parameters to existing run. "
                  f"This is common when resuming with a modified config. Error: {e}")
            
            # Optional: Attempt to log them as tags instead so the new values are still recorded
            try:
                # Convert values to strings for tags
                tags = {f"resumed.{k}": str(v) for k, v in flat_params.items()}
                mlflow.set_tags(tags)
            except Exception:
                pass
        
    def log_metrics(self, metrics: Dict[str, float], step: int):
        if self.active_run is None:
            raise RuntimeError("Must start a run before logging")
        mlflow.log_metrics(metrics, step=step)
        
    def log_artifact(self, local_path: str, artifact_path: str = None):
        if self.active_run is None:
            raise RuntimeError("Must start a run before logging")
        mlflow.log_artifact(local_path, artifact_path)
            
    def end_run(self):
        if self.active_run is not None:
            mlflow.end_run()
            self.active_run = None
            
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Recursively flattens nested dictionaries or namespaces for MLflow."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif hasattr(v, '__dict__'):
                 items.extend(self._flatten_dict(v.__dict__, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
