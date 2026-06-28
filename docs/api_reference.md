# API Reference

This reference details the core API interfaces for **HealthFraudML**.

## `FraudDetector`

The main unified wrapper for all fraud detection algorithms.

```python
class FraudDetector:
    def __init__(self, model, threshold=0.5):
        """
        Args:
            model: An instance of a model detector class.
            threshold (float): Decision threshold for flagging claims (0.0 to 1.0).
        """
        pass

    def fit(self, X, y):
        """Train the detector model."""
        pass

    def predict(self, X, explain=False):
        """
        Predict fraud on new claims.
        
        Returns:
            PredictionResults object containing predictions, scores, and explanations.
        """
        pass

    def evaluate(self, X, y):
        """Evaluate performance metrics."""
        pass
```

## `ReadinessAssessment`

Evaluate organizational capabilities before implementation.

```python
class ReadinessAssessment:
    def evaluate(self, **kwargs):
        """
        Evaluate and score organizational preparedness.
        
        Returns:
            ReadinessReport object.
        """
        pass
```
