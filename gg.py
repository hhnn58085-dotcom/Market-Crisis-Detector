
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


def main():
    # Generate sample market data for educational purposes
    np.random.seed(42)

    normal_days = 1000
    crisis_days = 50

    normal_returns = np.random.normal(0.001, 0.015, normal_days)
    crisis_returns = np.random.normal(-0.04, 0.06, crisis_days)

    returns = np.concatenate([normal_returns, crisis_returns])

    volatility = np.abs(
        np.random.normal(0.015, 0.005, len(returns))
    )

    # Create features
    data = pd.DataFrame({
        "daily_return": returns,
        "volatility": volatility
    })

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    # Detect unusual market behavior
    model = IsolationForest(
        contamination=crisis_days / len(data),
        random_state=42
    )

    predictions = model.fit_predict(X)

    # -1 means anomaly, 1 means normal
    data["detected_anomaly"] = (predictions == -1).astype(int)

    # Synthetic labels used only for demonstration
    actual_labels = np.concatenate([
        np.zeros(normal_days),
        np.ones(crisis_days)
    ]).astype(int)

    print("Market Crisis Detector")
    print("=" * 35)
    print(f"Total observations: {len(data)}")
    print(f"Detected unusual observations: {data['detected_anomaly'].sum()}")

    print("\nEvaluation on synthetic demonstration data:")
    print(classification_report(
        actual_labels,
        data["detected_anomaly"],
        target_names=["Normal", "Potential Crisis"],
        zero_division=0
    ))

    # Plot returns and detected anomalies
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["daily_return"], label="Daily Returns")
    anomalies = data[data["detected_anomaly"] == 1]

    plt.scatter(
        anomalies.index,
        anomalies["daily_return"],
        color="red",
        label="Detected Anomalies",
        s=20
    )

    plt.title("Market Crisis Detection (Synthetic Data)")
    plt.xlabel("Observation")
    plt.ylabel("Daily Return")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
