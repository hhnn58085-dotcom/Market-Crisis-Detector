# -*- coding: utf-8 -*-
"""مشروع تحليل أسعار السوق وكشف الأزمات والتنبؤ بالأسعار.
ملاحظة: مولّد البيانات أدناه اصطناعي بديل لأن Starter Notebook الرسمي لم يكن مرفقًا.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

SEED = 42
OUT = Path(".")
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

ITEMS = ["طحيـن 25ك", "سكر", "رز", "بندورة", "بطاطا", "زيت"]
AVAIL_MAP = {"شحـيح": 0, "ناقص": 1, "متوفر": 2}
AVAIL_ORDER = ["متوفر", "ناقص", "شحـيح"]

def generate_market_data(seed=SEED):
    """بيانات اصطناعية قابلة لإعادة الإنتاج، وليست أسعارًا فعلية."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    base_prices = {"طحيـن 25ك": 120, "سكر": 18, "رز": 30, "بندورة": 7, "بطاطا": 4, "زيت": 35}
    rows = []
    usd = 3.65
    for day, date in enumerate(dates):
        crisis = int(2 if 55 <= day <= 69 or 80 <= day <= 85 else (1 if 30 <= day <= 54 or 70 <= day <= 79 else 0))
        border = int(not (60 <= day <= 67 or 82 <= day <= 85))
        usd += rng.normal(0.001, 0.012) + 0.002 * crisis
        for item in ITEMS:
            # اتجاه تدريجي + أثر الأزمة وإغلاق المعبر + ضوضاء بسيطة
            trend = 1 + day * 0.0012
            stress = 1 + 0.045 * crisis + 0.08 * (1 - border) + 0.018 * crisis * (1-border)
            price = base_prices[item] * trend * stress * (1 + (usd - 3.65) * 0.22) + rng.normal(0, base_prices[item] * 0.025)
            price = max(price, base_prices[item] * 0.5)
            shortage_score = crisis + (1-border) + (price / base_prices[item] - 1 > 0.15)
            if shortage_score >= 2:
                availability = rng.choice(["ناقص", "شحـيح"], p=[0.35, 0.65])
            elif shortage_score == 1:
                availability = rng.choice(["متوفر", "ناقص"], p=[0.45, 0.55])
            else:
                availability = rng.choice(["متوفر", "ناقص"], p=[0.88, 0.12])
            rows.append({
                "date": date, "item": item, "price_shekel": round(float(price), 2),
                "usd_rate": round(float(usd), 4), "border_open": border,
                "crisis_level": crisis, "availability": availability
            })
    return pd.DataFrame(rows)

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close()

def main():
    sns.set_theme(style="whitegrid")
    df = generate_market_data()
    df["date"] = pd.to_datetime(df["date"])
    df["avail_num"] = df["availability"].map({"شحـيح": 0, "ناقص": 1, "متوفر": 2}).astype(int)
    df.to_csv(OUT / "market_data.csv", index=False, encoding="utf-8-sig")

    print("=" * 72, "\nالمرحلة 0: الاستكشاف والإحصاء\n", "=" * 72)
    print("Shape:", df.shape, "(المتوقع 540 صفًا و7 أعمدة أصلية)")
    print("\nأول 5 صفوف:\n", df.head())
    print("\nآخر 5 صفوف:\n", df.tail())
    print("\nمعلومات البيانات:")
    df.info()
    print("\nالإحصاءات الوصفية:\n", df.describe(include="all"))
    print("\nالقيم المفقودة:\n", df.isna().sum())
    print("\nإحصاءات السعر لكل صنف:")
    print(df.groupby("item")["price_shekel"].agg(["mean", "median", "std", "min", "max"]).round(3))
    print("\nمتوسط السعر حسب حالة المعبر:\n", df.groupby("border_open")["price_shekel"].mean().round(3))
    print("\nمتوسط السعر حسب مستوى الأزمة:\n", df.groupby("crisis_level")["price_shekel"].mean().round(3))
    print("\nPivot متوسط السعر (المعبر × الأزمة):\n",
          df.pivot_table(index="border_open", columns="crisis_level", values="price_shekel", aggfunc="mean").round(3))
    print("\nنسب حالات التوفر (%):\n", (df["availability"].value_counts(normalize=True)*100).round(2))

    print("\n" + "=" * 72, "\nالمرحلة 1: K-Means Clustering\n", "=" * 72)
    pivot = df.pivot_table(index="date", columns="item", values="price_shekel", aggfunc="mean").reindex(columns=ITEMS)
    normalized = pivot / pivot.median(axis=0)
    normalized["price_index"] = normalized.mean(axis=1)
    inertias = []
    for k in range(2, 7):
        model = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        model.fit(normalized[["price_index"]])
        inertias.append(model.inertia_)
    plt.figure(figsize=(7, 4))
    plt.plot(range(2, 7), inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    savefig("elbow_method.png")

    kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=10)
    labels = kmeans.fit_predict(normalized[["price_index"]])
    centers = pd.DataFrame({"label": labels, "price_index": normalized["price_index"].values}).groupby("label")["price_index"].mean().sort_values()
    names = {label: name for label, name in zip(centers.index, ["رخاء", "غلاء متوسط", "أزمة"])}
    daily = normalized[["price_index"]].copy()
    daily["cluster_id"] = labels
    daily["state"] = pd.Series(labels, index=daily.index).map(names)
    daily = daily.join(df.groupby("date").agg(border_open=("border_open", "first"), crisis_level=("crisis_level", "first")))
    print("ملخص العناقيد:\n", daily.groupby("state").agg(days=("price_index", "size"), mean_price_index=("price_index", "mean")).round(4))
    print("\nمتوسط حالة المعبر والأزمة حسب العنقود:\n", daily.groupby("state")[["border_open", "crisis_level"]].mean().round(3))
    crisis_days = daily["state"].eq("أزمة")
    high_stress = daily["border_open"].eq(0) & daily["crisis_level"].eq(2)
    overlap = (crisis_days & high_stress).sum() / max(crisis_days.sum(), 1) * 100
    print(f"\nنسبة أيام عنقود الأزمة التي تزامنت مع إغلاق المعبر ومستوى أزمة 2: {overlap:.2f}%")
    print("هذه نسبة وصفية وليست دليلًا قاطعًا على السببية.")
    print("Silhouette Score (k=3):", round(silhouette_score(normalized[["price_index"]], labels), 4))
    print("Inertia (k=3):", round(kmeans.inertia_, 4))
    daily.to_csv(OUT / "daily_market_states.csv", encoding="utf-8-sig")

    print("\n" + "=" * 72, "\nالمرحلة 2: كشف الشذوذ بالمسافة الإقليدية\n", "=" * 72)
    features_anom = ["price_shekel", "usd_rate", "border_open", "crisis_level", "avail_num"]
    scaler_anom = StandardScaler()
    Z = scaler_anom.fit_transform(df[features_anom])
    df["anomaly_score"] = np.linalg.norm(Z, axis=1)
    threshold = np.percentile(df["anomaly_score"], 95)
    df["anomaly"] = np.where(df["anomaly_score"] >= threshold, -1, 1)
    anomalies = df[df["anomaly"] == -1]
    print(f"Threshold (95th percentile): {threshold:.4f}")
    print(f"Anomalies: {len(anomalies)} / {len(df)} = {len(anomalies)/len(df)*100:.2f}%")
    print("\nأول 10 صفوف شاذة:\n", anomalies.head(10)[["date","item","price_shekel","usd_rate","border_open","crisis_level","availability","anomaly_score"]])
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="usd_rate", y="price_shekel", hue=df["anomaly"].map({1:"طبيعي", -1:"شاذ"}), palette={"طبيعي":"steelblue", "شاذ":"red"})
    plt.title("Price vs USD rate — anomalies highlighted")
    plt.xlabel("USD rate")
    plt.ylabel("Price (ILS)")
    savefig("anomaly_scatter.png")
    print("الطريقة غير موجّهة لأنها لا تتعلم من تسميات صحيحة مسبقًا؛ تستخدم المسافات عن مركز البيانات بعد المعايرة.")
    print("عتبة 1% قد تفوّت شواذ أقل حدة، و20% قد تفوّت كثيرًا من الحالات المهمة.")

    print("\n" + "=" * 72, "\nالمرحلة 3: Linear Regression — التنبؤ بسعر اليوم التالي\n", "=" * 72)
    df = df.sort_values(["item", "date"]).copy()
    df["price_next"] = df.groupby("item")["price_shekel"].shift(-1)
    reg = df.dropna(subset=["price_next"]).copy()
    X_cols = ["price_shekel", "usd_rate", "border_open", "crisis_level", "avail_num"]
    X_train_all, X_test_all, y_train, y_test = [], [], [], []
    predictions, actuals = [], []
    # تقسيم زمني مستقل لكل صنف لتجنب خلط تواريخ المستقبل مع التدريب
    coefs = []
    metrics = []
    for item in ITEMS:
        sub = reg[reg["item"] == item].sort_values("date")
        split = int(len(sub) * 0.8)
        X_train, X_test = sub[X_cols].iloc[:split], sub[X_cols].iloc[split:]
        ytr, yte = sub["price_next"].iloc[:split], sub["price_next"].iloc[split:]
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
        model = LinearRegression()
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        mae = mean_absolute_error(yte, pred)
        rmse = np.sqrt(mean_squared_error(yte, pred))
        r2 = r2_score(yte, pred)
        mape = np.mean(np.abs((yte.to_numpy() - pred) / np.maximum(np.abs(yte.to_numpy()), 1e-8))) * 100
        metrics.append({"item": item, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_percent": mape})
        predictions.extend(pred.tolist()); actuals.extend(yte.tolist())
        coefs.extend([{"item": item, "feature": col, "coefficient": coef} for col, coef in zip(X_cols, model.coef_)])
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df.round(4).to_string(index=False))
    metrics_df.to_csv(OUT / "regression_metrics.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals, predictions, alpha=.75)
    lo, hi = min(actuals + predictions), max(actuals + predictions)
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="red", label="Ideal prediction")
    plt.xlabel("Actual next-day price (ILS)")
    plt.ylabel("Predicted price (ILS)")
    plt.title("Actual vs Predicted")
    plt.legend()
    savefig("actual_vs_predicted.png")
    coef_df = pd.DataFrame(coefs)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=coef_df, x="feature", y="coefficient", hue="item")
    plt.xticks(rotation=25, ha="right")
    plt.title("Linear regression coefficients (standardized features)")
    savefig("regression_coefficients.png")
    print("التقسيم الزمني يحاكي المستقبل. R² المرتفع قد ينتج عن استمرارية الأسعار في البيانات الاصطناعية، ولا يعني ضمانًا للتنبؤ الواقعي.")
    print("MAE يعبّر عن متوسط الخطأ بالشيكل؛ MAPE يعبّر عنه كنسبة مئوية.")

    print("\n" + "=" * 72, "\nالمرحلة 4: Logistic Regression — توقع التوفر\n", "=" * 72)
    X = df[["price_shekel", "usd_rate", "border_open", "crisis_level"]]
    y = df["availability"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    cls_scaler = StandardScaler()
    X_train_scaled = cls_scaler.fit_transform(X_train)
    X_test_scaled = cls_scaler.transform(X_test)
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    classifier.fit(X_train_scaled, y_train)
    pred_cls = classifier.predict(X_test_scaled)
    print("Accuracy:", round(accuracy_score(y_test, pred_cls), 4))
    print("Precision weighted:", round(precision_score(y_test, pred_cls, average="weighted", zero_division=0), 4))
    print("Recall weighted:", round(recall_score(y_test, pred_cls, average="weighted", zero_division=0), 4))
    print("F1 weighted:", round(f1_score(y_test, pred_cls, average="weighted", zero_division=0), 4))
    print("\nClassification report:\n", classification_report(y_test, pred_cls, labels=AVAIL_ORDER, zero_division=0))
    cm = confusion_matrix(y_test, pred_cls, labels=AVAIL_ORDER)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=AVAIL_ORDER, yticklabels=AVAIL_ORDER)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — availability")
    savefig("confusion_matrix.png")
    coef_table = pd.DataFrame(classifier.coef_, columns=X.columns, index=classifier.classes_)
    print("معاملات النموذج:\n", coef_table.round(4))
    coef_table.to_csv(OUT / "logistic_coefficients.csv", encoding="utf-8-sig")

    print("\n" + "=" * 72, "\nالمرحلة 5: محاكاة السيناريوهات\n", "=" * 72)
    def predict_price_next(price_today, usd, border, crisis, avail):
        avail_num = AVAIL_MAP.get(avail, 2)
        results = {}
        for item in ITEMS:
            sub = reg[reg["item"] == item].sort_values("date")
            split = int(len(sub) * 0.8)
            scaler = StandardScaler().fit(sub[X_cols].iloc[:split])
            model = LinearRegression().fit(scaler.transform(sub[X_cols].iloc[:split]), sub["price_next"].iloc[:split])
            row = pd.DataFrame([[price_today, usd, border, crisis, avail_num]], columns=X_cols)
            results[item] = float(model.predict(scaler.transform(row))[0])
        return results
    def predict_availability(price_today, usd, border, crisis):
        row = pd.DataFrame([[price_today, usd, border, crisis]], columns=X.columns)
        return classifier.predict(cls_scaler.transform(row))[0]
    scenarios = [
        ("استقرار ورخاء", 120, 3.70, 1, 0, "متوفر"),
        ("تصعيد متوسط", 150, 3.85, 1, 1, "ناقص"),
        ("أزمة وإغلاق كامل", 180, 3.95, 0, 2, "شحـيح"),
    ]
    for title, price, usd, border, crisis, avail in scenarios:
        price_predictions = predict_price_next(price, usd, border, crisis, avail)
        availability_prediction = predict_availability(price, usd, border, crisis)
        print(f"\n{title}: سعر اليوم={price}₪، الدولار={usd}، المعبر={'مفتوح' if border else 'مغلق'}، الأزمة={crisis}")
        print("التوفر المتوقع:", availability_prediction)
        print("تنبؤات اليوم التالي لكل صنف:", {k: round(v, 2) for k, v in price_predictions.items()})
    print("\nاكتمل التنفيذ. راجع ملفات CSV ومجلد figures/، ولا تتعامل مع النتائج كبيانات سوق حقيقية.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
