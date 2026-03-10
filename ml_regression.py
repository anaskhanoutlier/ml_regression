

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# SECTION 1: LINEAR REGRESSION FROM SCRATCH
# ─────────────────────────────────────────

class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch using:
    - Normal Equation: θ = (X^T X)^{-1} X^T y
    - Gradient Descent: θ := θ - α * ∇J(θ)
    """

    def __init__(self, method='normal_eq', learning_rate=0.01, n_iterations=1000):
        self.method = method
        self.lr     = learning_rate
        self.n_iter = n_iterations
        self.theta  = None
        self.cost_history = []

    def _add_bias(self, X):
        """Add column of ones for intercept term."""
        return np.c_[np.ones((X.shape[0], 1)), X]

    def _cost(self, X, y):
        """Mean Squared Error cost J(θ) = (1/2m) Σ(h(x) - y)²"""
        m = len(y)
        predictions = X @ self.theta
        return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

    def fit(self, X, y):
        X_b = self._add_bias(X)
        m, n = X_b.shape

        if self.method == 'normal_eq':
            # Normal Equation
            self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        elif self.method == 'gradient_descent':
            self.theta = np.zeros(n)
            self.cost_history = []
            for i in range(self.n_iter):
                gradient = (1 / m) * X_b.T @ (X_b @ self.theta - y)
                self.theta -= self.lr * gradient
                self.cost_history.append(self._cost(X_b, y))

        return self

    def predict(self, X):
        X_b = self._add_bias(X)
        return X_b @ self.theta

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot

    def rmse(self, X, y):
        return np.sqrt(np.mean((self.predict(X) - y) ** 2))


# ─────────────────────────────────────────
# SECTION 2: DATASET
# ─────────────────────────────────────────

def generate_dataset(n=300, seed=42):
    """
    Simulate house price dataset.
    price = 500 + 1200*area + 300*rooms - 100*age + noise
    """
    np.random.seed(seed)
    area   = np.random.uniform(500, 3000, n)
    rooms  = np.random.randint(1, 6, n).astype(float)
    age    = np.random.uniform(0, 50, n)
    noise  = np.random.normal(0, 15000, n)
    price  = 500 + 1200 * area + 300 * rooms - 100 * age + noise

    return pd.DataFrame({
        'area_sqft' : area,
        'rooms'     : rooms,
        'age_years' : age,
        'price_inr' : price
    })


# ─────────────────────────────────────────
# SECTION 3: MODEL EVALUATION PIPELINE
# ─────────────────────────────────────────

def evaluate_model(name, y_true, y_pred):
    """Print comprehensive evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n  ── {name} ──")
    print(f"  R² Score   : {r2:.4f}")
    print(f"  RMSE       : ₹{rmse:,.2f}")
    print(f"  MAE        : ₹{mae:,.2f}")
    print(f"  MAPE       : {mape:.2f}%")
    return {'name': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae}


# ─────────────────────────────────────────
# SECTION 4: VISUALIZATION
# ─────────────────────────────────────────

def visualize_all(df, results, gd_model, X_test, y_test, y_pred_lr):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Machine Learning — Linear & Polynomial Regression\n"
                 "BSc Mathematics Project | Python · Scikit-learn · NumPy",
                 fontsize=13, fontweight='bold')

    # ── Plot 1: Area vs Price (Simple) ──
    ax = axes[0, 0]
    ax.scatter(df['area_sqft'], df['price_inr'] / 1e6, alpha=0.3, s=10, color='steelblue')
    x_line = np.linspace(df['area_sqft'].min(), df['area_sqft'].max(), 100).reshape(-1, 1)
    lr_simple = LinearRegression().fit(df[['area_sqft']], df['price_inr'])
    ax.plot(x_line, lr_simple.predict(x_line) / 1e6, 'r-', linewidth=2, label='Linear Fit')
    ax.set_xlabel("Area (sqft)")
    ax.set_ylabel("Price (₹ Million)")
    ax.set_title("Simple Linear Regression\nArea vs Price", fontweight='bold')
    ax.legend()

    # ── Plot 2: Gradient Descent Convergence ──
    ax = axes[0, 1]
    ax.plot(gd_model.cost_history, color='tomato', linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost J(θ)")
    ax.set_title("Gradient Descent Convergence\nCost vs Iterations", fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.4)

    # ── Plot 3: Predicted vs Actual ──
    ax = axes[0, 2]
    ax.scatter(y_test / 1e6, y_pred_lr / 1e6, alpha=0.4, s=15, color='green')
    lims = [min(y_test.min(), y_pred_lr.min()) / 1e6,
            max(y_test.max(), y_pred_lr.max()) / 1e6]
    ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect Prediction')
    ax.set_xlabel("Actual Price (₹M)")
    ax.set_ylabel("Predicted Price (₹M)")
    ax.set_title("Actual vs Predicted Price", fontweight='bold')
    ax.legend()

    # ── Plot 4: Residuals ──
    ax = axes[1, 0]
    residuals = y_test - y_pred_lr
    ax.scatter(y_pred_lr / 1e6, residuals / 1000, alpha=0.4, s=15, color='purple')
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Predicted Price (₹M)")
    ax.set_ylabel("Residual (₹ thousands)")
    ax.set_title("Residual Plot\n(Random = Good Fit)", fontweight='bold')

    # ── Plot 5: Model Comparison ──
    ax = axes[1, 1]
    model_names = [r['name'] for r in results]
    r2_vals = [r['R2'] for r in results]
    colors = ['steelblue', 'tomato', 'green', 'orange', 'purple'][:len(results)]
    bars = ax.barh(model_names, r2_vals, color=colors)
    ax.set_xlabel("R² Score")
    ax.set_title("Model Comparison — R² Score", fontweight='bold')
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, r2_vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)

    # ── Plot 6: Polynomial Fit Comparison ──
    ax = axes[1, 2]
    x_1d = df['area_sqft'].values.reshape(-1, 1)
    y_1d = df['price_inr'].values
    x_plot = np.linspace(x_1d.min(), x_1d.max(), 200).reshape(-1, 1)
    ax.scatter(x_1d / 1000, y_1d / 1e6, alpha=0.2, s=8, color='gray', label='Data')
    for deg, color in zip([1, 2, 3], ['red', 'blue', 'green']):
        pipe = Pipeline([('poly', PolynomialFeatures(deg)), ('lr', LinearRegression())])
        pipe.fit(x_1d, y_1d)
        ax.plot(x_plot / 1000, pipe.predict(x_plot) / 1e6,
                color=color, linewidth=2, label=f'Degree {deg}')
    ax.set_xlabel("Area (×1000 sqft)")
    ax.set_ylabel("Price (₹M)")
    ax.set_title("Polynomial Regression Comparison", fontweight='bold')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("project7_regression.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Plot saved as 'project7_regression.png'")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 65)
    print("PROJECT 7: Machine Learning — Linear & Polynomial Regression")
    print("BSc Mathematics + CS | IGNOU | Python · Scikit-learn")
    print("=" * 65)

    df = generate_dataset(n=400)
    print(f"\n  Dataset: {df.shape}  |  Sample:")
    print(df.head(3).to_string())

    features = ['area_sqft', 'rooms', 'age_years']
    X = df[features].values
    y = df['price_inr'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("\n━━━ TRAINING MODELS ━━━")
    results = []

    # 1. Scratch — Normal Equation
    lr_scratch = LinearRegressionScratch(method='normal_eq')
    lr_scratch.fit(X_train_s, y_train)
    y_pred_scratch = lr_scratch.predict(X_test_s)
    results.append(evaluate_model("Linear Reg (Scratch - Normal Eq)", y_test, y_pred_scratch))
    print(f"  Coefficients: {np.round(lr_scratch.theta, 2)}")

    # 2. Scratch — Gradient Descent
    gd_model = LinearRegressionScratch(method='gradient_descent', learning_rate=0.1, n_iterations=500)
    gd_model.fit(X_train_s, y_train)
    y_pred_gd = gd_model.predict(X_test_s)
    results.append(evaluate_model("Linear Reg (Scratch - Gradient Descent)", y_test, y_pred_gd))

    # 3. Scikit-learn Linear Regression
    lr_sk = LinearRegression()
    lr_sk.fit(X_train_s, y_train)
    y_pred_lr = lr_sk.predict(X_test_s)
    results.append(evaluate_model("Scikit-learn LinearRegression", y_test, y_pred_lr))

    # 4. Ridge Regression
    ridge = Ridge(alpha=10)
    ridge.fit(X_train_s, y_train)
    results.append(evaluate_model("Ridge Regression (α=10)", y_test, ridge.predict(X_test_s)))

    # 5. Lasso Regression
    lasso = Lasso(alpha=100)
    lasso.fit(X_train_s, y_train)
    results.append(evaluate_model("Lasso Regression (α=100)", y_test, lasso.predict(X_test_s)))

    # 6. Polynomial Regression (degree=2)
    poly_pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    poly_pipe.fit(X_train, y_train)
    results.append(evaluate_model("Polynomial Regression (deg=2)", y_test, poly_pipe.predict(X_test)))

    # Cross-Validation
    print("\n━━━ CROSS-VALIDATION (5-fold) ━━━")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(lr_sk, X_train_s, y_train, cv=kfold, scoring='r2')
    print(f"  CV R² scores: {np.round(cv_scores, 4)}")
    print(f"  Mean R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature Importance
    print("\n━━━ FEATURE COEFFICIENTS ━━━")
    for feat, coef in zip(features, lr_sk.coef_):
        print(f"  {feat:<20} : {coef:>12.2f}")
    print(f"  Intercept: {lr_sk.intercept_:>12.2f}")

    print("\n📊 Generating visualizations...")
    visualize_all(df, results, gd_model, X_test, y_test, y_pred_lr)


if __name__ == "__main__":
    main()
