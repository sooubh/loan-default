# Architectural Plan: SHAP Explainability Integration

This document outlines the architectural and implementation plan for integrating **SHAP (SHapley Additive exPlanations)** explainability into [app.py](file:///data/data/com.termux/files/home/loan-default/app.py). The loaded model is a custom `WrappedModel` consisting of a `ColumnTransformer` preprocessor and a tree-based classifier model (like `XGBClassifier` or `RandomForestClassifier`).

---

## 1. Environment & Dependencies

To support SHAP computations and premium visualizations, we need to add the following libraries to [requirements.txt](file:///data/data/com.termux/files/home/loan-default/requirements.txt):

1. **`shap`**: Core explainability framework.
2. **`plotly`**: For interactive, premium responsive visualizations.
3. **`matplotlib`**: Required by SHAP's native waterfall/summary plots.

### Proposed Additions to `requirements.txt`
```text
shap>=0.42.0
plotly>=5.18.0
matplotlib>=3.8.0
```

---

## 2. SHAP Computation Strategy

The prediction row is a single-row pandas DataFrame. Since the model is a pipeline wrapped in `WrappedModel`, we must explain the estimator (`model.model`) using the preprocessed feature representation.

### Detailed Steps:
1. **Extract components**: Retrieve `model.model` (the estimator) and `model.preprocess` (the pipeline preprocessor).
2. **Preprocess the input row**: Transform the 1-row DataFrame using `model.preprocess.transform(row)`.
3. **Handle Sparse Matrices**: Since `OneHotEncoder` may produce sparse matrices depending on its settings, we must convert it to a dense array to avoid errors with SHAP:
   ```python
   X_preprocessed = model.preprocess.transform(row)
   if hasattr(X_preprocessed, "toarray"):
       X_preprocessed = X_preprocessed.toarray()
   ```
4. **Initialize Explainer**: Initialize `shap.TreeExplainer` on `model.model` using `@st.cache_resource` to avoid re-initializing the explainer on every rerun (which is computationally expensive).
5. **Extract predictions & base values**: Extract class 1 (default probability/log-odds) SHAP values robustly, handling variations in output shapes for `RandomForest` vs. `XGBoost`.

### Python Code for Computation
```python
import shap
import numpy as np

@st.cache_resource
def get_shap_explainer(_model):
    # Uses TreeExplainer which is optimized for XGBoost and RandomForest
    return shap.TreeExplainer(_model.model)

# Run explanation inside the prediction block
explainer = get_shap_explainer(model)
raw_shap_values = explainer.shap_values(X_preprocessed)

# Robustly extract SHAP values for class 1 (Default Risk)
if isinstance(raw_shap_values, list):
    # RandomForest: returns a list [class_0_shap, class_1_shap]
    shap_contribs = raw_shap_values[1][0]
else:
    # XGBoost: returns (num_samples, num_features) representing log-odds
    shap_contribs = raw_shap_values[0] if len(raw_shap_values.shape) == 2 else raw_shap_values

# Retrieve corresponding base value (average model prediction)
base_val = explainer.expected_value
if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1:
    base_val = base_val[1]
```

---

## 3. Feature Name Mapping & Aggregation

When the preprocessor runs `transform()`, numerical features are scaled/imputed and categorical features are one-hot encoded. The feature output size changes from $D_{\text{orig}}$ to $D_{\text{prep}}$. We must map these preprocessed feature SHAP values back to meaningful names.

### ColumnTransformer Feature Names
In scikit-learn `>=1.0`, we can retrieve the preprocessed feature names via:
```python
feature_names = model.preprocess.get_feature_names_out()
```
This generates names like `num__credit_score` or `cat__gender_Male`.

We propose two options for presenting this data:

### Option A: Clean Direct Mapping (Detailed Category Impact)
Keeps categories separated so the user sees the specific impact of the active category (e.g. `gender (Female) = 1` vs `gender (Male) = 0`).

```python
clean_names = []
feature_values = []

for name in feature_names:
    if name.startswith("num__"):
        orig_col = name.replace("num__", "")
        clean_names.append(orig_col.replace("_", " ").title())
        feature_values.append(f"{row[orig_col].values[0]:.2f}")
    elif name.startswith("cat__"):
        raw_cat = name.replace("cat__", "")
        # Find which categorical column it matches
        matched = False
        for col in cat_cols:
            if raw_cat.startswith(col + "_"):
                category_val = raw_cat[len(col)+1:]
                clean_names.append(f"{col.replace('_', ' ').title()} ({category_val})")
                is_active = "Yes" if str(row[col].values[0]) == str(category_val) else "No"
                feature_values.append(is_active)
                matched = True
                break
        if not matched:
            clean_names.append(raw_cat)
            feature_values.append("")
```

### Option B: Additive Aggregation (Grouped Original Features)
Sums the SHAP values of all one-hot encoded categories back to their parent feature. This is mathematically correct since SHAP values are additive:
$$\text{SHAP}(X_{\text{original}}) = \sum_{j \in \text{OHE}(X_{\text{original}})} \text{SHAP}(X_j)$$

```python
aggregated_shap = {}
for name, val in zip(feature_names, shap_contribs):
    if name.startswith("num__"):
        orig_col = name.replace("num__", "")
        aggregated_shap[orig_col] = aggregated_shap.get(orig_col, 0.0) + val
    elif name.startswith("cat__"):
        raw_cat = name.replace("cat__", "")
        for col in cat_cols:
            if raw_cat.startswith(col + "_"):
                aggregated_shap[col] = aggregated_shap.get(col, 0.0) + val
                break
```

---

## 4. UI Visualization Design

We recommend displaying explanations in a dedicated expander titled **"🔍 Risk Explainability (SHAP Analysis)"** below the probability spectrum.

We propose two interactive visualization designs:

### Design 1: Premium Interactive Plotly Bar Chart (Recommended)
This matches the dark-themed premium styling of the Streamlit app. It uses custom colors (crimson red for risk-increasing factors, and emerald green for risk-reducing factors).

```python
import plotly.graph_objects as go

def plot_shap_plotly(names, values, top_n=10):
    df_shap = pd.DataFrame({"Feature": names, "Impact": values})
    df_shap["AbsImpact"] = df_shap["Impact"].abs()
    
    # Sort and take top N
    df_top = df_shap.sort_values(by="AbsImpact", ascending=True).tail(top_n)
    
    colors = ["#ef4444" if val >= 0 else "#10b981" for val in df_top["Impact"]]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_top["Feature"],
        x=df_top["Impact"],
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>SHAP Impact: %{x:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="<b>Top Feature Contributions to Risk</b>",
        xaxis_title="SHAP Value (Positive increases risk, Negative reduces risk)",
        yaxis_title="Features",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=40),
        height=350 + top_n * 15,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif")
    )
    return fig
```

### Design 2: Matplotlib Waterfall Plot
Using SHAP's native waterfall visualizer.

```python
import matplotlib.pyplot as plt

def plot_shap_waterfall(values, base_value, names, data_row):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create an Explanation object
    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=data_row,
        feature_names=names
    )
    
    # Generate plot
    shap.plots.waterfall(explanation, max_display=10, show=False)
    
    # Customize matplotlib look for dark theme compatibility
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    plt.title("SHAP Waterfall Prediction Path", color="white", fontsize=14)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    
    return fig
```

---

## 5. Reference Integration Block

To integrate this plan, the following snippet can be appended directly inside the `predict_btn` block in `app.py`:

```python
# --- SHAP EXPLAINABILITY INTEGRATION ---
st.markdown("### 🔍 Risk Explainability (SHAP Analysis)")
with st.expander("Show detailed feature contributions", expanded=True):
    try:
        # 1. Preprocess row & handle sparse
        X_preprocessed = model.preprocess.transform(row)
        if hasattr(X_preprocessed, "toarray"):
            X_preprocessed = X_preprocessed.toarray()
            
        # 2. Initialize and run TreeExplainer
        explainer = shap.TreeExplainer(model.model)
        raw_shap_values = explainer.shap_values(X_preprocessed)
        
        # 3. Shape Extraction for Class 1 (Default Risk)
        if isinstance(raw_shap_values, list):
            # RandomForest
            shap_contribs = raw_shap_values[1][0]
        elif len(raw_shap_values.shape) == 3:
            # 3D Array
            shap_contribs = raw_shap_values[0, :, 1]
        else:
            # XGBoost
            shap_contribs = raw_shap_values[0] if len(raw_shap_values.shape) == 2 else raw_shap_values
            
        # 4. Feature Names Extraction
        feature_names = model.preprocess.get_feature_names_out()
        
        # 5. Map preprocessed names back to clean readable names
        clean_names = []
        for name in feature_names:
            if name.startswith("num__"):
                clean_names.append(name.replace("num__", "").replace("_", " ").title())
            elif name.startswith("cat__"):
                raw_cat = name.replace("cat__", "")
                matched = False
                for col in schema['categorical'].keys():
                    if raw_cat.startswith(col + "_"):
                        cat_val = raw_cat[len(col)+1:]
                        clean_names.append(f"{col.replace('_', ' ').title()}: {cat_val}")
                        matched = True
                        break
                if not matched:
                    clean_names.append(raw_cat.replace("_", " ").title())
            else:
                clean_names.append(name)
                
        # 6. Plot premium Plotly visualization
        fig = plot_shap_plotly(clean_names, shap_contribs, top_n=10)
        st.plotly_chart(fig, use_container_width=True)
        
        # 7. Add interpretation text
        st.markdown("""
        * **Red Bars** represent factors that increased the default risk probability.
        * **Green Bars** represent factors that decreased the default risk probability.
        * The size of the bar corresponds to the magnitude of the feature's influence.
        """)
    except Exception as shap_err:
        st.error(f"Could not compute SHAP explanations: {shap_err}")
```
