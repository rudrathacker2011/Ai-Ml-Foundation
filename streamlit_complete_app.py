# ========== COMPLETE STREAMLIT ML DASHBOARD ==========
# Save this as: streamlit_complete_app.py
# Run with: streamlit run streamlit_complete_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            mean_squared_error, mean_absolute_error, r2_score)
import pickle
from io import StringIO
import time

# ========== PAGE CONFIG (MUST BE FIRST) ==========
st.set_page_config(
    page_title="ML Dashboard Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS STYLING ==========
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ========== CACHING FUNCTIONS ==========
@st.cache_data
def load_sample_data(dataset_name):
    """Load sample datasets"""
    if dataset_name == "Iris (Classification)":
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif dataset_name == "Wine (Classification)":
        from sklearn.datasets import load_wine
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif dataset_name == "Boston Housing (Regression)":
        # Create synthetic housing data
        np.random.seed(42)
        n_samples = 500
        df = pd.DataFrame({
            'rooms': np.random.uniform(3, 10, n_samples),
            'age': np.random.uniform(1, 100, n_samples),
            'distance': np.random.uniform(1, 12, n_samples),
            'price': np.random.uniform(100000, 800000, n_samples)
        })
        return df
    else:
        return None

@st.cache_data
def generate_synthetic_data(n_samples, n_features, task_type):
    """Generate synthetic data for testing"""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    
    if task_type == "Classification":
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
    else:  # Regression
        y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(n_samples) * 0.1
    
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

# ========== SIDEBAR NAVIGATION ==========
st.sidebar.markdown("# 🎯 Navigation")
page = st.sidebar.radio(
    "Choose a page:",
    ["🏠 Home", "📊 Data Explorer", "🤖 Model Training", "🔮 Predictions", "📈 Visualizations"]
)

st.sidebar.markdown("---")

# ========== SESSION STATE INITIALIZATION ==========
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = []
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = "Classification"

# ========== PAGE 1: HOME ==========
if page == "🏠 Home":
    st.markdown('<div class="main-header">🤖 ML Dashboard Pro</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Complete Machine Learning Dashboard!
    
    This interactive app allows you to:
    - 📊 **Explore your data** with advanced statistics and visualizations
    - 🤖 **Train ML models** with multiple algorithms
    - 🔮 **Make predictions** on new data
    - 📈 **Visualize results** with professional charts
    - 💾 **Save and load models** for later use
    
    ---
    
    #### 🚀 Quick Start Guide:
    1. **Upload your data** or use sample datasets in the Data Explorer
    2. **Train a model** in the Model Training page
    3. **Make predictions** on new data
    4. **Visualize** your results
    
    ---
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**📊 Data Explorer**\n\nUpload CSV files, view statistics, and explore your data")
    
    with col2:
        st.success("**🤖 Model Training**\n\nTrain multiple ML models and compare performance")
    
    with col3:
        st.warning("**🔮 Predictions**\n\nMake predictions on new data using trained models")
    
    st.markdown("---")
    
    # Quick stats if data is loaded
    if st.session_state.df is not None:
        st.success("✅ Data is loaded and ready!")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Rows", st.session_state.df.shape[0])
        with col2:
            st.metric("Columns", st.session_state.df.shape[1])
        with col3:
            st.metric("Features", len(st.session_state.feature_cols) if st.session_state.feature_cols else 0)
        with col4:
            st.metric("Model Status", "Trained ✅" if st.session_state.model else "Not Trained ❌")
    else:
        st.info("👆 No data loaded yet. Go to Data Explorer to get started!")

# ========== PAGE 2: DATA EXPLORER ==========
elif page == "📊 Data Explorer":
    st.markdown('<div class="main-header">📊 Data Explorer</div>', unsafe_allow_html=True)
    
    # Data source selection
    st.markdown("### 📁 Choose Data Source")
    
    data_source = st.radio(
        "Select data source:",
        ["Upload CSV File", "Use Sample Dataset", "Generate Synthetic Data"],
        horizontal=True
    )
    
    df = None
    
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
    
    elif data_source == "Use Sample Dataset":
        dataset_choice = st.selectbox(
            "Choose a sample dataset:",
            ["Iris (Classification)", "Wine (Classification)", "Boston Housing (Regression)"]
        )
        
        if st.button("Load Sample Dataset"):
            df = load_sample_data(dataset_choice)
            st.session_state.df = df
            
            # Auto-detect task type
            if "Regression" in dataset_choice:
                st.session_state.task_type = "Regression"
            else:
                st.session_state.task_type = "Classification"
            
            st.success(f"✅ Loaded {dataset_choice}! Shape: {df.shape}")
    
    else:  # Generate Synthetic Data
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_samples = st.number_input("Number of samples:", 100, 10000, 500, 100)
        with col2:
            n_features = st.number_input("Number of features:", 2, 20, 5, 1)
        with col3:
            task_type = st.selectbox("Task type:", ["Classification", "Regression"])
        
        if st.button("Generate Data"):
            df = generate_synthetic_data(n_samples, n_features, task_type)
            st.session_state.df = df
            st.session_state.task_type = task_type
            st.success(f"✅ Generated {task_type} data! Shape: {df.shape}")
    
    # Display data if loaded
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.markdown("---")
        st.markdown("### 📋 Data Preview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            n_rows = st.slider("Number of rows to display:", 5, 100, 10)
            st.dataframe(df.head(n_rows), use_container_width=True)
        
        with col2:
            st.markdown("**Dataset Info**")
            st.write(f"**Rows:** {df.shape[0]}")
            st.write(f"**Columns:** {df.shape[1]}")
            st.write(f"**Memory:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
            st.write(f"**Missing values:** {df.isnull().sum().sum()}")
        
        st.markdown("---")
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🔍 Column Information")
        
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(col_info, use_container_width=True)
        
        # Missing data visualization
        if df.isnull().sum().sum() > 0:
            st.markdown("### 🚨 Missing Data")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            
            ax.barh(missing.index, missing.values, color='red', alpha=0.7)
            ax.set_xlabel('Number of Missing Values')
            ax.set_title('Missing Values by Column')
            ax.grid(axis='x', alpha=0.3)
            
            st.pyplot(fig)
            plt.close()

# ========== PAGE 3: MODEL TRAINING ==========
elif page == "🤖 Model Training":
    st.markdown('<div class="main-header">🤖 Model Training</div>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.error("❌ No data loaded! Please go to Data Explorer first.")
    else:
        df = st.session_state.df
        
        st.markdown("### 🎯 Configure Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Select Target Column**")
            columns = df.columns.tolist()
            target_col = st.selectbox("Target (what to predict):", columns, 
                                     index=len(columns)-1)  # Default to last column
            st.session_state.target_col = target_col
        
        with col2:
            st.markdown("**Select Feature Columns**")
            available_features = [col for col in columns if col != target_col]
            feature_cols = st.multiselect("Features (input data):", 
                                         available_features,
                                         default=available_features)
            st.session_state.feature_cols = feature_cols
        
        if len(feature_cols) > 0 and target_col:
            # Task type detection
            unique_targets = df[target_col].nunique()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if unique_targets <= 10:
                    task_type = st.radio("Task type:", ["Classification", "Regression"])
                else:
                    task_type = "Regression"
                    st.info("Auto-detected: Regression (many unique values)")
                
                st.session_state.task_type = task_type
            
            with col2:
                test_size = st.slider("Test size (%):", 10, 50, 20) / 100
            
            with col3:
                use_scaling = st.checkbox("Use feature scaling", value=True)
            
            st.markdown("---")
            st.markdown("### 🔧 Model Selection")
            
            if task_type == "Classification":
                model_choice = st.selectbox(
                    "Choose algorithm:",
                    ["Logistic Regression", "Random Forest", "Decision Tree", 
                     "K-Nearest Neighbors", "Support Vector Machine"]
                )
                
                # Model-specific parameters
                if model_choice == "Random Forest":
                    n_estimators = st.slider("Number of trees:", 10, 200, 100, 10)
                elif model_choice == "K-Nearest Neighbors":
                    n_neighbors = st.slider("Number of neighbors:", 1, 20, 5)
            
            else:  # Regression
                model_choice = st.selectbox(
                    "Choose algorithm:",
                    ["Linear Regression", "Random Forest Regressor"]
                )
                
                if model_choice == "Random Forest Regressor":
                    n_estimators = st.slider("Number of trees:", 10, 200, 100, 10)
            
            st.markdown("---")
            
            # Train button
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model... Please wait..."):
                    progress_bar = st.progress(0)
                    
                    # Prepare data
                    progress_bar.progress(20)
                    X = df[feature_cols].values
                    y = df[target_col].values
                    
                    # Split data
                    progress_bar.progress(40)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Store in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    # Feature scaling
                    if use_scaling:
                        progress_bar.progress(50)
                        scaler = StandardScaler()
                        scaler.fit(X_train)
                        X_train = scaler.transform(X_train)
                        X_test = scaler.transform(X_test)
                        st.session_state.scaler = scaler
                    else:
                        st.session_state.scaler = None
                    
                    # Create model
                    progress_bar.progress(60)
                    
                    if task_type == "Classification":
                        if model_choice == "Logistic Regression":
                            model = LogisticRegression(random_state=42, max_iter=1000)
                        elif model_choice == "Random Forest":
                            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                        elif model_choice == "Decision Tree":
                            model = DecisionTreeClassifier(random_state=42)
                        elif model_choice == "K-Nearest Neighbors":
                            model = KNeighborsClassifier(n_neighbors=n_neighbors)
                        elif model_choice == "Support Vector Machine":
                            model = SVC(random_state=42)
                    else:
                        if model_choice == "Linear Regression":
                            model = LinearRegression()
                        elif model_choice == "Random Forest Regressor":
                            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                    
                    # Train
                    progress_bar.progress(80)
                    model.fit(X_train, y_train)
                    st.session_state.model = model
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    
                    progress_bar.progress(100)
                    time.sleep(0.5)
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 📊 Model Performance")
                    
                    if task_type == "Classification":
                        # Metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Accuracy", f"{accuracy * 100:.2f}%")
                        with col2:
                            st.metric("Precision", f"{precision * 100:.2f}%")
                        with col3:
                            st.metric("Recall", f"{recall * 100:.2f}%")
                        with col4:
                            st.metric("F1-Score", f"{f1 * 100:.2f}%")
                        
                        # Confusion Matrix
                        st.markdown("#### Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                   cbar_kws={'label': 'Count'}, ax=ax)
                        ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
                        ax.set_xlabel('Predicted Label', fontsize=12)
                        ax.set_ylabel('True Label', fontsize=12)
                        
                        st.pyplot(fig)
                        plt.close()
                        
                        # Classification Report
                        st.markdown("#### Detailed Classification Report")
                        report = classification_report(y_test, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                    
                    else:  # Regression
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("R² Score", f"{r2:.4f}")
                        with col2:
                            st.metric("RMSE", f"{rmse:.4f}")
                        with col3:
                            st.metric("MAE", f"{mae:.4f}")
                        with col4:
                            st.metric("MSE", f"{mse:.4f}")
                        
                        # Actual vs Predicted
                        st.markdown("#### Actual vs Predicted")
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
                        
                        # Perfect prediction line
                        min_val = min(y_test.min(), y_pred.min())
                        max_val = max(y_test.max(), y_pred.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 
                               'r--', linewidth=2, label='Perfect Prediction')
                        
                        ax.set_xlabel('Actual Values', fontsize=12)
                        ax.set_ylabel('Predicted Values', fontsize=12)
                        ax.set_title('Actual vs Predicted Values', fontweight='bold', fontsize=14)
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        plt.close()
                    
                    # Feature Importance (if available)
                    if hasattr(model, 'feature_importances_'):
                        st.markdown("#### Feature Importance")
                        
                        importance = model.feature_importances_
                        importance_df = pd.DataFrame({
                            'Feature': feature_cols,
                            'Importance': importance
                        }).sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(importance_df['Feature'], importance_df['Importance'], 
                               color='green', alpha=0.7, edgecolor='black')
                        ax.set_xlabel('Importance', fontsize=12)
                        ax.set_title('Feature Importance', fontweight='bold', fontsize=14)
                        ax.grid(axis='x', alpha=0.3)
                        
                        st.pyplot(fig)
                        plt.close()
                    
                    # Save model option
                    st.markdown("---")
                    st.markdown("### 💾 Save Model")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Download Model (.pkl)"):
                            model_bytes = pickle.dumps(model)
                            st.download_button(
                                label="📥 Download Model",
                                data=model_bytes,
                                file_name="trained_model.pkl",
                                mime="application/octet-stream"
                            )
                    
                    with col2:
                        if st.session_state.scaler and st.button("Download Scaler (.pkl)"):
                            scaler_bytes = pickle.dumps(st.session_state.scaler)
                            st.download_button(
                                label="📥 Download Scaler",
                                data=scaler_bytes,
                                file_name="scaler.pkl",
                                mime="application/octet-stream"
                            )

# ========== PAGE 4: PREDICTIONS ==========
elif page == "🔮 Predictions":
    st.markdown('<div class="main-header">🔮 Make Predictions</div>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.error("❌ No model trained yet! Please go to Model Training first.")
    else:
        st.success("✅ Model is ready for predictions!")
        
        st.markdown("### 🎯 Input Method")
        
        input_method = st.radio(
            "Choose input method:",
            ["Manual Input", "Upload CSV"],
            horizontal=True
        )
        
        if input_method == "Manual Input":
            st.markdown("#### Enter feature values:")
            
            input_data = {}
            
            cols = st.columns(3)
            for i, feature in enumerate(st.session_state.feature_cols):
                with cols[i % 3]:
                    input_data[feature] = st.number_input(
                        f"{feature}:", 
                        value=0.0,
                        step=0.1,
                        format="%.4f"
                    )
            
            if st.button("🔮 Predict", type="primary"):
                # Prepare input
                X_input = np.array([list(input_data.values())])
                
                # Scale if needed
                if st.session_state.scaler:
                    X_input = st.session_state.scaler.transform(X_input)
                
                # Predict
                prediction = st.session_state.model.predict(X_input)[0]
                
                # Display prediction
                st.markdown("---")
                st.markdown("### 🎯 Prediction Result")
                
                if st.session_state.task_type == "Classification":
                    st.success(f"**Predicted Class:** {prediction}")
                    
                    # Probability if available
                    if hasattr(st.session_state.model, 'predict_proba'):
                        proba = st.session_state.model.predict_proba(X_input)[0]
                        
                        st.markdown("#### Class Probabilities")
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        classes = range(len(proba))
                        ax.bar(classes, proba, color='blue', alpha=0.7, edgecolor='black')
                        ax.set_xlabel('Class', fontsize=12)
                        ax.set_ylabel('Probability', fontsize=12)
                        ax.set_title('Prediction Probabilities', fontweight='bold')
                        ax.set_ylim(0, 1)
                        ax.grid(axis='y', alpha=0.3)
                        
                        st.pyplot(fig)
                        plt.close()
                else:
                    st.success(f"**Predicted Value:** {prediction:.4f}")
        
        else:  # Upload CSV
            st.markdown("#### Upload CSV file with feature values")
            
            pred_file = st.file_uploader("Upload CSV for batch predictions", type=['csv'])
            
            if pred_file is not None:
                pred_df = pd.read_csv(pred_file)
                
                st.markdown("**Input Data Preview:**")
                st.dataframe(pred_df.head(), use_container_width=True)
                
                if st.button("🔮 Predict All", type="primary"):
                    # Prepare data
                    X_pred = pred_df[st.session_state.feature_cols].values
                    
                    # Scale if needed
                    if st.session_state.scaler:
                        X_pred = st.session_state.scaler.transform(X_pred)
                    
                    # Predict
                    predictions = st.session_state.model.predict(X_pred)
                    
                    # Add predictions to dataframe
                    pred_df['Prediction'] = predictions
                    
                    st.markdown("---")
                    st.markdown("### 📊 Prediction Results")
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Download results
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions (CSV)",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

# ========== PAGE 5: VISUALIZATIONS ==========
elif page == "📈 Visualizations":
    st.markdown('<div class="main-header">📈 Data Visualizations</div>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.error("❌ No data loaded! Please go to Data Explorer first.")
    else:
        df = st.session_state.df
        
        st.markdown("### 📊 Visualization Options")
        
        viz_type = st.selectbox(
            "Choose visualization type:",
            ["Distribution Plot", "Correlation Heatmap", "Scatter Plot", 
             "Box Plot", "Pair Plot (Sample)", "Feature Comparison"]
        )
        
        if viz_type == "Distribution Plot":
            col = st.selectbox("Select column:", df.columns)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if df[col].dtype in ['int64', 'float64']:
                ax.hist(df[col].dropna(), bins=30, color='skyblue', 
                       edgecolor='black', alpha=0.7)
                ax.axvline(df[col].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean = {df[col].mean():.2f}')
                ax.axvline(df[col].median(), color='green', linestyle='--', 
                          linewidth=2, label=f'Median = {df[col].median():.2f}')
                ax.legend()
            else:
                df[col].value_counts().plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            
            ax.set_title(f'Distribution of {col}', fontweight='bold', fontsize=14)
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        
        elif viz_type == "Correlation Heatmap":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0,
                           fmt='.2f', square=True, linewidths=0.5, ax=ax,
                           cbar_kws={"shrink": 0.8})
                ax.set_title('Correlation Heatmap', fontweight='bold', fontsize=16)
                
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Not enough numeric columns for correlation analysis")
        
        elif viz_type == "Scatter Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("X-axis:", numeric_cols)
            with col2:
                y_col = st.selectbox("Y-axis:", [c for c in numeric_cols if c != x_col])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df[x_col], df[y_col], alpha=0.6, s=50, 
                      edgecolors='black', linewidths=0.5)
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(f'{y_col} vs {x_col}', fontweight='bold', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        
        elif viz_type == "Box Plot":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_cols = st.multiselect("Select columns:", numeric_cols, 
                                          default=numeric_cols[:min(5, len(numeric_cols))])
            
            if selected_cols:
                fig, ax = plt.subplots(figsize=(12, 6))
                df[selected_cols].boxplot(ax=ax, patch_artist=True)
                ax.set_title('Box Plot Comparison', fontweight='bold', fontsize=14)
                ax.set_ylabel('Values', fontsize=12)
                ax.grid(axis='y', alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                
                st.pyplot(fig)
                plt.close()
        
        elif viz_type == "Pair Plot (Sample)":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:4]
            
            if len(numeric_cols) >= 2:
                sample_df = df[numeric_cols].sample(min(100, len(df)))
                
                fig = plt.figure(figsize=(12, 12))
                pd.plotting.scatter_matrix(sample_df, alpha=0.6, figsize=(12, 12), 
                                         diagonal='hist', edgecolors='black', linewidths=0.5)
                plt.suptitle('Pair Plot (Sample)', fontweight='bold', fontsize=16, y=0.995)
                
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Not enough numeric columns for pair plot")
        
        elif viz_type == "Feature Comparison":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_features = st.multiselect("Select features to compare:", 
                                              numeric_cols,
                                              default=numeric_cols[:min(3, len(numeric_cols))])
            
            if selected_features:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                df[selected_features].plot(kind='box', ax=ax, patch_artist=True)
                ax.set_title('Feature Comparison', fontweight='bold', fontsize=14)
                ax.set_ylabel('Normalized Values', fontsize=12)
                ax.grid(axis='y', alpha=0.3)
                
                st.pyplot(fig)
                plt.close()

# ========== FOOTER ==========
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 About
**ML Dashboard Pro v1.0**

Complete machine learning platform for:
- Data exploration
- Model training
- Predictions
- Visualizations

Built with Streamlit, scikit-learn, and Pandas.

---
**Quick Tips:**
- Start with Data Explorer
- Train models with your data
- Make predictions instantly
- Visualize everything!

---
💡 **Need help?** Check the Home page for guides.
""")

# Session info in sidebar
if st.session_state.df is not None:
    st.sidebar.success(f"✅ Data loaded: {st.session_state.df.shape[0]} rows")
else:
    st.sidebar.info("ℹ️ No data loaded")

if st.session_state.model is not None:
    st.sidebar.success("✅ Model trained")
else:
    st.sidebar.info("ℹ️ No model trained")