import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os
import random

# Page Configuration
st.set_page_config(
    page_title="AI Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. DATA GENERATION MODULE ---

def generate_simulated_data(n_samples: int = 5000) -> pd.DataFrame:
    """Generates synthetic network traffic data for demonstration."""
    np.random.seed(42)
    random.seed(42)
    
    data = {
        'packet_length': np.random.randint(50, 1501, n_samples),
        'duration': np.random.uniform(0.01, 10.0, n_samples),
        'protocol': np.random.choice([0, 1, 2], n_samples), # 0=TCP, 1=UDP, 2=ICMP
        'source_port': np.random.randint(1024, 65536, n_samples),
        'destination_port': np.random.choice([80, 443, 22, 21, 25, 53, 3306, 8080], n_samples),
        'packet_count': np.random.randint(1, 101, n_samples),
        'byte_count': np.random.randint(100, 100001, n_samples),
        'flow_duration': np.random.uniform(0.1, 60.0, n_samples),
    }
    
    # Simulate Flags (Simplified)
    flags_list = ['SYN', 'ACK', 'FIN', 'RST', 'SYN-ACK', 'PUSH-ACK']
    data['flags'] = [random.choice(flags_list) for _ in range(n_samples)]
    
    df = pd.DataFrame(data)
    
    # Encode flags numerically for the model
    df['flags_encoded'] = df['flags'].astype('category').cat.codes
    
    # Labeling (15% Malicious)
    df['Label'] = 0
    malicious_indices = np.random.choice(df.index, size=int(0.15 * n_samples), replace=False)
    df.loc[malicious_indices, 'Label'] = 1
    
    # Introduce some correlations for "malicious" traffic to make it learnable
    # For example, malicious traffic might have high packet_count or specific ports
    df.loc[df['Label'] == 1, 'packet_count'] = np.random.randint(80, 200, len(malicious_indices))
    df.loc[df['Label'] == 1, 'duration'] = np.random.uniform(5.0, 15.0, len(malicious_indices))
    
    return df

# --- 2. DATA LOADING MODULE ---

def load_data() -> pd.DataFrame:
    """Attempts to load CIC-IDS2017 dataset or falls back to simulated data."""
    csv_path = "network_traffic_data.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            # Basic cleanup for CIC-IDS2017 style datasets
            df.columns = df.columns.str.strip()
            if 'Label' in df.columns:
                df['Label'] = df['Label'].apply(lambda x: 1 if str(x).lower() != 'benign' else 0)
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
    
    return generate_simulated_data()

# --- 3. MACHINE LEARNING MODULE ---

def prepare_features(df: pd.DataFrame):
    """Separates features (X) from labels (y)."""
    # Drop non-numeric or irrelevant columns for training
    X = df.drop(['Label', 'flags'] if 'flags' in df.columns else ['Label'], axis=1)
    y = df['Label']
    return X, y

def train_model(X, y):
    """Trains a Random Forest classifier."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    
    with st.spinner("Training Random Forest Model..."):
        model.fit(X_train, y_train)
    
    metrics = evaluate_model(model, X_test, y_test)
    return model, (X_train, X_test, y_train, y_test), metrics

def evaluate_model(model, X_test, y_test) -> dict:
    """Calculates model performance metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'report': classification_report(y_test, y_pred, output_dict=True),
        'conf_matrix': confusion_matrix(y_test, y_pred)
    }
    return metrics

# --- 4. STREAMLIT DASHBOARD COMPONENTS ---

def display_sidebar(df, model_state):
    st.sidebar.title("⚙️ Controls")
    
    if st.sidebar.button("🚀 Train Model Now", help="Click to train the Random Forest classifier"):
        X, y = prepare_features(df)
        model, data_split, metrics = train_model(X, y)
        st.session_state['nids_model'] = model
        st.session_state['nids_metrics'] = metrics
        st.session_state['nids_features'] = X.columns.tolist()
        st.sidebar.success("Model Trained!")

    if 'nids_model' in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Training Metrics")
        m = st.session_state['nids_metrics']
        st.sidebar.metric("Accuracy", f"{m['accuracy']:.2%}")
        st.sidebar.metric("Precision", f"{m['precision']:.2%}")
        st.sidebar.metric("Recall", f"{m['recall']:.2%}")
        st.sidebar.metric("F1-Score", f"{m['f1']:.2%}")
        
        # Feature Importance
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔝 Feature Importance")
        importances = st.session_state['nids_model'].feature_importances_
        feat_df = pd.DataFrame({
            'Feature': st.session_state['nids_features'],
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feat_df, hue='Feature', palette='viridis', ax=ax, legend=False)
        st.sidebar.pyplot(fig)
    else:
        st.sidebar.warning("⚠️ Model not trained yet.")

def tab_traffic_simulator():
    st.header("🕵️ Live Traffic Simulator")
    st.info("Input network packet details to analyze if the traffic is suspicious.")
    
    if 'nids_model' not in st.session_state:
        st.error("Please train the model first using the sidebar button.")
        return

    col1, col2 = st.columns(2)
    
    with col1:
        packet_len = st.slider("Packet Length (bytes)", 50, 1500, 500)
        duration = st.slider("Duration (seconds)", 0.01, 10.0, 1.0)
        protocol = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"])
        protocol_map = {"TCP": 0, "UDP": 1, "ICMP": 2}
        
    with col2:
        src_port = st.number_input("Source Port", 1024, 65535, 49152)
        dst_port = st.number_input("Destination Port", 1, 65535, 80)
        packet_count = st.slider("Packet Count", 1, 100, 10)
        byte_count = st.slider("Byte Count", 100, 100000, 5000)
    
    # Use fixed values for other fields used in training but not in manual input
    # In a real scenario, we'd need flow_duration and flags_encoded as well
    flow_duration = st.slider("Flow Duration", 0.1, 60.0, 5.0)
    
    # Create input data (must match features used during training)
    # nids_features: ['packet_length', 'duration', 'protocol', 'source_port', 'destination_port', 'packet_count', 'byte_count', 'flow_duration', 'flags_encoded']
    input_data = pd.DataFrame([{
        'packet_length': packet_len,
        'duration': duration,
        'protocol': protocol_map[protocol],
        'source_port': src_port,
        'destination_port': dst_port,
        'packet_count': packet_count,
        'byte_count': byte_count,
        'flow_duration': flow_duration,
        'flags_encoded': 0 # Defaulting to first flag category
    }])

    if st.button("🔍 Analyze Traffic"):
        prediction = st.session_state['nids_model'].predict(input_data)[0]
        prob = st.session_state['nids_model'].predict_proba(input_data)[0]
        
        st.markdown("---")
        if prediction == 1:
            st.error(f"### Malicious Traffic Detected! ⚠️ \nConfidence: {prob[1]:.2%}")
        else:
            st.success(f"### Normal Traffic ✓ \nConfidence: {prob[0]:.2%}")

def tab_dataset_overview(df):
    st.header("📊 Dataset Overview")
    
    st.subheader("Data Preview (First 100 samples)")
    st.dataframe(df.head(100), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Dataset Shape", df.shape)
        st.write("#### Statistics", df.describe())
        
    with col2:
        st.write("#### Class Distribution")
        class_counts = df['Label'].value_counts()
        labels = ['Normal', 'Malicious']
        fig, ax = plt.subplots()
        ax.pie(class_counts, labels=labels, autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Distributions")
    feature_to_plot = st.selectbox("Select Feature to Visualize Distribution", 
                                  ['packet_length', 'duration', 'packet_count', 'byte_count'])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(data=df, x=feature_to_plot, hue='Label', multiple="stack", palette={0: "#2ecc71", 1: "#e74c3c"}, ax=ax)
    st.pyplot(fig)

def tab_performance():
    st.header("📈 Model Performance Details")
    
    if 'nids_metrics' not in st.session_state:
        st.warning("Train the model to see performance results.")
        return
        
    m = st.session_state['nids_metrics']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(m['conf_matrix'], annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Malicious'], yticklabels=['Normal', 'Malicious'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
        
    with col2:
        st.subheader("Classification Report")
        report_df = pd.DataFrame(m['report']).transpose()
        st.table(report_df)

def tab_about():
    st.header("📖 About the Project")
    st.markdown("""
    ### AI-Based Network Intrusion Detection System
    This project demonstrates how Machine Learning (specifically **Random Forest**) can be used to identify suspicious network activity.
    
    #### Technical Stack
    - **Frontend:** Streamlit
    - **Machine Learning:** Scikit-Learn
    - **Data Handling:** Pandas, Numpy
    - **Visualization:** Seaborn, Matplotlib
    
    #### How it works:
    1. **Data Loading:** The system loads a simulated dataset of network traffic including features like packet length, duration, and protocol.
    2. **Training:** A Random Forest classifier is trained on the data to find patterns associated with 'Malicious' vs 'Normal' traffic.
    3. **Detection:** Users can manually input traffic parameters in the simulator to see if the model flags them as threats.
    
    #### Author
    Developed by Antigravity (Expert AI Coding Assistant)
    """)

# --- 5. MAIN EXECUTION FLOW ---

def main():
    st.title("🛡️ AI-Powered Network Intrusion Detection System")
    st.markdown("""
    Monitor and detect network threats using advanced Machine Learning. 
    *Simulate traffic patterns, train classifiers, and analyze performance in real-time.*
    """)
    st.markdown("---")

    # Load data
    if 'nids_data' not in st.session_state:
        st.session_state['nids_data'] = load_data()
    
    df = st.session_state['nids_data']
    
    # Sidebar
    display_sidebar(df, st.session_state)
    
    # Main Tabs
    tabs = st.tabs(["🕵️ Live Traffic Simulator", "📊 Dataset Overview", "📈 Model Performance", "📖 About"])
    
    with tabs[0]:
        tab_traffic_simulator()
    
    with tabs[1]:
        tab_dataset_overview(df)
        
    with tabs[2]:
        tab_performance()
        
    with tabs[3]:
        tab_about()

if __name__ == "__main__":
    main()
