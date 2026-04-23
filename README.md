FairGuard AI is a specialized platform designed to audit and mitigate algorithmic bias in machine learning models. 
The application provides a comprehensive suite of tools for data ingestion, bias detection across protected attributes, and the application of fairness-enhancing interventions like reweighing.
The system is built as a multi-page Streamlit application integrating industry-standard fairness libraries such as holisticai and fairlearn to provide both quantitative metrics and qualitative visualizations.

Core Mission: 
The primary goal of FairGuard AI is to bridge the gap between raw data and ethical AI deployment. It allows developers and auditors to:

Detect: Identify disparate impact and statistical parity differences in model predictions.
Measure: Quantify bias using standardized metrics like Equalized Odds and Demographic Parity.
Fix: Apply pre-processing mitigation techniques to balance datasets before final model training.

System Architecture: 
The application follows a modular workflow where data flows from ingestion through a transformation pipeline into a session-persistent state.

High-Level Data Flow: 
This diagram illustrates how data moves from the user interface through the core analysis logic.

<img width="751" height="832" alt="image" src="https://github.com/user-attachments/assets/15e43643-f6a7-4a02-a580-0f3e0e3e3a8f" />

Functional Modules
1. Data Ingestion: 
The system supports two primary entry points for data:
Automated Fetching: Retrieves the UCI Adult Income dataset directly from the UCI Machine Learning Repository 
Custom Upload: Allows users to provide their own CSV files, which are then cleaned and stored in the global session state.

3. Bias Scanning: 
The Bias Scan module performs the heavy lifting of auditing. It uses LabelEncoder to prepare categorical features 
and trains a baseline LogisticRegression model .
It then calculates bias metrics for a selected protected attribute, such as sex or race 
For details, see Fairness Concepts & Bias Metrics.

3. Mitigation Strategies: 
When significant bias is detected, the Mitigation module utilizes the Reweighing algorithm from the holisticai library 
This algorithm adjusts the weights of training samples to ensure the model learns a more equitable representation of different groups 
For details, see Reweighing Mitigation Algorithm.

4. Reporting and XAI: 
The project includes a technology stack for generating comprehensive reports, including reportlab for PDF generation and groq for LLM-based narrative summaries 
