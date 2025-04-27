Distributed IDS with Apache Spark

    A distributed Intrusion Detection System leveraging Apache Spark and Random Forests, trained on the NSL-KDD dataset.

🚀 Project Overview

This project demonstrates a scalable Intrusion Detection System (IDS) prototype built using Apache Spark. It processes network traffic data from the NSL-KDD dataset to detect potential threats in real-time, utilizing a Random Forest classifier for robust anomaly detection.
🛠️ Features

    Distributed training using Apache Spark's MLlib

    Efficient data preprocessing and feature engineering

    Random Forest classification model

    Evaluation metrics including F1-score

    Easily extensible for other datasets or models

📚 Dataset

    NSL-KDD: A refined version of the classic KDD'99 dataset used for network intrusion detection research.

    Includes:

        KDDTrain+.txt

        KDDTest+.txt

    Stored in the /data directory.

🏗️ Project Structure

Distributed-IDS-Apache-Spark/
├── data/               # NSL-KDD dataset files
├── src/
│   └── main.py         # Main application script
├── requirements.txt    # Python dependencies
└── IDS-1.pdf           # Project documentation/report

⚙️ Installation

    Clone the repository:

git clone https://github.com/UsEr-1337-1/Distributed-IDS-Apache-Spark.git
cd Distributed-IDS-Apache-Spark

    Create a virtual environment (optional but recommended):

python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

    Install the required dependencies:

pip install -r requirements.txt

    Note: You must have Apache Spark installed. If not, follow this guide to set it up.

🚀 Running the Project

Simply execute:

python src/main.py

The script will:

    Start a Spark session

    Load and preprocess the data

    Train a Random Forest classifier

    Evaluate the model and display the F1 score

📈 Example Output

F1-score on test set: 0.875

📋 Requirements

    Python 3.7+

    Apache Spark 3.0+

    PySpark

    NumPy
