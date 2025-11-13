## Guidelines to run this repository

### Environment Setup

#### 1. Create virtual environment
```
python -m venv venv
venv\Scripts\activate     # Windows
source venv\bin\activate  # MacOS
```
#### 2. Clone this repository
```
git clone https://github.com/jerrryw/NLP_Homework3.git
```

#### 3. Install required packages
```
pip install -r requirements.txt
```

#### 4. In the directory, run this command
```
python -m src.main
```
Note: no need to cd into src folder to run main.py.

### Expected runtime and output files

The expected runtime with CUDA acceleration is around 1 hour. A smaller example to run the code is presented in main.py on line 48-53.

You can choose to comment the full example (line 42-46) and uncomment the smaller example (line 48-53) to complete a smaller simulation that can be done within 3 minutes. However, plots may not be fully accurate. Need to check metrics.csv for more details.

## Folder Structure
```
├── data/
│   ├── IMDB Dataset.csv
├── results/
│   └── plots/
|       └── BiLSTM_accuracy_f1_vs_seq.png
|       └── loss_curve_best.png
|       └── loss_curve_worst.png
|       └── LSTM_accuracy_f1_vs_seq.png
|       └── optimizer_performance_vs_seq.png
|       └── RNN_accuracy_f1_vs_seq.png
│   ├── metrics.csv
├── src/
│   ├── evaluate.py
│   ├── main.py
│   ├── models.py
│   ├── preprocess.py
│   ├── train.py
│   └── utils.py
├── report.pdf
└── README.md
├── requirements.txt
```