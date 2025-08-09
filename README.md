# Pinterest AI Scorer

This tool predicts the likelihood of a Pinterest ad campaign being a WINNER based on past campaign metrics.  
It includes both a command-line script and a web interface.

---

## Requirements
- Python 3.9+
- pip (Python package manager)
- Git (if cloning from GitHub)

---

## Installation

1. Clone the repository
git clone https://github.com/armaanatkar-cmyk/pinterest_ai.git
cd pinterest_ai

2. After allat install this shi

pip install -r requirements.txt

3. run it
python predict.py --model pinterest_baseline_logreg.pkl --in sample_input.csv --out predictions.csv --threshold 0.5

4. u can also run this john in web browser with dis

streamlit run streamlit_app.py



