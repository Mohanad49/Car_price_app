# 🚗 Used Car Price Predictor

A modern web application that predicts the price of used cars based on various specifications, built with Streamlit.

## 🚀 Live Demo

Try it now: [carpricegenie.streamlit.app](https://carpricegenie.streamlit.app)

![App Screenshot](https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?q=80&w=400&auto=format&fit=crop&ixlib=rb-4.0.3)

---

## Features

- **Price Prediction**: Get estimated market prices for used cars based on various specifications
- **Multiple Currency Support**: View predictions in USD, EUR, GBP, JPY, CAD, AUD, or EGP
- **Modern UI**: Sleek design with animations, transitions, and responsive layout
- **Interactive Elements**: User-friendly input controls and visual feedback

---

## Optional: Local Installation

If you want to run the app locally, follow these steps:

1. Clone this repository:
   ```
   git clone <repository-url>
   cd Car_price_app
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

5. Open your web browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

---

## Model Details

The prediction model is a Random Forest regressor trained on US used car data. The model takes into account various factors such as:

- Car age and mileage
- Engine specifications (displacement, cylinders, type)
- Body type and dimensions
- Fuel economy
- Vehicle condition (accidents, frame damage, etc.)
- And more...

---

## Requirements

- Python 3.8+
- Streamlit 1.45.1+
- pandas 2.2.3+
- numpy 1.26.4+
- joblib 1.4.2+
- scikit-learn 1.6.1+
- requests 2.32.3+

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
