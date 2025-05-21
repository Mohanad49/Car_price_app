import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="US Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# --- Load Model, Preprocessor, and Original Column Names ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('rf_model.joblib')
        preprocessor = joblib.load('preprocessor.joblib')
        original_cols = joblib.load('original_feature_columns.joblib')
        print("Resources loaded successfully.")
        # print(f"Original columns expected by preprocessor: {original_cols}")
        return model, preprocessor, original_cols
    except FileNotFoundError as e:
        st.error(f"Error loading model/preprocessor/columns files: {e}. Ensure 'rf_model.joblib', 'preprocessor.joblib', and 'original_feature_columns.joblib' are in the same directory as app.py.")
        return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading resources: {e}")
        return None, None, None

model, preprocessor, original_feature_columns = load_resources()

# --- Define Currency Conversion Function ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_exchange_rates():
    try:
        # Using ExchangeRate-API for demo purposes
        response = requests.get("https://open.er-api.com/v6/latest/USD")
        data = response.json()
        if data["result"] == "success":
            rates = data["rates"]
            # Add EGP if not already in the API response
            if "EGP" not in rates:
                rates["EGP"] = 50  # Approximate rate as of May 2024
            return rates
        else:
            # Default fallback rates including EGP
            return {"USD": 1.0, "EUR": 0.85, "GBP": 0.75, "JPY": 110.0, "CAD": 1.25, "AUD": 1.35, "EGP": 50}
    except:
        # Default fallback rates including EGP
        return {"USD": 1.0, "EUR": 0.85, "GBP": 0.75, "JPY": 110.0, "CAD": 1.25, "AUD": 1.35, "EGP": 50}

# --- Define Theme Settings ---
def set_theme():
    # Dark theme colors only
    primary_color = "#3498db"  # Blue
    secondary_color = "#2ecc71"  # Green
    background_color = "#121212"
    text_color = "#f1f1f1"
    card_bg_color = "#1e1e1e"
    bg_image = "https://images.unsplash.com/photo-1583121274602-3e2820c69888?q=80&w=1920&auto=format&fit=crop&ixlib=rb-4.0.3"
    fallback_bg = "linear-gradient(135deg, #121212, #1e1e1e, #121212)"
    
    # Store theme colors in session state for later use
    st.session_state.theme = {
        "primary_color": primary_color,
        "secondary_color": secondary_color,
        "background_color": background_color,
        "text_color": text_color,
        "card_bg_color": card_bg_color
    }
    
    # Apply theme
    st.markdown(f"""
    <style>
    /* Base Theme */
    body {{
        color: {text_color};
        transition: all 0.5s ease;
    }}
    
    /* Background image with fallback */
    .stApp {{
        background: {fallback_bg};
    }}
    
    /* Apply background image with overlay */
    @media only screen and (min-width: 768px) {{
        .stApp {{
            background: linear-gradient(rgba(18,18,18, 0.85), 
                      rgba(18,18,18, 0.85)),
                      url('{bg_image}');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
    }}
    
    /* Headers */
    h1, h2, h3 {{
        color: {primary_color} !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
        position: relative;
    }}
    
    /* Buttons */
    .stButton > button {{
        background-color: {primary_color};
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        position: relative;
    }}
    
    .stButton > button:hover {{
        background-color: {secondary_color};
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }}
    
    /* Cards */
    .card {{
        background-color: {card_bg_color}CC;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
    }}
    
    .card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }}
    
    /* Input fields */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div,
    .stTextInput > div > div > input {{
        border-radius: 8px;
        transition: all 0.3s ease;
    }}
    
    /* Success message */
    .success-msg {{
        background-color: {secondary_color};
        color: white !important;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        animation: fadeIn 0.5s ease-in-out;
        position: relative;
    }}
    
    /* Animations */
    @keyframes fadeIn {{
        0% {{ opacity: 0; transform: translateY(20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}
    
    .animate-fadeIn {{
        animation: fadeIn 0.5s ease-in-out;
    }}
    
    .animate-pulse {{
        animation: pulse 2s infinite;
    }}
    
    /* Dividers */
    hr {{
        border-top: 1px solid {primary_color}40;
        margin: 30px 0;
    }}

    /* Make sidebar more readable with background */
    .css-1d391kg, .css-1wrcr25, .css-ocqkz7, .css-1v3fvcr, [data-testid="stSidebar"] {{
        background-color: {card_bg_color}E6 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }}
    
    /* Container styling for better readability */
    .stApp > header {{
        background-color: transparent !important;
    }}
    
    .main .block-container {{
        background-color: {background_color}80;
        padding: 2rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }}

    /* Fix text visibility in both themes */
    .dark-text {{
        color: #333333 !important;
    }}
    
    .light-text {{
        color: #f1f1f1 !important;
    }}

    /* Fix scrolling issue with input fields */
    .stNumberInput, .stSelectbox {{
        pointer-events: auto;
    }}
    .stApp {{
        overflow-y: auto !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- Define Unique Categories for Selectboxes (from your Kaggle output) ---
# Helper function to prepare selectbox options (handles nan -> "Unknown", sorts)
def prepare_options(raw_list, nan_replacement="Unknown", sort=True):
    # Replace actual np.nan with the string replacement, then convert all to string
    options_str = [nan_replacement if pd.isna(item) else str(item) for item in raw_list]
    if sort:
        # Get unique sorted list
        unique_sorted_options = sorted(list(set(options_str)))
        return unique_sorted_options
    return list(set(options_str)) # Just unique if no sort needed

body_type_opts_raw = ['Pickup Truck', 'Sedan', 'SUV / Crossover', 'Van', 'Hatchback', 'Coupe', 'Minivan', 'Convertible', 'Wagon', np.nan]
engine_cylinders_opts_raw = ['V8', 'I4', 'V6 Hybrid', 'V6', 'V6 Flex Fuel Vehicle', 'I3', 'H4', 'V8 Flex Fuel Vehicle', np.nan, 'I4 Flex Fuel Vehicle', 'I6 Diesel', 'I4 Hybrid', 'I6', 'I4 Diesel', 'V8 Biodiesel', 'V6 Biodiesel', 'V8 Diesel', 'V6 Diesel', 'I5', 'H6', 'V10', 'W12', 'V12', 'I4 Compressed Natural Gas', 'I2', 'V8 Hybrid', 'V8 Compressed Natural Gas', 'I5 Diesel', 'H4 Hybrid', 'I5 Biodiesel', 'W12 Flex Fuel Vehicle', 'R2', 'I6 Hybrid', 'V8 Propane', 'V6 Compressed Natural Gas', 'V10 Diesel', 'W8', 'I3 Hybrid']
engine_type_opts_raw = engine_cylinders_opts_raw # You mentioned they are the same
fuel_type_opts_raw = ['Gasoline', 'Hybrid', 'Flex Fuel Vehicle', np.nan, 'Diesel', 'Electric', 'Biodiesel', 'Compressed Natural Gas', 'Propane']
listing_color_opts_raw = ['SILVER', 'GRAY', 'WHITE', 'BLACK', 'BLUE', 'RED', 'UNKNOWN', 'GREEN', 'YELLOW', 'BROWN', 'GOLD', 'TEAL', 'ORANGE', 'PURPLE', 'PINK'] # UNKNOWN is already a string
transmission_opts_raw = ['A', 'CVT', 'M', np.nan, 'Dual Clutch']
wheel_system_opts_raw = ['4WD', 'FWD', 'AWD', np.nan, 'RWD', '4X2']

# Prepare options for UI
body_type_options = prepare_options(body_type_opts_raw)
engine_cylinders_options = prepare_options(engine_cylinders_opts_raw)
engine_type_options = prepare_options(engine_type_opts_raw)
fuel_type_options = prepare_options(fuel_type_opts_raw)
listing_color_options = prepare_options(listing_color_opts_raw, nan_replacement="UNKNOWN") # Use UNKNOWN if it's a category
transmission_options_map = {'A': 'Automatic', 'M': 'Manual', 'CVT': 'CVT', 'Dual Clutch': 'Dual Clutch', 'Unknown': 'Unknown'}
prepared_trans_opts = prepare_options(transmission_opts_raw)
display_trans_opts = [transmission_options_map.get(opt, opt) for opt in prepared_trans_opts]
wheel_system_options = prepare_options(wheel_system_opts_raw)


numerical_cols = [
    'back_legroom', 'city_fuel_economy', 'daysonmarket', 'engine_displacement', 
    'fleet', 'frame_damaged', 'franchise_dealer', 'front_legroom', 
    'fuel_tank_volume', 'has_accidents', 'height', 'highway_fuel_economy', 
    'horsepower', 'isCab', 'is_new', 'length', 'maximum_seating', 
    'mileage', 'owner_count', 'salvage', 'savings_amount', 'seller_rating', 
    'theft_title', 'wheelbase', 'width', 'car_age'
]

# Define expected dtypes for conversion (simplified)
# We'll assume columns NOT in categorical_widget_map are numeric if they are in original_feature_columns
# This map helps us know which columns get string inputs from selectboxes
categorical_widget_map = {
    'body_type': body_type_options,
    'engine_cylinders': engine_cylinders_options,
    'engine_type': engine_type_options,
    'fuel_type': fuel_type_options,
    'listing_color': listing_color_options,
    'transmission': display_trans_opts, # Use display options here for widget
    'wheel_system': wheel_system_options
}
# Boolean flags will be handled separately as they map to 0/1
boolean_flags = ['fleet', 'frame_damaged', 'franchise_dealer', 'has_accidents', 'isCab', 'is_new', 'salvage', 'theft_title']


# --- Main App Interface ---
st.markdown('<div class="animate-fadeIn">', unsafe_allow_html=True)

# How it works section
with st.expander("How it works"):
    st.markdown("""
    1. Enter the specifications of the used car in the form below.
    2. Click **Predict Price** to get an estimated market value in your selected currency.
    3. You can change the currency and re-predict as needed.
    4. Use the **Reset** button to start over.
    """)

# Custom CSS for the title area with a semi-transparent backdrop
st.markdown("""
<div style="background-color: rgba(0,0,0,0.6); padding: 20px; border-radius: 10px; margin-bottom: 20px; backdrop-filter: blur(5px); -webkit-backdrop-filter: blur(5px); box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);">
    <h1 style="text-align: center; color: white !important; margin: 0;">🚗 US Used Car Price Predictor</h1>
</div>
""", unsafe_allow_html=True)

# App settings in sidebar
with st.sidebar:
    st.markdown("### App Settings")
    
    # Set dark theme (no toggle)
    set_theme()
    
    # Currency selection
    currencies = {
        "USD": "US Dollar ($)",
        "EUR": "Euro (€)",
        "GBP": "British Pound (£)",
        "JPY": "Japanese Yen (¥)",
        "CAD": "Canadian Dollar (CA$)",
        "AUD": "Australian Dollar (A$)",
        "EGP": "Egyptian Pound (EGP)"
    }
    
    st.markdown("<span style='font-size: 0.95em; color: #aaa;'>💡 Select your preferred currency. The prediction will update automatically.</span>", unsafe_allow_html=True)
    selected_currency = st.selectbox(
        "Select Currency",
        options=list(currencies.keys()),
        format_func=lambda x: currencies[x],
        key="currency_select"
    )
    
    # Get currency symbols for display
    currency_symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "CAD": "CA$",
        "AUD": "A$",
        "EGP": "EGP"
    }
    
    # Get exchange rates
    exchange_rates = get_exchange_rates()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app predicts the price of used cars based on various specifications.
    
    The model was trained on US used car data.
    """)
    
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%B %d, %Y')}")

# Introduction
st.markdown("""
<div class="card animate-fadeIn" style="background-color: rgba(0,0,0,0.5); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);">
    <p style="font-size: 1.2em; text-align: center; color: white !important;">Enter the specifications of a used car to get an estimated market price in your selected currency.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

if model is not None and preprocessor is not None and original_feature_columns is not None:
    # Reset button logic
    if st.button("Reset", type="secondary", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in ["theme"]:
                del st.session_state[key]
        st.rerun()

    col1, col2, col3 = st.columns(3)
    input_values_from_widgets = {} # To store raw widget outputs
    valid_inputs = True

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Basic Info")
        # Mileage
        if 'mileage' in original_feature_columns:
            mileage = st.number_input("Mileage", min_value=0, value=50000, step=1000, help="Total miles the car has been driven.")
            if mileage > 300000:
                st.warning("High mileage: This is above typical values for most used cars.")
            elif mileage < 1000:
                st.info("Very low mileage: Is this a nearly new car?")
            input_values_from_widgets['mileage'] = mileage
        # Car Age
        if 'car_age' in original_feature_columns:
            car_age = st.number_input("Car Age (years)", min_value=0, max_value=100, value=5, step=1, help="How many years since the car was manufactured.")
            if car_age > 30:
                st.warning("This is an unusually old car.")
            elif car_age < 1:
                st.info("Is this a new or nearly new car?")
            input_values_from_widgets['car_age'] = car_age
        # Horsepower
        if 'horsepower' in original_feature_columns:
            horsepower = st.number_input("Horsepower (HP)", min_value=10, max_value=1200, value=200, step=10, help="Engine power. Typical cars range from 70 to 400 HP.")
            if horsepower > 1200 or horsepower < 10:
                st.error("Horsepower must be between 10 and 1200.")
                valid_inputs = False
            elif horsepower > 600:
                st.warning("High horsepower: This is above typical values for most cars.")
            elif horsepower < 50:
                st.info("Very low horsepower: Is this a compact or economy car?")
            input_values_from_widgets['horsepower'] = horsepower
        # Engine Displacement
        if 'engine_displacement' in original_feature_columns:
            engine_displacement = st.number_input("Engine Displacement (L)", min_value=0.1, max_value=10.0, value=2.5, step=0.1, format="%.1f", help="Total engine size in liters.")
            if engine_displacement > 10.0 or engine_displacement < 0.1:
                st.error("Engine displacement must be between 0.1 and 10.0.")
                valid_inputs = False
            elif engine_displacement > 6.0:
                st.warning("Large engine displacement: This is above typical values for most cars.")
            elif engine_displacement < 1.0:
                st.info("Small engine: Is this a compact or hybrid car?")
            input_values_from_widgets['engine_displacement'] = engine_displacement
        # Fuel Tank Volume
        if 'fuel_tank_volume' in original_feature_columns:
            fuel_tank_volume = st.number_input("Fuel Tank Volume (gal)", min_value=1.0, max_value=100.0, value=15.0, step=0.1, format="%.1f", help="Capacity of the fuel tank in gallons.")
            if fuel_tank_volume < 1.0 or fuel_tank_volume > 100.0:
                st.error("Fuel tank volume must be between 1.0 and 100.0 gallons.")
                valid_inputs = False
            input_values_from_widgets['fuel_tank_volume'] = fuel_tank_volume
        # City Fuel Economy
        if 'city_fuel_economy' in original_feature_columns:
            city_fuel_economy = st.number_input("City Fuel Economy (MPG)", min_value=1, max_value=150, value=20, step=1, help="Miles per gallon in city driving conditions.")
            if city_fuel_economy < 1 or city_fuel_economy > 150:
                st.error("City fuel economy must be between 1 and 150 MPG.")
                valid_inputs = False
            input_values_from_widgets['city_fuel_economy'] = city_fuel_economy
        # Highway Fuel Economy
        if 'highway_fuel_economy' in original_feature_columns:
            highway_fuel_economy = st.number_input("Highway Fuel Economy (MPG)", min_value=1, max_value=150, value=30, step=1, help="Miles per gallon on highways.")
            if highway_fuel_economy < 1 or highway_fuel_economy > 150:
                st.error("Highway fuel economy must be between 1 and 150 MPG.")
                valid_inputs = False
            input_values_from_widgets['highway_fuel_economy'] = highway_fuel_economy
        # Days on Market
        if 'daysonmarket' in original_feature_columns:
            daysonmarket = st.number_input("Days on Market", min_value=0, value=30, step=1, help="How many days the car has been listed for sale.")
            if daysonmarket < 0:
                st.error("Days on market cannot be negative.")
                valid_inputs = False
            input_values_from_widgets['daysonmarket'] = daysonmarket
        # Previous Owners
        if 'owner_count' in original_feature_columns:
            owner_count = st.number_input("Previous Owners", min_value=0, max_value=10, value=1, step=1, help="Number of previous owners.")
            if owner_count > 10 or owner_count < 0:
                st.error("Value must be between 0 and 10.")
                valid_inputs = False
            input_values_from_widgets['owner_count'] = owner_count
        # Savings Amount
        if 'savings_amount' in original_feature_columns:
            savings_amount = st.number_input("Savings Amount ($)", min_value=0, value=0, step=100, help="Discount or savings on the car price, if any.")
            if savings_amount < 0:
                st.error("Savings amount cannot be negative.")
                valid_inputs = False
            input_values_from_widgets['savings_amount'] = savings_amount
        # Seller Rating
        if 'seller_rating' in original_feature_columns:
            seller_rating = st.number_input("Seller Rating (0-5)", min_value=0.0, max_value=5.0, value=4.0, step=0.1, format="%.1f", help="Rating of the seller (0 = worst, 5 = best).")
            if seller_rating < 0 or seller_rating > 5:
                st.error("Seller rating must be between 0 and 5.")
                valid_inputs = False
            input_values_from_widgets['seller_rating'] = seller_rating
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Engine & Dimensions")
        if 'back_legroom' in original_feature_columns: input_values_from_widgets['back_legroom'] = st.number_input("Back Legroom (in)", min_value=10.0, max_value=60.0, value=35.0, step=0.1, format="%.1f")
        if 'front_legroom' in original_feature_columns: input_values_from_widgets['front_legroom'] = st.number_input("Front Legroom (in)", min_value=20.0, max_value=70.0, value=40.0, step=0.1, format="%.1f")
        if 'height' in original_feature_columns: input_values_from_widgets['height'] = st.number_input("Height (in)", min_value=30.0, max_value=120.0, value=60.0, step=0.1, format="%.1f")
        if 'length' in original_feature_columns: input_values_from_widgets['length'] = st.number_input("Length (in)", min_value=80.0, max_value=300.0, value=180.0, step=0.1, format="%.1f")
        if 'wheelbase' in original_feature_columns: input_values_from_widgets['wheelbase'] = st.number_input("Wheelbase (in)", min_value=50.0, max_value=200.0, value=100.0, step=0.1, format="%.1f")
        if 'width' in original_feature_columns: input_values_from_widgets['width'] = st.number_input("Width (in)", min_value=40.0, max_value=120.0, value=70.0, step=0.1, format="%.1f")
        if 'maximum_seating' in original_feature_columns: input_values_from_widgets['maximum_seating'] = st.number_input("Max Seating", min_value=1, max_value=15, value=5, step=1)

        if 'body_type' in original_feature_columns: input_values_from_widgets['body_type'] = st.selectbox("Body Type", options=body_type_options)
        if 'engine_cylinders' in original_feature_columns: input_values_from_widgets['engine_cylinders'] = st.selectbox("Engine Cylinders", options=engine_cylinders_options)
        if 'engine_type' in original_feature_columns: input_values_from_widgets['engine_type'] = st.selectbox("Engine Type", options=engine_type_options)
        if 'fuel_type' in original_feature_columns: input_values_from_widgets['fuel_type'] = st.selectbox("Fuel Type", options=fuel_type_options)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Condition & Status")
        bool_options_map_display = {"No": 0, "Yes": 1} # For display
        for flag_col, flag_label in [
            ('fleet', "Fleet Vehicle?"), ('frame_damaged', "Frame Damaged?"),
            ('franchise_dealer', "Franchise Dealer?"), ('has_accidents', "Accidents Reported?"),
            ('isCab', "Was a Cab/Taxi?"), ('is_new', "Is New (<2 yrs old)?"),
            ('salvage', "Salvage Title?"), ('theft_title', "Theft on Title?")
        ]:
            if flag_col in original_feature_columns:
                selected_display = st.selectbox(flag_label, options=list(bool_options_map_display.keys()))
                input_values_from_widgets[flag_col] = bool_options_map_display[selected_display]

        if 'listing_color' in original_feature_columns: input_values_from_widgets['listing_color'] = st.selectbox("Listing Color Group", options=listing_color_options)
        
        if 'transmission' in original_feature_columns:
            selected_display_trans = st.selectbox("Transmission", options=display_trans_opts)
            # Map back to original value for the model (A, M, CVT etc. or Unknown)
            original_trans_value = selected_display_trans # Default if not in map
            for k, v in transmission_options_map.items():
                if v == selected_display_trans:
                    original_trans_value = k
                    break
            input_values_from_widgets['transmission'] = original_trans_value
            
        if 'wheel_system' in original_feature_columns: input_values_from_widgets['wheel_system'] = st.selectbox("Wheel System", options=wheel_system_options)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Predict button with animation, only enabled if valid_inputs is True
    if not valid_inputs:
        st.error("Please correct the highlighted input errors before predicting.")
    predict_btn = st.button("Predict Price", type="primary", use_container_width=True, disabled=not valid_inputs)
    
    # Results container
    results_container = st.container()
    
    if predict_btn:
        with st.spinner("Calculating price..."):
            # Progress bar for animation
            import time
            progress = st.progress(0, text="Preparing input...")
            for percent in range(1, 101, 20):
                time.sleep(0.08)
                progress.progress(percent, text=f"Processing... {percent}%")
            progress.empty()
            # Construct the input DataFrame exactly as the preprocessor expects it
            input_df_data_final = {}
            for col_name in original_feature_columns:
                if col_name in input_values_from_widgets:
                    value = input_values_from_widgets[col_name]
                    if col_name not in categorical_widget_map and col_name not in boolean_flags:
                        input_df_data_final[col_name] = pd.to_numeric(value, errors='coerce')
                    else:
                        input_df_data_final[col_name] = value
                else:
                    st.warning(f"Input for feature '{col_name}' was not collected. Using NaN or 'Unknown'.")
                    if col_name not in categorical_widget_map and col_name not in boolean_flags:
                        input_df_data_final[col_name] = np.nan
                    else:
                        input_df_data_final[col_name] = 'Unknown'
            try:
                input_df = pd.DataFrame([input_df_data_final], columns=original_feature_columns)
                for col in input_df.columns:
                    if col in numerical_cols:
                        try:
                            if input_df[col].iloc[0] is not None and not pd.isna(input_df[col].iloc[0]):
                                 input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                        except:
                            input_df[col] = np.nan
                with results_container:
                    with st.expander("View Input Data"):
                        st.dataframe(input_df.T.astype(str))
                processed_input = preprocessor.transform(input_df)
                prediction = model.predict(processed_input)
                final_price_usd = round(prediction[0], 2)
                # Show toast feedback
                st.toast("Prediction complete!", icon="🎉")
                # Show prediction
                exchange_rate = get_exchange_rates().get(selected_currency, 1.0)
                final_price = final_price_usd * exchange_rate
                if selected_currency == "JPY":
                    formatted_price = f"{int(final_price):,}"
                else:
                    formatted_price = f"{final_price:,.2f}"
                currency_symbol = currency_symbols.get(selected_currency, "$")
                theme_colors = st.session_state.get("theme", {})
                secondary_color = theme_colors.get("secondary_color", "#27ae60")
                with results_container:
                    st.markdown(f"""
                    <div class="success-msg animate-pulse" style="background: linear-gradient(135deg, {secondary_color}, #1a5276); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2);">
                        <div style="display: flex; align-items: center; justify-content: center;">
                            <span style="font-size: 32px; margin-right: 10px;">💰</span>
                            <span>Predicted Car Price: {currency_symbol}{formatted_price}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if selected_currency != "USD":
                        st.markdown(f"""
                        <div style="text-align: center; margin-bottom: 20px; background-color: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px; backdrop-filter: blur(5px); -webkit-backdrop-filter: blur(5px);">
                            (USD: ${final_price_usd:,.2f})
                        </div>
                        """, unsafe_allow_html=True)
                    # Car image based on body type
                    car_images = {
                        "Sedan": "https://img.icons8.com/color/96/000000/sedan.png",
                        "SUV / Crossover": "https://img.icons8.com/color/96/000000/suv.png",
                        "Pickup Truck": "https://img.icons8.com/color/96/000000/pickup.png",
                        "Coupe": "https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/external-coupe-automotive-ecommerce-flaticons-lineal-color-flat-icons-3.png",
                        "Convertible": "https://img.icons8.com/color/96/000000/convertible.png",
                        "Wagon": "https://img.icons8.com/color/96/000000/station-wagon.png",
                        "Minivan": "https://img.icons8.com/external-flaticons-lineal-color-flat-icons/64/external-minivan-automotive-ecommerce-flaticons-lineal-color-flat-icons.png",
                        "Van": "https://img.icons8.com/color/96/000000/van.png",
                        "Hatchback": "https://img.icons8.com/color/96/000000/hatchback.png"
                    }
                    body_type = input_values_from_widgets.get('body_type', 'Sedan')
                    car_image_url = car_images.get(body_type, "https://img.icons8.com/color/96/000000/car.png")
                    st.markdown(f"""
                    <div style="display: flex; justify-content: center; margin: 20px 0; background-color: rgba(255,255,255,0.2); padding: 20px; border-radius: 50%; width: 120px; height: 120px; margin-left: auto; margin-right: auto; backdrop-filter: blur(5px); -webkit-backdrop-filter: blur(5px);">
                        <img src="{car_image_url}" width="96" height="96" alt="{body_type}" style="object-fit: contain;">
                    </div>
                    <div style="text-align: center; margin-bottom: 30px; background-color: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px; backdrop-filter: blur(5px); -webkit-backdrop-filter: blur(5px);">
                        <span style="font-weight: bold; color: white;">{body_type}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
            except Exception as e:
                with results_container:
                    st.error(f"An error occurred during prediction: {e}")
                    st.error(f"Expected columns by preprocessor: {original_feature_columns}")
                    if 'input_df' in locals():
                        st.error(f"Columns in DataFrame sent to preprocessor: {input_df.columns.tolist()}")
                        st.error(f"Data types of input_df sent to preprocessor: \n{input_df.dtypes}")

# Add a modern footer
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.95em; margin-top: 40px; padding-bottom: 10px;">
    Made with ❤️ by Mohanad &middot; Powered by Streamlit
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("Note: This prediction is based on a machine learning model and should be used as an estimate.")
st.markdown('</div>', unsafe_allow_html=True)
