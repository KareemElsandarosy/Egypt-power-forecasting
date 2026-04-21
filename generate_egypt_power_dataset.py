import urllib.request
import json
import ssl
import random
import datetime
import csv

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Autumn'

def generate_dataset():
    governorates = {
        'Cairo': {'lat': 30.0444, 'lon': 31.2357, 'pop_weight': 0.20, 'adj_factor': 1.3},
        'Alexandria': {'lat': 31.2001, 'lon': 29.9187, 'pop_weight': 0.10, 'adj_factor': 1.1},
        'Dakahlia': {'lat': 31.0364, 'lon': 31.3807, 'pop_weight': 0.08, 'adj_factor': 0.9},
        'Sharqia': {'lat': 30.5877, 'lon': 31.5020, 'pop_weight': 0.08, 'adj_factor': 0.9},
        'Port_Said': {'lat': 31.2565, 'lon': 32.2841, 'pop_weight': 0.03, 'adj_factor': 1.2},
        'Asyut': {'lat': 27.1810, 'lon': 31.1837, 'pop_weight': 0.05, 'adj_factor': 0.9},
        'Aswan': {'lat': 24.0889, 'lon': 32.8998, 'pop_weight': 0.02, 'adj_factor': 0.9}
    }

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    filename = "Egypt_Governorates_Load_Dataset_Advanced.csv"
    
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Header with Lags and Advanced Features
        writer.writerow([
            "Datetime", "Governorate", "Temperature_C", "Humidity_Percent", 
            "Hour", "Month", "Day_of_Year", "Day_of_Week", "Is_Weekend", "Season",
            "CDH", "Temp_x_Hour", "CDH_x_Is_Weekend", 
            "Lag_1h", "Lag_24h", "Lag_168h", "Electricity_Demand_MW"
        ])
        
        total_records = 0

        for gov_name, data in governorates.items():
            print(f"Fetching data for {gov_name}...")
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={data['lat']}&longitude={data['lon']}&start_date=2024-01-01&end_date=2026-04-20&hourly=temperature_2m,relative_humidity_2m&timezone=Africa%2FCairo"
            
            try:
                req = urllib.request.urlopen(url, context=ctx)
                api_data = json.loads(req.read())
            except Exception as e:
                print(f"Error fetching data for {gov_name}: {e}")
                continue

            hourly_time = api_data['hourly']['time']
            hourly_temp = api_data['hourly']['temperature_2m']
            hourly_humidity = api_data['hourly']['relative_humidity_2m']
            
            weight = data['pop_weight'] * data['adj_factor']
            
            # Temporary list to store loads for lag computation
            gov_records = []
            
            for i in range(len(hourly_time)):
                dt_str = hourly_time[i]
                temp = hourly_temp[i] if hourly_temp[i] is not None else 25.0
                humidity = hourly_humidity[i] if hourly_humidity[i] is not None else 50.0

                dt = datetime.datetime.fromisoformat(dt_str)
                hour = dt.hour
                month = dt.month
                day_of_year = dt.timetuple().tm_yday
                day_of_week = dt.weekday()
                is_weekend = 1 if day_of_week in [4, 5] else 0
                season = get_season(month)
                
                # Feature Engineering
                cdh = max(0, temp - 24)
                temp_x_hour = temp * hour
                cdh_x_is_weekend = cdh * is_weekend
                
                # Non-linear Load Simulation Logic
                base_load = 22000 * weight
                
                temp_effect = 0
                if temp > 25:
                    # Exponential relationship for AC load
                    temp_effect = 400 * ((temp - 25) ** 1.5) * weight
                elif temp < 15:
                    # Linear for heating
                    temp_effect = (15 - temp) * 300 * weight
                    
                time_effect = 0
                if 18 <= hour <= 22: time_effect = 2500 * weight
                elif 2 <= hour <= 6: time_effect = -2000 * weight
                    
                weekend_effect = -1500 * weight if is_weekend else 0
                noise = random.randint(int(-100 * weight), int(100 * weight))
                
                total_mw = round(base_load + temp_effect + time_effect + weekend_effect + noise, 2)
                
                gov_records.append({
                    "dt_str": dt_str, "temp": temp, "humidity": humidity, 
                    "hour": hour, "month": month, "day_of_year": day_of_year,
                    "day_of_week": day_of_week, "is_weekend": is_weekend, "season": season,
                    "cdh": cdh, "temp_x_hour": temp_x_hour, "cdh_x_is_weekend": cdh_x_is_weekend,
                    "mw": total_mw
                })
            
            # Now loop through gov_records and compute Lags, starting from index 168
            for i in range(168, len(gov_records)):
                rec = gov_records[i]
                lag_1h = gov_records[i-1]["mw"]
                lag_24h = gov_records[i-24]["mw"]
                lag_168h = gov_records[i-168]["mw"]
                
                writer.writerow([
                    rec["dt_str"], gov_name, rec["temp"], rec["humidity"], 
                    rec["hour"], rec["month"], rec["day_of_year"], rec["day_of_week"], 
                    rec["is_weekend"], rec["season"], rec["cdh"], rec["temp_x_hour"], 
                    rec["cdh_x_is_weekend"], lag_1h, lag_24h, lag_168h, rec["mw"]
                ])
                total_records += 1

    print(f"\n[SUCCESS] Advanced Dataset created: {filename}")
    print(f"[INFO] Total Records Generated (excluding first week per gov): {total_records:,}")

if __name__ == '__main__':
    generate_dataset()
