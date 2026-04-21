import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import urllib.request
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# ── PAGE CONFIG ──
st.set_page_config(page_title="EgyPower Forecast", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# ── CSS (Light Theme) ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif !important; }
    .stApp { background: linear-gradient(160deg, #f8f9fc 0%, #eef1f8 40%, #f0f4fa 70%, #f8f9fc 100%); }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding-top: 1rem; }

    .hero { text-align: center; padding: 20px 0 10px 0; position: relative; }
    .hero::before { content: ''; position: absolute; top: -50px; left: 50%; transform: translateX(-50%); width: 400px; height: 400px; background: radial-gradient(circle, rgba(37,99,235,0.06) 0%, transparent 70%); pointer-events: none; }
    .hero-badge { display: inline-block; background: rgba(37,99,235,0.08); border: 1px solid rgba(37,99,235,0.2); color: #2563EB; padding: 6px 18px; border-radius: 50px; font-size: 0.8rem; font-weight: 600; letter-spacing: 1px; margin-bottom: 10px; }
    .hero h1 { font-size: 2.5rem; font-weight: 900; color: #1e293b; margin: 0 0 5px 0; line-height: 1.2; }
    .hero h1 span { background: linear-gradient(135deg, #2563EB, #7C3AED); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero .subtitle { color: #64748b; font-size: 1rem; direction: rtl; }

    .g-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px 22px; direction: rtl; text-align: right; transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1); position: relative; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    .g-card:hover { transform: translateY(-4px); border-color: rgba(37,99,235,0.3); box-shadow: 0 12px 40px rgba(37,99,235,0.1); }
    .g-card .icon { font-size: 1.5rem; margin-bottom: 6px; display: inline-block; }
    .g-card .label { color: #64748b; font-size: 0.82rem; font-weight: 400; margin: 0; }
    .g-card .value { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; margin: 4px 0 2px 0; background: linear-gradient(135deg, #2563EB, #7C3AED); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .g-card .unit { color: #94a3b8; font-size: 0.78rem; }
    .g-card.peak-warn { border-right: 3px solid #ef4444; }
    .g-card.peak-warn .value { background: linear-gradient(135deg, #ef4444, #f97316); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .g-card.confidence .value { background: linear-gradient(135deg, #0ea5e9, #7C3AED); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

    .sec-title { color: #1e293b; font-size: 1.1rem; font-weight: 700; direction: rtl; text-align: right; margin: 15px 0 8px 0; display: flex; align-items: center; gap: 10px; flex-direction: row-reverse; }
    .sec-title .dot { width: 8px; height: 8px; border-radius: 50%; background: #2563EB; display: inline-block; box-shadow: 0 0 8px rgba(37,99,235,0.4); }

    .mapping-banner { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.25); border-radius: 10px; padding: 10px 16px; direction: rtl; text-align: right; color: #b45309; font-size: 0.85rem; margin-bottom: 15px; }

    .footer-honesty { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 20px; direction: rtl; text-align: right; color: #64748b; font-size: 0.8rem; margin-top: 30px; line-height: 1.8; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    .footer-honesty b { color: #334155; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; color: #64748b; padding: 8px 18px; }
    .stTabs [aria-selected="true"] { background: rgba(37,99,235,0.08) !important; border-color: rgba(37,99,235,0.3) !important; color: #2563EB !important; }
    div[data-baseweb="select"] { direction: rtl; }

    .model-row { display: flex; gap: 12px; margin-bottom: 10px; direction: rtl; }
    .model-chip { flex: 1; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 16px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
    .model-chip.winner { border-color: rgba(37,99,235,0.4); background: rgba(37,99,235,0.04); box-shadow: 0 4px 12px rgba(37,99,235,0.1); }
    .model-chip .mname { color: #1e293b; font-weight: 700; font-size: 0.9rem; margin-bottom: 6px; }
    .model-chip .mval { font-family: 'JetBrains Mono', monospace; color: #2563EB; font-size: 1.1rem; font-weight: 700; }
    .model-chip .mlab { color: #94a3b8; font-size: 0.75rem; }

    [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
    [data-testid="stSidebar"] .stMarkdown h3 { color: #1e293b; direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

# ── LOAD ASSETS ──
@st.cache_resource
def load_assets():
    try:
        md = joblib.load('model_pipeline.pkl')
        with open('mapping_config.json', 'r', encoding='utf-8') as f:
            mp = json.load(f)
        return md, mp
    except FileNotFoundError:
        return None, None

model_data, mapping_config = load_assets()
if not model_data:
    st.error("model_pipeline.pkl not found. Run the notebook first.")
    st.stop()

pipelines = model_data.get('pipelines', {})
if not pipelines:
    pipelines = {model_data.get('best_model', 'Default'): model_data['pipeline']}

margins = model_data.get('margins', {})
feature_order = model_data['feature_order']
comparison_results = model_data.get('comparison_results', [])
best_model_name = model_data.get('best_model', 'Unknown')

BASE_COORDS = {
    'Cairo': {'lat': 30.0444, 'lon': 31.2357},
    'Alexandria': {'lat': 31.2001, 'lon': 29.9187},
    'Dakahlia': {'lat': 31.0364, 'lon': 31.3807},
    'Sharqia': {'lat': 30.5877, 'lon': 31.5020},
    'Port_Said': {'lat': 31.2565, 'lon': 32.2841},
    'Asyut': {'lat': 27.1810, 'lon': 31.1837},
    'Aswan': {'lat': 24.0889, 'lon': 32.8998}
}

GOV_ARABIC = {
    "Total (National)": None,
    "Cairo": "Cairo", "Giza": "Giza", "Qalyubia": "Qalyubia",
    "Alexandria": "Alexandria", "Matrouh": "Matrouh", "Beheira": "Beheira",
    "Dakahlia": "Dakahlia", "Damietta": "Damietta", "Kafr El Sheikh": "Kafr El Sheikh", "Gharbia": "Gharbia",
    "Sharqia": "Sharqia", "Monufia": "Monufia",
    "Port_Said": "Port_Said", "Ismailia": "Ismailia", "Suez": "Suez",
    "Asyut": "Asyut", "Sohag": "Sohag", "Minya": "Minya", "Beni Suef": "Beni Suef",
    "Faiyum": "Faiyum", "Qena": "Qena",
    "Aswan": "Aswan", "Luxor": "Luxor", "Red Sea": "Red Sea", "New Valley": "New Valley",
    "North Sinai": "North Sinai", "South Sinai": "South Sinai"
}

def get_season(m):
    if m in [12, 1, 2]: return 'Winter'
    if m in [3, 4, 5]: return 'Spring'
    if m in [6, 7, 8]: return 'Summer'
    return 'Autumn'

ARABIC_DAYS = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

# ── WEATHER API WITH RETRY ──
@st.cache_data(ttl=3600)
def fetch_weather(lat, lon, days):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,relative_humidity_2m&forecast_days={days}&timezone=Africa%2FCairo"
    last_err = None
    for attempt in range(3):
        try:
            req = urllib.request.urlopen(url, timeout=8)
            data = json.loads(req.read())
            df = pd.DataFrame(data['hourly'])
            df['time'] = pd.to_datetime(df['time'])
            df.rename(columns={'temperature_2m': 'Temperature', 'relative_humidity_2m': 'Humidity'}, inplace=True)
            return df
        except Exception as e:
            last_err = e
            time.sleep(1)
    return None

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### Control Panel")
    st.markdown("---")
    selected_name = st.selectbox("Governorate", list(GOV_ARABIC.keys()), index=0)
    forecast_days = st.slider("Forecast Days", 1, 7, 3)
    st.markdown("---")
    selected_model = st.selectbox("Prediction Model", list(pipelines.keys()), index=list(pipelines.keys()).index(best_model_name) if best_model_name in pipelines else 0)
    pipeline = pipelines[selected_model]
    confidence_margin = margins.get(selected_model, 0)
    
    st.markdown("---")
    st.markdown(f"**Selected:** {selected_model}")
    if confidence_margin:
        st.markdown(f"**Confidence:** +/- {confidence_margin:.0f} MW")
    if comparison_results:
        st.markdown("---")
        st.markdown("### Model Scores")
        for r in comparison_results:
            emoji = "🏆" if r['Model'] == best_model_name else "  "
            st.markdown(f"{emoji} **{r['Model']}**  \nR2={r['R2']:.4f} | MAE={r['MAE']:.0f}")

selected_en = GOV_ARABIC[selected_name]
is_national = (selected_name == "Total (National)")

# ── HERO ──
st.markdown("""
<div class="hero">
    <div class="hero-badge">POWERED BY ML + REAL WEATHER DATA</div>
    <h1>Egypt Power <span>Load Forecasting</span></h1>
    <p class="subtitle">Intelligent hourly electricity demand prediction across all governorates</p>
</div>
""", unsafe_allow_html=True)

# ── PREDICTION ENGINE ──
INIT_LOAD = 5000

def predict_single_gov(base_gov, multiplier, weather_df):
    results = []
    history = [INIT_LOAD] * 24
    for _, row in weather_df.iterrows():
        dt, temp = row['time'], row['Temperature']
        h, m = dt.hour, dt.month
        inp = pd.DataFrame([{
            'Governorate': base_gov, 'Temperature_C': temp, 'Humidity_Percent': row['Humidity'],
            'Hour': h, 'Day_of_Week': dt.weekday(), 'Month': m,
            'Is_Weekend': 1 if dt.weekday() in [4, 5] else 0,
            'Season': get_season(m), 'CDH': max(0, temp - 24),
            'Lag_1h': history[-1], 'Lag_24h': history[-24],
            'Temp_x_Hour': temp * h
        }])[feature_order]
        p = pipeline.predict(inp)[0]
        results.append({'Datetime': dt, 'Temperature': temp, 'Predicted_Load_MW': p * multiplier})
        history.append(p)
    return results

forecast_results = []
try:
    with st.spinner('Building forecasts...'):
        if not is_national:
            gov_info = mapping_config[selected_en]
            base_gov, mult = gov_info['base'], gov_info['multiplier']
            coords = BASE_COORDS[base_gov]
            if base_gov != selected_en:
                st.markdown(f'<div class="mapping-banner">Prediction for <b>{selected_name}</b> using regional similarity model with <b>{base_gov}</b> | Multiplier: <b>{mult:.2f}</b></div>', unsafe_allow_html=True)
            weather_df = fetch_weather(coords['lat'], coords['lon'], forecast_days)
            if weather_df is None:
                st.error("Weather API unavailable. Please try again later.")
                st.stop()
            forecast_results = predict_single_gov(base_gov, mult, weather_df)
        else:
            base_weather = {}
            for bg, c in BASE_COORDS.items():
                w = fetch_weather(c['lat'], c['lon'], forecast_days)
                if w is None:
                    st.error(f"Weather API failed for {bg}.")
                    st.stop()
                base_weather[bg] = w
            time_idx = base_weather['Cairo']['time']
            histories = {bg: [INIT_LOAD] * 24 for bg in BASE_COORDS}
            for i in range(len(time_idx)):
                dt = time_idx.iloc[i]
                h, m = dt.hour, dt.month
                iw = 1 if dt.weekday() in [4, 5] else 0
                s = get_season(m)
                base_preds = {}
                avg_t = 0
                for bg in BASE_COORDS:
                    r = base_weather[bg].iloc[i]
                    t = r['Temperature']
                    avg_t += t
                    inp = pd.DataFrame([{
                        'Governorate': bg, 'Temperature_C': t, 'Humidity_Percent': r['Humidity'],
                        'Hour': h, 'Day_of_Week': dt.weekday(), 'Month': m,
                        'Is_Weekend': iw, 'Season': s, 'CDH': max(0, t - 24),
                        'Lag_1h': histories[bg][-1], 'Lag_24h': histories[bg][-24],
                        'Temp_x_Hour': t * h
                    }])[feature_order]
                    p = pipeline.predict(inp)[0]
                    base_preds[bg] = p
                    histories[bg].append(p)
                total = sum(base_preds[inf['base']] * inf['multiplier'] for inf in mapping_config.values())
                forecast_results.append({'Datetime': dt, 'Temperature': avg_t / 7, 'Predicted_Load_MW': total})
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

res_df = pd.DataFrame(forecast_results)

# ── KPIs ──
total_mwh = res_df['Predicted_Load_MW'].sum()
total_kwh = total_mwh * 1000
peak_load = res_df['Predicted_Load_MW'].max()
min_load = res_df['Predicted_Load_MW'].min()
peak_time = res_df.loc[res_df['Predicted_Load_MW'].idxmax(), 'Datetime']
avg_temp = res_df['Temperature'].mean()
cost_egp = total_kwh * 2.14
is_peak_danger = (is_national and peak_load > 40000) or (not is_national and peak_load > 6000)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f'<div class="g-card"><div class="icon">⚡</div><p class="label">Total Energy</p><p class="value">{total_mwh:,.0f}</p><p class="unit">MWh | {total_kwh:,.0f} kWh</p></div>', unsafe_allow_html=True)
with k2:
    cls = "peak-warn" if is_peak_danger else ""
    st.markdown(f'<div class="g-card {cls}"><div class="icon">🔺</div><p class="label">Peak Load</p><p class="value">{peak_load:,.0f}</p><p class="unit">MW | {peak_time.strftime("%Y-%m-%d %H:00")}</p></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="g-card"><div class="icon">🌡</div><p class="label">Avg Temperature</p><p class="value">{avg_temp:.1f} C</p><p class="unit">Celsius</p></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="g-card"><div class="icon">💳</div><p class="label">Estimated Cost</p><p class="value">{cost_egp:,.0f}</p><p class="unit">EGP | 2.14 EGP/kWh</p></div>', unsafe_allow_html=True)
with k5:
    st.markdown(f'<div class="g-card confidence"><div class="icon">🎯</div><p class="label">Current Model</p><p class="value">{selected_model[:8]}</p><p class="unit">MAE: {next((r["MAE"] for r in comparison_results if r["Model"] == selected_model), 0)} MW</p></div>', unsafe_allow_html=True)

# ── CHART TEMPLATE ──
CT = dict(plot_bgcolor='#ffffff', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#334155', family='Cairo'), hovermode='x unified', margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
GR = dict(showgrid=True, gridcolor='rgba(0,0,0,0.06)')

# ── TABS ──
tab1, tab2, tab3, tab4 = st.tabs(["Load Forecast", "Dual-Axis", "Model Comparison", "Data & Export"])

with tab1:
    st.markdown(f'<div class="sec-title"><span class="dot"></span> Load Forecast - {selected_name}</div>', unsafe_allow_html=True)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=res_df['Datetime'], y=res_df['Predicted_Load_MW'], mode='lines', name='Predicted Load', line=dict(color='#2563EB', width=2.5, shape='spline'), fill='tozeroy', fillcolor='rgba(37,99,235,0.07)'))
    fig1.add_annotation(x=peak_time, y=peak_load, text=f"Peak: {peak_load:,.0f} MW", showarrow=True, arrowhead=2, arrowcolor='#ef4444', font=dict(color='#ef4444', size=12, family='JetBrains Mono'), bgcolor='rgba(239,68,68,0.08)', bordercolor='#ef4444', borderwidth=1, borderpad=4)
    if confidence_margin:
        fig1.add_trace(go.Scatter(x=res_df['Datetime'], y=res_df['Predicted_Load_MW'] + confidence_margin, mode='lines', name='Upper CI', line=dict(color='rgba(124,58,237,0.3)', width=1, dash='dot'), showlegend=False))
        fig1.add_trace(go.Scatter(x=res_df['Datetime'], y=res_df['Predicted_Load_MW'] - confidence_margin, mode='lines', name='Lower CI', line=dict(color='rgba(124,58,237,0.3)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(124,58,237,0.05)', showlegend=False))
    fig1.update_layout(**CT, xaxis={**GR, 'title': 'Timeline'}, yaxis={**GR, 'title': 'Load (MW)', 'side': 'right'})
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.markdown(f'<div class="sec-title"><span class="dot"></span> Temperature vs Load Analysis</div>', unsafe_allow_html=True)
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=res_df['Datetime'], y=res_df['Predicted_Load_MW'], name='Load (MW)', line=dict(color='#2563EB', width=2), fill='tozeroy', fillcolor='rgba(37,99,235,0.05)'), secondary_y=False)
    fig2.add_trace(go.Scatter(x=res_df['Datetime'], y=res_df['Temperature'], name='Temp (C)', line=dict(color='#ef4444', width=2, dash='dot')), secondary_y=True)
    fig2.update_layout(**CT)
    fig2.update_yaxes(title_text="Load (MW)", secondary_y=False, **GR, side='right')
    fig2.update_yaxes(title_text="Temperature (C)", secondary_y=True, **GR, side='left')
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown('<div class="sec-title"><span class="dot"></span> Model Competition Results</div>', unsafe_allow_html=True)
    if comparison_results:
        comp_df = pd.DataFrame(comparison_results).sort_values('MAE')
        best = comp_df.iloc[0]['Model']
        html = '<div class="model-row">'
        for _, r in comp_df.iterrows():
            w = "winner" if r['Model'] == best else ""
            badge = "🏆 " if r['Model'] == best else ""
            html += f'<div class="model-chip {w}"><div class="mname">{badge}{r["Model"]}</div><div class="mval">R2: {r["R2"]:.4f}</div><div class="mlab">MAE: {r["MAE"]:.1f} | RMSE: {r["RMSE"]:.1f}</div></div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        fig3, axes_data = go.Figure(), comp_df
        colors = ['#2563EB' if m == best else '#cbd5e1' for m in axes_data['Model']]
        fig3.add_trace(go.Bar(x=axes_data['Model'], y=axes_data['R2'], marker_color=colors, text=[f"{v:.4f}" for v in axes_data['R2']], textposition='outside', textfont=dict(color='#334155')))
        fig3.update_layout(**CT, title='R2 Score Comparison', yaxis=dict(title='R2', **GR))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No comparison data. Re-run the notebook to generate model competition results.")

with tab4:
    st.markdown(f'<div class="sec-title"><span class="dot"></span> Full Data Table</div>', unsafe_allow_html=True)
    display_df = res_df.copy()
    display_df['Predicted_Load_MW'] = display_df['Predicted_Load_MW'].round(2)
    display_df['Temperature'] = display_df['Temperature'].round(1)
    st.dataframe(display_df, use_container_width=True, height=400)
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, f"forecast_{selected_name}_{forecast_days}d.csv", "text/csv")

# ── DAILY BREAKDOWN ──
st.markdown('<div class="sec-title"><span class="dot"></span> Daily Breakdown</div>', unsafe_allow_html=True)
res_df['Date'] = res_df['Datetime'].dt.date
daily = res_df.groupby('Date').agg(peak=('Predicted_Load_MW', 'max'), low=('Predicted_Load_MW', 'min'), avg=('Predicted_Load_MW', 'mean'), energy=('Predicted_Load_MW', 'sum'), temp=('Temperature', 'mean')).reset_index()

for _, row in daily.iterrows():
    d = row['Date']
    dn = ARABIC_DAYS.get(d.weekday(), '')
    with st.expander(f"{d} ({dn}) | Peak: {row['peak']:,.0f} MW | Energy: {row['energy']:,.0f} MWh"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peak", f"{row['peak']:,.0f} MW")
        c2.metric("Low", f"{row['low']:,.0f} MW")
        c3.metric("Average", f"{row['avg']:,.0f} MW")
        c4.metric("Avg Temp", f"{row['temp']:.1f} C")

# ── FOOTER ──
st.markdown("""
<div class="footer-honesty">
    <b>Academic Honesty Statement</b><br><br>
    Weather data is <b>100% real</b> from Open-Meteo API.
    Electricity load data is <b>synthetic</b>, engineered to simulate real consumption patterns
    using CAPMAS population weights and thermal correlation relationships.<br><br>
    <b>Region mapping is based on predefined geographic and consumption similarity groups.</b>
</div>
""", unsafe_allow_html=True)
