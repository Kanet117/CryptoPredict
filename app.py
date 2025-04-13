import requests
import yfinance as yf
from transformers import pipeline
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import torch
from bs4 import BeautifulSoup
import datetime
import json

plt.switch_backend('Agg')

TOP_5_CRYPTOS = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Cardano": "ADA-USD",
    "Ripple": "XRP-USD"
}
PROFIT_TARGET = 0.025

MODEL_MAP = {
    "GPT-2 (Standard)": "gpt2",
    "Cerebras BTLM 3B (Detailed Reasoning)": "cerebras/btlm-3b-8k-base"
}
local_generador = None
current_local_model_name = None

HORIZON_MAP = {
    "Short Term (1 month)": {"period": "1mo", "days_lookback": 5, "days_compare": 10, "fetch_period_label": "1 Month", "future_days": 3},
    "Medium Term (1 year)": {"period": "1y", "days_lookback": 30, "days_compare": 60, "fetch_period_label": "1 Year", "future_days": 30},
    "Long Term (5 years)": {"period": "5y", "days_lookback": 90, "days_compare": 180, "fetch_period_label": "5 Years", "future_days": 90}
}

def obtener_datos_cripto(nombre_ticker, period="1y", interval="1d"):
    try:
        print(f"Fetching data for {nombre_ticker} with period: {period}, interval: {interval}")
        cripto = yf.Ticker(nombre_ticker)
        data = cripto.history(period=period, interval=interval, auto_adjust=True)
        if data.empty or 'Close' not in data.columns:
            print(f"Warning: Data for {nombre_ticker} (period: {period}) is empty or missing 'Close' column.")
            return pd.DataFrame()
        if isinstance(data.index, pd.DatetimeIndex):
             if data.index.tz is not None:
                  data.index = data.index.tz_localize(None)
        else:
             data.index = pd.to_datetime(data.index)
        print(f"Successfully fetched {len(data)} data points.")
        return data.dropna(subset=['Close'])
    except Exception as e:
        print(f"Error fetching data for {nombre_ticker} (period: {period}): {e}")
        return pd.DataFrame()

def obtener_noticias(tema='bitcoin'):
    try:
        query_term = f"{tema} cryptocurrency price news"
        url = f"https://news.google.com/rss/search?q={query_term}&hl=en-US&gl=US&ceid=US:en"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, features="xml")
        items = soup.find_all('item', limit=5)
        if not items:
            return "📰 No recent news found matching the criteria."
        noticias = []
        for item in items:
            title = item.title.text if item.title else "No Title"
            pub_date_str = item.pubDate.text if item.pubDate else ""
            try:
                pub_date = pd.to_datetime(pub_date_str, utc=True).tz_convert(None).strftime('%Y-%m-%d %H:%M')
            except Exception:
                 try:
                      pub_date = pd.to_datetime(pub_date_str).strftime('%Y-%m-%d %H:%M')
                 except:
                      noticias.append(f"- {title}")
                      continue
            noticias.append(f"- {title} ({pub_date})")
        return "\n".join(noticias)
    except requests.exceptions.RequestException as e:
         print(f"Error fetching news (Network/HTTP): {e}")
         return f"❌ Error fetching news feed: Network or HTTP issue ({e})."
    except Exception as e:
        print(f"Error parsing news: {e}")
        return "❌ Error parsing news feed."

def load_local_model(model_choice_name):
    global local_generador, current_local_model_name
    model_id = MODEL_MAP.get(model_choice_name)
    if not model_id:
        print(f"Error: Model '{model_choice_name}' not found in MODEL_MAP.")
        local_generador = None
        current_local_model_name = None
        return False
    if model_id == current_local_model_name and local_generador is not None:
        print(f"Local model '{model_choice_name}' ({model_id}) is already loaded.")
        return True
    print(f"Loading local model: {model_choice_name} ({model_id})... This might take a while.")
    try:
        if local_generador is not None:
            print(f"Unloading previous model: {current_local_model_name}")
            del local_generador
            local_generador = None
            current_local_model_name = None
            if torch.cuda.is_available():
                print("Clearing CUDA cache...")
                torch.cuda.empty_cache()
            print("Previous model unloaded.")
        trust_code = True if "cerebras" in model_id.lower() else False
        print(f"Setting trust_remote_code={trust_code} for model {model_id}")
        local_generador = pipeline(
            "text-generation",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1,
            trust_remote_code=trust_code
        )
        current_local_model_name = model_id
        print(f"Local model '{model_choice_name}' ({model_id}) loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading local AI model '{model_choice_name}' ({model_id}): {e}")
        local_generador = None
        current_local_model_name = None
        if torch.cuda.is_available():
             torch.cuda.empty_cache()
        return False

def generar_grafica(precios_df, moneda, horizon_label):
    if precios_df.empty or len(precios_df['Close']) < 2:
        print("Not enough data to generate plot.")
        return None
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(precios_df.index, precios_df['Close'], label='Closing Price', color='dodgerblue', linewidth=1.5)
        ax.set_title(f'Historical Price ({horizon_label}) - {moneda.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Price (USD)', fontsize=10)
        last_date = precios_df.index[-1]
        last_price = precios_df['Close'].iloc[-1]
        ax.plot(last_date, last_price, 'ro', markersize=6, label=f'Last: ${last_price:.2f}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error generating plot: {e}")
        if 'fig' in locals() and isinstance(fig, plt.Figure) and plt.fignum_exists(fig.number):
            plt.close(fig)
        return None

def obtener_recomendacion_basica(precios, horizon_params):
    days_lookback = horizon_params['days_lookback']
    days_compare = horizon_params['days_compare']
    total_days_needed = days_compare
    if len(precios) < total_days_needed:
        return "Wait", f"Insufficient data ({len(precios)} days) for {horizon_params['fetch_period_label']} trend analysis (need {total_days_needed} days)."
    try:
        if days_lookback <= 0 or days_compare <= days_lookback:
             return "Wait", "Invalid lookback/comparison periods configuration."
        if days_lookback >= len(precios) or days_compare > len(precios):
            return "Wait", f"Not enough data points ({len(precios)}) for lookback ({days_lookback})/comparison ({days_compare}) periods."
        recent_period = precios.iloc[-days_lookback:]
        previous_period = precios.iloc[-days_compare:-days_lookback]
        if recent_period.empty or previous_period.empty:
            return "Wait", "Could not define comparison periods (empty slices)."
        precio_reciente_avg = recent_period.mean()
        precio_anterior_avg = previous_period.mean()
        if pd.isna(precio_anterior_avg) or precio_anterior_avg == 0:
            return "Wait", "Previous comparison period average is zero or NaN."
        if pd.isna(precio_reciente_avg):
             return "Wait", "Recent comparison period average is NaN."
        cambio_porcentual = ((precio_reciente_avg - precio_anterior_avg) / precio_anterior_avg) * 100
        threshold = 2.5
        if cambio_porcentual > threshold:
            reason = f"Upward trend detected ({cambio_porcentual:.2f}% avg price increase between recent {days_lookback} days and previous {days_compare-days_lookback} days for {horizon_params['fetch_period_label']} horizon)."
            return "Buy", reason
        elif cambio_porcentual < -threshold:
            reason = f"Downward trend detected ({cambio_porcentual:.2f}% avg price decrease between recent {days_lookback} days and previous {days_compare-days_lookback} days for {horizon_params['fetch_period_label']} horizon)."
            return "Sell", reason
        else:
            reason = f"Price relatively stable ({cambio_porcentual:.2f}% avg price change between recent {days_lookback} days and previous {days_compare-days_lookback} days for {horizon_params['fetch_period_label']} horizon)."
            return "Wait", reason
    except Exception as e:
        print(f"Error calculating recommendation: {e}")
        return "Wait", f"Error during trend calculation: {e}"

def predecir_tiempos_ganancia(precios_series, horizon_params, data_lookback_days=None):
    global PROFIT_TARGET
    period_label = horizon_params['fetch_period_label']
    future_days_est = horizon_params['future_days']

    min_data_points = data_lookback_days if data_lookback_days else 20
    if len(precios_series) < min_data_points:
        return "N/A", "N/A", 0.0, "N/A", f"Insufficient data ({len(precios_series)} points) for predictive estimation (need {min_data_points})."

    try:
        current_price = precios_series.iloc[-1]

        if data_lookback_days:
             lookback_days_actual = min(data_lookback_days, len(precios_series))
        else:
             lookback_days_actual = min(horizon_params['days_lookback'] * 2, len(precios_series))

        x = np.arange(lookback_days_actual)
        y = precios_series.iloc[-lookback_days_actual:].values

        if len(x) < 2 or len(y) < 2:
             return "N/A", "N/A", 0.0, "N/A", "Not enough points for trend calculation after slicing."

        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]

        target_sell_price = current_price * (1 + PROFIT_TARGET)
        target_buy_price = current_price * (1 - PROFIT_TARGET)

        days_to_sell_target = (target_sell_price - current_price) / slope if slope > 0 else float('inf')
        days_to_buy_target = (target_buy_price - current_price) / slope if slope < 0 else float('inf')

        today = datetime.datetime.now().date()
        buy_recommendation = "N/A"
        sell_recommendation = "N/A"
        potential_profit = 0.0
        reasoning = ""
        buy_days_estimate = "N/A"

        if period_label == "1 Month" or data_lookback_days is not None: # Apply day logic for short term or specific lookback
            if 0 < days_to_sell_target <= 30:
                 sell_date = today + datetime.timedelta(days=int(days_to_sell_target))
                 hourly_volatility = precios_series.iloc[-24:].std() if len(precios_series) >= 24 else precios_series.iloc[-lookback_days_actual:].std()
                 time_estimate = "around market open/close" if hourly_volatility > precios_series.mean() * 0.005 else "during mid-day trading"
                 sell_recommendation = f"Potentially around {sell_date.strftime('%Y-%m-%d')} ({time_estimate})"
                 potential_profit = PROFIT_TARGET * 100
                 reasoning += f" Based on recent upward trend (slope: {slope:.2f}/day over last {lookback_days_actual} days), target sell price ${target_sell_price:.2f} might be reached in ~{int(days_to_sell_target)} days."
            elif 0 < days_to_buy_target <= 30:
                 buy_date = today + datetime.timedelta(days=int(days_to_buy_target))
                 time_estimate = "potentially during a dip"
                 buy_recommendation = f"Consider buying dip around {buy_date.strftime('%Y-%m-%d')} ({time_estimate})"
                 potential_profit = ((current_price * (1 + PROFIT_TARGET)) / target_buy_price - 1) * 100 if target_buy_price > 0 else 0.0
                 buy_days_estimate = int(days_to_buy_target) # Store the number of days
                 reasoning += f" Based on recent downward trend (slope: {slope:.2f}/day over last {lookback_days_actual} days), a potential buy target near ${target_buy_price:.2f} might occur in ~{buy_days_estimate} days."
            else:
                 reasoning += f" Current trend (slope: {slope:.2f}/day over last {lookback_days_actual} days) doesn't strongly indicate reaching a +/-{PROFIT_TARGET*100:.1f}% target within the short term (approx. 30 days)."
        else: # Medium/Long term
             if 0 < days_to_sell_target <= future_days_est:
                 sell_estimate = f"within ~{int(days_to_sell_target / 7)}-{int(days_to_sell_target / 7)+4} weeks" if period_label == "1 Year" else f"within ~{int(days_to_sell_target / 30)}-{int(days_to_sell_target / 30)+2} months"
                 sell_recommendation = f"Potentially {sell_estimate}"
                 potential_profit = PROFIT_TARGET * 100
                 reasoning += f" Based on {period_label} trend (slope: {slope:.2f}/day over last {lookback_days_actual} days), target sell price ${target_sell_price:.2f} might be reached {sell_estimate}."
             elif 0 < days_to_buy_target <= future_days_est:
                 buy_estimate = f"within ~{int(days_to_buy_target / 7)}-{int(days_to_buy_target / 7)+4} weeks" if period_label == "1 Year" else f"within ~{int(days_to_buy_target / 30)}-{int(days_to_buy_target / 30)+2} months"
                 buy_recommendation = f"Consider buying dip {buy_estimate}"
                 potential_profit = ((current_price * (1 + PROFIT_TARGET)) / target_buy_price - 1) * 100 if target_buy_price > 0 else 0.0
                 buy_days_estimate = int(days_to_buy_target) # Store days even for longer term if calculated
                 reasoning += f" Based on {period_label} trend (slope: {slope:.2f}/day over last {lookback_days_actual} days), a potential buy target near ${target_buy_price:.2f} might occur {buy_estimate}."
             else:
                 reasoning += f" Current trend (slope: {slope:.2f}/day over last {lookback_days_actual} days) doesn't strongly indicate reaching a +/-{PROFIT_TARGET*100:.1f}% target within the {period_label} horizon ({future_days_est} days)."

        if not reasoning:
             reasoning = f"Based on current trend ({slope:.2f}/day over last {lookback_days_actual} days), reaching a +/-{PROFIT_TARGET*100:.1f}% target within the {period_label} horizon seems unlikely with this simple model."

        return buy_recommendation, sell_recommendation, potential_profit, buy_days_estimate, reasoning.strip()

    except Exception as e:
        print(f"Error during prediction calculation: {e}")
        return "N/A", "N/A", 0.0, "N/A", f"Error during predictive estimation: {e}"

def predecir_cripto_con_grafica(selected_model_name, investment_horizon, moneda_nombre):
    global local_generador, current_local_model_name
    if not moneda_nombre:
        return None, "⚠️ Please enter a cryptocurrency name (e.g., bitcoin, ETH)."
    if not selected_model_name or selected_model_name not in MODEL_MAP:
         return None, "⚠️ Please select a valid AI model."
    if not investment_horizon or investment_horizon not in HORIZON_MAP:
        return None, "⚠️ Please select a valid investment horizon."

    model_id = MODEL_MAP[selected_model_name]
    ai_analysis_available = True # Assume true initially
    if not load_local_model(selected_model_name):
        if local_generador is None:
             error_msg = f"❌ Failed to load local model: {selected_model_name}. Check logs. AI analysis will be skipped."
             print(error_msg)
             ai_analysis_available = False
    else:
         ai_analysis_available = True

    horizon_params = HORIZON_MAP[investment_horizon]
    fetch_period = horizon_params['period']
    fetch_period_label = horizon_params['fetch_period_label']

    moneda_nombre_lower = moneda_nombre.strip().lower()
    simbolos = {
        "bitcoin": "BTC-USD", "ethereum": "ETH-USD", "dogecoin": "DOGE-USD",
        "solana": "SOL-USD", "cardano": "ADA-USD", "ripple": "XRP-USD", "litecoin": "LTC-USD",
    }
    if "-" in moneda_nombre_lower and moneda_nombre_lower.endswith("-usd"):
         simbolo = moneda_nombre_lower.upper()
    else:
         simbolo = simbolos.get(moneda_nombre_lower, moneda_nombre_lower.upper() + "-USD")

    print(f"Fetching data for {simbolo}...")
    datos_precios_df = obtener_datos_cripto(simbolo, period=fetch_period)
    if datos_precios_df.empty:
        error_msg = f"❌ Could not fetch price data for '{moneda_nombre}' (Ticker: {simbolo}, Period: {fetch_period_label}). Please check inputs or try later."
        print(error_msg)
        return None, error_msg
    precios_series = datos_precios_df['Close']

    print("Generating plot...")
    grafica_fig = generar_grafica(datos_precios_df, moneda_nombre, fetch_period_label)

    print(f"Fetching news for {moneda_nombre_lower}...")
    noticias = obtener_noticias(moneda_nombre_lower)

    print("Generating basic recommendation...")
    recomendacion, razon_basica = obtener_recomendacion_basica(precios_series, horizon_params)

    print("Generating timing prediction (speculative)...")
    pred_compra, pred_venta, pred_ganancia, _, razon_prediccion = predecir_tiempos_ganancia(precios_series, horizon_params)
    print(f"Prediction Results: Buy={pred_compra}, Sell={pred_venta}, Profit={pred_ganancia:.2f}%, Reason={razon_prediccion}")

    respuesta_ia_completa = "AI analysis skipped or failed."
    if ai_analysis_available and local_generador is not None:
        print("Preparing prompt for AI...")
        precios_ultimos_df = precios_series[-5:].reset_index()
        precios_ultimos_df.columns = ['Date', 'Price']
        precios_ultimos_df['Date'] = pd.to_datetime(precios_ultimos_df['Date']).dt.strftime('%Y-%m-%d')
        noticias_resumen = noticias.replace('\n', ' ')

        prompt = (
            f"Cryptocurrency: {moneda_nombre.upper()} ({simbolo})\n"
            f"Investment Horizon: {investment_horizon}\n"
            f"Basic Trend Recommendation ({fetch_period_label}): {recomendacion} ({razon_basica})\n"
            f"Speculative Timing Estimation ({PROFIT_TARGET*100:.1f}% Target):\n"
            f"  - Potential Buy Timing: {pred_compra}\n"
            f"  - Potential Sell Timing: {pred_venta}\n"
            f"  - Estimated Potential Profit: {pred_ganancia:.2f}%\n"
            f"  - Basis for Estimation: {razon_prediccion}\n"
            f"Recent Prices (last 5 days):\n{precios_ultimos_df.to_string(index=False)}\n"
            f"Recent News Headlines: {noticias_resumen}\n\n"
            f"Task: Provide a detailed reasoning for the {recomendacion} recommendation considering the {investment_horizon} horizon. Analyze the price trend, the speculative timing estimation, and the recent news. Explain *why* the recommendation makes sense (or doesn't) in this context. Focus ONLY on the reasoning.\n\n"
            f"Detailed Reasoning:"
        )

        try:
            print(f"Generating text with model: {current_local_model_name}")
            max_new_toks = 250
            eos_token_id = getattr(local_generador.tokenizer, 'eos_token_id', 50256)

            generated_output = local_generador(
                prompt,
                max_new_tokens=max_new_toks,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                pad_token_id=eos_token_id
            )[0]['generated_text']

            respuesta_ia_raw = generated_output[len(prompt):].strip()
            razonamiento_detallado = respuesta_ia_raw if respuesta_ia_raw else "AI failed to generate reasoning."
            display_model_name = selected_model_name

            respuesta_ia_completa = f"🧠 **AI Analysis ({display_model_name} - {investment_horizon}):**\n   - **Detailed Reasoning:** {razonamiento_detallado}"

        except Exception as e:
            print(f"Error during AI generation with {current_local_model_name}: {e}")
            respuesta_ia_completa = f"❌ Error generating AI analysis with {current_local_model_name}. Details: {e}"
    elif not ai_analysis_available:
         respuesta_ia_completa = f"⚠️ AI analysis skipped because the model '{selected_model_name}' could not be loaded."

    recomendacion_emoji = {"Buy": "✅", "Sell": "❌", "Wait": "🤔"}
    precios_ultimos_tabla_html = precios_series[-5:].reset_index().rename(columns={'index': 'Date', 'Close': 'Price (USD)'})
    precios_ultimos_tabla_html['Date'] = pd.to_datetime(precios_ultimos_tabla_html['Date']).dt.strftime('%Y-%m-%d')
    precios_ultimos_tabla_html['Price (USD)'] = precios_ultimos_tabla_html['Price (USD)'].map('{:.2f}'.format)
    tabla_html_output = precios_ultimos_tabla_html.to_html(index=False, justify='center', border=0, classes='dataframe', escape=False)

    prediccion_md = f"""
### 📈 Speculative Timing Estimation ({PROFIT_TARGET*100:.1f}% Profit Target)
*Warning: Highly speculative, not financial advice.*
- **Potential Buy Timing:** {pred_compra}
- **Potential Sell Timing:** {pred_venta}
- **Estimated Potential Profit:** {pred_ganancia:.2f}% (if targets met)
- **Basis for Estimation:** {razon_prediccion}
"""

    final_model_name_used = selected_model_name

    resultado_md = f"""
## Analysis for {moneda_nombre.upper()} ({simbolo}) - {investment_horizon}

---

### {recomendacion_emoji[recomendacion]} Basic Recommendation: **{recomendacion}**
*{razon_basica}*

---

{prediccion_md}

---

### 💰 Last 5 Closing Prices:
<div style="width: fit-content; margin: auto; font-family: monospace;">
{tabla_html_output}
</div>
<br/>

---

### 📰 Recent News:
{noticias}

---

{respuesta_ia_completa}

---
*Disclaimer: This is not financial advice. Analysis uses historical data, news, and AI generation for the selected horizon ({investment_horizon}). Predictions are speculative. Model used: {final_model_name_used}.*
"""
    print("Analysis complete.")
    return grafica_fig, resultado_md


def analizar_mejores_opciones():
    global TOP_5_CRYPTOS
    print("Analyzing top 5 cryptocurrencies...")
    results = []
    short_term_horizon = HORIZON_MAP["Short Term (1 month)"]
    lookback_days_for_top5 = 10

    for name, ticker in TOP_5_CRYPTOS.items():
        print(f"Analyzing {name} ({ticker})...")
        data_10d = obtener_datos_cripto(ticker, period=f"{lookback_days_for_top5}d", interval="1d")
        data_2d = obtener_datos_cripto(ticker, period="2d", interval="1d")

        if data_10d.empty or data_2d.empty or len(data_2d) < 2 or len(data_10d) < lookback_days_for_top5:
            results.append({
                "Name": name, "Ticker": ticker, "Price": "N/A", "Change (24h)": "N/A",
                "Buy Days Est.": "N/A", "Reason": f"Could not fetch sufficient data ({len(data_10d)}/{lookback_days_for_top5} days)."
            })
            continue

        current_price = data_10d['Close'].iloc[-1]
        price_yesterday = data_2d['Close'].iloc[-2]
        change_24h = ((current_price - price_yesterday) / price_yesterday) * 100 if price_yesterday else 0

        _, _, _, buy_days_est, reason = predecir_tiempos_ganancia(data_10d['Close'], short_term_horizon, data_lookback_days=lookback_days_for_top5)

        results.append({
            "Name": name, "Ticker": ticker, "Price": f"${current_price:.2f}",
            "Change (24h)": f"{change_24h:+.2f}%", "Buy Days Est.": buy_days_est, "Reason": reason
        })

    md_output = f"## 📊 Top 5 Crypto Short-Term Outlook (Speculative - based on last {lookback_days_for_top5} days)\n"
    md_output += "*(Analysis might take a minute...)*\n\n"
    md_output += "| Name | Ticker | Current Price | Change (24h) | Est. Buy Dip (Days) | Basis |\n"
    md_output += "|---|---|---|---|---|---|\n"
    for r in results:
        reason_short = r['Reason'].split('.')[0] + '.' if '.' in r['Reason'] else r['Reason']
        reason_short = (reason_short[:100] + '...') if len(reason_short) > 100 else reason_short
        md_output += f"| {r['Name']} | {r['Ticker']} | {r['Price']} | {r['Change (24h)']} | {r['Buy Days Est.']} | {reason_short} |\n"

    md_output += f"\n*Disclaimer: Analysis based on limited {lookback_days_for_top5}-day data and simple trend estimation. Estimated buy days predict time to reach a potential {PROFIT_TARGET*100:.1f}% lower price based *only* on the downward trend in the lookback period. Highly speculative. Not financial advice.*"
    print("Top 5 analysis complete.")
    return md_output


print("Setting up Gradio interface...")

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky")) as iface:
    gr.Markdown("# 🔮 Crypto Analyzer AI (v2 - Local)")
    gr.Markdown("Enter a cryptocurrency, select an AI model (Cerebras for more detailed reasoning) and an investment horizon. Get price charts, news, recommendations, speculative timing estimates, and AI-powered reasoning. You can also analyze the top 5 coins.")

    with gr.Row():
        with gr.Column(scale=1):
            input_model = gr.Radio(
                choices=list(MODEL_MAP.keys()),
                label="🤖 Select AI Model",
                value="GPT-2 (Standard)"
            )
            gr.Markdown("---") # Spacer
            input_horizon = gr.Radio(
                choices=list(HORIZON_MAP.keys()),
                label="⏱️ Select Investment Horizon",
                value="Medium Term (1 year)"
            )
            gr.Markdown("---") # Spacer
            input_crypto = gr.Textbox(
                label="Enter Cryptocurrency Name or Symbol",
                placeholder="E.g.: bitcoin, ETH-USD, Solana..."
            )
            gr.Markdown("---") # Spacer
            analyze_button = gr.Button("Analyze Cryptocurrency", variant="primary")
            top5_button = gr.Button("Analyze Top 5 Coins", variant="secondary")

        with gr.Column(scale=2):
            output_plot = gr.Plot(label="Price Chart")
            output_markdown = gr.Markdown(label="Analysis and Recommendation")
            output_top5 = gr.Markdown(label="Top 5 Analysis")

    analyze_button.click(
        fn=predecir_cripto_con_grafica,
        inputs=[input_model, input_horizon, input_crypto],
        outputs=[output_plot, output_markdown]
    )

    top5_button.click(
        fn=analizar_mejores_opciones,
        inputs=[],
        outputs=[output_top5]
    )

    gr.Examples(
         examples=[
              ["GPT-2 (Standard)", "Medium Term (1 year)", "bitcoin"],
              ["Cerebras BTLM 3B (Detailed Reasoning)", "Long Term (5 years)", "ethereum"],
              ["GPT-2 (Standard)", "Short Term (1 month)", "DOGE-USD"]
         ],
         inputs=[input_model, input_horizon, input_crypto]
    )

print("Launching Gradio app...")
if __name__ == "__main__":
    iface.launch(debug=True)

