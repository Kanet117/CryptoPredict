# Import necessary libraries
import requests
import yfinance as yf
from transformers import pipeline
from bs4 import BeautifulSoup
import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io # Needed for handling plot image in memory

# --- Configuration ---
# Set matplotlib backend to Agg to avoid GUI conflicts in environments without a display
plt.switch_backend('Agg')

# --- Data Fetching Functions ---

def obtener_datos_cripto(nombre_ticker):
    """
    Fetches historical cryptocurrency data for the last year.

    Args:
        nombre_ticker (str): The ticker symbol (e.g., "BTC-USD").

    Returns:
        pandas.DataFrame: DataFrame with historical data, or empty DataFrame on error.
    """
    try:
        cripto = yf.Ticker(nombre_ticker)
        # Fetch data for 1 year
        data = cripto.history(period="1y", interval="1d", auto_adjust=True)
        if data.empty or 'Close' not in data.columns:
            print(f"Warning: Data for {nombre_ticker} is empty or missing 'Close' column.")
            return pd.DataFrame() # Return empty DataFrame
        # Ensure index is datetime for plotting
        data.index = pd.to_datetime(data.index)
        return data.dropna(subset=['Close'])
    except Exception as e:
        print(f"Error fetching data for {nombre_ticker}: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

def obtener_noticias(tema='bitcoin'):
    """
    Fetches recent news headlines related to a cryptocurrency topic from Google News RSS (English).

    Args:
        tema (str): The topic to search for (e.g., "bitcoin").

    Returns:
        str: A string containing formatted news headlines, or an error message.
    """
    try:
        # Format the search query for Google News RSS in English
        url = f"https://news.google.com/rss/search?q={tema}+cryptocurrency&hl=en-US&gl=US&ceid=US:en" # Changed to English
        headers = { # Add a user-agent to mimic a browser
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        r = requests.get(url, headers=headers, timeout=10) # Added timeout
        r.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Use 'lxml' for parsing XML content
        soup = BeautifulSoup(r.text, features="xml") # Changed parser to 'xml' for RSS

        # Find all 'item' tags (news articles), limit to 5
        items = soup.find_all('item', limit=5)
        if not items:
            return "📰 No recent news found."

        # Extract titles and format them
        noticias = [f"- {item.title.text}" for item in items]
        return "\n".join(noticias)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching news (Request failed): {e}")
        return "❌ Could not fetch news due to network error."
    except Exception as e:
        print(f"Error parsing news: {e}")
        return "❌ Error processing news feed."

# --- AI Model Initialization ---
# Load the text generation pipeline from Hugging Face
print("Loading AI model...")
try:
    # Using gpt2 as before. Consider 'distilgpt2' for faster inference if needed.
    generador = pipeline("text-generation", model="gpt2")
    print("AI model loaded successfully.")
except Exception as e:
    print(f"Error loading AI model: {e}")
    generador = None # Set generator to None if loading fails

# --- Plotting Function ---
def generar_grafica(precios_df, moneda):
    """
    Generates a line plot of closing prices.

    Args:
        precios_df (pd.DataFrame): DataFrame containing price data with a DatetimeIndex.
        moneda (str): Name of the cryptocurrency for the title.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object, or None if data is insufficient.
    """
    if precios_df.empty or len(precios_df['Close']) < 2:
        print("Not enough data to generate plot.")
        return None # Cannot plot without data

    try:
        fig, ax = plt.subplots(figsize=(10, 5)) # Create figure and axes
        ax.plot(precios_df.index, precios_df['Close'], label='Closing Price', color='dodgerblue')

        # Formatting the plot (in English)
        ax.set_title(f'Historical Price (1 Year) - {moneda.upper()}', fontsize=14)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Price (USD)', fontsize=10)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig # Return the figure object
    except Exception as e:
        print(f"Error generating plot: {e}")
        return None
    finally:
        # Close the plot to free memory, Gradio takes care of displaying the figure object
        if 'fig' in locals():
             plt.close(fig)


# --- Recommendation Logic ---
def obtener_recomendacion_basica(precios):
    """
    Generates a basic buy/sell/wait recommendation and a brief reason based on recent price trend.

    Args:
        precios (pd.Series): Series of closing prices.

    Returns:
        tuple: (str, str) - Recommendation ("Buy", "Sell", "Wait") and a brief reason.
    """
    if len(precios) < 10: # Need enough data for a trend
        return "Wait", "Insufficient data for trend analysis."

    # Compare the average of the last 5 days with the average of the 5 days before that
    precio_reciente = precios[-5:].mean()
    precio_anterior = precios[-10:-5].mean()

    # Avoid division by zero if precio_anterior is 0
    if precio_anterior == 0:
         return "Wait", "Previous price period average is zero, cannot calculate trend."

    cambio_porcentual = ((precio_reciente - precio_anterior) / precio_anterior) * 100

    if cambio_porcentual > 2.5: # If price increased by more than 2.5% recently
        reason = f"Recent upward trend detected ({cambio_porcentual:.2f}% change in last 10 days)."
        return "Buy", reason
    elif cambio_porcentual < -2.5: # If price decreased by more than 2.5% recently
        reason = f"Recent downward trend detected ({cambio_porcentual:.2f}% change in last 10 days)."
        return "Sell", reason
    else: # Otherwise, wait
        reason = f"Price relatively stable ({cambio_porcentual:.2f}% change in last 10 days)."
        return "Wait", reason

# --- Main Prediction Function ---
def predecir_cripto_con_grafica(moneda_nombre):
    """
    Main function to analyze a cryptocurrency, generate plot, news, recommendation, and AI analysis.

    Args:
        moneda_nombre (str): Name or symbol of the cryptocurrency (e.g., "bitcoin", "ETH").

    Returns:
        tuple: (matplotlib.figure.Figure or None, str) - Plot object and Markdown formatted results.
    """
    if not moneda_nombre:
        return None, "⚠️ Please enter a cryptocurrency name (e.g., bitcoin, ETH)."

    # Standardize input and find ticker symbol
    moneda_nombre_lower = moneda_nombre.strip().lower()
    simbolos = { # Common symbols map
        "bitcoin": "BTC-USD",
        "ethereum": "ETH-USD",
        "dogecoin": "DOGE-USD",
        "solana": "SOL-USD",
        "cardano": "ADA-USD",
        "ripple": "XRP-USD",
        "litecoin": "LTC-USD",
    }
    simbolo = simbolos.get(moneda_nombre_lower, moneda_nombre_lower.upper() + "-USD")

    # 1. Get price data
    print(f"Fetching data for {simbolo}...")
    datos_precios_df = obtener_datos_cripto(simbolo)

    if datos_precios_df.empty:
        error_msg = f"❌ Could not fetch price data for '{moneda_nombre}' (Ticker: {simbolo}). Please check the name/symbol."
        print(error_msg)
        return None, error_msg # Return None for plot, and the error message

    precios_series = datos_precios_df['Close']

    # 2. Generate plot
    print("Generating plot...")
    grafica_fig = generar_grafica(datos_precios_df, moneda_nombre)

    # 3. Get news (English)
    print(f"Fetching news for {moneda_nombre_lower}...")
    noticias = obtener_noticias(moneda_nombre_lower) # Fetches English news now

    # 4. Get basic recommendation and reason
    print("Generating recommendation...")
    recomendacion, razon_basica = obtener_recomendacion_basica(precios_series)

    # 5. Prepare prompt for AI (if model loaded)
    respuesta_ia_completa = "AI analysis not available (model loading failed)."
    if generador:
        print("Generating AI analysis...")
        # Prepare data for prompt
        precios_ultimos_df = precios_series[-5:].reset_index() # Get last 5 prices as DataFrame
        precios_ultimos_df.columns = ['Date', 'Price'] # Rename columns
        precios_ultimos_df['Date'] = precios_ultimos_df['Date'].dt.strftime('%Y-%m-%d') # Format date
        # Create HTML table for prices
        tabla_precios_html = precios_ultimos_df.to_html(index=False, justify='center', classes='price-table')


        noticias_resumen = noticias.replace('\n', ' ') # Flatten news for prompt

        # Modified prompt asking AI to elaborate on the basic recommendation
        prompt = (
            f"Cryptocurrency: {moneda_nombre.upper()}\n"
            f"Basic Recommendation based on recent trend: {recomendacion} ({razon_basica})\n"
            f"Recent Prices (last 5 days):\n{precios_ultimos_df.to_string(index=False)}\n" # Use string format for prompt context
            f"Recent News Headlines: {noticias_resumen}\n\n"
            f"Task: Elaborate on the basic recommendation ({recomendacion}). Provide a more detailed reasoning considering the recent prices and news headlines. "
            f"Should one really {recomendacion.lower()}? Why or why not? "
            f"Also, include one interesting cryptocurrency fun fact.\n\n"
            f"Format your response as:\n"
            f"Detailed Reasoning: [Your detailed analysis here, explaining why the recommendation makes sense or might need caution, based on news/prices]\n"
            f"Fun Fact: [Your fun fact here]"
        )

        try:
            generated_output = generador(
                prompt,
                max_length=250,          # Max total length (prompt + new tokens) - increased slightly
                max_new_tokens=150,     # Max tokens to generate - increased slightly
                num_return_sequences=1,
                do_sample=True,
                temperature=0.75,       # Slightly increased temperature for potentially more nuanced reasoning
                top_k=50,
                pad_token_id=generador.tokenizer.eos_token_id
            )[0]['generated_text']

            respuesta_ia_raw = generated_output[len(prompt):].strip()

            # Parsing the AI response
            razonamiento_detallado = "Could not extract detailed reasoning."
            dato_curioso = "Could not extract fun fact."

            if "Detailed Reasoning:" in respuesta_ia_raw:
                parts = respuesta_ia_raw.split("Detailed Reasoning:", 1)[1]
                if "Fun Fact:" in parts:
                    razonamiento_detallado = parts.split("Fun Fact:", 1)[0].strip()
                    dato_curioso = parts.split("Fun Fact:", 1)[1].strip()
                else:
                    razonamiento_detallado = parts.strip()
            elif "Fun Fact:" in respuesta_ia_raw:
                 dato_curioso = respuesta_ia_raw.split("Fun Fact:", 1)[1].strip()

            respuesta_ia_completa = f"🧠 **AI Analysis:**\n   - **Detailed Reasoning:** {razonamiento_detallado}\n   - **Fun Fact:** {dato_curioso}"

        except Exception as e:
            print(f"Error during AI generation: {e}")
            respuesta_ia_completa = "❌ Error generating AI analysis."
    else:
         print("AI model not loaded, skipping generation.")


    # 6. Format final output in Markdown (English)
    recomendacion_emoji = {"Buy": "✅", "Sell": "❌", "Wait": "🤔"}

    # Prepare HTML table for Markdown output
    precios_ultimos_tabla_html = precios_series[-5:].reset_index().rename(columns={'index': 'Date', 'Close': 'Price (USD)'})
    precios_ultimos_tabla_html['Date'] = precios_ultimos_tabla_html['Date'].dt.strftime('%Y-%m-%d')
    precios_ultimos_tabla_html['Price (USD)'] = precios_ultimos_tabla_html['Price (USD)'].map('{:.2f}'.format)
    tabla_html_output = precios_ultimos_tabla_html.to_html(index=False, justify='center', border=0, classes='table table-sm table-striped') # Basic styling classes


    resultado_md = f"""
## Analysis for {moneda_nombre.upper()} ({simbolo})

---

### {recomendacion_emoji[recomendacion]} Recommendation: **{recomendacion}**
*{razon_basica}*

---

### 💰 Last 5 Closing Prices:
{tabla_html_output}

<br/> ---

### 📰 Recent News:
{noticias}

---

{respuesta_ia_completa}

---

"""
    print("Analysis complete.")
    return grafica_fig, resultado_md # Return plot figure and markdown string

# --- Gradio Interface (English) ---
print("Setting up Gradio interface...")
iface = gr.Interface(
    fn=predecir_cripto_con_grafica,
    inputs=gr.Textbox(
        label="Cryptocurrency Name", # English Label
        placeholder="E.g.: bitcoin, ETH, DOGE, Solana..." # English Placeholder
    ),
    outputs=[
        gr.Plot(label="Price Chart (1 Year)"), # English Label
        gr.Markdown(label="Analysis and Recommendation") # English Label
    ],
    title="🔮 Crypto Analyzer AI (Hugging Face)", # English Title
    description="Enter a cryptocurrency name to get historical prices, recent news, a basic recommendation, and AI-generated analysis.", # English Description
    examples=[["bitcoin"], ["ethereum"], ["solana"]],
    allow_flagging='never',
    theme=gr.themes.Soft()
)

# Launch the interface
print("Launching Gradio app...")
if __name__ == "__main__":
    iface.launch()

