import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
from auth import init_auth_db, login_user

# Lista de ETFs populares para analizar
def get_default_etfs():
    """Retorna la lista de ETFs por defecto"""
    return {
        'VOO': 'Vanguard S&P 500 ETF',
        'VTI': 'Vanguard Total Stock Market ETF',
        'QQQ': 'Invesco QQQ Trust',
        'VGT': 'Vanguard Information Technology ETF',
        'VYM': 'Vanguard High Dividend Yield ETF',
        'VNQ': 'Vanguard Real Estate ETF',
        'ARKK': 'ARK Innovation ETF',
        'IEMG': 'iShares Core MSCI Emerging Markets ETF',
        'VEA': 'Vanguard FTSE Developed Markets ETF',
        'SCHD': 'Schwab US Dividend Equity ETF',
        'GOOGL': 'Google',
        'EPD': 'iShares ESG Preferred ETF',
    }

# Configuración de la página
st.set_page_config(page_title="ETF Investment Dashboard", layout="wide")

# Inicializar la base de datos
def init_db():
    conn = sqlite3.connect('etf.db')
    c = conn.cursor()
    
    # Tabla para inversiones
    c.execute('''CREATE TABLE IF NOT EXISTS investments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT,
                  etf_symbol TEXT,
                  amount REAL,
                  shares REAL,
                  price REAL)''')
    
    # Tabla para ETFs
    c.execute('''CREATE TABLE IF NOT EXISTS etfs
                 (symbol TEXT PRIMARY KEY,
                  name TEXT,
                  last_price REAL,
                  market_cap REAL,
                  volume REAL,
                  last_update TEXT)''')
    
    # Insertar ETFs por defecto si la tabla está vacía
    c.execute("SELECT COUNT(*) FROM etfs")
    if c.fetchone()[0] == 0:
        default_etfs = get_default_etfs()
        for symbol, name in default_etfs.items():
            c.execute('''INSERT OR IGNORE INTO etfs (symbol, name) 
                        VALUES (?, ?)''', (symbol, name))
    
    conn.commit()
    return conn

def load_etfs(conn):
    """Carga los ETFs desde la base de datos"""
    c = conn.cursor()
    c.execute("SELECT symbol, name FROM etfs")
    results = c.fetchall()
    return {row[0]: row[1] for row in results} if results else get_default_etfs()

def get_all_etfs():
    """Obtiene una lista de todos los ETFs disponibles en yfinance"""
    try:
        # Lista base de ETFs populares para asegurar que siempre tengamos algunos buenos ETFs
        base_etfs = {
            'VOO': 'Vanguard S&P 500 ETF',
            'VTI': 'Vanguard Total Stock Market ETF',
            'QQQ': 'Invesco QQQ Trust',
            'SPY': 'SPDR S&P 500 ETF',
            'IVV': 'iShares Core S&P 500 ETF'
        }
        
        # Lista de proveedores principales de ETFs
        providers = [
            'Vanguard', 'iShares', 'SPDR', 'Invesco', 'Schwab',
            'First Trust', 'Global X', 'ARK', 'ProShares', 'Direxion'
        ]
        
        etfs = {}
        
        # Primero agregamos los ETFs base
        etfs.update(base_etfs)
        
        # Buscamos ETFs por proveedor
        for provider in providers:
            try:
                # Usar yfinance para buscar ETFs
                ticker = yf.Ticker(provider.replace(' ', '-'))
                if hasattr(ticker, 'info'):
                    similar = ticker.recommendations if hasattr(ticker, 'recommendations') else pd.DataFrame()
                    if not similar.empty:
                        for symbol in similar.index[:50]:  # Tomar los primeros 50 de cada proveedor
                            try:
                                t = yf.Ticker(symbol)
                                if hasattr(t, 'info'):
                                    info = t.info
                                    if 'ETF' in info.get('quoteType', '').upper():
                                        etfs[symbol] = info.get('longName', f"{symbol} ETF")
                            except:
                                continue
            except:
                continue
        
        # Buscar ETFs adicionales por categorías populares
        categories = ['Technology', 'Financial', 'Energy', 'Healthcare', 'Real Estate']
        for category in categories:
            search_term = f"{category} ETF"
            try:
                # Aquí podrías usar una API más específica para búsqueda de ETFs
                # Por ahora usamos una aproximación simple
                ticker = yf.Ticker(search_term.replace(' ', '-'))
                if hasattr(ticker, 'info'):
                    similar = ticker.recommendations if hasattr(ticker, 'recommendations') else pd.DataFrame()
                    if not similar.empty:
                        for symbol in similar.index[:20]:  # Tomar los primeros 20 de cada categoría
                            try:
                                t = yf.Ticker(symbol)
                                if hasattr(t, 'info'):
                                    info = t.info
                                    if 'ETF' in info.get('quoteType', '').upper():
                                        etfs[symbol] = info.get('longName', f"{symbol} ETF")
                            except:
                                continue
            except:
                continue
        
        return etfs
    
    except Exception as e:
        print(f"Error al obtener lista de ETFs: {str(e)}")
        return get_default_etfs()

def update_etfs(conn, force_update=False):
    """Actualiza la información de los ETFs y encuentra los mejores para invertir"""
    c = conn.cursor()
    current_time = datetime.now()
    
    # Verificar última actualización
    c.execute("SELECT MAX(last_update) FROM etfs")
    last_update = c.fetchone()[0]
    
    # Actualizar si no hay datos o si han pasado más de 24 horas
    if force_update or not last_update or (
        current_time - datetime.strptime(last_update, '%Y-%m-%d %H:%M:%S')
    ).total_seconds() > 86400:
        
        # Obtener todos los ETFs disponibles
        all_etfs = get_all_etfs()
        
        # Lista para almacenar resultados de análisis
        analysis_results = []
        
        # Analizar cada ETF
        with st.spinner(f"Analizando {len(all_etfs)} ETFs..."):
            for symbol, name in all_etfs.items():
                try:
                    df = get_etf_data(symbol, period='1y')
                    if df is not None and not df.empty:
                        current_price = df['Close'].iloc[-1]
                        volume = df['Volume'].mean()
                        market_cap = current_price * volume
                        
                        # Calcular métricas
                        returns = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
                        volatility = df['Close'].pct_change().std()
                        rsi = calculate_rsi(df)[-1]
                        sma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
                        sma_200 = df['Close'].rolling(window=200).mean().iloc[-1]
                        
                        # Calcular puntuación
                        score = 0
                        # Tendencia alcista
                        score += 1 if current_price > sma_50 else 0
                        score += 1 if sma_50 > sma_200 else 0
                        # RSI en rango óptimo (30-70)
                        score += 1 if 30 <= rsi <= 70 else 0
                        # Rendimiento positivo
                        score += 1 if returns > 0 else 0
                        # Volatilidad moderada
                        score += 1 if volatility < 0.02 else 0
                        # Volumen significativo
                        score += 1 if volume > 100000 else 0
                        # Precio cerca del SMA 50
                        score += 1 if abs((current_price - sma_50) / sma_50) < 0.05 else 0
                        
                        analysis_results.append({
                            'symbol': symbol,
                            'name': name,
                            'score': score,
                            'current_price': current_price,
                            'volume': volume,
                            'market_cap': market_cap,
                            'returns': returns,
                            'volatility': volatility,
                            'rsi': rsi
                        })
                except Exception as e:
                    print(f"Error al analizar {symbol}: {str(e)}")
                    continue
        
        # Ordenar por puntuación y seleccionar los 20 mejores
        df_analysis = pd.DataFrame(analysis_results)
        if not df_analysis.empty:
            df_analysis = df_analysis.sort_values(['score', 'market_cap'], ascending=[False, False]).head(20)
            
            # Actualizar la base de datos con los mejores ETFs
            for _, row in df_analysis.iterrows():
                c.execute("""INSERT OR REPLACE INTO etfs 
                           (symbol, name, last_price, volume, last_update)
                           VALUES (?, ?, ?, ?, ?)""",
                        (row['symbol'], row['name'], row['current_price'], 
                         row['volume'], current_time.strftime('%Y-%m-%d %H:%M:%S')))
            
            conn.commit()
            
            # Mostrar resumen de actualización
            st.success(f"✅ Se han encontrado y analizado los mejores {len(df_analysis)} ETFs")
            
            # Mostrar tabla de resultados
            st.write("📊 Top ETFs encontrados:")
            summary_df = df_analysis[['symbol', 'name', 'score', 'returns', 'rsi', 'volatility']].copy()
            summary_df['returns'] = summary_df['returns'].map('{:,.2f}%'.format)
            summary_df['volatility'] = summary_df['volatility'].map('{:,.2%}'.format)
            summary_df['rsi'] = summary_df['rsi'].map('{:,.2f}'.format)
            st.dataframe(summary_df)
    
    return load_etfs(conn)

# Función para calcular el RSI
def calculate_rsi(data, periods=14):
    """Calcula el RSI (Relative Strength Index) para una serie de precios"""
    # Calcular cambios
    delta = data['Close'].diff()
    
    # Separar ganancias y pérdidas
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    
    # Calcular RS y RSI
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Función para obtener datos históricos de ETF
def get_etf_data(symbol, period='1y'):
    try:
        etf = yf.Ticker(symbol)
        hist = etf.history(period=period)
        return hist
    except:
        return None

# Función para calcular métricas técnicas
def calculate_technical_metrics(df):
    if df is None or df.empty:
        return None
    
    metrics = {}
    
    # Medias móviles
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    df['RSI'] = calculate_rsi(df)
    
    # Volatilidad
    df['Daily_Return'] = df['Close'].pct_change()
    volatility = df['Daily_Return'].std() * (252 ** 0.5)  # Anualizada
    
    # Rendimiento
    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    
    metrics['current_price'] = df['Close'].iloc[-1]
    metrics['sma_50'] = df['SMA_50'].iloc[-1]
    metrics['sma_200'] = df['SMA_200'].iloc[-1]
    metrics['rsi'] = df['RSI'].iloc[-1]
    metrics['volatility'] = volatility
    metrics['total_return'] = total_return
    
    return metrics

# Función para evaluar precio de entrada
def evaluate_entry_price(df, metrics):
    if df is None or metrics is None:
        return None
    
    current_price = metrics['current_price']
    sma_200 = metrics['sma_200']
    rsi = metrics['rsi']
    
    # Calcular niveles de soporte y resistencia
    recent_low = df['Low'].tail(50).min()
    recent_high = df['High'].tail(50).max()
    
    # Evaluar precio objetivo de entrada
    if current_price < sma_200:
        entry_target = current_price  # Buen momento para entrar
    else:
        entry_target = sma_200  # Esperar pullback a la media móvil de 200 días
    
    # Ajustar según RSI
    if rsi < 30:  # Sobrevendido
        entry_target = current_price  # Buen momento para entrar
    elif rsi > 70:  # Sobrecomprado
        entry_target = recent_low  # Esperar corrección
    
    return {
        'entry_target': entry_target,
        'support': recent_low,
        'resistance': recent_high
    }

# Función para analizar ETF y dar recomendaciones
def analyze_etf(df):
    if df is None or df.empty:
        return "No hay datos suficientes para analizar"
    
    metrics = calculate_technical_metrics(df)
    if metrics is None:
        return []
    
    signals = []
    
    # Análisis de tendencia
    if metrics['sma_50'] > metrics['sma_200']:
        signals.append("✅ Tendencia alcista (Golden Cross)")
    elif metrics['sma_50'] < metrics['sma_200']:
        signals.append("⚠️ Tendencia bajista (Death Cross)")
    
    # Análisis RSI
    if metrics['rsi'] < 30:
        signals.append("💹 ETF sobrevendido (RSI < 30) - Posible oportunidad de compra")
    elif metrics['rsi'] > 70:
        signals.append("⚠️ ETF sobrecomprado (RSI > 70) - Considerar esperar")
    
    # Análisis de volatilidad
    if metrics['volatility'] < 0.15:
        signals.append("📊 Baja volatilidad - Más estable")
    elif metrics['volatility'] > 0.25:
        signals.append("📈 Alta volatilidad - Mayor riesgo")
    
    # Rendimiento
    if metrics['total_return'] > 0:
        signals.append(f"📈 Rendimiento total: +{metrics['total_return']:.2f}%")
    else:
        signals.append(f"📉 Rendimiento total: {metrics['total_return']:.2f}%")
    
    # Evaluación de precio de entrada
    entry_analysis = evaluate_entry_price(df, metrics)
    if entry_analysis:
        if metrics['current_price'] <= entry_analysis['entry_target']:
            signals.append(f"🎯 Buen momento para entrar - Precio actual: ${metrics['current_price']:.2f}")
        else:
            signals.append(f"⏳ Precio objetivo de entrada: ${entry_analysis['entry_target']:.2f}")
        signals.append(f"📊 Soporte: ${entry_analysis['support']:.2f} | Resistencia: ${entry_analysis['resistance']:.2f}")
    
    return signals

# Función para analizar múltiples ETFs
def analyze_multiple_etfs(etfs_dict):
    """Analiza múltiples ETFs y retorna un DataFrame con los resultados"""
    results = []
    
    for symbol, name in etfs_dict.items():
        df = get_etf_data(symbol)
        if df is not None and not df.empty:
            current_price = df['Close'].iloc[-1]
            returns = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
            volatility = df['Close'].pct_change().std()
            rsi = calculate_rsi(df)[-1]
            sma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
            sma_200 = df['Close'].rolling(window=200).mean().iloc[-1]
            
            # Calcular puntuación
            score = 0
            # Tendencia alcista
            score += 1 if current_price > sma_50 else 0
            score += 1 if sma_50 > sma_200 else 0
            # RSI en rango óptimo (30-70)
            score += 1 if 30 <= rsi <= 70 else 0
            # Rendimiento positivo
            score += 1 if returns > 0 else 0
            # Volatilidad moderada
            score += 1 if volatility < 0.02 else 0
            # Volumen significativo
            score += 1 if df['Volume'].mean() > 100000 else 0
            # Precio cerca del SMA 50
            score += 1 if abs((current_price - sma_50) / sma_50) < 0.05 else 0
            
            # Calcular precio objetivo
            if current_price < sma_50:
                entry_target = min(current_price * 1.02, sma_50)
            else:
                entry_target = max(current_price * 0.98, sma_50)
            
            results.append({
                'symbol': symbol,
                'name': name,
                'price': current_price,
                'return': returns,
                'volatility': volatility,
                'rsi': rsi,
                'score': score,
                'entry_target': entry_target
            })
    
    return pd.DataFrame(results).sort_values('score', ascending=False)

# Función para actualizar una inversión
def update_investment(conn, id, date, etf_symbol, amount, shares, price):
    c = conn.cursor()
    c.execute("""UPDATE investments 
                 SET date = ?, etf_symbol = ?, amount = ?, shares = ?, price = ?
                 WHERE id = ?""",
              (date, etf_symbol, amount, shares, price, id))
    conn.commit()

# Función para eliminar una inversión
def delete_investment(conn, id):
    c = conn.cursor()
    c.execute("DELETE FROM investments WHERE id = ?", (id,))
    conn.commit()

# Función para agregar una nueva inversión
def add_investment(conn, date, etf_symbol, amount, shares, price):
    c = conn.cursor()
    c.execute("""INSERT INTO investments (date, etf_symbol, amount, shares, price)
                 VALUES (?, ?, ?, ?, ?)""",
              (date, etf_symbol, amount, shares, price))
    conn.commit()

# Función para analizar una posición de inversión
def analyze_investment_position(investment_data, current_data):
    """Analiza una posición de inversión y genera recomendaciones"""
    
    # Calcular métricas básicas
    entry_price = investment_data['price']
    current_price = current_data['Close'].iloc[-1]
    profit_loss = ((current_price / entry_price) - 1) * 100
    
    # Calcular métricas técnicas
    rsi = calculate_rsi(current_data)[-1]
    sma_50 = current_data['Close'].rolling(window=50).mean().iloc[-1]
    sma_200 = current_data['Close'].rolling(window=200).mean().iloc[-1]
    volatility = current_data['Close'].pct_change().std()
    
    # Sistema de puntuación para la recomendación
    score = 0
    reasons = []
    
    # 1. Tendencia de precio
    if current_price > sma_50 and sma_50 > sma_200:
        score += 2
        reasons.append("✅ Tendencia alcista fuerte")
    elif current_price > sma_50:
        score += 1
        reasons.append("✅ Tendencia alcista moderada")
    else:
        reasons.append("⚠️ Tendencia bajista")
    
    # 2. RSI
    if 30 <= rsi <= 70:
        score += 1
        reasons.append("✅ RSI en rango normal")
    elif rsi > 70:
        reasons.append("⚠️ RSI sobrecomprado")
    else:
        reasons.append("⚠️ RSI sobrevendido")
    
    # 3. Rentabilidad
    if profit_loss > 20:
        score += 2
        reasons.append("✅ Excelente rentabilidad")
    elif profit_loss > 10:
        score += 1
        reasons.append("✅ Buena rentabilidad")
    elif profit_loss < -10:
        score -= 1
        reasons.append("⚠️ Pérdida significativa")
    
    # 4. Volatilidad
    if volatility < 0.02:
        score += 1
        reasons.append("✅ Baja volatilidad")
    elif volatility > 0.03:
        score -= 1
        reasons.append("⚠️ Alta volatilidad")
    
    # Determinar recomendación final
    if score >= 3:
        recommendation = "MANTENER"
        color = "green"
    elif score >= 1:
        recommendation = "MANTENER CON PRECAUCIÓN"
        color = "yellow"
    else:
        recommendation = "CONSIDERAR CERRAR"
        color = "red"
    
    return {
        'recommendation': recommendation,
        'color': color,
        'score': score,
        'profit_loss': profit_loss,
        'current_price': current_price,
        'rsi': rsi,
        'reasons': reasons
    }

# Función para mostrar análisis de inversiones
def show_investment_analysis(conn):
    """Muestra análisis detallado de las inversiones actuales"""
    # Obtener inversiones actuales
    investments_df = pd.read_sql_query("""
        SELECT id, date, etf_symbol, amount, shares, price 
        FROM investments 
        ORDER BY date DESC
    """, conn)
    
    if investments_df.empty:
        st.info("No hay inversiones registradas")
        return
    
    st.subheader("📊 Análisis de Inversiones Actuales")
    
    # Analizar cada inversión
    analysis_results = []
    
    for _, investment in investments_df.iterrows():
        current_data = get_etf_data(investment['etf_symbol'])
        if current_data is not None:
            analysis = analyze_investment_position(investment, current_data)
            
            analysis_results.append({
                'Fecha': investment['date'],
                'ETF': investment['etf_symbol'],
                'Inversión': f"${investment['amount']:.2f}",
                'Acciones': f"{investment['shares']:.4f}",
                'Precio Entrada': f"${investment['price']:.2f}",
                'Precio Actual': f"${analysis['current_price']:.2f}",
                'Rendimiento': f"{analysis['profit_loss']:.2f}%",
                'RSI': f"{analysis['rsi']:.2f}",
                'Recomendación': analysis['recommendation'],
                'Razones': "\n".join(analysis['reasons'])
            })
    
    # Crear DataFrame con los resultados
    analysis_df = pd.DataFrame(analysis_results)
    
    # Mostrar tabla de análisis
    st.write("### 📈 Resumen de Posiciones")
    
    # Aplicar colores a las recomendaciones
    def color_recommendations(val):
        if 'MANTENER' in val:
            if 'PRECAUCIÓN' in val:
                return 'background-color: #ffd700'
            return 'background-color: #90EE90'
        return 'background-color: #ffcccb'
    
    # Mostrar tabla con formato
    st.dataframe(
        analysis_df,
        column_config={
            "Fecha": "Fecha",
            "ETF": "ETF",
            "Inversión": st.column_config.NumberColumn(
                "Inversión",
                help="Monto invertido",
                format="$%.2f"
            ),
            "Acciones": "Acciones",
            "Precio Entrada": "Precio Entrada",
            "Precio Actual": "Precio Actual",
            "Rendimiento": st.column_config.NumberColumn(
                "Rendimiento",
                help="Rendimiento actual",
                format="%.2f%%"
            ),
            "RSI": "RSI",
            "Recomendación": st.column_config.Column(
                "Recomendación",
                help="Recomendación basada en análisis técnico"
            ),
            "Razones": st.column_config.TextColumn(
                "Razones",
                help="Razones para la recomendación",
                width="large"
            )
        },
        hide_index=True
    )
    
    # Mostrar estadísticas generales
    total_invested = investments_df['amount'].sum()
    current_value = sum(
        float(row['Acciones'].replace(',', '')) * float(row['Precio Actual'].replace('$', ''))
        for row in analysis_results
    )
    total_return = ((current_value / total_invested) - 1) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="💰 Inversión Total",
            value=f"${total_invested:,.2f}"
        )
    with col2:
        st.metric(
            label="📈 Valor Actual",
            value=f"${current_value:,.2f}"
        )
    with col3:
        st.metric(
            label="✨ Rendimiento Total",
            value=f"{total_return:.2f}%",
            delta=f"{total_return:.2f}%"
        )

# Interfaz principal
def main():
    if not login_user():
        return
        
    st.title("📊 Dashboard de Inversión en ETFs")
    
    # Inicializar la base de datos
    conn = init_db()  # Get a fresh connection
    
    try:
        # Cargar ETFs inicialmente
        etfs_dict = load_etfs(conn)
        
        # Crear pestañas
        tab1, tab2, tab3 = st.tabs(["📈 Análisis Individual", "🔄 Comparación de ETFs", "💰 Registro de inversiones"])
        
        with tab1:
            # Sidebar para análisis de ETFs
            st.sidebar.header("🎯 Seleccionar ETF")
            
            # Seleccionar ETF para analizar
            selected_etf = st.sidebar.selectbox(
                "Escoge un ETF para analizar",
                list(etfs_dict.keys()),
                format_func=lambda x: f"{x} - {etfs_dict[x]}"
            )
            
            # Resto del código del tab1...
            
            if selected_etf:
                df = get_etf_data(selected_etf)
                if df is not None:
                    # Mostrar precio actual
                    current_price = df['Close'].iloc[-1]
                    st.metric(
                        label=f"Precio actual de {selected_etf}",
                        value=f"${current_price:.2f}",
                        delta=f"{((current_price/df['Close'].iloc[-2])-1)*100:.2f}%"
                    )
                    
                    # Mostrar gráfico de precios
                    st.subheader(f"📈 Precio histórico de {selected_etf}")
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Precio'))
                    
                    # Agregar medias móviles
                    df['SMA_50'] = df['Close'].rolling(window=50).mean()
                    df['SMA_200'] = df['Close'].rolling(window=200).mean()
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='Media 50 días', line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='Media 200 días', line=dict(color='blue')))
                    
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar análisis
                    st.subheader("📊 Análisis técnico")
                    signals = analyze_etf(df)
                    for signal in signals:
                        st.write(signal)
        
        with tab2:
            col1, col2 = st.columns([0.2, 0.8])
            with col1:
                if st.button("🔄 Actualizar ETFs"):
                    with st.spinner("Actualizando lista de ETFs..."):
                        etfs_dict.update(update_etfs(conn, force_update=True))
                    st.success("✅ ETFs actualizados exitosamente")
            
            st.subheader("📊 Comparación de ETFs")
            st.write("Analizando los mejores ETFs para invertir...")
            
            # Usar etfs_dict en lugar de POPULAR_ETFS
            df_comparison = analyze_multiple_etfs(etfs_dict)
            
            # Mostrar tabla de comparación
            st.subheader("📊 Ranking de ETFs")
            formatted_df = df_comparison.copy()
            formatted_df['price'] = formatted_df['price'].map('${:,.2f}'.format)
            formatted_df['return'] = formatted_df['return'].map('{:,.2f}%'.format)
            formatted_df['volatility'] = formatted_df['volatility'].map('{:,.2%}'.format)
            formatted_df['rsi'] = formatted_df['rsi'].map('{:,.2f}'.format)
            formatted_df['entry_target'] = formatted_df['entry_target'].map('${:,.2f}'.format)
            
            # Usar una visualización más simple sin background_gradient
            st.dataframe(
                formatted_df,
                column_config={
                    "symbol": "Símbolo",
                    "name": "Nombre",
                    "price": "Precio Actual",
                    "return": "Rendimiento",
                    "rsi": "RSI",
                    "volatility": "Volatilidad",
                    "score": st.column_config.NumberColumn(
                        "Puntuación",
                        help="Puntuación basada en análisis técnico (máx. 7)",
                        format="%.1f ⭐"
                    ),
                    "entry_target": "Precio Objetivo"
                },
                hide_index=True,
            )
            
            # Mostrar mejores oportunidades
            st.subheader("🎯 Mejores oportunidades de inversión")
            top_opportunities = df_comparison.head(3)
            
            for _, row in top_opportunities.iterrows():
                st.write(f"### {row['symbol']} - {row['name']}")
                st.write(f"💵 Precio actual: ${row['price']:.2f} | 🎯 Precio objetivo: ${row['entry_target']:.2f}")
                st.write(f"📊 Puntuación: {row['score']}/7 | RSI: {row['rsi']:.2f}")
                st.write("---")
        
        with tab3:
            # Sección de inversiones
            st.header("💰 Gestión de Inversiones")
            
            # Crear tres pestañas
            inv_tab1, inv_tab2, inv_tab3 = st.tabs(["📝 Nueva Inversión", "📊 Análisis de Posiciones", "✏️ Editar Inversiones"])
            
            with inv_tab1:
                col1, col2 = st.columns(2)
                with col1:
                    # Inicializar valores en el estado de la sesión si no existen
                    if 'new_investment' not in st.session_state:
                        st.session_state.new_investment = 30.0
                    if 'custom_price' not in st.session_state:
                        st.session_state.custom_price = 0.0
                    
                    new_investment = st.number_input("Monto a invertir ($)", 
                                                  min_value=0.0, 
                                                  value=st.session_state.new_investment,
                                                  key="new_investment_input")
                    selected_etf = st.selectbox("Seleccionar ETF para invertir", 
                                              list(etfs_dict.keys()), 
                                              format_func=lambda x: f"{x} - {etfs_dict[x]}",
                                              key="selected_etf_input")
                    custom_price = st.number_input("Precio de entrada personalizado (opcional)", 
                                                 min_value=0.0, 
                                                 value=st.session_state.custom_price,
                                                 key="custom_price_input")
                    investment_date = st.date_input("Fecha de inversión", 
                                                  datetime.now(), 
                                                  key="investment_date_input")
                    
                    def clear_form():
                        st.session_state.new_investment_input = 30.0
                        st.session_state.custom_price_input = 0.0
                    
                    if st.button("📝 Registrar inversión", key="register_button", on_click=clear_form):
                        if selected_etf and new_investment > 0:
                            df = get_etf_data(selected_etf, period='1d')
                            if df is not None:
                                current_price = custom_price if custom_price > 0 else df['Close'].iloc[-1]
                                shares = new_investment / current_price
                                add_investment(
                                    conn,
                                    investment_date.strftime('%Y-%m-%d'),
                                    selected_etf,
                                    new_investment,
                                    shares,
                                    current_price
                                )
                                st.success("✅ Inversión registrada exitosamente")
        
            with inv_tab2:
                show_investment_analysis(conn)
        
            with inv_tab3:
                # Cargar inversiones existentes
                investments = pd.read_sql_query("""
                    SELECT id, date, etf_symbol, amount, shares, price 
                    FROM investments 
                    ORDER BY date DESC""", conn)
                
                if not investments.empty:
                    st.write("📋 Editar inversiones existentes:")
                    
                    # Convertir la columna de fecha a datetime
                    investments['date'] = pd.to_datetime(investments['date'])
                    
                    # Convertir la tabla a un formato editable
                    edited_df = st.data_editor(
                        investments,
                        column_config={
                            "id": "ID",
                            "date": st.column_config.DatetimeColumn(
                                "Fecha",
                                help="Fecha de la inversión",
                                format="YYYY-MM-DD",
                                step=86400  # Un día en segundos
                            ),
                            "etf_symbol": st.column_config.SelectboxColumn(
                                "ETF",
                                help="Símbolo del ETF",
                                options=list(etfs_dict.keys()),
                                required=True
                            ),
                            "amount": st.column_config.NumberColumn(
                                "Monto ($)",
                                help="Monto invertido",
                                min_value=0,
                                format="$%.2f",
                                required=True
                            ),
                            "shares": st.column_config.NumberColumn(
                                "Acciones",
                                help="Número de acciones",
                                min_value=0,
                                format="%.4f",
                                required=True
                            ),
                            "price": st.column_config.NumberColumn(
                                "Precio",
                                help="Precio de entrada",
                                min_value=0,
                                format="$%.2f",
                                required=True
                            )
                        },
                        hide_index=True,
                        num_rows="dynamic"
                    )
                    
                    # Detectar cambios y actualizar la base de datos
                    if not edited_df.equals(investments):
                        # Encontrar filas eliminadas
                        deleted_ids = set(investments['id']) - set(edited_df['id'])
                        for id in deleted_ids:
                            delete_investment(conn, id)
                        
                        # Actualizar filas modificadas y agregar nuevas
                        for _, row in edited_df.iterrows():
                            if pd.isna(row['id']):  # Nueva fila
                                if not pd.isna(row['date']) and not pd.isna(row['etf_symbol']):
                                    add_investment(
                                        conn,
                                        row['date'].strftime('%Y-%m-%d'),
                                        row['etf_symbol'],
                                        row['amount'],
                                        row['shares'],
                                        row['price']
                                    )
                            else:  # Fila existente
                                update_investment(
                                    conn,
                                    row['id'],
                                    row['date'].strftime('%Y-%m-%d'),
                                    row['etf_symbol'],
                                    row['amount'],
                                    row['shares'],
                                    row['price']
                                )
                        
                        st.success("✅ Cambios guardados exitosamente")
                        # Recargar los datos actualizados
                        investments = pd.read_sql_query("""
                            SELECT id, date, etf_symbol, amount, shares, price 
                            FROM investments 
                            ORDER BY date DESC""", conn)
            
                # Mostrar resumen del portafolio
                if not investments.empty:
                    st.subheader("📊 Resumen del Portafolio")
                    
                    # Agrupar por ETF
                    portfolio_summary = investments.groupby('etf_symbol').agg({
                        'amount': 'sum',
                        'shares': 'sum'
                    }).reset_index()
                    
                    # Calcular valores actuales
                    portfolio_summary['current_price'] = portfolio_summary['etf_symbol'].apply(
                        lambda x: get_etf_data(x, period='1d')['Close'].iloc[-1] if get_etf_data(x, period='1d') is not None else 0
                    )
                    portfolio_summary['current_value'] = portfolio_summary['shares'] * portfolio_summary['current_price']
                    portfolio_summary['profit_loss'] = portfolio_summary['current_value'] - portfolio_summary['amount']
                    portfolio_summary['return_pct'] = (portfolio_summary['current_value'] / portfolio_summary['amount'] - 1) * 100
                    
                    # Mostrar tabla de resumen por ETF
                    st.write("Resumen por ETF:")
                    summary_df = portfolio_summary.copy()
                    summary_df['amount'] = summary_df['amount'].map('${:,.2f}'.format)
                    summary_df['current_value'] = summary_df['current_value'].map('${:,.2f}'.format)
                    summary_df['profit_loss'] = summary_df['profit_loss'].map('${:,.2f}'.format)
                    summary_df['return_pct'] = summary_df['return_pct'].map('{:,.2f}%'.format)
                    summary_df.columns = ['ETF', 'Total Invertido', 'Acciones', 'Precio Actual', 'Valor Actual', 'Ganancia/Pérdida', 'Rendimiento']
                    st.dataframe(summary_df)
                    
                    # Mostrar métricas totales
                    total_invested = portfolio_summary['amount'].sum()
                    total_current_value = portfolio_summary['current_value'].sum()
                    total_return = ((total_current_value / total_invested) - 1) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("💰 Total invertido", f"${total_invested:,.2f}")
                    col2.metric("💰 Valor actual", f"${total_current_value:,.2f}")
                    col3.metric("📈 Rendimiento total", f"{total_return:.2f}%")
        
                else:
                    st.info("No hay inversiones registradas todavía.")
    
    finally:
        conn.close()

if __name__ == "__main__":
    from auth import init_auth_db, login_user
    init_auth_db()
    main()
