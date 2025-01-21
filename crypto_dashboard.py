import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sqlite3
import requests
from datetime import datetime, timedelta
from auth import init_auth_db, login_user
import ta
import numpy as np
import time
import json
import os
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *
from plyer import notification
import schedule
import threading
import time
import streamlit as st
# Cargar variables de entorno
load_dotenv()



# Configuración de Binance
#BINANCE_API_KEY = st.secrets["BINANCE_API_KEY"]
#BINANCE_API_SECRET = st.secrets["BINANCE_API_SECRET"]
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)


# Configuración de la página
st.set_page_config(page_title="Crypto Futures Trading Dashboard", layout="wide")

# Cache para datos
CACHE_DIR = "cache"
CACHE_DURATION = 60  # 1 minuto en segundos

def get_futures_symbols():
    """Obtiene la lista de los 50 pares de futuros más importantes por volumen"""
    try:
        # Obtener información de todos los pares
        exchange_info = client.futures_exchange_info()
        symbols = {}
        
        # Obtener datos de 24h para todos los pares USDT
        tickers = client.futures_ticker()
        
        # Filtrar solo pares USDT y ordenar por volumen
        usdt_tickers = [
            t for t in tickers 
            if t['symbol'].endswith('USDT') and 
            float(t['volume']) > 0 and
            any(s['symbol'] == t['symbol'] and s['status'] == 'TRADING' 
                for s in exchange_info['symbols'])
        ]
        
        # Ordenar por volumen y tomar los top 50
        top_50 = sorted(
            usdt_tickers,
            key=lambda x: float(x['volume']) * float(x['lastPrice']),
            reverse=True
        )[:50]
        
        # Crear diccionario con información relevante
        for ticker in top_50:
            symbol = ticker['symbol']
            symbol_info = next(s for s in exchange_info['symbols'] if s['symbol'] == symbol)
            symbols[symbol] = {
                'baseAsset': symbol_info['baseAsset'],
                'pricePrecision': symbol_info['pricePrecision'],
                'quantityPrecision': symbol_info['quantityPrecision'],
                'volume24h': float(ticker['volume']) * float(ticker['lastPrice']),
                'price': float(ticker['lastPrice'])
            }
        
        return symbols
    except Exception as e:
        st.error(f"Error al obtener símbolos de futuros: {str(e)}")
        return {}

def get_funding_rate(symbol):
    """Obtiene la tasa de financiamiento actual"""
    try:
        funding_rate = client.futures_funding_rate(symbol=symbol, limit=1)[0]
        return {
            'rate': float(funding_rate['fundingRate']),
            'time': datetime.fromtimestamp(funding_rate['fundingTime']/1000)
        }
    except Exception as e:
        st.error(f"Error al obtener tasa de financiamiento: {str(e)}")
        return None

def get_futures_data(symbol, interval='1d', limit=100):
    """Obtiene datos históricos de futuros"""
    try:
        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ])
        
        # Convertir tipos de datos
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.set_index('timestamp', inplace=True)
        
        # Agregar datos de posiciones largas/cortas
        try:
            long_short_ratio = client.futures_long_short_ratio(
                symbol=symbol,
                period='1d',
                limit=limit
            )
            
            df_ratio = pd.DataFrame(long_short_ratio)
            df_ratio['timestamp'] = pd.to_datetime(df_ratio['timestamp'], unit='ms')
            df_ratio.set_index('timestamp', inplace=True)
            
            df['long_ratio'] = df_ratio['longAccount']
            df['short_ratio'] = df_ratio['shortAccount']
        except:
            pass
        
        return df
    
    except Exception as e:
        st.error(f"Error al obtener datos de futuros: {str(e)}")
        return None

def get_liquidations(symbol, start_time=None, limit=100):
    """Obtiene datos de liquidaciones"""
    try:
        if start_time is None:
            start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)
        
        liquidations = client.futures_liquidation_orders(
            symbol=symbol,
            startTime=start_time,
            limit=limit
        )
        
        if liquidations:
            df = pd.DataFrame(liquidations)
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df['price'] = pd.to_numeric(df['price'])
            df['qty'] = pd.to_numeric(df['qty'])
            return df
        return pd.DataFrame()
    
    except Exception as e:
        st.error(f"Error al obtener liquidaciones: {str(e)}")
        return pd.DataFrame()

def calculate_optimal_leverage(df, risk_percentage=1):
    """Calcula el apalancamiento óptimo basado en la volatilidad"""
    try:
        # Calcular volatilidad diaria
        returns = df['close'].pct_change()
        daily_volatility = returns.std()
        
        # Calcular apalancamiento óptimo
        # Usando la fórmula: leverage = risk_percentage / (daily_volatility * 2)
        optimal_leverage = risk_percentage / (daily_volatility * 2)
        
        # Limitar el apalancamiento máximo a 20x
        return min(optimal_leverage, 20)
    
    except Exception as e:
        st.error(f"Error al calcular apalancamiento óptimo: {str(e)}")
        return 1

class Strategy:
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def analyze(self, df, symbol_info):
        pass

class RSIMACDStrategy(Strategy):
    def __init__(self):
        super().__init__(
            "RSI + MACD",
            "Estrategia basada en RSI y MACD con confirmación de tendencia EMA"
        )
    
    def analyze(self, df, symbol_info):
        signals = []
        
        if len(df) < 14:
            return signals
        
        # Calcular indicadores
        df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['EMA20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['EMA50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        
        current_price = df['close'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        
        # Señales LONG
        if (current_rsi < 30 and 
            df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and
            df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1]):
            
            signals.append({
                'tipo': 'LONG',
                'entrada': current_price,
                'stop_loss': current_price * 0.98,  # 2% SL
                'take_profit': current_price * 1.04,  # 4% TP
                'razones': [
                    "RSI en sobreventa",
                    "Cruce alcista MACD",
                    "Tendencia alcista (EMA20 > EMA50)"
                ]
            })
        
        # Señales SHORT
        elif (current_rsi > 70 and 
              df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] and
              df['EMA20'].iloc[-1] < df['EMA50'].iloc[-1]):
            
            signals.append({
                'tipo': 'SHORT',
                'entrada': current_price,
                'stop_loss': current_price * 1.02,  # 2% SL
                'take_profit': current_price * 0.96,  # 4% TP
                'razones': [
                    "RSI en sobrecompra",
                    "Cruce bajista MACD",
                    "Tendencia bajista (EMA20 < EMA50)"
                ]
            })
        
        return signals

class BBandsStrategy(Strategy):
    def __init__(self):
        super().__init__(
            "Bollinger Bands + Volumen",
            "Estrategia basada en Bandas de Bollinger con confirmación de volumen"
        )
    
    def analyze(self, df, symbol_info):
        signals = []
        
        if len(df) < 20:
            return signals
        
        # Calcular Bandas de Bollinger
        bbands = ta.volatility.BollingerBands(df['close'])
        df['BB_High'] = bbands.bollinger_hband()
        df['BB_Low'] = bbands.bollinger_lband()
        df['BB_Mid'] = bbands.bollinger_mavg()
        
        # Calcular Media de Volumen
        df['Volume_MA'] = df['volume'].rolling(window=20).mean()
        
        current_price = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        
        # Señales LONG
        if (current_price < df['BB_Low'].iloc[-1] and 
            current_volume > df['Volume_MA'].iloc[-1] * 1.5):
            
            signals.append({
                'tipo': 'LONG',
                'entrada': current_price,
                'stop_loss': current_price * 0.985,  # 1.5% SL
                'take_profit': df['BB_Mid'].iloc[-1],  # TP en la media
                'razones': [
                    "Precio por debajo de BB inferior",
                    "Volumen 50% superior a la media",
                    f"Distancia a BB Media: {((df['BB_Mid'].iloc[-1]/current_price - 1) * 100):.2f}%"
                ]
            })
        
        # Señales SHORT
        elif (current_price > df['BB_High'].iloc[-1] and 
              current_volume > df['Volume_MA'].iloc[-1] * 1.5):
            
            signals.append({
                'tipo': 'SHORT',
                'entrada': current_price,
                'stop_loss': current_price * 1.015,  # 1.5% SL
                'take_profit': df['BB_Mid'].iloc[-1],  # TP en la media
                'razones': [
                    "Precio por encima de BB superior",
                    "Volumen 50% superior a la media",
                    f"Distancia a BB Media: {((current_price/df['BB_Mid'].iloc[-1] - 1) * 100):.2f}%"
                ]
            })
        
        return signals

class BreakoutStrategy(Strategy):
    def __init__(self):
        super().__init__(
            "Breakout + Momentum",
            "Estrategia de ruptura de niveles con confirmación de momentum"
        )
    
    def analyze(self, df, symbol_info):
        signals = []
        
        if len(df) < 20:
            return signals
        
        # Calcular niveles
        df['High_20'] = df['high'].rolling(window=20).max()
        df['Low_20'] = df['low'].rolling(window=20).min()
        df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        current_price = df['close'].iloc[-1]
        current_adx = df['ADX'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        
        # Breakout alcista
        if (current_price > df['High_20'].iloc[-2] and  # Ruptura del máximo
            current_adx > 25 and  # Tendencia fuerte
            current_rsi > 50):  # Momentum alcista
            
            signals.append({
                'tipo': 'LONG',
                'entrada': current_price,
                'stop_loss': df['Low_20'].iloc[-1],  # SL en el mínimo
                'take_profit': current_price + (current_price - df['Low_20'].iloc[-1]),  # TP simétrico
                'razones': [
                    "Ruptura de máximo de 20 períodos",
                    f"ADX fuerte: {current_adx:.1f}",
                    f"RSI alcista: {current_rsi:.1f}"
                ]
            })
        
        # Breakout bajista
        elif (current_price < df['Low_20'].iloc[-2] and  # Ruptura del mínimo
              current_adx > 25 and  # Tendencia fuerte
              current_rsi < 50):  # Momentum bajista
            
            signals.append({
                'tipo': 'SHORT',
                'entrada': current_price,
                'stop_loss': df['High_20'].iloc[-1],  # SL en el máximo
                'take_profit': current_price - (df['High_20'].iloc[-1] - current_price),  # TP simétrico
                'razones': [
                    "Ruptura de mínimo de 20 períodos",
                    f"ADX fuerte: {current_adx:.1f}",
                    f"RSI bajista: {current_rsi:.1f}"
                ]
            })
        
        return signals

def send_notification(title, message):
    """Envía una notificación al sistema"""
    try:
        notification.notify(
            title=title,
            message=message,
            app_icon=None,
            timeout=10,
        )
    except Exception as e:
        st.error(f"Error al enviar notificación: {str(e)}")

def analyze_with_strategies(symbol_info, timeframe='15m'):
    """Analiza todos los pares con múltiples estrategias y guarda las señales en la base de datos"""
    # Inicializar estrategias
    strategies = [
        RSIMACDStrategy(),       
        BreakoutStrategy()
    ]
    
    all_signals = []
    current_time = datetime.now()
    
    for symbol in symbol_info.keys():
        df = get_futures_data(symbol, interval=timeframe)
        if df is not None and len(df) >= 20:
            # Calcular apalancamiento óptimo
            leverage = calculate_optimal_leverage(df)
            
            for strategy in strategies:
                signals = strategy.analyze(df, symbol_info[symbol])
                for signal in signals:
                    # Verificar si la señal ya existe
                    if not signal_exists(symbol, signal['entrada'], signal['tipo'], strategy.name):
                        signal_data = {
                            'symbol': symbol,
                            'strategy': strategy.name,
                            'price': df['close'].iloc[-1],
                            'timestamp': current_time,
                            'leverage': leverage,
                            **signal
                        }
                        all_signals.append(signal_data)
                        # Guardar la señal en la base de datos
                        store_signal(signal_data, timeframe)
    
    return all_signals

def signal_exists(symbol, entry_price, signal_type, strategy):
    """Verifica si una señal similar ya existe en la base de datos en los últimos 5 minutos"""
    conn = sqlite3.connect('trading_signals.db')
    c = conn.cursor()
    
    # Buscar señales idénticas en los últimos 5 minutos
    c.execute('''
        SELECT id FROM signals 
        WHERE symbol = ? 
        AND entry_price = ? 
        AND signal_type = ? 
        AND strategy = ?
        AND timestamp >= datetime('now', '-5 minutes')
    ''', (symbol, entry_price, signal_type, strategy))
    
    exists = c.fetchone() is not None
    conn.close()
    return exists

def store_signal(signal, timeframe):
    """Almacena una señal en la base de datos"""
    conn = sqlite3.connect('trading_signals.db')
    c = conn.cursor()
    
    # Calcular R/R
    if signal['tipo'] == 'LONG':
        risk_reward = (signal['take_profit'] - signal['entrada']) / (signal['entrada'] - signal['stop_loss'])
    else:
        risk_reward = (signal['entrada'] - signal['take_profit']) / (signal['stop_loss'] - signal['entrada'])
    
    # Insertar señal
    c.execute('''
        INSERT INTO signals (
            timestamp, symbol, strategy, signal_type, entry_price, current_price,
            stop_loss, take_profit, risk_reward, leverage, timeframe, reasons
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        signal['timestamp'],
        signal['symbol'],
        signal['strategy'],
        signal['tipo'],
        signal['entrada'],
        signal['price'],
        signal['stop_loss'],
        signal['take_profit'],
        risk_reward,
        signal['leverage'],
        timeframe,
        '\n'.join(signal['razones'])
    ))
    
    conn.commit()
    conn.close()

def update_signal_status():
    """Actualiza el estado de las señales basado en los precios actuales"""
    conn = sqlite3.connect('trading_signals.db')
    c = conn.cursor()
    
    # Obtener señales activas
    c.execute('SELECT id, symbol, signal_type, entry_price, stop_loss, take_profit FROM signals WHERE status = "ACTIVE"')
    active_signals = c.fetchall()
    
    for signal in active_signals:
        signal_id, symbol, signal_type, entry, sl, tp = signal
        
        # Obtener precio actual
        try:
            current_price = float(client.futures_symbol_ticker(symbol=symbol)['price'])
            
            # Verificar si se alcanzó SL o TP
            if signal_type == 'LONG':
                if current_price <= sl:  # Stop Loss
                    exit_type = 'STOP_LOSS'
                    pl_percent = (sl - entry) / entry * 100
                    win_loss = 'LOSS'
                elif current_price >= tp:  # Take Profit
                    exit_type = 'TAKE_PROFIT'
                    pl_percent = (tp - entry) / entry * 100
                    win_loss = 'WIN'
                else:
                    continue
            else:  # SHORT
                if current_price >= sl:  # Stop Loss
                    exit_type = 'STOP_LOSS'
                    pl_percent = (entry - sl) / entry * 100
                    win_loss = 'LOSS'
                elif current_price <= tp:  # Take Profit
                    exit_type = 'TAKE_PROFIT'
                    pl_percent = (entry - tp) / entry * 100
                    win_loss = 'WIN'
                else:
                    continue
            
            # Actualizar señal
            c.execute('''
                UPDATE signals 
                SET status = ?, exit_price = ?, exit_timestamp = ?, profit_loss = ?, win_loss = ?
                WHERE id = ?
            ''', (exit_type, current_price, datetime.now(), pl_percent, win_loss, signal_id))
            
            # Enviar notificación
            send_notification(
                f"💰 Señal Cerrada: {symbol}",
                f"Tipo: {signal_type}\nResultado: {exit_type} ({win_loss})\nP/L: {pl_percent:.2f}%"
            )
            
        except Exception as e:
            print(f"Error actualizando señal {signal_id}: {str(e)}")
    
    conn.commit()
    conn.close()

def show_signal_history():
    """Muestra el historial de señales"""
    st.subheader("📜 Historial de Señales")
    
    conn = sqlite3.connect('trading_signals.db')
    
    # Asegurarse de que la base de datos está actualizada
    init_database()
    
    # Filtros
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status_filter = st.multiselect(
            "Estado",
            ["ACTIVE", "STOP_LOSS", "TAKE_PROFIT"],
            default=["ACTIVE", "STOP_LOSS", "TAKE_PROFIT"]
        )
    
    with col2:
        days_filter = st.slider(
            "Últimos días",
            min_value=1,
            max_value=30,
            value=7
        )
    
    with col3:
        type_filter = st.multiselect(
            "Tipo",
            ["LONG", "SHORT"],
            default=["LONG", "SHORT"]
        )
    
    with col4:
        result_filter = st.multiselect(
            "Resultado",
            ["WIN", "LOSS", "ACTIVE"],
            default=["WIN", "LOSS", "ACTIVE"]
        )
    
    # Obtener señales filtradas
    query = f'''
        SELECT 
            timestamp,
            symbol,
            strategy,
            signal_type,
            entry_price,
            current_price,
            stop_loss,
            take_profit,
            risk_reward,
            leverage,
            timeframe,
            status,
            exit_price,
            exit_timestamp,
            profit_loss,
            win_loss
        FROM signals
        WHERE 
            status IN ({','.join(['?']*len(status_filter))})
            AND signal_type IN ({','.join(['?']*len(type_filter))})
            AND (win_loss IN ({','.join(['?']*len(result_filter))}) OR (win_loss IS NULL AND 'ACTIVE' IN ({','.join(['?']*len(result_filter))})))
            AND timestamp >= datetime('now', '-' || ? || ' days')
        ORDER BY timestamp DESC
    '''
    
    params = [*status_filter, *type_filter, *result_filter, *result_filter, days_filter]
    df = pd.read_sql_query(query, conn, params=params)
    
    if not df.empty:
        # Formatear columnas
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        
        # Manejar exit_timestamp
        def format_exit_timestamp(ts):
            if pd.isna(ts):
                return '-'
            try:
                return pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M')
            except:
                return '-'
        
        df['exit_timestamp'] = df['exit_timestamp'].apply(format_exit_timestamp)
        df['win_loss'] = df['win_loss'].fillna('ACTIVE')
        
        # Función para formatear P/L
        def format_pl(val):
            if pd.isna(val):
                return '-'
            return f'{val:,.2f}%'
        
        # Crear DataFrame para mostrar
        df_display = pd.DataFrame({
            'Fecha': df['timestamp'],
            'Par': df['symbol'],
            'Estrategia': df['strategy'],
            'Tipo': df['signal_type'],
            'Entrada': df['entry_price'].map('${:,.4f}'.format),
            'Stop Loss': df['stop_loss'].map('${:,.4f}'.format),
            'Take Profit': df['take_profit'].map('${:,.4f}'.format),
            'R/R': df['risk_reward'].map('{:,.2f}'.format),
            'Apalancamiento': df['leverage'].map('{:,.1f}x'.format),
            'Estado': df['status'],
            'Resultado': df['win_loss'],
            'P/L': df['profit_loss'].apply(format_pl)
        })
        
        # Aplicar colores
        def color_pl(val):
            if val == '-':
                return ''
            try:
                num = float(val.strip('%'))
                return 'color: green' if num > 0 else 'color: red'
            except:
                return ''
        
        def color_type(val):
            return 'color: green' if val == 'LONG' else 'color: red'
        
        def color_status(val):
            colors = {
                'ACTIVE': 'color: blue',
                'TAKE_PROFIT': 'color: green',
                'STOP_LOSS': 'color: red'
            }
            return colors.get(val, '')
            
        def color_result(val):
            colors = {
                'WIN': 'color: green',
                'LOSS': 'color: red',
                'ACTIVE': 'color: blue'
            }
            return colors.get(val, '')
        
        # Mostrar tabla con estilos
        st.dataframe(
            df_display.style
            .applymap(color_pl, subset=['P/L'])
            .applymap(color_type, subset=['Tipo'])
            .applymap(color_status, subset=['Estado'])
            .applymap(color_result, subset=['Resultado']),
            use_container_width=True
        )
        
        # Estadísticas
        st.subheader("📊 Resumen de Resultados")
        
        # Calcular estadísticas solo para señales cerradas
        closed_signals = df[df['status'] != 'ACTIVE'].copy()
        if not closed_signals.empty:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                total_trades = len(closed_signals)
                st.metric("Total Trades", f"{total_trades}")
            
            with col2:
                win_rate = (closed_signals['win_loss'] == 'WIN').mean() * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            
            with col3:
                avg_profit = closed_signals[closed_signals['profit_loss'] > 0]['profit_loss'].mean()
                st.metric("Ganancia Media", f"{avg_profit:.2f}%" if not pd.isna(avg_profit) else "0.00%")
            
            with col4:
                avg_loss = closed_signals[closed_signals['profit_loss'] < 0]['profit_loss'].mean()
                st.metric("Pérdida Media", f"{avg_loss:.2f}%" if not pd.isna(avg_loss) else "0.00%")
            
            with col5:
                total_pl = closed_signals['profit_loss'].sum()
                st.metric("P/L Total", f"{total_pl:.2f}%" if not pd.isna(total_pl) else "0.00%")
            
            # Gráfico de resultados
            if len(closed_signals) > 0:
                fig = go.Figure()
                
                # Agregar línea de P/L acumulado
                closed_signals = closed_signals.sort_values('exit_timestamp')
                cumulative_pl = closed_signals['profit_loss'].fillna(0).cumsum()
                
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(closed_signals['exit_timestamp']),
                    y=cumulative_pl,
                    mode='lines',
                    name='P/L Acumulado'
                ))
                
                fig.update_layout(
                    title="P/L Acumulado",
                    xaxis_title="Fecha",
                    yaxis_title="P/L %",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Gráfico de distribución de resultados
                fig2 = go.Figure()
                
                wins = len(closed_signals[closed_signals['win_loss'] == 'WIN'])
                losses = len(closed_signals[closed_signals['win_loss'] == 'LOSS'])
                
                fig2.add_trace(go.Bar(
                    x=['WIN', 'LOSS'],
                    y=[wins, losses],
                    name='Distribución de Resultados'
                ))
                
                fig2.update_layout(
                    title="Distribución de Resultados",
                    xaxis_title="Resultado",
                    yaxis_title="Número de Trades",
                    showlegend=False
                )
                
                st.plotly_chart(fig2, use_container_width=True)
    
    else:
        st.info("No hay señales que cumplan con los filtros seleccionados")
    
    conn.close()

def show_trading_signals_table(symbol_info):
    """Muestra una tabla con todas las señales de trading activas"""
    st.subheader("📊 Tabla de Señales de Trading")
    
    # Selector de timeframe
    timeframe = st.selectbox(
        "Timeframe",
        ['15m', '1h', '4h', '1d'],
        index=0
    )
    
    # Selector de estrategias
    strategies = st.multiselect(
        "Estrategias",
        ["RSI + MACD", "Bollinger Bands + Volumen", "Breakout + Momentum"],
        default=["RSI + MACD","Breakout + Momentum"]
    )
    
    # Filtro de tiempo
    hours_filter = st.slider(
        "Mostrar señales de las últimas X horas",
        min_value=1,
        max_value=48,
        value=24
    )
    
    # Mostrar información de volumen
    st.write("Top 50 pares por volumen 24h:")
    volume_df = pd.DataFrame([
        {
            'Par': symbol,
            'Volumen 24h': f"${info['volume24h']:,.0f}",
            'Precio': f"${info['price']:,.4f}"
        }
        for symbol, info in symbol_info.items()
    ])
    st.dataframe(volume_df, use_container_width=True)
    
    # Buscar nuevas señales
    analyze_with_strategies(symbol_info, timeframe)
    
    # Obtener todas las señales activas de la base de datos
    conn = sqlite3.connect('trading_signals.db')
    query = '''
        SELECT 
            timestamp,
            symbol,
            strategy,
            signal_type,
            entry_price,
            current_price,
            stop_loss,
            take_profit,
            risk_reward,
            leverage
        FROM signals 
        WHERE status = 'ACTIVE'
        AND strategy IN ({})
        AND timestamp >= datetime('now', '-' || ? || ' hours')
        ORDER BY timestamp DESC
    '''.format(','.join(['?']*len(strategies)))
    
    params = [*strategies, hours_filter]
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if not df.empty:
        # Formatear columnas
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Crear DataFrame para mostrar
        df_display = pd.DataFrame({
            'Fecha': df['timestamp'].dt.strftime('%Y-%m-%d %H:%M'),
            'Par': df['symbol'],
            'Estrategia': df['strategy'],
            'Tipo': df['signal_type'],
            'Precio': df['current_price'].map('${:,.4f}'.format),
            'Entrada': df['entry_price'].map('${:,.4f}'.format),
            'Stop Loss': df['stop_loss'].map('${:,.4f}'.format),
            'Take Profit': df['take_profit'].map('${:,.4f}'.format),
            'R/R': df['risk_reward'].map('{:,.2f}'.format),
            'Apalancamiento': df['leverage'].map('{:,.1f}x'.format)
        })
        
        # Aplicar colores
        def color_type(val):
            return 'color: green' if val == 'LONG' else 'color: red'
        
        # Mostrar tabla con estilos
        st.dataframe(
            df_display.style.applymap(color_type, subset=['Tipo']),
            use_container_width=True
        )
        
        # Mostrar conteo de señales por par
        st.subheader("📊 Resumen de Señales Activas")
        col1, col2 = st.columns(2)
        
        with col1:
            signals_by_pair = df.groupby('symbol').size().reset_index()
            signals_by_pair.columns = ['Par', 'Cantidad']
            st.write("Señales por Par:")
            st.dataframe(signals_by_pair, use_container_width=True)
        
        with col2:
            signals_by_type = df.groupby('signal_type').size().reset_index()
            signals_by_type.columns = ['Tipo', 'Cantidad']
            st.write("Señales por Tipo:")
            st.dataframe(signals_by_type, use_container_width=True)
    else:
        st.info("No hay señales activas que cumplan con los filtros seleccionados")

def show_futures_analysis(symbol_info):
    """Muestra el análisis de futuros"""
    if not symbol_info:
        st.error("No se pudo obtener la lista de símbolos. Por favor, intenta más tarde.")
        return
    
    # Seleccionar par de trading
    selected_symbol = st.selectbox(
        "Seleccionar Par de Trading",
        list(symbol_info.keys()),
        key="analysis_symbol_select"
    )
    
    if selected_symbol:
        with st.spinner('Cargando datos...'):
            # Obtener datos históricos
            df = get_futures_data(selected_symbol)
            
            if df is not None and not df.empty:
                try:
                    # Obtener funding rate
                    funding = get_funding_rate(selected_symbol)
                    if funding:
                        st.metric(
                            "Funding Rate",
                            f"{funding['rate']*100:.4f}%",
                            f"Próximo funding: {funding['time'].strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                    
                    # Mostrar precio actual y cambio
                    current_price = df['close'].iloc[-1]
                    price_change = ((current_price - df['close'].iloc[-2])/df['close'].iloc[-2]*100)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Precio Actual",
                            f"${current_price:.4f}",
                            f"{price_change:.2f}%"
                        )
                    
                    with col2:
                        rsi = ta.momentum.RSIIndicator(df['close']).rsi().iloc[-1]
                        st.metric(
                            "RSI",
                            f"{rsi:.2f}",
                            "Sobrecompra" if rsi > 70 else "Sobreventa" if rsi < 30 else "Neutral"
                        )
                    
                    with col3:
                        volume_change = ((df['volume'].iloc[-1] - df['volume'].iloc[-2])/df['volume'].iloc[-2]*100)
                        st.metric(
                            "Volumen 24h",
                            f"${df['volume'].iloc[-1]:,.0f}",
                            f"{volume_change:.2f}%"
                        )
                    
                    # Gráfico de precios
                    fig = go.Figure()
                    
                    # Candlesticks
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='OHLC'
                    ))
                    
                    # EMAs
                    df['EMA20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
                    df['EMA50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
                    
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name='EMA 20', line=dict(color='orange')))
                    fig.add_trace(go.Scatter(x=df.index, y=df['EMA50'], name='EMA 50', line=dict(color='blue')))
                    
                    fig.update_layout(
                        title=f'Análisis Técnico de {selected_symbol}',
                        yaxis_title='Precio (USDT)',
                        xaxis_title='Fecha',
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar liquidaciones
                    st.subheader("💥 Liquidaciones Recientes")
                    liquidations = get_liquidations(selected_symbol)
                    if not liquidations.empty:
                        total_long_liq = liquidations[liquidations['side'] == 'LONG']['qty'].sum()
                        total_short_liq = liquidations[liquidations['side'] == 'SHORT']['qty'].sum()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Liquidaciones LONG", f"{total_long_liq:.2f} {symbol_info[selected_symbol]['baseAsset']}")
                        with col2:
                            st.metric("Liquidaciones SHORT", f"{total_short_liq:.2f} {symbol_info[selected_symbol]['baseAsset']}")
                    
                    # Análisis de señales
                    signals = RSIMACDStrategy().analyze(df, symbol_info[selected_symbol])
                    if signals:
                        st.subheader("🎯 Señales de Trading")
                        for signal in signals:
                            with st.expander(f"Señal {signal['tipo']}"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("📊 Niveles")
                                    st.write(f"Entrada: ${signal['entrada']:.4f}")
                                    st.write(f"Stop Loss: ${signal['stop_loss']:.4f}")
                                    st.write(f"Take Profit: ${signal['take_profit']:.4f}")
                                
                                with col2:
                                    st.write("📈 Análisis")
                                    st.write(f"R/R Ratio: {((signal['take_profit'] - signal['entrada']).abs() / (signal['stop_loss'] - signal['entrada']).abs()):.2f}")
                                    for razon in signal['razones']:
                                        st.write(f"• {razon}")
                
                except Exception as e:
                    st.error(f"Error al procesar los datos: {str(e)}")
            else:
                st.error(f"No se pudieron obtener datos para {selected_symbol}")

def check_and_notify():
    """Función que se ejecuta cada 15 minutos para buscar señales"""
    symbol_info = get_futures_symbols()
    if not symbol_info:
        return
    
    # Actualizar estado de señales existentes
    update_signal_status()
    
    # Buscar nuevas señales
    signals = analyze_with_strategies(symbol_info)
    
    if signals:
        # Agrupar señales por tipo
        long_signals = [s for s in signals if s['tipo'] == 'LONG']
        short_signals = [s for s in signals if s['tipo'] == 'SHORT']
        
        # Almacenar y notificar nuevas señales
        for signal in signals:
            store_signal(signal, '15m')
        
        # Enviar notificaciones
        if long_signals:
            message = "\n".join([
                f"🟢 {s['symbol']}: {s['strategy']} - Entrada: ${s['entrada']:.4f}"
                for s in long_signals[:3]
            ])
            send_notification(
                "🚀 Señales LONG Detectadas",
                f"Se encontraron {len(long_signals)} señales LONG:\n{message}"
            )
        
        if short_signals:
            message = "\n".join([
                f"🔴 {s['symbol']}: {s['strategy']} - Entrada: ${s['entrada']:.4f}"
                for s in short_signals[:3]
            ])
            send_notification(
                "💫 Señales SHORT Detectadas",
                f"Se encontraron {len(short_signals)} señales SHORT:\n{message}"
            )

def run_schedule():
    """Ejecuta el schedule en un thread separado"""
    while True:
        schedule.run_pending()
        time.sleep(1)

def init_database():
    """Inicializa la base de datos de señales"""
    conn = sqlite3.connect('trading_signals.db')
    c = conn.cursor()
    
    # Crear tabla de señales si no existe
    c.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            symbol TEXT NOT NULL,
            strategy TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            entry_price REAL NOT NULL,
            current_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            take_profit REAL NOT NULL,
            risk_reward REAL NOT NULL,
            leverage REAL NOT NULL,
            timeframe TEXT NOT NULL,
            reasons TEXT,
            status TEXT DEFAULT 'ACTIVE',
            exit_price REAL,
            exit_timestamp DATETIME,
            profit_loss REAL,
            win_loss TEXT
        )
    ''')
    
    # Verificar si necesitamos agregar la columna leverage
    c.execute("PRAGMA table_info(signals)")
    columns = [column[1] for column in c.fetchall()]
    
    if 'leverage' not in columns:
        c.execute('ALTER TABLE signals ADD COLUMN leverage REAL DEFAULT 1.0')
    
    if 'win_loss' not in columns:
        c.execute('ALTER TABLE signals ADD COLUMN win_loss TEXT')
    
    conn.commit()
    conn.close()

def main():
    st.title("🚀 Crypto Futures Trading Dashboard")
    
    # Inicializar base de datos
    init_database()
    
    # Iniciar schedule para alertas
    schedule.every(15).minutes.do(check_and_notify)
    threading.Thread(target=run_schedule, daemon=True).start()
    
    # Obtener símbolos de futuros
    symbol_info = get_futures_symbols()
    
    if symbol_info:
        # Crear pestañas
        tab1, tab2, tab3 = st.tabs([
            "📊 Análisis Individual",
            "🎯 Señales de Trading",
            "📜 Historial de Señales"
        ])
        
        with tab1:
            show_futures_analysis(symbol_info)
        
        with tab2:
            show_trading_signals_table(symbol_info)
        
        with tab3:
            show_signal_history()
    else:
        st.error("No se pudieron obtener los símbolos de futuros. Por favor, verifica tu conexión.")

if __name__ == "__main__":
    main()
