"""VolatilityQuant: interactive research dashboard. Run with streamlit run app.py."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from modules.engine import (prices_for,forecast_from_prices,option_data,load_aapl_snapshot,surface_grid,
                            black_scholes,hedge,stress_prices,stress_statistics,STRESS)

PURPLE = '#A78BFA'
BLUE = '#48CAE4'
GREEN = '#A3DA91'
BG = '#090B13'
st.set_page_config(page_title='VolatilityQuant | Quantitative Risk Intelligence',page_icon='◈',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp{background:radial-gradient(ellipse at 75% 0%,#17132d 0%,#090b13 45%);}
.block-container{padding-top:2rem;max-width:1380px}
h1,h2,h3{letter-spacing:-.025em}
[data-testid="stMetric"]{background:#151725;border:1px solid #2b2b43;border-radius:16px;padding:15px}
[data-testid="stMetricValue"]{color:#A78BFA}
div.stButton>button[kind="primary"]{background:linear-gradient(100deg,#7746dd,#48CAE4);border:none;border-radius:10px;color:white}
[data-testid="stSidebar"]{background:#10111e}
.hero{padding:26px;border:1px solid #303050;border-radius:22px;background:linear-gradient(110deg,#21193d,#111827 75%);margin-bottom:20px}
.eyebrow{font-size:.72rem;letter-spacing:.22em;color:#9c92c9;font-weight:700}
.hero h1{font-size:2.7rem;margin:.3rem 0;color:#f7f5ff}
.hero p{color:#c3bfd6;margin-bottom:0}
</style>''',unsafe_allow_html=True)

def style(fig,height=470):
    fig.update_layout(template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#DADAF0',height=height,margin=dict(l=10,r=10,t=45,b=10),legend=dict(orientation='h',y=1.1))
    return fig

@st.cache_data(ttl=1800,show_spinner=False)
def get_prices(ticker):return prices_for(ticker)
@st.cache_data(ttl=1800,show_spinner=False)
def get_forecasts(ticker):return forecast_from_prices(get_prices(ticker))
@st.cache_data(ttl=1800,show_spinner=False)
def get_options(ticker):return option_data(ticker)
@st.cache_data(ttl=3600,show_spinner=False)
def get_stress():return stress_prices()
# Handle navigation requests before creating the sidebar widget
if "next_page" in st.session_state:
    st.session_state["page_nav"] = st.session_state.pop("next_page")
with st.sidebar:
    st.markdown('## ◈ VOLATILITYQUANT')
    st.caption('QUANTITATIVE RISK INTELLIGENCE')
    page=st.radio('Explore', ['Overview','01 · AI Forecasting','02 · IV Surface','03 · Options Hedging','04 · Stress Testing'],label_visibility='collapsed',key='page_nav')
    st.divider()
    st.caption('Research prototype · Not financial advice')

st.markdown('<div class="hero"><div class="eyebrow">VOLATILITYQUANT / RESEARCH PLATFORM</div><h1>Understand volatility.<br>Explore risk.</h1><p>Machine-learning forecasts · Options intelligence · Dynamic hedging · Historical stress analysis</p></div>',unsafe_allow_html=True)

if page=='Overview':
    a,b,c,d=st.columns(4)
    for col,name,val in [(a,'Forecast models','3'),(b,'Research securities','5'),(c,'Core modules','4'),(d,'Hedge scenarios','3')]:col.metric(name,val)
    st.subheader('Research modules')
    for title,desc in [('01 / AI Volatility Forecasting','Five-day historical, GARCH(1,1) and XGBoost volatility estimates.'),('02 / Implied Volatility Surface','Real option-chain quotes, call/put surfaces and volatility skew.'),('03 / Options Hedging','European option delta hedging with stock rebalancing and trading costs.'),('04 / Historical Stress Testing','SPY volatility and drawdown behavior across three market crises.')]:
        with st.container(border=True):st.markdown(f'**{title}**');st.caption(desc)
    st.info('Select a module from the sidebar. Data is fetched on demand; market quotes may be delayed or incomplete.')

elif page=='01 · AI Forecasting':
    st.header('AI Volatility Forecasting')
    ticker=st.text_input('Ticker symbol',value='NVDA',key='forecast_ticker').strip().upper()
    if st.button('Generate forecasts',type='primary'):
        try:
            with st.spinner('Downloading prices and fitting GARCH / XGBoost...'):
                p=get_prices(ticker);pred=get_forecasts(ticker)
            st.session_state['forecast_result']=(ticker,p,pred)
        except Exception as e:st.error(f'Forecast unavailable: {e}')
    if 'forecast_result' in st.session_state:
        t,p,pred=st.session_state['forecast_result']
        st.caption(f'Latest available data: {p.index[-1].date()} · {t} · Annualized volatility')
        cols=st.columns(3)
        for col,(name,v) in zip(cols,pred.items()):col.metric(name,f'{v:.2f}%')
        fig=go.Figure(go.Bar(x=list(pred),y=list(pred.values()),marker_color=[GREEN,BLUE,PURPLE],text=[f'{v:.2f}%' for v in pred.values()],textposition='outside'))
        fig.update_layout(yaxis_title='Annualized volatility (%)');st.plotly_chart(style(fig),use_container_width=True)
        st.caption('Historical = trailing 20-day mean squared returns. GARCH and XGBoost estimate next five trading days. These are volatility estimates, not expected stock returns.')
        st.subheader('Apply this forecast to options hedging')
        chosen_model=st.selectbox('Choose a forecast model',list(pred),key='transfer_model')
        st.caption(f'{t} · {chosen_model}: {pred[chosen_model]:.2f}% annualized · five-trading-day horizon')
        if st.button('Use forecast in hedging simulator',type='primary'):
            st.session_state['hedge_input']={'ticker':t,'model':chosen_model,'volatility_pct':float(pred[chosen_model]),'asof':str(p.index[-1].date())}
            st.session_state.pop('hedge_result',None)
            st.session_state['next_page']='03 · Options Hedging'
            st.rerun()

elif page=='02 · IV Surface':
    st.header('Implied Volatility Explorer')
    data_mode = st.radio(
        'Options data source',
        ['Live Yahoo Finance', 'Saved AAPL Snapshot'],
        horizontal=True,
        key='iv_data_mode'
    )
    if data_mode == 'Live Yahoo Finance':
        ticker = st.text_input('Ticker symbol', value='AAPL', key='iv_ticker').strip().upper()
    else:
