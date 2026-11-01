"""VolatilityQuant: interactive research dashboard. Run with streamlit run app.py."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from modules.engine import (prices_for,forecast_from_prices,option_data,surface_grid,
                            black_scholes,hedge,stress_prices,stress_statistics,STRESS)

PURPLE='#A78BFA';BLUE='#48CAE4';GREEN='#A3DA91';BG='#090B13'
st.set_page_config(page_title='VolatilityQuant | Quantitative Risk Intelligence',page_icon='◈',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.stApp{background:radial-gradient(ellipse at 75% 0%,#17132d 0%,#090b13 45%);}
.block-container{padding-top:2rem;max-width:1380px}
h1,h2,h3{letter-spacing:-.025em}
[data-testid="stMetric"]{background:#151725;border:1px solid #2b2b43;border-radius:16px;padding:15px}
[data-testid="stMetricValue"]{color:#b39aff}
div.stButton>button[kind="primary"]{background:linear-gradient(100deg,#7746dd,#3665e9);border:none;border-radius:10px;color:white}
[data-testid="stSidebar"]{background:#10111e}
.hero{padding:26px;border:1px solid #303050;border-radius:22px;background:linear-gradient(110deg,#21193d,#111827 75%);margin-bottom:20px}
.eyebrow{font-size:.72rem;letter-spacing:.22em;color:#9c92c9;font-weight:400}
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

with st.sidebar:
    st.markdown('## ◈ VOLATILITYQUANT')
    st.caption('QUANTITATIVE RISK INTELLIGENCE')
    page=st.radio('Explore', ['Overview','01 · AI Forecasting','02 · IV Surface','03 · Options Hedging','04 · Stress Testing'],label_visibility='collapsed')
    st.divider()
    st.caption('Research prototype · Not financial advice')

st.markdown('<div class="hero"><div class="eyebrow">VOLATILITYQUANT / RESEARCH PLATFORM</div><h1>Understand volatility.<br>Explore risk.</h1><p>Machine-learning forecasts · Options intelligence · Dynamic hedging · Historical stress analysis</p></div>',unsafe_allow_html=True)

if page=='Overview':
    st.markdown("""
    <div style="padding:30px 32px;border:1px solid #35304E;border-radius:20px;
                background:linear-gradient(115deg,#21173D,#101624 85%);margin:12px 0 24px;">
      <div style="font-size:12px;letter-spacing:3px;color:#A3DA91;font-weight:700;">
        VOLATILITYQUANT / RESEARCH PLATFORM</div>
      <div style="font-size:34px;line-height:1.2;font-weight:750;color:#F7F5FF;margin:14px 0;">
        From volatility forecasts to<br>informed risk analysis.</div>
      <div style="font-size:16px;color:#C5C3D8;max-width:760px;line-height:1.7;">
        Explore how statistical models, machine learning, and options analytics
        help us understand market uncertainty and evaluate financial risk.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('### Our research question')
    st.markdown(
        '**How do different volatility forecasting approaches behave across market conditions, '
        'and how might their estimates influence options hedging?**'
    )
    st.caption('A research prototype combining historical market data, predictive models, and simulated hedging.')

    st.markdown('### Explore the platform')
    cards = [
        ('01', 'AI Volatility Forecasting',
         'Compare historical volatility, GARCH(1,1), and XGBoost estimates.', '#A78BFA'),
        ('02', 'Implied Volatility Surface',
         'Explore option-chain implied volatility across strike and maturity.', '#48CAE4'),
        ('03', 'Options Hedging Simulator',
         'Examine delta hedging, rebalancing, and simulated trading costs.', '#A3DA91'),
        ('04', 'Historical Stress Testing',
         'Investigate SPY volatility and drawdowns during major market crises.', '#A78BFA'),
    ]
    for left, right in [(cards[0], cards[1]), (cards[2], cards[3])]:
        col1, col2 = st.columns(2, gap='medium')
        for col, (number, title, description, accent) in zip((col1, col2), (left, right)):
            with col:
                st.markdown(
                    f'<div style="min-height:170px;padding:22px;border:1px solid #302F49;'
                    f'border-top:3px solid {accent};border-radius:15px;background:#151725;'
                    f'margin-bottom:16px;">'
                    f'<div style="font-size:12px;color:{accent};font-weight:750;'
                    f'letter-spacing:2px;">MODULE {number}</div>'
                    f'<div style="font-size:19px;color:#F7F5FF;font-weight:700;'
                    f'margin:12px 0 8px;">{title}</div>'
                    f'<div style="color:#B6B6C9;font-size:14px;line-height:1.5;">'
                    f'{description}</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown('### How the research fits together')
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('**Forecast**')
        st.caption('Estimate future volatility using statistical and machine-learning methods.')
    with c2:
        st.markdown('**Analyze**')
        st.caption('Explore implied volatility and the sensitivity of option positions.')
    with c3:
        st.markdown('**Evaluate**')
        st.caption('Study simulated hedging outcomes and historical periods of market stress.')
    st.info('Choose a research module from the sidebar to begin. Market data may be delayed or unavailable.')

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

elif page=='02 · IV Surface':
    st.header('Implied Volatility Explorer')
    import yfinance as yf
    st.caption(f"yfinance version: {yf.__version__}")
    ticker=st.text_input('Ticker symbol',value='AAPL',key='iv_ticker').strip().upper()
    if st.button('Load option chains',type='primary'):
        try:
            with st.spinner('Retrieving options chains...'):spot,options=get_options(ticker)
            st.session_state['option_result']=(ticker,spot,options)
        except Exception as e:st.error(f'Options unavailable: {e}')
    if 'option_result' in st.session_state:
        t,spot,options=st.session_state['option_result']
        a,b,c=st.columns(3);a.metric('Reference stock price',f'${spot:.2f}');b.metric('Screened contracts',len(options));c.metric('Expirations',options.expiration.nunique())
        kind=st.selectbox('Surface', ['Combined OTM','Calls','Puts'])
        d=options.copy()
        if kind=='Calls':d=d[d.optionType=='Call']
        elif kind=='Puts':d=d[d.optionType=='Put']
        else:d=d[((d.optionType=='Call')&(d.strike>=spot))|((d.optionType=='Put')&(d.strike<=spot))]
        d=d[d.Moneyness.between(.85,1.15)]
        try:
            X,Y,Z,points=surface_grid(d)
            fig=go.Figure()
            fig.add_trace(go.Surface(x=X,y=Y,z=Z,colorscale=[[0,'#17396a'],[.5,'#7447cf'],[1,'#37d7b0']],opacity=.88,colorbar=dict(title='IV %'),name='Interpolated IV'))
            fig.add_trace(go.Scatter3d(x=points.Moneyness,y=points.daysToExpiration,z=points.IV_Percent,mode='markers',marker=dict(size=3,color='#e8e6ff'),name='Observed'))
            fig.update_layout(scene=dict(xaxis_title='Strike / spot',yaxis_title='Days to expiry',zaxis_title='IV (%)'))
            st.plotly_chart(style(fig,650),use_container_width=True)
        except Exception as e:st.warning(f'Surface could not be constructed with these quotes: {e}')
        exp=st.selectbox('Volatility smile expiration',sorted(d.expiration.dt.strftime('%Y-%m-%d').unique().tolist())) if not d.empty else None
        if exp:
            subset=d[d.expiration.dt.strftime('%Y-%m-%d')==exp].sort_values('Moneyness')
            fig=go.Figure()
            for label,color in [('Call',BLUE),('Put',PURPLE)]:
                sub=subset[subset.optionType==label]
                if not sub.empty:fig.add_trace(go.Scatter(x=sub.Moneyness,y=sub.IV_Percent,mode='lines+markers',name=label,line_color=color))
            fig.add_vline(x=1,line_dash='dash',line_color='#888')
            fig.update_layout(xaxis_title='Strike / spot',yaxis_title='Implied volatility (%)')
            st.plotly_chart(style(fig),use_container_width=True)
        st.caption('Exploratory interpolation only; not arbitrage-free. Last-trade timestamps do not prove current quote freshness. Thinly traded options can distort the surface.')

elif page=='03 · Options Hedging':
    st.header('Dynamic Options Hedging')
    st.caption('Educational five-day simulation; all paths are hypothetical. Volatility inputs can be model-based or manually chosen.')
    a,b,c=st.columns(3)
    with a:kind=st.selectbox('Option type',['call','put']);K=st.slider('Strike ($)',70,130,100,5)
    with b:vol_source=st.selectbox('Hedge volatility source',['Manual','Historical · NVDA','GARCH · NVDA','XGBoost · NVDA']);manual=st.slider('Manual hedge volatility (%)',10,90,35)
    with c:cost=st.slider('Trading cost (%)',0.0,0.5,0.10,0.05);market=st.slider('Option pricing volatility (%)',10,90,35)
    if st.button('Run hedge simulation',type='primary'):
        try:
            if vol_source=='Manual':hvol=manual/100
            else:
                pred=get_forecasts('NVDA');hvol=pred[vol_source.split(' · ')[0]]/100
            rng=np.random.default_rng(42);S0=100;T=5/252;r=.04;true_vol=market/100
            shocks=rng.standard_normal(5);path=[S0]
            for z in shocks:path.append(path[-1]*np.exp((r-.5*true_vol**2)/252+true_vol*np.sqrt(1/252)*z))
            result=hedge(path,K,T,r,hvol,true_vol,kind,cost/100)
            st.session_state['hedge_result']=(result,hvol,kind,K)
        except Exception as e:st.error(f'Simulation unavailable: {e}')
    if 'hedge_result' in st.session_state:
        df,hvol,kind,K=st.session_state['hedge_result']
        a,b,c=st.columns(3)
        metric_items = [
            (a, 'Final hedged P&amp;L', f'${df.PortfolioValue.iloc[-1]:,.2f}', PURPLE),
            (b, 'Total trading costs', f'${df.TransactionCosts.iloc[-1]:,.2f}', BLUE),
            (c, 'Hedge volatility', f'{hvol*100:.2f}%', GREEN),
        ]
        for col, label, value, accent in metric_items:
            with col:
                st.markdown(
                    f'<div style="background:#151725;border:1px solid #2b2b43;'
                    f'border-radius:16px;padding:20px 22px;min-height:140px;">'
                    f'<div style="color:#F8FAFC;font-size:16px;margin-bottom:14px;">'
                    f'{label}</div>'
                    f'<div style="color:{accent};font-size:40px;font-weight:600;'
                    f'line-height:1.2;">{value}</div></div>',
                    unsafe_allow_html=True,
                )
        fig=go.Figure();fig.add_trace(go.Scatter(x=df.Day,y=df.PortfolioValue,name='Hedged portfolio',line_color=PURPLE,mode='lines+markers'));fig.add_hline(y=0,line_dash='dash',line_color='#777')
        fig.update_layout(xaxis_title='Trading day',yaxis_title='Portfolio P&L ($)');st.plotly_chart(style(fig),use_container_width=True)
        st.caption('Simplified European-style cash settlement, no dividends, fractional shares, fixed model volatility and common Black–Scholes premium. Results are not historical option trades.')

elif page=='04 · Stress Testing':
    st.header('Historical Stress Testing')
    st.caption('Historical SPY price, 20-day realized volatility and peak-to-date drawdown. Forecasting crisis results remain documented in the research notebook.')
    crisis=st.selectbox('Select historical event',list(STRESS))
    if st.button('Analyze historical period',type='primary'):
        try:
            with st.spinner('Loading long-run SPY history...'):p=get_stress();vol,dd,summary=stress_statistics(p)
            st.session_state['stress_result']=(p,vol,dd,summary)
        except Exception as e:st.error(f'Stress data unavailable: {e}')
    if 'stress_result' in st.session_state:
        p,vol,dd,summary=st.session_state['stress_result'];start,end=STRESS[crisis]
        r=summary[summary.Crisis==crisis]
        if not r.empty:
            row=r.iloc[0];a,b,c=st.columns(3)
            a.metric('Period return',f'{row["Period return (%)"]:.2f}%');b.metric('Peak 20d volatility',f'{row["Peak 20d vol (%)"]:.2f}%');c.metric('Worst drawdown from earlier peak',f'{row["Worst drawdown (%)"]:.2f}%')
        fig=go.Figure(go.Scatter(x=vol.loc[start:end].index,y=vol.loc[start:end].values,line_color=PURPLE,name='20d volatility'))
        fig.update_layout(xaxis_title='Date',yaxis_title='Annualized volatility (%)');st.plotly_chart(style(fig),use_container_width=True)
        fig=go.Figure(go.Scatter(x=dd.loc[start:end].index,y=dd.loc[start:end].values,line_color=GREEN,fill='tozeroy',name='Drawdown'))
        fig.update_layout(xaxis_title='Date',yaxis_title='Drawdown (%)');st.plotly_chart(style(fig),use_container_width=True)
        st.dataframe(summary.round(2),hide_index=True,use_container_width=True)
        st.caption('Drawdown is measured from the running peak since the start of the historical dataset; crisis windows are illustrative selections.')

st.divider();st.caption('VOLATILITYLAB · Research & educational use only · No investment recommendations · Data providers may have delays')
