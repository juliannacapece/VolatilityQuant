"""VolatilityQuant: interactive research dashboard. Run with streamlit run app.py."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from modules.engine import (prices_for,forecast_from_prices,option_data,load_aapl_snapshot,surface_grid,
                            historical_nvda_prices,historical_episode,black_scholes,hedge,simulate_path,stress_prices, stress_statistics, stress_resilience, STRESS, RESILIENCE_WEIGHTS)

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
def get_options(ticker,rate):return option_data(ticker,rate=rate)
@st.cache_data(show_spinner=False)
def get_snapshot(rate):return load_aapl_snapshot(rate)
@st.cache_data(ttl=3600,show_spinner=False)
def get_stress(ticker):return stress_prices(ticker)
def get_stress_analysis(ticker):
    p=get_stress(ticker);vol,dd,summary=stress_statistics(p)
    return p,vol,dd,summary,stress_resilience(p,vol)

STRESS_TICKERS={'SPY':'S&P 500 ETF','QQQ':'Nasdaq-100 ETF','IWM':'Russell 2000 ETF','DIA':'Dow Jones ETF',
                'AAPL':'Apple','MSFT':'Microsoft','AMZN':'Amazon','NVDA':'NVIDIA','JPM':'JPMorgan Chase',
                'XOM':'ExxonMobil','JNJ':'Johnson & Johnson','TSLA':'Tesla'}

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



   
    # FEATURE 1: Dynamic GARCH Parameter Stability
    st.divider()

    with st.expander("GARCH(1,1) Parameter Stability Analysis"):
        st.subheader("Rolling GARCH Parameter Stability")

        st.caption(
            "Analyze how GARCH parameters change across rolling "
            "two-year estimation windows, updated every 63 trading days."
        )

        if "forecast_result" not in st.session_state:
            st.info("Generate forecasts first to load a ticker.")

        else:
            stability_ticker = st.session_state["forecast_result"][0]

            st.write(f"Selected ticker: **{stability_ticker}**")

            if st.button(
                "Run parameter stability analysis",
                key="run_garch_stability"
            ):
                try:
                    from arch import arch_model

                    with st.spinner(
                        f"Fitting rolling GARCH models for {stability_ticker}..."
                    ):
                        prices = get_prices(stability_ticker)

                        # Convert prices into daily log returns (%).
                        if isinstance(prices, pd.DataFrame):
                            if "Close" in prices.columns:
                                prices = prices["Close"]
                            elif "Adj Close" in prices.columns:
                                prices = prices["Adj Close"]
                            else:
                                raise ValueError(
                                    "Could not find a closing-price column."
                                )

                        prices = pd.Series(prices).dropna().astype(float)
                        returns_pct = (
                            np.log(prices / prices.shift(1)).dropna() * 100
                        )

                        window_days = 504
                        step_days = 63

                        if len(returns_pct) < window_days:
                            raise ValueError(
                                "Not enough price history. "
                                "At least 504 daily returns are required."
                            )

                        records = []

                        for end in range(
                            window_days,
                            len(returns_pct) + 1,
                            step_days
                        ):
                            sample = returns_pct.iloc[
                                end - window_days:end
                            ]

                            model = arch_model(
                                sample,
                                mean="Constant",
                                vol="GARCH",
                                p=1,
                                q=1,
                                dist="t"
                            )

                            fit = model.fit(
                                disp="off",
                                show_warning=False
                            )

                            if fit.convergence_flag != 0:
                                continue

                            params = fit.params
                            alpha = float(params["alpha[1]"])
                            beta = float(params["beta[1]"])

                            records.append({
                                "Date": sample.index[-1],
                                "omega": float(params["omega"]),
                                "alpha": alpha,
                                "beta": beta,
                                "persistence": alpha + beta
                            })

                        stability = pd.DataFrame(records)

                        if stability.empty:
                            raise ValueError(
                                "No rolling GARCH models converged."
                            )

                        st.session_state["garch_stability_result"] = (
                            stability_ticker,
                            stability
                        )

                except Exception as e:
                    st.error(f"Stability analysis failed: {e}")

            # Display saved results after the button is clicked.
            result = st.session_state.get(
                "garch_stability_result"
            )

            if result is not None:
                result_ticker, stability = result

                if result_ticker != stability_ticker:
                    st.info(
                        "Click Run parameter stability analysis "
                        "to calculate results for this ticker."
                    )

                else:
                    persistence = stability["persistence"]

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Average Persistence",
                        f"{persistence.mean():.4f}"
                    )
                    col2.metric(
                        "Minimum Persistence",
                        f"{persistence.min():.4f}"
                    )
                    col3.metric(
                        "Maximum Persistence",
                        f"{persistence.max():.4f}"
                    )

                    # Chart 1: Persistence
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=stability["Date"],
                        y=stability["persistence"],
                        mode="lines+markers",
                        name="Alpha + Beta",
                        line=dict(color=PURPLE, width=3)
                    ))

                    fig.add_hline(
                        y=1,
                        line_dash="dash",
                        line_color="#888888"
                    )

                    fig.update_layout(
                        title=f"{result_ticker} · GARCH Persistence",
                        xaxis_title="Window End Date",
                        yaxis_title="Alpha + Beta"
                    )

                    st.plotly_chart(
                        style(fig, 380),
                        use_container_width=True
                    )

                    # Chart 2: Alpha and Beta
                    fig2 = go.Figure()

                    for parameter, color in [
                        ("alpha", BLUE),
                        ("beta", GREEN)
                    ]:
                        fig2.add_trace(go.Scatter(
                            x=stability["Date"],
                            y=stability[parameter],
                            mode="lines+markers",
                            name=parameter,
                            line=dict(color=color, width=3)
                        ))

                    fig2.update_layout(
                        title="Individual GARCH Parameters",
                        xaxis_title="Window End Date",
                        yaxis_title="Parameter Estimate"
                    )

                    st.plotly_chart(
                        style(fig2, 350),
                        use_container_width=True
                    )

                    st.info(
                        f"{result_ticker}'s estimated GARCH persistence "
                        f"averaged {persistence.mean():.4f}, "
                        f"ranging from {persistence.min():.4f} "
                        f"to {persistence.max():.4f}. "
                        "Changes across windows indicate variation "
                        "in estimated volatility dynamics."
                    )

                    st.caption(
                        "Rolling two-year windows overlap. "
                        "This is a descriptive stability analysis, "
                        "not a formal statistical test."
                    )

                    if st.checkbox(
                        "Show parameter estimates",
                        key="show_garch_parameters"
                    ):
                        st.dataframe(
                            stability.round(4),
                            hide_index=True,
                            use_container_width=True
                        )



elif page=='02 · IV Surface':
    st.header('Implied Volatility Explorer')
    LIVE,SNAPSHOT='Live (Yahoo Finance)','Saved AAPL snapshot'
    src_col,ticker_col,rate_col=st.columns([1.4,1,1])
    with src_col:source=st.radio('Data source',[LIVE,SNAPSHOT],horizontal=True,key='iv_source')
    with ticker_col:ticker=st.text_input('Ticker symbol',value='AAPL',key='iv_ticker',disabled=source==SNAPSHOT).strip().upper()
    with rate_col:rate=st.number_input('Risk-free rate (%)',0.0,15.0,4.0,0.25,key='iv_rate')/100
    if st.button('Load option chains',type='primary'):
        result=None;live_error=None
        if source==LIVE:
            try:
                with st.spinner('Retrieving options chains...'):spot,options,as_of=get_options(ticker,rate)
                result=(ticker,spot,options,as_of,'Live · Yahoo Finance',None,rate)
            except Exception as e:live_error=f'live options for {ticker} were unavailable ({e})'
        if result is None:  # snapshot chosen, or automatic fallback after a live failure
            try:
                spot,options,as_of=get_snapshot(rate)
                result=('AAPL',spot,options,as_of,'Saved snapshot',live_error,rate)
            except Exception as e:st.error(f'Options unavailable. Saved snapshot also failed: {e}')
        if result:st.session_state['option_result']=result
    if len(st.session_state.get('option_result',()))==7:
        t,spot,options,as_of,label,live_error,run_rate=st.session_state['option_result']
        if live_error:st.warning(f'Showing the saved AAPL snapshot because {live_error}.')
        st.caption(f'{label} · {t} · quotes as of {as_of.tz_convert("America/New_York"):%Y-%m-%d %H:%M} New York time · risk-free rate {run_rate:.2%}')
        a,b,c=st.columns(3);a.metric('Reference stock price',f'${spot:.2f}');b.metric('Screened contracts',len(options));c.metric('Expirations',options.expiration.nunique())
        kind_col,iv_col=st.columns(2)
        with kind_col:kind=st.selectbox('Surface', ['Combined OTM','Calls','Puts'])
        with iv_col:iv_source=st.radio('Implied volatility',['Computed from mid-quotes','Yahoo-reported'],horizontal=True,
                                       help='Computed: Black-76 on bid/ask mid with a forward implied from put-call parity per expiry. Yahoo-reported values are shown for comparison.')
        d=options.copy()
        computed=iv_source=='Computed from mid-quotes'
        if not computed:d['IV_Percent']=d['YahooIV_Percent']
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
                if not sub.empty:
                    fig.add_trace(go.Scatter(x=sub.Moneyness,y=sub.IV_Percent,mode='lines+markers',name=label,line_color=color))
                    if computed and st.session_state.get('iv_compare_yahoo'):
                        fig.add_trace(go.Scatter(x=sub.Moneyness,y=sub.YahooIV_Percent,mode='markers',name=f'{label} · Yahoo-reported',
                                                 marker=dict(color=color,symbol='circle-open',size=7),opacity=.7))
            fig.add_vline(x=1,line_dash='dash',line_color='#888')
            fig.update_layout(xaxis_title='Strike / spot',yaxis_title='Implied volatility (%)')
            st.plotly_chart(style(fig),use_container_width=True)
            if computed:st.checkbox('Overlay Yahoo-reported IV for comparison',key='iv_compare_yahoo')
            fwd=subset[['forward','forwardSource']].iloc[0] if not subset.empty else None
            if fwd is not None:st.caption(f'Forward for {exp}: ${fwd.forward:.2f} ({fwd.forwardSource}).')
        st.caption('Computed IV: Black-76 on the bid/ask mid-quote, with each expiry\'s forward implied from put-call parity (so dividends need no separate input) and the chosen risk-free rate. '
                   'Listed equity options are American; out-of-the-money IVs are little affected, but deep in-the-money values (Calls/Puts views) can be biased. '
                   'Exploratory interpolation only; not arbitrage-free. Last-trade timestamps do not prove current quote freshness. Thinly traded options can distort the surface.')

elif page=='03 · Options Hedging':
    st.header('Dynamic Options Hedging')
    mode=st.radio('Experiment',['Historical NVDA','Controlled simulation'],horizontal=True)
    if mode=='Historical NVDA':
        st.caption('Latest five completed sessions available, conservatively excluding today. Six unadjusted closes define five stock moves. Hypothetical at-the-money European call, 100 shares, 4% rate, daily close rebalancing, no dividends.')
        cost=st.slider('Trading cost (%)',0.0,0.5,0.10,0.05,key='historical_cost')
        manual=None
        if st.checkbox('Include a manual hedge benchmark'):
            manual=st.slider('Manual annualized volatility (%)',1,200,35,key='historical_manual')/100
        st.caption('Historical, GARCH, and XGBoost update before each move using up to five years of past data. Forecast horizons shrink from five days to one. XGBoost uses only fully observed training targets and fixed model settings.')
        if st.button('Run historical comparison',type='primary'):
            try:
                with st.spinner('Loading completed NVDA sessions and fitting daily forecasts...'):
                    prices,fetched=historical_nvda_prices()
                    result=historical_episode(prices,cost/100,manual)
                st.session_state['historical_hedge_result']=(result,fetched,cost)
            except Exception as e:st.error(f'Historical comparison unavailable: {e}')
        if 'historical_hedge_result' in st.session_state:
            result,fetched,run_cost=st.session_state['historical_hedge_result'];path=result['path']
            st.caption(f'Last completed run · fetched {fetched.strftime("%Y-%m-%d %H:%M %Z")} · latest close {path.index[-1].date()} · trading cost {run_cost:.2f}%. Run again to apply changes.')
            st.write(f'Opening reference: {path.index[0].date()}. Five evaluated sessions: {path.index[1].date()} through {path.index[-1].date()}. Strike: ${result["strike"]:.2f}. Shared hypothetical premium: ${result["premium"]:.2f} per share.')
            fig=go.Figure(go.Scatter(x=path.index,y=path.values,name='NVDA close',mode='lines+markers',line_color=BLUE))
            fig.update_layout(xaxis_title='Session date',yaxis_title='NVDA close ($)')
            st.plotly_chart(style(fig,320),use_container_width=True)
            st.subheader('Terminal hedging comparison')
            comparison = result['summary'].assign(
                **{'Absolute hedging error ($)': result['summary']['Terminal P&L ($)'].abs()}
            )
            st.caption('Terminal P&L is signed hedging error relative to zero. Absolute error measures the distance from zero after fees.')
            fig = go.Figure(go.Bar(
                x=comparison['Strategy'], y=comparison['Absolute hedging error ($)'],
                marker_color=PURPLE,
            ))
            fig.update_layout(yaxis_title='Absolute hedging error ($)')
            st.plotly_chart(style(fig, 300), use_container_width=True)
            st.dataframe(comparison,hide_index=True,use_container_width=True)
            st.subheader('Daily forecasts · annualized volatility (%)')
            st.dataframe(result['forecasts'],hide_index=True,use_container_width=True)
            st.subheader('Stock hedge and cash ledger')
            st.dataframe(result['ledger'],hide_index=True,use_container_width=True)
            st.info('One historical episode cannot establish which strategy reduces risk consistently. The opening premium is shared and based on starting historical volatility; these are hypothetical options, not observed option trades. Daily cash balances exclude option value and are not daily portfolio P&L.')
    else:
        st.caption('Five-day hypothetical stock path, long option, and daily stock hedge. All volatility inputs are annualized.')
        a,b=st.columns(2)
        with a:
            kind=st.selectbox('Option type',['call','put'])
            K=st.slider('Strike ($)',70,130,100,5)
        with b:
            cost=st.slider('Trading cost (%)',0.0,0.5,0.10,0.05)

        def manual_schedule(prefix):
            mode=st.radio('Volatility schedule',['Constant','Daily'],horizontal=True,key=prefix+'_mode')
            if mode=='Constant':
                value=st.slider('Annualized volatility (%)',10,90,35,key=prefix+'_constant')
                return np.full(5,value/100)
            values=[]
            for day,col in enumerate(st.columns(5)):
                with col:
                    values.append(st.number_input(f'Day {day+1} (%)',min_value=1.0,max_value=200.0,value=35.0,step=1.0,key=f'{prefix}_{day}'))
            return np.asarray(values)/100

        st.subheader('1 · Simulated market')
        st.caption('These values generate stock moves. Each daily value applies to the next close-to-close interval. The full schedule is assumed known for option pricing, using remaining average variance.')
        market_vols=manual_schedule('market')
        st.subheader('2 · Hedging strategy')
        st.caption('These are the strategy’s beliefs, used to calculate delta and stock trades. They do not change the simulated stock path.')
        vol_source=st.selectbox('Hedge volatility source',['Manual'])
        hedge_vols=manual_schedule('hedge') if vol_source=='Manual' else None
        if vol_source!='Manual':
            st.caption('The selected NVDA estimate is calculated at the start and held fixed for all five days. It is not updated from the simulated returns.')
        st.caption('A daily manual hedge value is used before that day’s move and is treated as a flat volatility assumption over the option’s remaining life.')
        if st.button('Run hedge simulation',type='primary'):
            try:
                if hedge_vols is None:
                    pred=get_forecasts('NVDA')
                    hedge_vols=np.full(5,pred[vol_source.split(' · ')[0]]/100)
                path=simulate_path(market_vols)
                result=hedge(path,K,5/252,.04,hedge_vols,market_vols,kind,cost/100)
                schedule=pd.DataFrame({'Trading day':range(1,6),'Market volatility (%)':market_vols*100,'Hedge volatility (%)':hedge_vols*100})
                st.session_state['daily_hedge_result']=(result,schedule,vol_source)
            except Exception as e:st.error(f'Simulation unavailable: {e}')
        if 'daily_hedge_result' in st.session_state:
            df,schedule,source=st.session_state['daily_hedge_result']
            st.caption(f'Last completed run · {source}. Run again to apply changed settings. Stock starts at $100; the same random shocks are used across runs.')
            final_error = df.PortfolioValue.iloc[-1]
            a,b,c=st.columns(3)
            a.metric('Signed hedging error (final P&L)',f'${final_error:,.2f}')
            b.metric('Absolute hedging error',f'${abs(final_error):,.2f}')
            c.metric('Total trading costs',f'${df.TransactionCosts.iloc[-1]:,.2f}')
            st.caption('Absolute error is the distance from zero at expiration, after trading costs.')
            fig=go.Figure()
            for column,color in [('Market volatility (%)',BLUE),('Hedge volatility (%)',PURPLE)]:
                fig.add_trace(go.Scatter(x=schedule['Trading day'],y=schedule[column],name=column,line_color=color,mode='lines+markers'))
            fig.update_layout(xaxis_title='Trading day',yaxis_title='Annualized volatility (%)')
            st.plotly_chart(style(fig,300),use_container_width=True)
            st.dataframe(schedule,hide_index=True,use_container_width=True)
            fig=go.Figure(go.Scatter(x=df.Day,y=df.PortfolioValue,name='Hedged portfolio',line_color=PURPLE,mode='lines+markers'))
            fig.add_hline(y=0,line_dash='dash',line_color='#888')
            fig.update_layout(xaxis_title='Trading day',yaxis_title='Portfolio P&L ($)')
            st.plotly_chart(style(fig),use_container_width=True)
            st.caption('European-style cash settlement, no dividends, fractional shares, 4% interest, and daily rebalancing. Market volatility follows a deterministic user-defined schedule; this is not a stochastic-volatility pricing model or historical option trade.')

elif page=='04 · Stress Testing':
    st.header('Historical Stress Testing')
    st.caption('Historical price, 20-day realized volatility, peak-to-date drawdown and a crisis resilience score for a chosen ticker. Forecasting crisis results remain documented in the research notebook.')
    sel_a,sel_b=st.columns(2)
    with sel_a:
        choice=st.selectbox('Ticker',list(STRESS_TICKERS)+['Custom…'],key='stress_ticker_choice',
                            format_func=lambda t:t if t=='Custom…' else f'{t} · {STRESS_TICKERS[t]}')
        ticker=(st.text_input('Custom ticker symbol',value='GOOGL',key='stress_custom_ticker').strip().upper()
                if choice=='Custom…' else choice)
    with sel_b:
        crisis=st.selectbox('Select historical event',list(STRESS))
    if st.button('Analyze historical period',type='primary'):
        if not ticker:st.error('Enter a ticker symbol.')
        else:
            try:
                with st.spinner(f'Loading long-run {ticker} history...'):
                    p,vol,dd,summary,res=get_stress_analysis(ticker)
                st.session_state['stress_result']=(ticker,p,vol,dd,summary,res)
            except Exception as e:st.error(f'Stress data unavailable for {ticker}: {e}')
    if len(st.session_state.get('stress_result',()))==6:
        t,p,vol,dd,summary,res=st.session_state['stress_result'];start,end=STRESS[crisis]
        if t!=ticker:st.info(f'Showing results for {t}. Click "Analyze historical period" to load {ticker}.')
        st.subheader(f'{t} · {crisis}')
        rr=res[res.Crisis==crisis]
        if rr.empty:
            st.warning(f'{t} needs at least one year of price history before the {crisis} window, so no resilience score is shown.')
        else:
            x=rr.iloc[0];score=x['Resilience score']
            band,color=('Strong',GREEN) if score>=60 else ('Moderate',BLUE) if score>=40 else ('Weak','#F87171')
            rec='Not yet recovered' if x['Recovery censored'] else f"Recovered {x['Recovery date']}"
            st.markdown(
                f'<div style="background:linear-gradient(110deg,#21193d,#111827 75%);border:1px solid {color};'
                f'border-radius:20px;padding:24px 28px;margin-bottom:18px;display:flex;align-items:center;gap:32px;flex-wrap:wrap">'
                f'<div><div class="eyebrow">CRISIS RESILIENCE SCORE</div>'
                f'<div style="font-size:64px;color:{color};line-height:1">{score:.0f}<span style="font-size:24px;color:#9c92c9">/100</span></div>'
                f'<div style="color:{color};font-size:15px;margin-top:4px">{band} recovery</div></div>'
                f'<div style="color:#c3bfd6;font-size:15px;line-height:1.7">'
                f'Peak {x["Peak date"]} → trough {x["Trough date"]} · {rec}<br>'
                f'{x["Max drawdown (%)"]:.1f}% drawdown · {x["Days to recover"]} trading days to recover · '
                f'{x["6M retracement (%)"]:.0f}% of loss regained in 6 months</div></div>',
                unsafe_allow_html=True)
            spy_score=None
            if t!='SPY':
                try:
                    spy_res=get_stress_analysis('SPY')[4];spy_row=spy_res[spy_res.Crisis==crisis]
                    if not spy_row.empty:spy_score=spy_row['Resilience score'].iloc[0]
                except Exception:pass
            cols=st.columns(len(RESILIENCE_WEIGHTS)+(spy_score is not None))
            for col,(k,w) in zip(cols,RESILIENCE_WEIGHTS.items()):
                col.metric(f'{k} ({w:.0%})',f"{x[f'{k} score']:.0f}")
            if spy_score is not None:
                cols[-1].metric('Resilience vs SPY',f'{score:.0f}',delta=f'{score-spy_score:+.0f} pts')
            if x['Recovery censored']:
                st.caption('Price has not regained its pre-crisis peak; the recovery score is an upper bound.')
        r=summary[summary.Crisis==crisis]
        if not r.empty:
            row=r.iloc[0];a,b,c=st.columns(3)
            a.metric('Period return',f'{row["Period return (%)"]:.2f}%');b.metric('Peak 20d volatility',f'{row["Peak 20d vol (%)"]:.2f}%');c.metric('Worst drawdown from earlier peak',f'{row["Worst drawdown (%)"]:.2f}%')
        fig=go.Figure(go.Scatter(x=vol.loc[start:end].index,y=vol.loc[start:end].values,line_color=PURPLE,name='20d volatility'))
        fig.update_layout(xaxis_title='Date',yaxis_title='Annualized volatility (%)');st.plotly_chart(style(fig),use_container_width=True)
        fig=go.Figure(go.Scatter(x=dd.loc[start:end].index,y=dd.loc[start:end].values,line_color=GREEN,fill='tozeroy',name='Drawdown'))
        fig.update_layout(xaxis_title='Date',yaxis_title='Drawdown (%)');st.plotly_chart(style(fig),use_container_width=True)
        st.dataframe(summary.merge(res[['Crisis','Resilience score','Days to recover']],on='Crisis',how='left').round(2),hide_index=True,use_container_width=True)
        st.caption('Drawdown is measured from the running peak since the start of the historical dataset (2005 or first listing); crisis windows are market-wide illustrative selections, so a single stock may bottom outside them. '
                   'Resilience score (0–100) weights depth 30%, time to regain the pre-crisis peak 30%, 6-month retracement 20% and volatility normalization 20%; '
                   'it uses prices after each window and is descriptive, not predictive. Bands: Strong ≥ 60, Moderate 40–60, Weak < 40. '
                   'Yahoo Finance mostly covers companies that still trade, so firms that failed in a crisis are missing (survivorship bias); adjusted prices include dividends.')

st.divider();st.caption('VOLATILITYLAB · Research & educational use only · No investment recommendations · Data providers may have delays')
