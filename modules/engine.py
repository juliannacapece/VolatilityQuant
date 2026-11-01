"""Reusable quantitative calculations extracted from VolatilityLab research notebooks."""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.interpolate import griddata
from arch import arch_model
from xgboost import XGBRegressor

HORIZON=5
EPS=1e-6

def prices_for(ticker, period='5y'):
    data=yf.download(ticker,period=period,auto_adjust=True,progress=False)
    if data.empty: raise ValueError(f'No market prices for {ticker}')
    if isinstance(data.columns,pd.MultiIndex):data.columns=data.columns.get_level_values(0)
    p=data['Close'].dropna()
    if len(p)<100:raise ValueError('At least 100 trading days are required')
    return p

def build_features(r, horizon=HORIZON):
    x=pd.DataFrame(index=r.index)
    for w in (5,10,20,60):
        x[f'var_{w}d']=r.pow(2).rolling(w).mean()
        x[f'return_{w}d']=r.rolling(w).sum()
    x['abs_return_1d']=r.abs();x['return_1d']=r
    x['negative_return_1d']=(r<0).astype(int)
    future=pd.concat([r.shift(-i).pow(2) for i in range(1,horizon+1)],axis=1)
    x['target']=future.mean(axis=1)
    x.iloc[-horizon:,x.columns.get_loc('target')]=np.nan
    return x

def forecast_from_prices(p, horizon=HORIZON):
    r=np.log(p/p.shift(1)).dropna()*100
    hist=np.sqrt(r.iloc[-20:].pow(2).mean()*252)
    fit=arch_model(r,mean='Constant',vol='GARCH',p=1,q=1,dist='t').fit(disp='off',show_warning=False)
    f=fit.forecast(horizon=horizon)
    garch=np.sqrt(np.mean(f.variance.iloc[-1].to_numpy()+f.mean.iloc[-1].to_numpy()**2)*252)
    features=build_features(r, horizon);train=features.dropna()
    cols=[c for c in features if c!='target']
    if len(train)<250:raise ValueError('Insufficient labeled data for XGBoost')
    model=XGBRegressor(n_estimators=50,max_depth=2,learning_rate=.03,subsample=.8,colsample_bytree=.8,reg_lambda=10,min_child_weight=10,objective='reg:squarederror',random_state=42,n_jobs=2)
    model.fit(train[cols],np.log(train['target']+EPS))
    x=np.maximum(np.exp(model.predict(features[cols].dropna().iloc[[-1]])[0])-EPS,1e-8)
    return {'Historical':float(hist),'GARCH':float(garch),'XGBoost':float(np.sqrt(x*252))}


def option_data(ticker, max_expirations=18):
    import time

    stock = yf.Ticker(ticker)

    # Get recent stock price
    h = stock.history(period="5d")

    if h.empty:
        raise ValueError(
            f"No recent stock prices available for {ticker}"
        )

    spot = float(h["Close"].dropna().iloc[-1])

    # Retrieve expiration dates
    try:
        expirations = stock.options
    except Exception as e:
        raise RuntimeError(
            f"Could not retrieve expiration dates: {e}"
        ) from e

    if not expirations:
        raise ValueError(
            f"No expiration dates returned for {ticker}. "
            "Yahoo Finance may be temporarily unavailable."
        )

    parts = []
    errors = []

    for exp in expirations[:max_expirations]:

        chain = None

        # Retry temporary download failures
        for attempt in range(2):
            try:
                chain = stock.option_chain(exp)
                break

            except Exception as e:
                errors.append(
                    f"{exp} attempt {attempt + 1}: "
                    f"{type(e).__name__}: {e}"
                )

                if attempt == 0:
                    time.sleep(1.5)

        if chain is None:
            continue

        for typ, df in [
            ("Call", chain.calls),
            ("Put", chain.puts)
        ]:
            if not df.empty:
                q = df.copy()
                q["optionType"] = typ
                q["expiration"] = exp
                parts.append(q)

    if not parts:
        details = "; ".join(errors[:3])

        raise RuntimeError(
            f"No options chains could be downloaded "
            f"for {ticker}. "
            f"Errors: {details or 'No contracts returned'}"
        )

    d = pd.concat(parts, ignore_index=True)

    # Convert numerical columns
    for c in [
        "strike",
        "bid",
        "ask",
        "impliedVolatility"
    ]:
        d[c] = pd.to_numeric(
            d[c], errors="coerce"
        )

    # Initial quote filtering
    d = d[
        (d["bid"] > 0)
        & (d["ask"] > d["bid"])
        & d["impliedVolatility"].between(.01, 3)
    ].copy()

    d["relativeSpread"] = (
        (d["ask"] - d["bid"])
        / ((d["ask"] + d["bid"]) / 2)
    )

    d["expiration"] = pd.to_datetime(
        d["expiration"]
    )

    now = (
        pd.Timestamp.now(tz="America/New_York")
        .normalize()
        .tz_localize(None)
    )

    d["daysToExpiration"] = (
        d["expiration"] - now
    ).dt.days

    d["Moneyness"] = d["strike"] / spot

    d["IV_Percent"] = (
        d["impliedVolatility"] * 100
    )

    # Final data-quality filters
    d = d[
        d["relativeSpread"].lt(.30)
        & d["daysToExpiration"].between(7, 365)
        & d["Moneyness"].between(.70, 1.30)
        & d["IV_Percent"].between(5, 150)
    ].copy()

    if d.empty:
        raise ValueError(
            f"Options were downloaded for {ticker}, "
            "but no contracts passed the quality filters."
        )

    return spot, d


def surface_grid(df,xcol='Moneyness'):
    d=df.groupby([xcol,'daysToExpiration'],as_index=False)['IV_Percent'].median()
    if len(d)<10 or d[xcol].nunique()<3 or d.daysToExpiration.nunique()<2:raise ValueError('Too few distinct strikes or expirations for interpolation')
    X,Y=np.meshgrid(np.linspace(d[xcol].min(),d[xcol].max(),55),np.linspace(d.daysToExpiration.min(),d.daysToExpiration.max(),55))
    Z=griddata((d[xcol],d.daysToExpiration),d.IV_Percent,(X,Y),method='linear')
    if np.isnan(Z).all():raise ValueError('Interpolation unavailable')
    return X,Y,Z,d

def black_scholes(S,K,T,r,sigma,kind):
    if S<=0 or K<=0 or sigma<=0:raise ValueError('Positive prices and volatility required')
    if T<=0:
        return (max(S-K,0),float(S>K)) if kind=='call' else (max(K-S,0),-float(S<K))
    d1=(np.log(S/K)+(r+.5*sigma*sigma)*T)/(sigma*np.sqrt(T));d2=d1-sigma*np.sqrt(T)
    if kind=='call':return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2),norm.cdf(d1)
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1),norm.cdf(d1)-1

def volatility_schedule(value, days):
    """Annualized decimal volatility, one value per daily interval."""
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        values = np.full(days, float(values))
    if values.shape != (days,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f'Provide {days} positive finite daily volatility values')
    return values


def simulate_path(volatility, days=5, spot=100, r=.04, seed=42):
    vols = volatility_schedule(volatility, days)
    shocks = np.random.default_rng(seed).standard_normal(days)
    returns = (r - .5 * vols**2) / 252 + vols / np.sqrt(252) * shocks
    return np.r_[spot, spot * np.exp(np.cumsum(returns))]


def hedge(path,K,T,r,hedge_vol,market_vol,kind='call',cost=.001,size=100):
    """Long option with offsetting stock; deterministic market variance schedule.

    Day i's hedge input is available before interval i. Market option marks use
    remaining integrated variance; hedge deltas use the current strategy input
    as a flat remaining-life assumption. Neither input is an automatic forecast.
    """
    path=np.asarray(path,dtype=float);n=len(path)-1
    if n<1 or not np.all(np.isfinite(path)) or np.any(path<=0) or T<=0:
        raise ValueError('Invalid price path or maturity')
    hv=volatility_schedule(hedge_vol,n);mv=volatility_schedule(market_vol,n)
    dt=T/n
    def pricing_vol(day):
        return float(np.sqrt(np.mean(mv[day:]**2))) if day<n else float(mv[-1])
    premium,_=black_scholes(path[0],K,T,r,pricing_vol(0),kind)
    _,delta=black_scholes(path[0],K,T,r,hv[0],kind)
    shares=-delta*size;fees=abs(shares)*path[0]*cost
    cash=-premium*size-shares*path[0]-fees;records=[]
    for day,S in enumerate(path):
        rem=max(T-day*dt,0)
        current_hv=float(hv[min(day,n-1)])
        price,_=black_scholes(S,K,rem,r,pricing_vol(day),kind)
        _,d=black_scholes(S,K,rem,r,current_hv,kind)
        if day:
            cash*=np.exp(r*dt)
            if day<n:
                target=-d*size;trade=target-shares;fee=abs(trade)*S*cost
                cash-=trade*S+fee;fees+=fee;shares=target
        if day==n:
            payoff=max(S-K,0) if kind=='call' else max(K-S,0)
            fee=abs(shares)*S*cost;cash+=payoff*size+shares*S-fee;fees+=fee;shares=0;value=cash
        else:value=price*size+shares*S+cash
        records.append({'Day':day,'Stock':S,'PortfolioValue':value,'TransactionCosts':fees,'Delta':d})
    return pd.DataFrame(records)


def stress_prices():
    p=prices_for('SPY',period='max')
    return p.loc['2005-01-01':]

STRESS={'2008 Financial Crisis':('2008-09-01','2009-03-31'),'2020 COVID Crash':('2020-02-01','2020-04-30'),'2022 Market Selloff':('2022-01-01','2022-12-31')}

def stress_statistics(p):
    ret=np.log(p/p.shift(1)).dropna();vol=ret.rolling(20).std()*np.sqrt(252)*100;dd=(p/p.cummax()-1)*100
    rows=[]
    for name,(start,end) in STRESS.items():
        pp=p.loc[start:end]
        if len(pp)<2:continue
        rows.append({'Crisis':name,'Period return (%)':(pp.iloc[-1]/pp.iloc[0]-1)*100,'Peak 20d vol (%)':vol.loc[start:end].max(),'Worst drawdown (%)':dd.loc[start:end].min()})
    return vol,dd,pd.DataFrame(rows)



from pathlib import Path

def load_aapl_snapshot():
    """Load saved AAPL option quotes for demo use."""

    file_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "AAPL_options_snapshot.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            "AAPL options snapshot was not found."
        )

    d = pd.read_csv(file_path)

    # Use the approximate stock price from collection
    spot = 344.63

    # Read the snapshot timestamp
    snapshot_time = pd.to_datetime(
        d["snapshotTimestampUTC"].iloc[0],
        utc=True
    )

    d["expiration"] = pd.to_datetime(
        d["expiration"]
    )

    for column in [
        "strike", "bid", "ask", "impliedVolatility"
    ]:
        d[column] = pd.to_numeric(
            d[column], errors="coerce"
        )

    # Apply original quote-quality rules
    d = d[
        (d["bid"] > 0) &
        (d["ask"] > d["bid"]) &
        d["impliedVolatility"].between(0.01, 3)
    ].copy()

    d["relativeSpread"] = (
        (d["ask"] - d["bid"]) /
        ((d["ask"] + d["bid"]) / 2)
    )

    # Calculate maturity relative to snapshot date
    snapshot_date = snapshot_time.tz_convert(
        "America/New_York"
    ).date()

    d["daysToExpiration"] = (
        d["expiration"] -
        pd.Timestamp(snapshot_date)
    ).dt.days

    d["Moneyness"] = d["strike"] / spot
    d["IV_Percent"] = d["impliedVolatility"] * 100

    d = d[
        (d["relativeSpread"] < 0.30) &
        d["daysToExpiration"].between(7, 365) &
        d["Moneyness"].between(0.70, 1.30) &
        d["IV_Percent"].between(5, 150)
    ].copy()

    if d.empty:
        raise ValueError(
            "No saved contracts passed the filters."
        )

    return spot, d, snapshot_time



def historical_nvda_prices(now=None):
    """Exclude today's session conservatively, including on early-close days."""
    now = pd.Timestamp.now(tz='America/New_York') if now is None else pd.Timestamp(now).tz_convert('America/New_York')
    # Before the next NY calendar day, today's bar is never considered final.
    end = now.normalize().date()
    start = (now - pd.DateOffset(years=6)).date()
    data = yf.download('NVDA', start=str(start), end=str(end), auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError('No NVDA data returned')
    if isinstance(data.columns, pd.MultiIndex):
        data.columns=data.columns.get_level_values(0)
    prices=data['Close'].dropna().sort_index()
    dates=pd.DatetimeIndex(prices.index)
    if dates.tz is not None:
        dates=dates.tz_convert('America/New_York').tz_localize(None)
    prices.index=dates.normalize()
    prices=prices.loc[prices.index < pd.Timestamp(end)]
    if len(prices)<320:
        raise ValueError('Insufficient completed NVDA history')
    return prices, now


def historical_episode(prices, cost=.001, manual=None, forecaster=forecast_from_prices):
    """Five observed moves; each decision forecasts only its remaining horizon.

    Closing positions approximate execution at observed closes. Rates and time
    use trading-day conventions. No intermediate option market prices implied.
    """
    prices=prices.sort_index().dropna()
    if len(prices)<320 or not prices.index.is_unique or not np.all(np.isfinite(prices)) or (prices<=0).any():
        raise ValueError('Need unique dates and at least 320 positive finite closes')
    path=prices.iloc[-6:]; strike=float(path.iloc[0]); rate=.04; size=100
    forecasts=[]
    for i,date in enumerate(path.index[:-1]):
        history=prices.loc[:date].loc[lambda x:x.index >= date-pd.DateOffset(years=5)]
        pred=forecaster(history,horizon=5-i)
        forecasts.append({'Decision date':date,'Next close':path.index[i+1],'Days remaining':5-i,**pred})
    forecasts=pd.DataFrame(forecasts)
    if manual is not None:
        forecasts['Manual']=volatility_schedule(manual,5)*100
    premium=black_scholes(strike,strike,5/252,rate,forecasts.Historical.iloc[0]/100,'call')[0]
    rows=[]; ledger=[]
    for name in ['Historical','GARCH','XGBoost']+(['Manual'] if manual is not None else []):
        cash=-premium*size; shares=0.; fees=0.; turnover=0.; financed_fees=0.
        for i,(date,spot) in enumerate(path.items()):
            if i:
                cash*=np.exp(rate/252)
                financed_fees*=np.exp(rate/252)
            before=shares
            if i<5:
                vol=float(forecasts[name].iloc[i])/100
                _,delta=black_scholes(spot,strike,(5-i)/252,rate,vol,'call')
                shares=-delta*size
            else:
                shares=0.;delta=np.nan
                cash+=max(spot-strike,0)*size
            trade=shares-before; notional=abs(trade)*spot;fee=notional*cost
            cash-=trade*spot+fee;fees+=fee;turnover+=notional;financed_fees+=fee
            ledger.append({'Strategy':name,'Date':date,'Stock close ($)':spot,'Shares after trade':shares,'Shares traded':trade,'Cash balance ($)':cash,'Cumulative costs ($)':fees})
        rows.append({'Strategy':name,'Terminal P&L ($)':cash,'Trading costs ($)':fees,'P&L before costs ($)':cash+financed_fees,'Stock turnover ($)':turnover})
    return {'path':path,'forecasts':forecasts,'summary':pd.DataFrame(rows),'ledger':pd.DataFrame(ledger),'premium':premium,'strike':strike}
