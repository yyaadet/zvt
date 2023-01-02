import click
import pandas as pd
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import datetime
import math
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.tsa.stattools import adfuller
from statsmodels.api import OLS
from constants import FUND_OUTPUT, STOCK_OUTPUT
from dash import Dash, html, dcc, Output, Input
from dash import dash_table
import plotly.express as px
import scipy.stats as sps


all_df = None
TRADE_SELL = "sell"
TRADE_BUY = "buy"
TRADE_NONE = "no"

app = Dash(__name__)
app.logger.handlers.clear()
handler = RotatingFileHandler(filename="trading.log", mode="a", maxBytes=1024*1024*128, backupCount=2)
logging.basicConfig(level=logging.INFO, handlers=[handler], format="[%(asctime)s %(funcName)s:%(lineno)d %(levelname)s]:: %(message)s")
logger = logging.getLogger(__name__)



class AlphaModel:

    def __init__(self, entity_id: str, df: pd.DataFrame, use_adx:bool) -> None:
        self.entity_id = entity_id
        self.df = df[df["entity_id"] == entity_id]
        self.df = self.df.reset_index()
        self.use_adx = use_adx
        self.expected_positive_prop = 0.5
        self.weekday_positive_test_df = None
        logger.info("{} df size {}".format(self.entity_id, self.df.shape))

    def initial_MA(self):
        self.df["ma10"] = self.df["close"].rolling(10).mean()
        self.df["ma20"] = self.df["close"].rolling(20).mean()
        self.df["ma50"] = self.df["close"].rolling(50).mean()
        self.df["ma100"] = self.df["close"].rolling(100).mean()
        self.df["ma200"] = self.df["close"].rolling(200).mean()
        logger.info("MA df\n{}".format(self.df[["name", "timestamp", "ma10", "ma20", "ma50", "ma100", "ma200"]]))

    def initial_ADX(self):
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        lookback = 14
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
        atr = tr.rolling(lookback).mean()
        
        plus_di = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
        adx_smooth = adx.ewm(alpha = 1/lookback).mean()

        self.df['plus_di'] = plus_di
        self.df['minus_di'] = minus_di
        self.df['adx'] = adx_smooth
        logger.info("ADX df \n{}".format(self.df[["name", "timestamp", "adx", "plus_di", "minus_di"]].tail()))
        return plus_di, minus_di, adx_smooth

    def initial_return(self):
        self.df['return'] = self.df['close'].pct_change()
        logger.info("Returns df \n{}".format(self.df[['entity_id', 'name', 'timestamp', 'return']]))

    def initial_weekday_positive_hypothesis_test(self):
        ''' Must have column return
        '''
        expected_prop = self.expected_positive_prop
        self.df['weekday'] = self.df['timestamp'].apply(lambda x: x.weekday())
        self.df['is_return_positive'] = self.df['return'].apply(lambda x: x > 0)
        weekday_df = self.df.groupby('weekday').agg(
            n_total = pd.NamedAgg(column="is_return_positive", aggfunc='count'),
            n_pos = pd.NamedAgg(column="is_return_positive", aggfunc="sum")
        )
        weekday_df['prop'] = weekday_df['n_pos'] / weekday_df['n_total']
        weekday_df['expected_prop'] = expected_prop
        weekday_df['test_statistics'] = weekday_df.apply(lambda x: proportions_ztest(x['n_pos'], x['n_total'], x['expected_prop'], "larger")[0], axis=1)
        weekday_df['pvalue'] = weekday_df.apply(lambda x: proportions_ztest(x['n_pos'], x['n_total'], x['expected_prop'], "larger")[1], axis=1)
        logger.info("H0: prop = {}, HA: prop > {}\n{}".format(expected_prop, expected_prop, weekday_df))
        self.weekday_positive_test_df = weekday_df
        weekday_pvalue_map = {}
        for idx, row in weekday_df.iterrows():
            weekday_pvalue_map[idx] = row['pvalue']
        
        for idx, row in self.df.iterrows():
            self.df.loc[idx, 'pvalue'] = weekday_pvalue_map.get(row['weekday'])
        logger.info("After weekday positive hypothesis test\n{}".format(self.df[['entity_id', 'name', 'timestamp', 'weekday', 'pvalue']]))
        return weekday_df

    def get_adf_test(self, col='close'):
        res = adfuller(self.df[col])
        data = pd.DataFrame({
            "test_statistics": [res[0]],
            'pvalue': [res[1]],
            'used_lag': [res[2]],
            'n_observation': [res[3]],
        })
        return data

    def get_hurst_exponent(self, col='close', max_lags=[10, 20, 50, 100, 200]):
        """The values of the Hurst exponent range between 0 and 1. 
        Based on the value of H, we can classify any time series into one of the three categories:

        1. H < 0.5 — a mean-reverting (anti-persistent) series. The closer the value is to 0, the stronger the mean-reversion process is. 
           In practice, it means that a high value is followed by a low value and vice-versa.
        2. H = 0.5 — a geometric random walk.
        3. H > 0.5 — a trending (persistent) series. The closer the value is to 1, the stronger the trend. 
           In practice, it means that a high value is followed by a higher one.

        Args:
            col (str, optional): used for computation. Defaults to 'close'.
            max_lag (int, optional): max lag. Defaults to 20.
        """
        hes = []
        variances = []
        for max_lag in max_lags:
            ts = np.asarray(self.df[col].values)
            lagvec = []
            tau = []
            lags = range(2, max_lag)
            for lag in lags:
                pdiff = np.subtract(ts[lag:], ts[:-lag])
                lagvec.append(pdiff)
                tau.append(np.sqrt(np.std(pdiff)))

            '''
            tau = [
                np.std(np.subtract(ts[lag: ], ts[:-lag])) for lag in lags
            ]
            logger.info("tau {}".format(tau))
            '''
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            logger.info("max lag {} hurst exponent {}".format(max_lag, reg))
            hes.append(reg[0] * 2.0)
            variances.append(self._get_variance_ratio_test(ts, max_lag))
        return pd.DataFrame({"max_lags": max_lags, 'hurst_exponent': hes, "variance_ratio_test": variances})

    def _get_variance_ratio_test(self, ts, lag=2):
        """Because of finite sample size, we need to know the statistical significance and MacKinlay of an estimated value of H to be sure whether we can reject the null hypothesis that H is really 0.5.
        This hypothesis test is provided by the Variance Ratio test (Lo, 2001).

        Args:
            ts (_type_): _description_
            lag (int, optional): _description_. Defaults to 2.

        Returns:
            flaot: pvalue 
        """
        ts = np.asarray(ts)
        n = len(ts)
        mu = sum(ts[1:n] - ts[:n-1])/ n
        m = (n - lag + 1) * (1 - lag/n)
        b = sum(np.square(ts[1:n] - ts[:n-1] - mu)) / (n-1)
        t = sum(np.square(ts[lag:n] - ts[:n-lag] - lag*mu)) / m
        return t / (lag * b)

    def get_variance_ratio(self):
        returns = self.df['close'].pct_change()[1:]
        n = len(returns)
        lags = []
        var0s = []
        variances = []
        variance_ratios = []
        z_stats = []
        p_values = []
        var0 = returns.var()
        for t in [10, 20, 50, 100, 200]:
            variance = returns.rolling(t).sum().var()
            variance_ratio = variance / (t * var0)
            se = (2 * (2*t - 1)*(t-1)/(3*t*n))**(1/2)
            z_stat = variance_ratio / se
            p_value = 2 * (1 - sps.norm.cdf(abs(z_stat)))
            lags.append(t)
            var0s.append(round(var0, 6))
            variances.append(round(variance, 6))
            variance_ratios.append(round(variance_ratio, 6))
            z_stats.append(round(z_stat, 6))
            p_values.append(round(p_value, 6))
        
        return pd.DataFrame({'lag': lags, 'var0': var0, 'variance': variances, 'variance_ratio': variance_ratios, 'z_stat': z_stats, 'p_value': p_values}) 
    
    def get_half_life(self, col='close'):
        ts = np.asarray(self.df[col].values)
        delta_ts = np.diff(ts)
        lag_ts = np.vstack([ts[1:], np.ones(len(ts[1:]))]).T
        logger.info("ts {}, delta ts {}, lag ts {}".format(ts, delta_ts, lag_ts))
        beta = np.linalg.lstsq(lag_ts, delta_ts, rcond=None)
        logger.info("beta {}".format(beta))
        half_life = (np.log(2) / beta[0])[0]
        return pd.DataFrame({'half_life': [half_life]})

    def initial_half_life(self, col='close'):
        half_life = self.get_half_life(col).at[0, 'half_life']
        half_life = int(math.ceil(half_life))
        self.df['ma_half'] = self.df[col].rolling(half_life).mean()
        logger.info("After initial half life\n{}".format(self.df[['entity_id', 'name', 'timestamp', 'ma_half']]))


class Strategy:
    def __init__(self, cash=1000, risk_ratio=0.1) -> None:
        self.risk_ratio = risk_ratio
        self.initial_cash = 1000
        self.initial_position = 0
        self.left_cash = self.initial_cash
        self.left_position = self.initial_position
        self.costs = []
        self.positions = []
        self.prices = []
        self.trade_days = []
        self.actions = []
        self.signals_list = []
        self.balances = []
        self.cashs = []
        self.hold_positions = []

    def justify(self, trade_day, alpha_model: AlphaModel, use_adx: bool, lookback: int = 5):
        """Justify trade action.

        Args:
            trade_day (DateTime): trade datetime
            lookback (int): lookback days 
            alpha_model (AlphaModel): AlphaModel Object
        
        Returns:
            str, signals: TRADE_SELL or TRADE_BUY or TRADE_NO, list of signals
        """
        adx_signals, trade_signals, positive_signals = self.get_trade_signal(alpha_model, trade_day, lookback)
        signals = {
            'adx': adx_signals,
            'trade': trade_signals,
            'positive': positive_signals,
        }
        if use_adx and all(adx_signals) is False:
            logger.info("{} adx_signals is False, ignore".format(trade_day))
            return TRADE_NONE, signals

        if len(trade_signals) == 0:
            logger.warning("{} trade signals is empty, ignore".format(trade_day))
            return TRADE_NONE, signals
        
        is_buy = all(list(map(lambda x: x == TRADE_BUY, trade_signals)))
        is_sell = all(list(map(lambda x: x == TRADE_SELL, trade_signals)))
        logger.info("Day {}, is buy {}, is sell {}".format(trade_day.date(), is_buy, is_sell))
        if is_buy:
            return TRADE_BUY, signals
        elif is_sell:
            return TRADE_SELL, signals
        else:
            logger.info("{} {}".format(trade_day, TRADE_NONE))
            return TRADE_NONE, signals
    
    def get_trade_signal(self, alpha_model: AlphaModel, when: datetime.datetime, lookback: int =5):
        """Get signals by look back.

        Args:
            when (datetime): _description_
            lookback (int, optional): Look back day. Defaults to 5.

        Returns:
            (): (adx signals, trade signals, positive signals)
        """
        when_df = alpha_model.df[alpha_model.df['timestamp'] <= when].tail(1)
        positive_signals = []
        for idx, row in when_df.iterrows():
            if row['pvalue'] < 0.05:
                positive_signals.append(True)
            else:
                positive_signals.append(False)

        prev_df = alpha_model.df[alpha_model.df["timestamp"] < when].tail(lookback)
        logger.info("{} tail df\n{}".format(
            when.date(), 
            prev_df[["name", "timestamp", "adx", "ma10", "ma20", "ma50", "ma100", "ma200"]]))
        if len(prev_df) < lookback:
            return [], [], positive_signals

        adx_signals = []
        trade_signals = []
        for idx, row in prev_df.iterrows():
            if row['ma10'] > row['ma20'] > row['ma50'] > row['ma100'] > row['ma200']:
                trade_signal = TRADE_BUY
            elif row['ma10'] < row['ma20'] < row['ma50'] < row['ma100'] < row['ma200']:
                trade_signal = TRADE_SELL
            else:
                trade_signal = TRADE_NONE
            
            adx_signal = True if row['adx'] > 20 else False
            adx_signals.append(adx_signal)
            trade_signals.append(trade_signal)
        
        logger.info("Strategy {}, weekday {}, trade {}, adx {}, positive {}".format(
            when.date(), when.weekday(), trade_signals, adx_signals, positive_signals
            ))
        return adx_signals, trade_signals, positive_signals 

    def buy(self, trade_day, price, signals):
        last_position = self.get_hold_position()
        if last_position > 0:
            buy_position = math.ceil(last_position * self.risk_ratio)
        else:
            buy_position = math.ceil(self.left_cash * self.risk_ratio / price)

        if buy_position <= 0:
            logger.warning("buy position is less than 0, ignore buy")
            return False

        cost = price * buy_position
        if cost > self.left_cash:
            logger.warning("Cost {} is larger than left cash {}, ignore buy".format(cost, self.left_cash))
            return False
        self.left_cash -= cost
        self.left_position += buy_position
        self.costs.append(cost)
        self.trade_days.append(trade_day)
        self.positions.append(buy_position)
        self.actions.append(TRADE_BUY)
        self.signals_list.append(signals)
        self.prices.append(price)
        balance = self.left_cash + self.get_hold_position() * price
        self.balances.append(balance)
        self.cashs.append(self.left_cash)
        self.hold_positions.append(self.get_hold_position())
        return True

    def sell(self, trade_day, price, signals):
        last_position = self.get_hold_position()
        if last_position <= 0:
            logger.warning("you don't have position, ignore sell")
            return False

        sell_position = max(1, math.ceil(last_position * self.risk_ratio))
        cost = sell_position * price
        self.left_cash += cost
        self.costs.append(-cost)
        self.trade_days.append(trade_day)
        self.positions.append(-sell_position)
        self.actions.append(TRADE_SELL)
        self.signals_list.append(signals)
        self.prices.append(price)
        self.cashs.append(self.left_cash)
        balance = self.left_cash + self.get_hold_position() * price
        self.balances.append(balance)
        self.hold_positions.append(self.get_hold_position())
        return True

    def get_hold_position(self):
        return sum(self.positions)

    def get_trade_df(self):
        df = pd.DataFrame({
            "cost": self.costs,
            "day": self.trade_days,
            "position": self.positions,
            "action": self.actions,
            "signals": self.signals_list,
            "price": self.prices,
            "balance": self.balances,
            "cash": self.cashs,
            'hold_position': self.hold_positions,
        })
        return df

    def summary(self, last_price):
        initial_balance = self.initial_cash + 0
        final_balance = self.left_cash + self.left_position * last_price
        profit = final_balance - initial_balance
        ret = 100 * profit / initial_balance
        trade_df = self.get_trade_df()
        print("{} - Initial balance ${:.2f}, final balance ${:.2f}, final position price ${:.2f}, profit {:.2f}, return {:.2f}%".format(
            self.__class__.__name__, initial_balance, final_balance, last_price, profit, ret
        ))
        if trade_df.empty is False:
            print(trade_df)
        return trade_df


class MeanRevertStrategy(Strategy):

    def __init__(self, cash=1000, risk_ratio=0.1, half_life=30) -> None:
        super().__init__(cash, risk_ratio)

    def justify(self, trade_day, alpha_model: AlphaModel, use_adx: bool, lookback: int = 5):
        """ AlphaModel has initial half life.
        """
        df = alpha_model.df
        last_record = df[df['timestamp'] <= trade_day].tail(1).iloc[0, :]
        signals = [{'close': last_record['close'], 'ma_half': last_record['ma_half']}]
        if last_record['close'] < last_record['ma_half']:
            action = TRADE_BUY 
        elif last_record['close'] > last_record['ma_half']:
            action = TRADE_SELL 
        else:
            action = TRADE_NONE

        logger.info("MeanRevertStrategy {}, action {}, signals {}".format(trade_day.date(), action, signals))
        return action, signals

class BacktestEngine:

    def __init__(self, entity_id, start, end, use_adx, strategy: Strategy, alpha_model: AlphaModel) -> None:
        self.entity_id = entity_id
        self.start = start
        self.end = end
        self.use_adx = use_adx
        self.strategy = strategy
        self.alpha_model = alpha_model
        self.is_output = False

    def backtest(self):
        df = self.alpha_model.df
        df = df[df["timestamp"] >= self.start]
        df = df[df['timestamp'] <= self.end]
        df = df.reset_index()
        last_close = None
        logger.info("back test df shape {}".format(df.shape))
        if df.empty:
            logger.warning("back test df is empty, return")
            return

        for idx, row in df.iterrows():
            day = row['timestamp']
            close = row['close']
            last_close = close

            trade_signal, signals = self.strategy.justify(day, self.alpha_model, self.use_adx)
            if trade_signal == TRADE_BUY:
                self.strategy.buy(day, close, signals)
            elif trade_signal == TRADE_SELL:
                self.strategy.sell(day, close, signals)
            else:
                logger.info("{} {}".format(day, trade_signal))
        
        summary_df = self.strategy.summary(last_close)
        if self.is_output:
            dt_format = "%Y%m%d"
            summary_output = "summary_{}_{}_{}_{}.csv".format(self.entity_id, self.start.strftime(dt_format), self.end.strftime(dt_format), self.use_adx)
            summary_df.to_csv(summary_output)


@click.group()
@click.pass_context
def cli(ctx):
    ctx.ensure_object(dict)


@cli.command()
@click.option("--path", default="./rmd/coal.csv", help="The daily stock data")
@click.option("--entity-id", default="stock_sz_300750", help="The entity id")
@click.option("--start", default="2020-01-01", help="format: yyyy-mm-dd")
@click.option("--end", default="2022-12-12", help="format: yyyy-mm-dd")
@click.option("--use-adx", default=1, help="You can input 0 or 1")
@click.option("--risk-ratio", default=0.1, help="Risk ratio from 0 to 1")
def backtest(path, entity_id, start, end, use_adx, risk_ratio):
    global all_df

    use_adx = bool(int(use_adx))
    all_df = pd.read_csv(path, dtype={'entity_id': str}, index_col=False)
    all_df['timestamp'] = pd.to_datetime(all_df['timestamp'], format="%Y-%m-%d")

    alpha = AlphaModel(entity_id, all_df, use_adx)
    alpha.initial_MA()
    alpha.initial_ADX()
    alpha.initial_return()
    alpha.initial_weekday_positive_hypothesis_test()
    alpha.initial_half_life()

    start = datetime.datetime.strptime(start, "%Y-%m-%d")
    end = datetime.datetime.strptime(end, "%Y-%m-%d")

    engine = BacktestEngine(entity_id, start, end, use_adx, Strategy(risk_ratio=risk_ratio), alpha)
    engine.backtest()

    mean_revert_engine = BacktestEngine(entity_id, start, end, use_adx, MeanRevertStrategy(risk_ratio=risk_ratio), alpha)
    mean_revert_engine.backtest()


def get_summary(df):
    s_df = df.groupby(["entity_id", "name"]).agg(
        list_date = pd.NamedAgg(column="timestamp", aggfunc=min),
        open = pd.NamedAgg(column="open", aggfunc=np.mean),
        open_min = pd.NamedAgg(column="open", aggfunc=min),
        open_max = pd.NamedAgg(column="open", aggfunc=max),
        close = pd.NamedAgg(column="close", aggfunc=np.mean),
        close_min = pd.NamedAgg(column="close", aggfunc=min),
        close_max = pd.NamedAgg(column="close", aggfunc=max),
        last_date = pd.NamedAgg(column="timestamp", aggfunc=max),
        market_value = pd.NamedAgg(column="market_value", aggfunc=np.mean),
    ).sort_values("market_value", ascending=False)
    return s_df



@cli.command()
def summary():
    global all_df

    stock_df = pd.read_csv(STOCK_OUTPUT, dtype={'entity_id': str}, index_col=False)
    fund_df = pd.read_csv(FUND_OUTPUT, dtype={'entity_id': str}, index_col=False)
    all_df = pd.concat([stock_df, fund_df])
    logger.info(all_df.head())
    all_df['timestamp'] = pd.to_datetime(all_df['timestamp'])
    all_df.loc[:, "market_value"] = all_df["close"] * all_df["volume"]

    s_df = get_summary(all_df)
    for idx, row in s_df.iterrows():
        entity_id = idx[0]
        alpha = AlphaModel(entity_id, all_df, 1)
        alpha.initial_return()
        alpha.initial_weekday_positive_hypothesis_test()
    
    print(s_df)



@cli.command()
def runserver():
    global all_df
    
    stock_df = pd.read_csv(STOCK_OUTPUT, dtype={'entity_id': str}, index_col=False)
    fund_df = pd.read_csv(FUND_OUTPUT, dtype={'entity_id': str}, index_col=False)
    all_df = pd.concat([stock_df, fund_df])
    all_df['timestamp'] = pd.to_datetime(all_df['timestamp'])
    all_df.loc[:, "market_value"] = all_df["close"] * all_df["volume"]

    summary_df = get_summary(all_df)
    names = list(summary_df.index.unique())
    names = list(map(lambda x: {'label': f"{x[0]}-{x[1]}", 'value': x[0]}, names))

    app.layout = html.Div(
        children=[
            html.H1("Quant Trading"),
            dcc.Dropdown(options=names, id="entity_id", value=names[0]['value']),
            dcc.Graph(id="MA"),
            dcc.Graph(id="return"),
            dcc.Graph(id="log_close"),
            dcc.Graph(id="ma_half"),
            html.H2("ADF Test"),
            dash_table.DataTable(id="adf"),
            html.H2("Hurst Exponent"),
            dash_table.DataTable(id="hurst_exponent"),
            html.H2("Half Life"),
            dash_table.DataTable(id="half_life"),
            html.H2("Variance Ratio Test"),
            dash_table.DataTable(id="variance_ratio"),
            html.H2("Weekday Positive Proportion Test"),
            dash_table.DataTable(id="weekday_positive_test"),
        ]
    )

    app.run_server(debug=True)



@app.callback(
    Output("MA", "figure"),
    Output("return", "figure"),
    Output("log_close", "figure"),
    Output("ma_half", "figure"),
    Input("entity_id", 'value')
)
def update_figures(entity_id):
    df = all_df[all_df['entity_id'] == entity_id]
    alpha = AlphaModel(entity_id, df, True)
    alpha.initial_MA()
    alpha.initial_return()
    alpha.initial_half_life()

    fig = px.line(title="Move Average")
    fig.add_scatter(x=alpha.df['timestamp'], y=alpha.df['close'], name='close')
    fig.add_scatter(x=alpha.df["timestamp"], y=alpha.df["ma10"], name="ma10")
    fig.add_scatter(x=alpha.df["timestamp"], y=alpha.df["ma20"], name="ma20")
    fig.add_scatter(x=alpha.df["timestamp"], y=alpha.df["ma50"], name="ma50")
    fig.add_scatter(x=alpha.df["timestamp"], y=alpha.df["ma100"], name="ma100")
    fig.add_scatter(x=alpha.df["timestamp"], y=alpha.df["ma200"], name="ma200")

    colors = np.where(alpha.df['return'] < 0, 'Down', 'Up')
    return_fig = px.bar(alpha.df, x="timestamp", y="return", title="Day Return", color=colors)

    alpha.df['log_close'] = np.log(alpha.df['close'])
    log_close_fig = px.line(alpha.df, title="Log Close", x='timestamp', y='log_close')

    ma_half_fig = px.line(title="Move Average Of Half Life")
    ma_half_fig.add_scatter(x=alpha.df['timestamp'], y=alpha.df['ma_half'], name='ma_half')
    ma_half_fig.add_scatter(x=alpha.df['timestamp'], y=alpha.df['close'], name='close')
    return fig, return_fig, log_close_fig, ma_half_fig



@app.callback(
    Output("adf", "data"),
    Output("hurst_exponent", "data"),
    Output("half_life", "data"),
    Output("variance_ratio", "data"),
    Output("weekday_positive_test", "data"),
    Input("entity_id", 'value')
)
def update_tables(entity_id):
    df = all_df[all_df['entity_id'] == entity_id]
    alpha = AlphaModel(entity_id, df, True)
    alpha.initial_return()

    adf = alpha.get_adf_test()
    hurst_exponent_df = alpha.get_hurst_exponent()
    half_life_df = alpha.get_half_life()
    weekday_df = alpha.initial_weekday_positive_hypothesis_test()
    weekday_df['weekday'] = weekday_df.index
    variance_ratio = alpha.get_variance_ratio()

    return adf.to_dict('records'), \
        hurst_exponent_df.to_dict('records'), \
        half_life_df.to_dict('records'), \
        variance_ratio.to_dict('records'), \
        weekday_df.to_dict("records")



if __name__ == "__main__":
    cli()
