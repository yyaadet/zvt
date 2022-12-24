import click
import pandas as pd
import numpy as np
import logging
import datetime
import math
from constants import FUND_OUTPUT, STOCK_OUTPUT


logging.basicConfig(level=logging.INFO, format="%(asctime)s-%(levelname)s %(filename)s:%(lineno)s:: %(message)s")
logger = logging.getLogger(__name__)
all_df = None
TRADE_SELL = "sell"
TRADE_BUY = "buy"
TRADE_NONE = "no"


class Strategy:
    def __init__(self, cash=1000) -> None:
        self.initial_cash = 1000
        self.initial_position = 0
        self.left_cash = self.initial_cash
        self.left_position = self.initial_position
        self.costs = []
        self.positions = []
        self.prices = []
        self.trade_days = []
        self.actions = []
        self.trade_signals_list = []
        self.adx_signals_list = []
        self.balances = []
        self.cashs = []

    def buy(self, trade_day, price, trade_signals, adx_signals):
        last_position = self.get_position()
        if last_position > 0:
            buy_position = math.ceil(last_position * 0.1)
        else:
            buy_position = math.ceil(self.left_cash * 0.2 / price)

        if buy_position <= 0:
            logger.warning("buy position is less than 0, ignore")
            return False

        cost = price * buy_position
        if cost > self.left_cash:
            logger.warning("Cost {} is larger than left cash {}".format(cost, self.left_cash))
            return False
        self.left_cash -= cost
        self.left_position += buy_position
        self.costs.append(cost)
        self.trade_days.append(trade_day)
        self.positions.append(buy_position)
        self.actions.append(TRADE_BUY)
        self.trade_signals_list.append(trade_signals)
        self.adx_signals_list.append(adx_signals)
        self.prices.append(price)
        balance = self.left_cash + self.get_position() * price
        self.balances.append(balance)
        self.cashs.append(self.left_cash)
        return True

    def sell(self, trade_day, price, trade_signals, adx_signals):
        last_position = self.get_position()
        if last_position <= 0:
            logger.warning("you don't have position")
            return False

        sell_position = max(1, math.ceil(last_position * 0.1))
        cost = sell_position * price
        self.left_cash += cost
        self.costs.append(-cost)
        self.trade_days.append(trade_day)
        self.positions.append(-sell_position)
        self.actions.append(TRADE_SELL)
        self.trade_signals_list.append(trade_signals)
        self.adx_signals_list.append(adx_signals)
        self.prices.append(price)
        self.cashs.append(self.left_cash)
        balance = self.left_cash + self.get_position() * price
        self.balances.append(balance)
        return True

    def get_position(self):
        return sum(self.positions)

    def get_trade_df(self):
        df = pd.DataFrame({
            "cost": self.costs,
            "day": self.trade_days,
            "position": self.positions,
            "action": self.actions,
            "trade_signals": self.trade_signals_list,
            "adx_signals": self.adx_signals_list,
            "price": self.prices,
            "balance": self.balances,
            "cash": self.cashs
        })
        return df

    def summary(self, last_price):
        initial_balance = self.initial_cash + 0
        final_balance = self.left_cash + self.left_position * last_price
        profit = 100 * (final_balance - initial_balance) / initial_balance
        trade_df = self.get_trade_df()
        logger.info("initial balance ${:.2f}, final balance ${:.2f}, final position price ${:.2f}, profit {:.2f}%".format(
            initial_balance, final_balance, last_price, profit
            ))
        print(trade_df)
        return trade_df


class AlphaModel:

    def __init__(self, entity_id: str, df: pd.DataFrame, use_adx:bool) -> None:
        self.entity_id = entity_id
        self.df = df[df["entity_id"] == entity_id]
        self.df = self.df.reset_index()
        self.use_adx = use_adx
        print("{} df size {}".format(self.entity_id, self.df.shape))

    def get_MAs(self):
        self.df["ma10"] = self.df["close"].rolling(10).mean()
        self.df["ma20"] = self.df["close"].rolling(20).mean()
        self.df["ma50"] = self.df["close"].rolling(50).mean()
        self.df["ma100"] = self.df["close"].rolling(100).mean()
        self.df["ma200"] = self.df["close"].rolling(200).mean()
        logger.info("MA df\n{}".format(self.df[["name", "timestamp", "ma10", "ma20", "ma50", "ma100", "ma200"]]))

    def get_ADX(self):
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

    def get_trade_signal(self, when, lookback=5):
        prev_df = self.df[self.df["timestamp"] < when].tail(lookback)
        logger.info("{} tail df\n{}".format(
            when.date(), 
            prev_df[["name", "timestamp", "adx", "ma10", "ma20", "ma50", "ma100", "ma200"]]))

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
        
        logger.info("{} adx signals {}, trade signals {}".format(when, adx_signals, trade_signals))

        return adx_signals, trade_signals

    def back_test(self, start, end, strategy:Strategy):
        df = self.df[self.df["timestamp"] >= start]
        df = df[df['timestamp'] <= end]
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
            adx_signals, trade_signals = self.get_trade_signal(day)
            if self.use_adx and all(adx_signals) is False:
                logger.info("{} adx_signals is False, ignore".format(day))
                continue
            
            is_buy = all(list(map(lambda x: x == TRADE_BUY, trade_signals)))
            is_sell = all(list(map(lambda x: x == TRADE_SELL, trade_signals)))
            logger.info("is buy {}, is sell {}".format(is_buy, is_sell))
            if is_buy:
                strategy.buy(day, close, trade_signals, adx_signals)
            elif is_sell:
                strategy.sell(day, close, trade_signals, adx_signals)
            else:
                logger.info("{} {}".format(day, TRADE_NONE))

        summary_df = strategy.summary(last_close)
        dt_format = "%Y%m%d"
        summary_output = "summary_{}_{}_{}_{}.csv".format(self.entity_id, start.strftime(dt_format), end.strftime(dt_format), self.use_adx)
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
def backtest(path, entity_id, start, end, use_adx):
    global all_df

    use_adx = bool(int(use_adx))
    all_df = pd.read_csv(path, dtype={'entity_id': str}, index_col=False)
    all_df['timestamp'] = pd.to_datetime(all_df['timestamp'], format="%Y-%m-%d")
    alpha = AlphaModel(entity_id, all_df, use_adx)
    alpha.get_MAs()
    alpha.get_ADX()

    start = datetime.datetime.strptime(start, "%Y-%m-%d")
    end = datetime.datetime.strptime(end, "%Y-%m-%d")
    alpha.back_test(start, end, Strategy())


@cli.command()
def summary():
    global all_df

    stock_df = pd.read_csv(STOCK_OUTPUT, dtype={'entity_id': str}, index_col=False)
    fund_df = pd.read_csv(FUND_OUTPUT, dtype={'entity_id': str}, index_col=False)
    all_df = pd.concat([stock_df, fund_df])
    print(all_df.head())
    all_df['timestamp'] = pd.to_datetime(all_df['timestamp'])
    all_df.loc[:, "market_value"] = all_df["close"] * all_df["volume"]

    s_df = all_df.groupby(["entity_id", "name"]).agg(
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
    print(s_df)


if __name__ == "__main__":
    cli()
