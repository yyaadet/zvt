'''Fetch stock and fund data. Output csv columns:
id,entity_id,timestamp,provider,code,name,level,open,close,high,low,volume,turnover,change_pct,turnover_rate
'''
from zvt.domain import Stock, Stock1dHfqKdata
from zvt.ml import MaStockMLMachine
import os
import akshare as ak
import pandas as pd
from constants import basedir, STOCK_OUTPUT, FUND_OUTPUT


STOCK_NAMES = [
    "陕西煤业", "中国神华", "兖矿能源", "永泰能源", "美锦能源", "山西焦煤", "宁德时代", "赣锋锂业", "华友钴业"
]

FUND_NAMES = {
    "008280": "国泰中证煤炭ETF联接C",
    "014111": "嘉实中证稀有金属主题ETF发起联接C",
    "161032": "富国中证煤炭指数LOF",
}



def download_stock():
    Stock.record_data(provider="joinquant")
    codes_df = Stock.query_data(provider="joinquant",  index="code")

    subdf = codes_df[codes_df.name.isin(STOCK_NAMES)]
    print(subdf)

    sub_codes = subdf['code'].to_list()
    provider = "joinquant"
    Stock1dHfqKdata.record_data(provider=provider, codes=sub_codes, sleeping_time=1)
    df = Stock1dHfqKdata.query_data(provider=provider, codes=sub_codes)
    df.to_csv(STOCK_OUTPUT)
    return df


def download_fund():
    fund_df_map = {}
    for fund_id, fund_name in FUND_NAMES.items():
        fund_df = ak.fund_open_fund_info_em(fund=fund_id, indicator="单位净值走势")
        fund_df = fund_df.rename(columns={"净值日期": "timestamp", "单位净值": "close"})
        fund_df = fund_df[['timestamp', "close"]]
        fund_df['entity_id'] = fund_id
        fund_df['name'] = fund_name
        fund_df['open'] = None
        fund_df['high'] = None
        fund_df['low'] = None
        fund_df['volume'] = None
        fund_df_map[fund_id] = fund_df
    
    all_df = pd.concat(fund_df_map.values())
    all_df = all_df.reset_index()
    all_df = all_df.drop(columns=['index'])
    print(all_df)
    all_df.to_csv(FUND_OUTPUT)
    return all_df


if __name__ == "__main__":
    download_fund()
    download_stock()

