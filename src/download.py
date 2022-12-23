from zvt.domain import Stock, Stock1dHfqKdata
from zvt.ml import MaStockMLMachine
import os

basedir = os.path.dirname(__file__)
STOCK_NAMES = [
    "陕西煤业", "中国神华", "兖矿能源", "永泰能源", "美锦能源", "山西焦煤", "宁德时代", "赣锋锂业", "华友钴业"
]

def download():
    Stock.record_data(provider="joinquant")
    codes_df = Stock.query_data(provider="joinquant",  index="code")

    subdf = codes_df[codes_df.name.isin(STOCK_NAMES)]
    print(subdf)

    sub_codes = subdf['code'].to_list()
    provider = "joinquant"
    Stock1dHfqKdata.record_data(provider=provider, codes=sub_codes, sleeping_time=1)
    df = Stock1dHfqKdata.query_data(provider=provider, codes=sub_codes)
    path = os.path.join(basedir, "./rmd/coal.csv")
    df.to_csv(path)
    return path



if __name__ == "__main__":
    download()
