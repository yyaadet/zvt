from zvt.domain import Stock, Stock1dHfqKdata
from zvt.ml import MaStockMLMachine


Stock.record_data(provider="joinquant")
codes_df = Stock.query_data(provider="joinquant",  index="code")

subdf = codes_df[codes_df.name.isin(["陕西煤业", "中国神华", "兖矿能源", "永泰能源", "美锦能源", "山西焦煤", "宁德时代", "赣锋锂业", "华友钴业"])]
print(subdf)

sub_codes = subdf['code'].to_list()
# 
#entity_ids = ['stock_sh_601225', 'stock_sh_600188', 'stock_sh_601088']
provider = "joinquant"
Stock1dHfqKdata.record_data(provider=provider, codes=sub_codes, sleeping_time=1)
df = Stock1dHfqKdata.query_data(provider=provider, codes=sub_codes)
df.to_csv("rmd/coal.csv")
