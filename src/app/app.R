library(dplyr)
library(ggplot2)
library(pinyin)
library(zoo)
library(plotly)
library(TTR)
library(quantmod)
library(dash)
library(dashCoreComponents)
library(dashTable)



setwd("/Users/pengxiaotao/Documents/gitroom/zvt/src/app")

# load all data
df = read.csv("../rmd/coal.csv")
df$timestamp = as.Date(df$timestamp)
df$name = py(df$name, dic = pydic(only_first_letter = T, method = "toneless", dic = "pinyin"), sep = "")
df$market_value = df$close * df$volume


get_summary = function() {
  df %>%
    group_by(entity_id, name) %>%
    summarise(count=n(), 
              list_date = min(timestamp),
              open = mean(open), 
              close = mean(close),
              last_date = max(timestamp),
              market_value = mean(market_value)) %>%
    arrange(desc(market_value))
}


entities = list()
s = get_summary()
for(i in 1:nrow(s)) {
  if (i == 1) {
    entities[[i]] = list(label = paste(s$name[i], "-", s$market_value[i]), value = s$name[i], selected = T)
  } else {
    entities[[i]] = list(label = paste(s$name[i], "-", s$market_value[i]), value = s$name[i])
  }
  
}



get_signals = function(data) {
  data$ma10 = round(rollmean(data$close, k=10, fill = NA, align = "right"), 4)
  data$ma20 = round(rollmean(data$close, k=20, fill = NA, align = "right"), 4)
  data$ma50 = round(rollmean(data$close, k=50, fill = NA, align = "right"), 4)
  data$ma100 = round(rollmean(data$close, k=100, fill = NA, align = "right"), 4)
  data$ma200 = round(rollmean(data$close, k=200, fill = NA, align = "right"), 4)
  
  adx = ADX(as.matrix(data[, c("high", "low", "close")]))
  data$adx = round(adx[, "ADX"], 4)
  
  last = tail(data, 100)
  
  for (i in 1:nrow(last)) {
    if (last$ma10[i] > last$ma20[i] & 
        last$ma20[i] > last$ma50[i] &
        last$ma50[i] > last$ma100[i] &
        last$ma100[i] > last$ma200[i]) {
      last$ma_signal[i] = "buy"
    } else if (last$ma10[i] < last$ma20[i] &
               last$ma20[i] < last$ma50[i] &
               last$ma50[i] < last$ma100[i] &
               last$ma100[i] < last$ma200[i]) {
      last$ma_signal[i] = "sell"
    } else {
      last$ma_signal[i] = "no"
    }
    
    last$adx_signal[i] = ifelse(last$adx[i] > 20, T, F)
    last$final_signal[i] = ifelse(last$adx_signal[i] == T & last$ma_signal[i] != "no", T, F)
    
  }
  
  last
}


app = dash_app()


getMAFigureData = function(name, start) {
  subdf = df[which(df$name == name), ]
  
  subdf = subdf %>%
    mutate(ma10 = rollmean(close, k=10, fill = NA, align = "right")) %>%
    mutate(ma20 = rollmean(close, k=20, fill = NA, align = "right")) %>%
    mutate(ma50 = rollmean(close, k=50, fill = NA, align = "right")) %>%
    mutate(ma100 = rollmean(close, k=100, fill = NA, align = "right")) %>%
    mutate(ma200 = rollmean(close, k = 200, fill = NA, align = "right"))
  
  subdf = subdf[which(subdf$timestamp >= start), ]
  
  list(
    data = list (
      list(
        x = subdf$timestamp, 
        y = subdf$close,
        name = "close",
        mode = "lines"
      ),
      list(
        x = subdf$timestamp,
        y = subdf$ma10,
        name = "ma10", 
        mode = "lines"
      ),
      list(
        x = subdf$timestamp,
        y = subdf$ma20,
        name = "ma20", 
        mode = "lines"
      ),
      list(
        x = subdf$timestamp,
        y = subdf$ma50,
        name = "ma50", 
        mode = "lines"
      ),
      list(
        x = subdf$timestamp,
        y = subdf$ma100,
        name = "ma100", 
        mode = "lines"
      ),
      list(
        x = subdf$timestamp,
        y = subdf$ma200,
        name = "ma200", 
        mode = "lines"
      )
    )
  )
}


app %>% set_layout(
  h1("Trading"),
  dccDropdown("entity", options = entities, value = "ndsd"),
  dccDatePickerSingle(
    "start",
    date = as.Date("2020-01-01")
  ),
  dccGraph(
    "mas",
    figure = list()
  ),
  h2("Signals"),
  dashDataTable(
    "signals",
    columns = list(
      list(id = "timestamp", name = "timestamp"),
      list(id = "adx", name = "adx"),
      list(id = "ma10", name = "ma10"),
      list(id = "ma20", name = "ma20"),
      list(id = "ma50", name = "ma50"),
      list(id = "ma100", name = "ma100"),
      list(id = "ma200", name = "ma200"),
      list(id = "ma_signal", name = "ma_signal"),
      list(id = "adx_signal", name = "adx_signal"),
      list(id = "final_signal", name = "final_signal")
    ),
    data = NULL,
    page_size = 100
  )
)

app %>%
  add_callback(
    output(id = "mas", property = "figure"),
    list(
      input(id = "entity", property = "value"),
      input(id = "start", property = "date")
    ),
    function(entity, start) {
      name = ifelse(is.null(entity), "ndsd", entity)
      print(paste(name, start, sep = "-"))
      
      getMAFigureData(name, start)
      
    }
  )


app %>%
  add_callback(
    output(id = "signals", property = "data"),
    list(
      input(id = "entity", property = "value"),
      input(id = "start", property = "date")
    ),
    function(entity, start) {
      name = ifelse(is.null(entity), "ndsd", entity)
      print(paste(name, start, sep = "-"))
      
      subdf = df[which(df$name == name), ]
      
      subdf = get_signals(subdf) %>%
        arrange(desc(timestamp))
      
      df_to_list(subdf)
    }
  )



run_app(app)

