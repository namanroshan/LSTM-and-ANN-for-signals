#!/usr/bin/env python
# coding: utf-8

#This strategy has been coded for Interactive Brokers

import pandas as pd
import ib_insync
from ib_insync import *
import time
from typing import NamedTuple
from collections import OrderedDict
import datetime
util.startLoop()
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=199)



import pytz
IST = pytz.timezone('Asia/Kolkata')
eastern = pytz.timezone('US/Eastern')
PST = pytz.timezone('US/Pacific')



def time_check(mode="REG"):
    a = list(map(int, datetime.datetime.now(PST).strftime("%H %M").split()))
    current_time = 60*a[0] + a[1]
    if mode == "REG":
        st = 60*6 + 35
        et = 60*8 + 45
    
    if current_time<=et and current_time>=st:
        return 1
    elif current_time>et and current_time<et+10*60:
        return -1
    else:
        return 0


def scanner():
    sub = ScannerSubscription(
        instrument='STK',
        locationCode='STK.US.MAJOR',
        scanCode='TOP_PERC_GAIN')

    tagValues = [
        TagValue('priceAbove', 1.5),
        TagValue('priceBelow', 13),
        TagValue('volumeAbove', pow(10,6))]

    scanData = ib.reqScannerData(sub, [], tagValues)

    syms = [sd.contractDetails.contract.symbol for sd in scanData]
    return syms[:3]


def BracketOrder(parentOrderId, childOrderId, action, quantity, stopPrice):
    parent = Order()
    parent.orderId = parentOrderId
    parent.action = action
    parent.orderType = "STP LMT"
    parent.totalQuantity = quantity
    parent.lmtPrice = stopPrice+0.02
    parent.auxPrice = stopPrice
    parent.transmit = False

    stopLoss = Order()
    stopLoss.orderId = childOrderId
    stopLoss.action = "SELL" if action == "BUY" else "BUY"
    stopLoss.orderType = "TRAIL"
    stopLoss.auxPrice = round(stopPrice*0.02,2)
    stopLoss.trailStopPrice = round(0.98*stopPrice,2)
    stopLoss.totalQuantity = quantity
    stopLoss.parentId = parentOrderId
    stopLoss.transmit = True

    bracketOrder = [parent, stopLoss]
    return bracketOrder 


def stop_price(contract,count):
    ticker = ib.ticker(contract)
    ib.sleep(1)
    data = ib.reqHistoricalData(contract, '', barSizeSetting="5 mins", durationStr='1 D', whatToShow='TRADES', useRTH=True, keepUpToDate=False)
    data = pd.DataFrame(data)
    five_min_hi = data['high'][0]
    day_hi = data['high'].max()
    if count>1:
        return day_hi
    else:
        if ticker.marketPrice() < five_min_hi:
            return five_min_hi
        else:
            return day_hi


quantity = 500
order_id = {}
trade_count = {}


while(1):
    if time_check()==1:
        symbols = scanner()
        contracts = [Stock(stk,"SMART","USD") for stk in symbols]
        ib.qualifyContracts(*contracts)
        ib.sleep(2)

        ib.reqMarketDataType(1)              
        for contract in contracts:
            ib.reqMktData(contract, '', False, False)
            ib.sleep(1)

        for i in range(len(symbols)):
            if symbols[i] in trade_count:
                if trade_count[symbols[i]]>=3:
                    continue
            if symbols[i] in order_id:
                ids = [sd.orderId for sd in ib.openOrders()]
                if (order_id[symbols[i]][0] in ids) or (order_id[symbols[i]][1] in ids):
                    continue
                else:
                    if symbols[i] in trade_count:
                        trade_count[symbols[i]] += 1
                    else:
                        trade_count[symbols[i]] = 1
                        
                    parent_id = ib.client.getReqId()
                    child_id = ib.client.getReqId()
                    order_id[contracts[i].symbol] = [parent_id, child_id]
                    price = stop_price(contracts[i],trade_count[symbols[i]])
                    bracket = BracketOrder(parent_id, child_id, "BUY", quantity, price)
                    for o in bracket:
                        ib.placeOrder(contracts[i], o)
            else:
                if symbols[i] in trade_count:
                        trade_count[symbols[i]] += 1
                else:
                    trade_count[symbols[i]] = 1
                parent_id = ib.client.getReqId()
                child_id = ib.client.getReqId()
                price = stop_price(contracts[i],trade_count[symbols[i]])
                order_id[contracts[i].symbol] = [parent_id, child_id]
                bracket = BracketOrder(parent_id, child_id, "BUY", quantity, price)
                for o in bracket:
                    ib.placeOrder(contracts[i], o)

        time.sleep(60)
    
    elif time_check == -1:
        ib.disconnect()
        break
    else:
        time.sleep(10)



