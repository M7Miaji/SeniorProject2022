import time
activited=True#user activited the automatic mode
max=50#max price by the user
min=20#min price by the user
favStock=[{'stockID':0,'name':"apple",'buyprice':20,'numberofstocks':10,'numberofstocksihave':0,'newprice':20},
          {'stockID':1,'name':"toyota",'buyprice':50,'numberofstocks':50,'numberofstocksihave':0,'newprice':50}
          ]
balance=500#user money

stockrising=True#to check if the stock is currently rising or not so we can use it in the sell method
risklevel="low"#risk will help us to but a timer for a user to wait until we see if the stock will rise again
leastpricestockicanbuy=20

buycounter=0#to make a perfect balance between buying and selling and that will be by making one buy one selling at the moment
sellcounter=1



##############methods####################
def buy (balance,min,max,favStock):
    favStocklength = len(favStock)
    canibuy=True
    for i in range(0,favStocklength):
        if favStock[i]['buyprice']>=min and favStock[i]['buyprice']<=max and favStock[i]['buyprice']<=balance and favStock[i]['numberofstocks']!=favStock[i]['numberofstocksihave'] :
                balance-=favStock[i]['buyprice']
                favStock[i]['numberofstocksihave']+=1
        else:
            canibuy=False
    return balance,favStock,canibuy

def sell(stockrising,balance,risklevel,favStock):

    favStocklength = len(favStock)
    for i in range(0, favStocklength):
        if favStock[i]['numberofstocksihave'] > 0 and stockrising ==True and favStock[i]['newprice']>favStock[i]['buyprice']:
            if risklevel=="low":
                if stockrising==True:
                    t=3#time in seconds
                    countdown(t)
                balance += favStock[i]['newprice']
                favStock[i]['numberofstocksihave'] -= 1

            elif risklevel=="medium":
                if stockrising==True:
                    t=5#time in seconds
                    countdown(t)
                balance += favStock[i]['newprice']
                favStock[i]['numberofstocksihave'] -= 1

            elif risklevel=="high":
                if stockrising==True:
                    t=10#time in seconds
                    countdown(t)
                balance += favStock[i]['newprice']
                favStock[i]['numberofstocksihave'] -= 1


    return balance,favStock


# define the countdown func.
#this method will help us to sell the stocks based on time and the time will be unique for each risk level
def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1
    print('Sell!!')



###############Main method################
#in this method will do everything automatically wether to selling or buying a stock
def automatic(favStock,min,max,risklevel,activited,balance,leastpricestockicanbuy):
    if balance>=leastpricestockicanbuy:
        canibuy=True
        while activited==True and canibuy ==True:
            balance,favStock,canibuy=buy(balance,min,max,favStock)
        return balance, favStock
    else:
        balance, favStock=sell(stockrising,balance,risklevel,favStock)
        return balance,favStock





#################################################3
countdown(15)
while activited==True:
    balance,favStock=automatic(favStock,min,max,risklevel,activited,balance,leastpricestockicanbuy)#buying as much as he can and  then will stop the loop
    print(balance,favStock)
    print("1| Change a spefic stock price")
    print("2| Quit program")
    choice = int(input("Please enter a number to change something in the settings\n"
                       "#########################################################"))
    if choice==1:#to change a new price in the program so the algorithm can sell
        choice=-1
        print(favStock)
        choice=int(input("\nplease choice one of the stocks id\n####################################"))
        print(favStock)
        newpricebyuser=int(input("Please enter the new price in intger"))
        favStock[choice]['newprice']=newpricebyuser
        print("####################################")
        for i in range(0, len(favStock)):
            if favStock[choice]['newprice'] > favStock[choice]['buyprice']:
                stockrising = True
            else:
                stockrising = False
    elif choice==2:#will stop the program
        activited=False
    else:#if you want to run the automatic method faster just press any other number of letter
        balance, favStock = automatic(favStock, min, max, risklevel, activited, balance, leastpricestockicanbuy)



