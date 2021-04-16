from fuzzywuzzy import process
from dataWrapper.load_russel3000_stocks import load_russel3000
companies=load_russel3000()
details=companies['Details'].tolist()
names=companies['Name'].tolist()



def search_companies(text,threshold=91):
    result=process.extractOne(text, names)
    #print(result)
    if(result[1]>=threshold):
        match=companies[companies['Name'] == result[0]]
        ticker=match["Ticker"].values[0]
        return [ticker,result[0]]
    #Using the details column doesn't work well,
    #else: 
        #If can't find a matching name look in the details
    #    result=process.extractOne(text, details)
    #    print(result)
    #    if(result[1]>=80):
    #        match=companies[companies['Details'] == result[0]]
    #        ticker=match["Ticker"].values[0]
    #        name=match["Name"].values[0]
    #        return [ticker,name]
    return None



if __name__ == '__main__': 
    print(search_companies("The Democratic Party  formally nominated Hillary Clinton  for president Tuesday,  making her the first woman chosen by a major American party.  (CNBC)   In a Democratic National Convention speech peppered with personal  stories, former president Bill Clinton on Tuesday  portrayed his wife  as  compassionate but tenacious, calling her the best darn change  maker I've ever met in my life. (CNBC) Nine black women whose children died in racially-charged incidents  took the convention stage to endorse  Hillary Clinton, saying they believe she cares deeply about racial injustice and would try to heal wounds between police departments and African American communities. (NBC \
News)   Wall Street has  become far less  certain Hillary Clinton will be the next  president. Just 52 percent of respondents to the July CNBC Fed  Survey now believe the Democratic nominee will prevail in  November, a sharp \
drop from 80 percent in the April and June  surveys. (CNBC)   Apple's (AAPL)  stock  is jumping  after quarterly profit of $1.42 beat estimates by  4 cents and revenue also beat forecasts.  Apple gave strong  current quarter \
revenue guidance and reported iPhone and iPad  shipments that were higher than analysts had been estimating.   Twitter (TWTR)  saw  its stock price sink  after the company beat profit  estimates, but reported revenue below forecasts. The company also  gave lower-than-expected current quarter revenue guidance, and  revenue growth was the slowest since Twitter went public in 2013.   Tesla's (TSLA) upcoming Model 3 car  could generate $20 billion in  revenue  per year and an annual gross profit of about $5 billion,  CEO Elon Musk said Tuesday. (CNBC)   Investors  yanked  $20.7 billion from global hedge funds  in June, bringing net  redemption for the second quarter to $10.7 billion following  inflows in April and May, the  Financial Times  reported.  This is the third consecutive quarter capital has fled the hedge  fund sector. (FT)   Third Point's Dan Loeb  compared the current investment \
environment  to the — spoiler alert — epic battle in the penultimate episode of  Game of Thrones season six: Surging enemies forming a  seemingly impossible perimeter, a crush of fellow soldiers on the  field, arrows coming in overhead, and the need to avoid panic.  (CNBC)   An employee at Ray Dalio's Bridgewater Associates described the  hedge fund as a  cauldron of fear and  intimidation  in a complaint filed in Connecticut  alleging sexual harassment,  The New York Times   reported. The complaint describes an environment of constant  surveillance, according to the newspaper. (NYT)   Goldman Sachs (GS) has been  slapped with a $510 million lawsuit  by a shareholder  of one of its former clients over alleged fraudulent  misrepresentations that involve links to the prime minister of  Malaysia and the country's troubled 1MDB development fund. (CNBC) The Fed's post-meeting statement comes \
at 2 p.m. ET, with no  interest rate change expected, but investors will as usual be  parsing the statement for any clues as to future rate moves.   Several economic reports come ahead of the Fed's pronouncement,  beginning at 8:30 a.m. ET with the government's report on durable  goods.  Economists are expecting a 1.3 percent drop for June  following a 2.3 percent decline in May.     At 10 a.m. ET, the National Association of Realtors will release  its June pending home sales report, with consensus forecasts  calling for a 1.9 percent increase following a 3.7 percent drop  in May.   Other reports out today include the weekly report on mortgage  applications from the \
Mortgage Bankers Association at 7 a.m. ET,  and the Energy Department's usual Wednesday look at oil and  gasoline inventories at 10:30 a.m. ET.   Dow components Boeing (BA) and Coca-Cola (KO), as well as  NBCUniversal and CNBC parent Comcast (CMCSA) highlight this  morning's list of corporate earnings.  Also out this  morning: Altria (MO), Anthem (ANTM), Corning (GLW), Dr Pepper  Snapple (DPS), Gannett (GCI), Garmin (GRMN), General Dynamics  (GD), Goodyear Tire (GT), Hess (HES), Hilton Worldwide (HLT),  Mondelez International (MDLZ), Nasdaq (NDAQ), Norfolk Southern  (NSC), Northrop Grumman (NOC), Owens Corning (OC), Southern  Company (SO), State Street (STT), T-Mobile US (TMUS), Waste  Management (WM), and Wyndham Worldwide (WYN).   Facebook (FB) and Whole Foods (WFM) are among today's  after-the-bell earnings reports, along with Amgen (AMGN), CA  Technologies (CA), GoPro (GPRO), Groupon (GRPN), IAC/Interactive  (IACI), Marriott (MAR), McKesson (MCK), O'Reilly Automotive  (ORLY), Public Storage (PSA), Realty Income (O), and Xilinx  (XLNX). Linear Technology (LLTC) agreed to be bought by rival  semiconductor \
maker Analog Devices (ADI) in a cash and stock deal  worth $14.8 billion. The transaction values Linear at $60  per share, representing a 24 percent premium.   Deutsche Bank (DB) said it may need more cost cuts to turn its  operations around, following a sharp decline in second-quarter  revenue.   Buffalo Wild Wings (BWLD) reported quarterly profit of $1.27 per  share, 2 cents above estimates, but the chicken wings  restaurant saw revenues come in below Street projections as  same-store sales fell from a year earlier.   Anadarko Petroleum (APC) lost 60 cents per share for its latest  quarter, a loss that was 20 cents smaller than expected, and the  oil and gas producer saw revenue slightly above estimates. Cost cutting helped Anadarko avoid a bigger loss.   U.S. Steel (X) reported a loss of 31 cents per share for its  latest quarter, smaller than the 49 cent consensus estimate, but  the \
steel maker's revenue fell below estimates. The company  said it is seeing an improving pricing environment and that its  European operations saw its best results in nearly eight years. A film production unit of Disney (DIS) \
on Tuesday  admitted to  health and safety breaches  during a hearing in Britain over an  accident on the set of Star Wars: Episode VII The Force Awakens  that left Harrison Ford with a broken leg."))