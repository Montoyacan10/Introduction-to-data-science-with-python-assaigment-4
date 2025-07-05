
#Description¶
#In this assignment you must read in a file of metropolitan regions and associated sports teams from assets/wikipedia_data.html and answer some questions about each metropolitan region. Each of these regions may have one or more teams from the "Big 4": NFL (football, in assets/nfl.csv), MLB (baseball, in assets/mlb.csv), NBA (basketball, in assets/nba.csv or NHL (hockey, in assets/nhl.csv). Please keep in mind that all questions are from the perspective of the metropolitan region, and that this file is the "source of authority" for the location of a given sports team. Thus teams which are commonly known by a different area (e.g. "Oakland Raiders") need to be mapped into the metropolitan region given (e.g. San Francisco Bay Area). This will require some human data understanding outside of the data you've been given (e.g. you will have to hand-code some names, and might need to google to find out where teams are)!

#For each sport I would like you to answer the question: what is the win/loss ratio's correlation with the population of the city it is in? Win/Loss ratio refers to the number of wins over the number of wins plus the number of losses. Remember that to calculate the correlation with pearsonr, so you are going to send in two ordered lists of values, the populations from the wikipedia_data.html file and the win/loss ratio for a given sport in the same order. Average the win/loss ratios for those cities which have multiple teams of a single sport. Each sport is worth an equal amount in this assignment (20%*4=80%) of the grade for this assignment. You should only use data from year 2018 for your analysis -- this is important!

#Notes
#Do not include data about the MLS or CFL in any of the work you are doing, we're only interested in the Big 4 in this assignment.
#I highly suggest that you first tackle the four correlation questions in order, as they are all similar and worth the majority of grades for this assignment. This is by design!
#It's fair game to talk with peers about high level strategy as well as the relationship between metropolitan areas and sports teams. However, do not post code solving aspects of the assignment (including such as dictionaries mapping areas to teams, or regexes which will clean up names).
#There may be more teams than the assert statements test, remember to collapse multiple teams in one city into a single value!
#As this assignment utilizes global variables in the skeleton code, to avoid having errors in your code you can either:

#You can place all of your code within the function definitions for all of the questions (other than import statements).
#You can create copies of all the global variables with the copy() method and proceed as usual.

import pandas as pd
import re
import numpy as np 
from scipy import stats

nhl_df=pd.read_csv("assets/nhl.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

#For this question, calculate the win/loss ratio's correlation with the population of the city it is in for the NHL using 2018 data.
def answer_one():
        # limpiando cities df 
    #.apply(lambda x:[re.sub(r'[^A-Za-z]','',team) for team in x ])

    regex_p =r'\[.*|\d+.+|—'

    def cleaning_names(row):

        row =  [re.sub(regex_p,'',team) for team in row]
        clean_row  = [item for item in row if item]

        return clean_row


    #este segmento se encarga de hacer una copia de cities, limpiieza de los datos en la columna NHL del df con datos metropolitanos 
    citiesdf = cities.copy()
    clean_nhl_citiesdf = citiesdf['NHL'].str.split(r' |,').apply(lambda x:cleaning_names(x))# limpieza 
    citiesdf['NHL'] =clean_nhl_citiesdf # reescribo la columna de NHL con el valor obtenido en la limipeza 
    clean_citiesdf_split= citiesdf.set_index('Metropolitan area').explode('NHL') # con esto consigo crear una fila para equipo de la columna 
    #—

    # este segmento se encarga de la limpiezada de los nombres de los equipos 

    nhl_dfc = nhl_df.copy()
    nhl_df_clean = nhl_df['team'].apply(lambda x: re.sub(r'\*','',x)) 
    nhl_dfc['team'] = nhl_df_clean

    def get_metropolitan_a(x):
        
        return x.split(' ')[-1] 

    nhl_dfc['team_s']=nhl_dfc['team'].apply(lambda x: get_metropolitan_a(x))
    megeddf = pd.merge(nhl_dfc,clean_citiesdf_split.reset_index(),left_on='team_s' , right_on= 'NHL',how='outer')
    row_to_keep = ['team_s','GP','W','L','OL','Population (2016 est.)[8]']
    mergeddf = megeddf[megeddf['year']==2018].dropna().set_index('Metropolitan area')[row_to_keep]

    mergeddf['GP'] = pd.to_numeric(mergeddf['GP'], errors='coerce')
    mergeddf['W'] = pd.to_numeric(mergeddf['W'], errors='coerce')
    mergeddf['L'] = pd.to_numeric(mergeddf['L'], errors='coerce')
    mergeddf['OL'] = pd.to_numeric(mergeddf['OL'], errors='coerce')
    mergeddf['Population (2016 est.)[8]'] = pd.to_numeric(mergeddf['Population (2016 est.)[8]'], errors='coerce')


    aggregation_rules = {
        'GP': 'mean',
        'W': 'mean',
        'L': 'mean',
        'OL': 'mean',
        'team_s': lambda x: ', '.join(x.astype(str)),  # Concatenate team names
        'Population (2016 est.)[8]': 'first' # Take the first value for population
    }

    NHL_FINAL_DF = mergeddf.groupby('Metropolitan area').agg(aggregation_rules)
    NHL_FINAL_DF['win/lost'] = NHL_FINAL_DF['W']/(NHL_FINAL_DF['W']+NHL_FINAL_DF['L'])
    return NHL_FINAL_DF



def nhl_correlation(): 
    NHL_FINAL_DF = answer_one()
    # YOUR CODE HERE
    #raise NotImplementedError()
    
    population_by_region = NHL_FINAL_DF['Population (2016 est.)[8]'] # pass in metropolitan area population from cities
    win_loss_by_region = NHL_FINAL_DF['win/lost'] # pass in win/loss ratio from nhl_df in the same order as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q1: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q1: There should be 28 teams being analysed for NHL"
    
    return stats.pearsonr(population_by_region, win_loss_by_region)[0]


import pandas as pd
import numpy as np
import scipy.stats as stats
import re

nba_df=pd.read_csv("assets/nba.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]

#For this question, calculate the win/loss ratio's correlation with the population of the city it is in for the NBA using 2018 data.

def answer_two():
    # limpiando cities df 
    #.apply(lambda x:[re.sub(r'[^A-Za-z]','',team) for team in x ])

    regex_p =r'\[.*|\d+\]|—'

    def cleaning_names(row):

        row =  [re.sub(regex_p,'',team) for team in row]
        clean_row  = [item for item in row if item]

        return clean_row


    #este segmento se encarga de hacer una copia de cities, limpiieza de los datos en la columna NHL del df con datos metropolitanos 
    citiesdf = cities.copy()
    clean_nba_citiesdf = citiesdf['NBA'].str.split(r' |,').apply(lambda x:cleaning_names(x))# limpieza 
    citiesdf['NBA'] =clean_nba_citiesdf # reescribo la columna de NHL con el valor obtenido en la limipeza 
    clean_citiesdf_split= citiesdf.set_index('Metropolitan area').explode('NBA') # con esto consigo crear una fila para equipo de la columna 

    nba_dfc = nba_df.copy()
    nba_df_clean = nba_dfc['team'].apply(lambda x: re.sub(r'\*|\([0-9]+\)+','',x)) 
    nba_dfc['team'] = nba_df_clean

    def get_metropolitan_a(x):
        
        return x.split(' ')[-1].strip()

    nba_dfc['team_s']=nba_dfc['team'].apply(lambda x: get_metropolitan_a(x))
    nba_dfc = nba_dfc[nba_dfc['year'] == 2018.0]
    megeddf = pd.merge(nba_dfc,clean_citiesdf_split.reset_index(),left_on='team_s' , right_on= 'NBA',how='outer')
    row_to_keep = ['team_s','W','L','Population (2016 est.)[8]','team']
    mergeddf = megeddf.set_index('Metropolitan area')[row_to_keep].dropna()

    mergeddf['W'] = pd.to_numeric(mergeddf['W'], errors='coerce')
    mergeddf['L'] = pd.to_numeric(mergeddf['L'], errors='coerce')
    mergeddf['Population (2016 est.)[8]'] = pd.to_numeric(mergeddf['Population (2016 est.)[8]'], errors='coerce')


    aggregation_rules = {
        'W': 'mean',
        'L': 'mean',
        'team_s': lambda x: ', '.join(x.astype(str)),  # Concatenate team names
        'Population (2016 est.)[8]': 'first' # Take the first value for population
    }

    NBA_FINAL_DF = mergeddf.groupby('Metropolitan area').agg(aggregation_rules)
    NBA_FINAL_DF['win/lost'] = NBA_FINAL_DF['W']/(NBA_FINAL_DF['W']+NBA_FINAL_DF['L'])
    return NBA_FINAL_DF
#mergeddf['team'].sort_values().dropna(), megeddf.set_index('Metropolitan area')['team'].dropna()

def nba_correlation():
    NBA_FINAL_DF = answer_two()
    # YOUR CODE HERE
    #raise NotImplementedError()
    
    population_by_region = NBA_FINAL_DF['Population (2016 est.)[8]'] # pass in metropolitan area population from cities
    win_loss_by_region = NBA_FINAL_DF['win/lost'] # pass in win/loss ratio from nhl_df in the same order as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q2: Your lists must be the same length"
    assert len(population_by_region) == 28, "Q2: There should be 28 teams being analysed for NBA"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]
  


import pandas as pd
import numpy as np
import scipy.stats as stats
import re

mlb_df=pd.read_csv("assets/mlb.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]


#For this question, calculate the win/loss ratio's correlation with the population of the city it is in for the MLB using 2018 data.

def answer_three():

        # limpiando cities df 
    #.apply(lambda x:[re.sub(r'[^A-Za-z]','',team) for team in x ])

    regex_p =r'\[.*|\d+\]|—'

    def cleaning_names(row):

        row =  [re.sub(regex_p,'',team) for team in row]


        clean_row  = [item for item in row if item]

        return clean_row


    #este segmento se encarga de hacer una copia de cities, limpiieza de los datos en la columna NHL del df con datos metropolitanos 
    citiesdf = cities.copy()
    clean_nba_citiesdf = citiesdf['MLB'].str.split(r' ').apply(lambda x:cleaning_names(x))# limpieza 

    citiesdf['MLB'] =clean_nba_citiesdf # reescribo la columna de NHL con el valor obtenido en la limipeza 
    citiesdf['MLB'].loc[3] = ['Cubs','White Sox']
    citiesdf['MLB'].loc[7] = ['Red Sox']
    citiesdf['MLB'].loc[13] = ['Blue Jays']
    clean_citiesdf_split= citiesdf.set_index('Metropolitan area').explode('MLB') # con esto consigo crear una fila para equipo de la columna 

    mlb_dfc = mlb_df.copy()
    mlb_df_clean = mlb_dfc['team'].apply(lambda x: re.sub(r'\*|\([0-9]+\)+','',x)) 
    mlb_dfc['team'] = mlb_df_clean

    def get_metropolitan_a(x):
        
        return x.split(' ')[-1]

    mlb_dfc['team_s']=mlb_dfc['team'].apply(lambda x: get_metropolitan_a(x))
    mlb_dfc['team_s'].loc[0]='Red Sox'
    mlb_dfc['team_s'].loc[3]='Blue Jays'
    mlb_dfc['team_s'].loc[8]='White Sox'


    
    mlb_dfc = mlb_dfc[mlb_dfc['year'] == 2018.0]
    megeddf = pd.merge(mlb_dfc,clean_citiesdf_split.reset_index(),left_on='team_s' , right_on= 'MLB',how='outer')
    row_to_keep = ['team_s','W','L','Population (2016 est.)[8]','team']
    mergeddf = megeddf.set_index('Metropolitan area')[row_to_keep].dropna()

    mergeddf['W'] = pd.to_numeric(mergeddf['W'], errors='coerce')
    mergeddf['L'] = pd.to_numeric(mergeddf['L'], errors='coerce')
    mergeddf['Population (2016 est.)[8]'] = pd.to_numeric(mergeddf['Population (2016 est.)[8]'], errors='coerce')


    aggregation_rules = {
        'W': 'mean',
        'L': 'mean',
        'team_s': lambda x: ', '.join(x.astype(str)),  # Concatenate team names
        'Population (2016 est.)[8]': 'first' # Take the first value for population
    }

    MLB_FINAL_DF = mergeddf.groupby('Metropolitan area').agg(aggregation_rules)
    MLB_FINAL_DF['win/lost'] = MLB_FINAL_DF['W']/(MLB_FINAL_DF['W']+MLB_FINAL_DF['L'])
    return     MLB_FINAL_DF

def mlb_correlation(): 
    # YOUR CODE HERE
    #raise NotImplementedError()
    MBL_FINAL_DF =answer_three()
    population_by_region = MBL_FINAL_DF['Population (2016 est.)[8]'] # pass in metropolitan area population from cities
    win_loss_by_region = MBL_FINAL_DF['win/lost'] # pass in win/loss ratio from nhl_df in the same order as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q3: Your lists must be the same length"
    assert len(population_by_region) == 26, "Q3: There should be 26 teams being analysed for MLB"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]


import pandas as pd
import numpy as np
import scipy.stats as stats
import re

nfl_df=pd.read_csv("assets/nfl.csv")
cities=pd.read_html("assets/wikipedia_data.html")[1]
cities=cities.iloc[:-1,[0,3,5,6,7,8]]


#For this question, calculate the win/loss ratio's correlation with the population of the city it is in for the NFL using 2018 data

def answer_four():
            # limpiando cities df 
    #.apply(lambda x:[re.sub(r'[^A-Za-z]','',team) for team in x ])

    regex_p =r'\[.*|\d+\]|—'

    def cleaning_names(row):

        row =  [re.sub(regex_p,'',team) for team in row]


        clean_row  = [item for item in row if item]

        return clean_row


    #este segmento se encarga de hacer una copia de cities, limpiieza de los datos en la columna NHL del df con datos metropolitanos 
    citiesdf = cities.copy()
    clean_nba_citiesdf = citiesdf['NFL'].str.split(r' ').apply(lambda x:cleaning_names(x))# limpieza 

    citiesdf['NFL'] =clean_nba_citiesdf # reescribo la columna de NHL con el valor obtenido en la limipeza 
    #citiesdf['NFL'].loc[3] = ['Cubs','White Sox']
    #citiesdf['NFL'].loc[7] = ['Red Sox']
    #citiesdf['NFL'].loc[13] = ['Blue Jays']
    clean_citiesdf_split= citiesdf.set_index('Metropolitan area').explode('NFL') # con esto consigo crear una fila para equipo de la columna 

    nfl_dfc = nfl_df.copy()
    nfl_df_clean = nfl_dfc['team'].apply(lambda x: re.sub(r'\*|\([0-9]+\)+|\+','',x)) 
    nfl_dfc['team'] = nfl_df_clean

    def get_metropolitan_a(x):
        
        return x.split(' ')[-1]

    nfl_dfc['team_s']=nfl_dfc['team'].apply(lambda x: get_metropolitan_a(x))
    #mlb_dfc['team_s'].loc[0]='Red Sox'
    #mlb_dfc['team_s'].loc[3]='Blue Jays'
    #mlb_dfc['team_s'].loc[8]='White Sox'


    
    nfl_dfc = nfl_dfc[nfl_dfc['year'] == 2018.0]
    megeddf = pd.merge(nfl_dfc,clean_citiesdf_split.reset_index(),left_on='team_s' , right_on= 'NFL',how='outer')
    row_to_keep = ['team_s','W','L','Population (2016 est.)[8]','team']
    mergeddf = megeddf.set_index('Metropolitan area')[row_to_keep].dropna()

    mergeddf['W'] = pd.to_numeric(mergeddf['W'], errors='coerce')
    mergeddf['L'] = pd.to_numeric(mergeddf['L'], errors='coerce')
    mergeddf['Population (2016 est.)[8]'] = pd.to_numeric(mergeddf['Population (2016 est.)[8]'], errors='coerce')


    aggregation_rules = {
        'W': 'mean',
        'L': 'mean',
        'team_s': lambda x: ', '.join(x.astype(str)),  # Concatenate team names
        'Population (2016 est.)[8]': 'first' # Take the first value for population
    }

    NFL_FINAL_DF = mergeddf.groupby('Metropolitan area').agg(aggregation_rules)
    NFL_FINAL_DF['win/lost'] = NFL_FINAL_DF['W']/(NFL_FINAL_DF['W']+NFL_FINAL_DF['L'])
    return     NFL_FINAL_DF


def nfl_correlation(): 
    # YOUR CODE HERE
    #raise NotImplementedError()
    NFL_FINAL_DF =answer_four()
    population_by_region = NFL_FINAL_DF['Population (2016 est.)[8]'] # pass in metropolitan area population from cities
    win_loss_by_region = NFL_FINAL_DF['win/lost'] # pass in win/loss ratio from nhl_df in the same order as cities["Metropolitan area"]

    assert len(population_by_region) == len(win_loss_by_region), "Q4: Your lists must be the same length"
    assert len(population_by_region) == 29, "Q4: There should be 29 teams being analysed for NFL"

    return stats.pearsonr(population_by_region, win_loss_by_region)[0]



print(nfl_correlation())
