#simple match context features
import pandas as pd
import numpy as np
import math

# defined here so not redefined every single row
ROUND_TO_PLAYERS = {
    "F": 2,
    "SF": 4,
    "QF": 8,
    "R16": 16,
    "R32": 32,
    "R64": 64,
    "R128": 128
}


TOURNAMENT_COUNTRY = {
    # australia
    "Australian Open": "AUS", "Adelaide": "AUS", "Sydney": "AUS", "Sydney Olympics": "AUS", "Brisbane": "AUS",

    # USA
    "US Open": "USA", "Indian Wells Masters": "USA", "Miami Masters": "USA", "Cincinnati Masters": "USA",
    "Atlanta": "USA", "Washington": "USA", "Delray Beach": "USA", "Houston": "USA", "Memphis": "USA",
    "San Jose": "USA", "Long Island": "USA", "Los Angeles": "USA", "Scottsdale": "USA", "Orlando": "USA",
    "Indianapolis": "USA", "New Haven": "USA", "Winston-Salem": "USA",

    # united kingdom
    "Wimbledon": "GBR", "Queen's Club": "GBR", "Nottingham": "GBR", "London": "GBR",

    # france
    "Roland Garros": "FRA", "Paris Masters": "FRA", "Marseille": "FRA", "Lyon": "FRA", "Metz": "FRA",

    # spain
    "Barcelona": "ESP", "Madrid Masters": "ESP", "Mallorca": "ESP", "Valencia": "ESP",

    # italy
    "Rome Masters": "ITA", "Palermo": "ITA", "Milan": "ITA",

    # germany
    "Hamburg Masters": "GER", "Munich": "GER", "Stuttgart": "GER", "Stuttgart Outdoor": "GER",

    # switzerland
    "Basel": "SUI", "Gstaad": "SUI", "Geneva": "SUI",

    # austria
    "Vienna": "AUT", "Kitzbuhel": "AUT", "St. Poelten": "AUT",

    # netherlands
    "Rotterdam": "NED", "s Hertogenbosch": "NED",

    # belgium
    "Antwerp": "BEL",

    # sweden
    "Stockholm": "SWE", "Bastad": "SWE",

    # denmark
    "Copenhagen": "DEN",

    # russia
    "Moscow": "RUS", "St. Petersburg": "RUS",

    # china
    "Beijing": "CHN", "Shanghai Masters": "CHN", "Shenzhen": "CHN",

    # japan
    "Tokyo": "JPN",

    # argentina
    "Buenos Aires": "ARG",

    # chile
    "Santiago": "CHI", "Vina del Mar": "CHI",

    # brazil
    "Rio de Janeiro": "BRA", "Sao Paulo": "BRA", "Costa Do Sauipe": "BRA",

    # mexico
    "Acapulco": "MEX",

    # middle east
    "Doha": "QAT", "Dubai": "UAE",

    # others from early years
    "Bogota": "COL", "Casablanca": "MAR", "Tashkent": "UZB", "Bangkok": "THA",
    "Ho Chi Minh City": "VNM", "San Marino": "SMR", "Brighton": "GBR", "Toulouse": "FRA",
    "Hong Kong": "HKG", "Sopot": "POL", "Umag": "CRO", "Zagreb": "CRO",
    "Estoril": "POR", "Belgrade": "SRB", "Johannesburg": "RSA", "Kuala Lumpur": "MYS",
    "Istanbul": "TUR", "Antalya": "TUR", "Budapest": "HUN", "Quito": "ECU",
    "Marrakech": "MAR", "Chengdu": "CHN",

    #others that were found after testing the whole dataset
    #some include renaming ect
    "Auckland": "NZL",
    "Newport": "USA",
    "Amsterdam": "NED",
    "Stuttgart Masters": "GER",
    "Monte Carlo Masters": "MCO",
    "Canada Masters": "CAN",
    "Halle": "GER",
    "Dusseldorf": "GER",
    "Bucharest": "ROU",
    "Mexico City": "MEX",
    "Shanghai": "CHN",
    "Chennai": "IND",
    "Amersfoort": "NED",
    "Poertschach": "AUT",
    "Mumbai": "IND",
    "Las Vegas": "USA",
    "Warsaw": "POL",
    "Hamburg": "GER",
    "Eastbourne": "GBR",
    "Nice": "FRA",
    "Montpellier": "FRA",
    "Sofia": "BUL",
    "Los Cabos": "MEX",
    "Pune": "IND",
    "New York": "USA",
    "Cordoba": "ARG",
    "Zhuhai": "CHN",
    "Atp Cup": "AUS",
    "ATP Rio de Janeiro": "BRA",
    "Us Open": "USA",
    "Cologne 1": "GER",
    "Cologne 2": "GER",
    "Sardinia": "ITA",
    "St Petersburg": "RUS",
    "Nur-Sultan": "KAZ",
    "Great Ocean Road Open": "AUS",
    "Murray River Open": "AUS",
    "Singapore": "SGP",
    "Marbella": "ESP",
    "Cagliari": "ITA",
    "Parma": "ITA",
    "Belgrade 2": "SRB",
    "Belgrade ": "SRB",  # trailing space
    "San Diego": "USA",
    "Adelaide 1": "AUS",
    "Adelaide 2": "AUS",
    "Melbourne": "AUS",
    "Dallas": "USA",
    "Seoul": "KOR",
    "Tel Aviv": "ISR",
    "Astana": "KAZ",
    "Florence": "ITA",
    "Gijon": "ESP",
    "Naples": "ITA",
    "Rio De Janeiro": "BRA",
    "Banja Luka": "BIH",
}


# ml model prefer numeric order features
def encode_round(row):
    name = row["tourney_name"]
    draw = row["draw_size"]
    rnd = row["round"]

    # ATP finals special case
    if name in ["Masters Cup", "Tour Finals", "ATP Finals"]:
        mapping = {"RR": 1, "SF": 2, "F": 3}
        return mapping.get(rnd, 0)

    # players remaining in this round
    players_remaining = ROUND_TO_PLAYERS.get(rnd)

    # check for values like RR
    if players_remaining is None:
        return None

    # drawsize/first round playerseffectivly rounded up to nearest power of 2 check notes for maths explanation
    first_round_players = 2 ** int(math.ceil(math.log2(draw)))

    #  round calculated be finding the number of times the player have been halved
    # +1 so that round 1 == 1 not 0
    round_number = int(np.log2(first_round_players / players_remaining)) + 1

    return round_number


def assign_tour_finals_country(row):
    name = row["tourney_name"]
    year = row["tourney_date"].year

    if name in ["Masters Cup", "Tour Finals", "ATP Finals"]:
        if 2000 <= year <= 2004:
            return "USA"
        elif 2005 <= year <= 2008:
            return "CHN"
        elif 2009 <= year <= 2020:
            return "GBR"
        elif 2021 <= year <= 2023:
            return "ITA"
        else:
            return None
    return None

def add_tourney_countries(df):
    # tournament country assignment
    df["tourney_country"] = df["tourney_name"].map(TOURNAMENT_COUNTRY)

    # override for atp finals
    df["tf_country"] = df.apply(assign_tour_finals_country, axis=1)
    df["tourney_country"] = df["tf_country"].fillna(df["tourney_country"])
    df.drop(columns=["tf_country"], inplace=True)

    # validation: ensure every tournament has a country
    missing = df[df["tourney_country"].isna()]["tourney_name"].unique()
    if len(missing) > 0:
        raise ValueError(f"Missing country mapping for tournaments: {missing}")

    return df

def add_indoor_feature(df):

    ALWAYS_INDOOR = {
        "Paris Masters",
        "Basel",
        "Vienna",
        "Rotterdam",
        "Marseille",
        "Metz",
        "St. Petersburg",
        "Stockholm",
        "Lyon",
        "Copenhagen",
        "Milan",
        "Antwerp",
        "Sofia",
        "Montpellier",
        "Zagreb",
        "Brighton",
        "Toulouse",
        "Hong Kong",
        "Memphis",   # indoor until 2014, ATP removed after 2014
        "San Jose",
    }

    def is_indoor(row):
        name = row["tourney_name"]
        year = row["tourney_date"].year

        if name in ALWAYS_INDOOR:
            return 1

        # specific cases
        if name == "Madrid Masters":
            return 1 if 2002 <= year <= 2008 else 0

        if name == "Shanghai":
            return 1 if 2002 <= year <= 2004 else 0

        if name == "Shanghai Masters":
            return 0

        if name in ["Masters Cup", "Tour Finals", "ATP Finals"]:
            if 2000 <= year <= 2003:
                return 0   # houston outdoor
            elif 2005 <= year <= 2008:
                return 1   # shanghai indoor
            elif 2009 <= year <= 2020:
                return 1   # london indoor
            elif 2021 <= year <= 2023:
                return 1   # turin indoor
            else:
                return 0

        # assume outdoor unless said
        return 0

    df["is_indoor"] = df.apply(is_indoor, axis=1).astype(int)

    return df


def add_pre_match_context_features(df: pd.DataFrame) -> pd.DataFrame:

    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")

    # rank and more specific rank points
    df["rank_diff"] = df["p2_rank"] - df["p1_rank"]
    df["rank_points_diff"] = df["p2_rank_points"] - df["p1_rank_points"]

    # o/1 colums required for machine learning models
    df["is_grand_slam"] = (df["tourney_level"] == "G").astype(int)
    df["is_masters"] = (df["tourney_level"] == "M").astype(int)
    # usefull as older master and other tournaments had 5 sets (not just slams)
    df["is_best_of_five"] = (df["best_of"] == 5).astype(int)


    #changes match pattern dynamics
    df["p1_lefty"] = (df["p1_hand"] == "L").astype(int)
    df["p2_lefty"] = (df["p2_hand"] == "L").astype(int)
    df["lefty_diff"] = df["p1_lefty"] - df["p2_lefty"]

    # takes surface columns makes 0/1 columns for each surface and fill them
    # keep original surface for predict match input
    df["surface_raw"] = df["surface"]
    df = pd.get_dummies(df, columns=["surface"], prefix="surface")

    # round encoder called
    df["round_encoded"] = df.apply(encode_round, axis=1)

    # can help era context
    df["match_year"] = (df["tourney_date"]).dt.year

    # younger usually more durable
    df["age_diff"] = df["p2_age"] - df["p1_age"]

    # heavy influence on serve
    df["height_diff"] = df["p2_ht"] - df["p1_ht"]

    # help tournament understand importance maybe has effect on pressure
    prestige_map = {"G": 5, "F":4,  "M": 3, "A": 2, "B": 1}
    df["tourney_prestige"] = df["tourney_level"].map(prestige_map)

    # get points for tournament round helpful for pressure calculations
    POINTS_TABLE = {
        "G":  {"F":2000, "SF":1200, "QF":720, "R16":360, "R32":180, "R64":90, "R128":45},
        "M":  {"F":1000, "SF":600,  "QF":360, "R16":180, "R32":90, "R64":45, "R128": 20},
        "A":  {"F":500,  "SF":300,  "QF":180, "R16":90,  "R32":45, "R64":20, "R128":10},
        "B":  {"F":250,  "SF":150,  "QF":90,  "R16":45,  "R32":20, "R64":10, "R128":5},

        "FNL": {"RR": 200, "SF": 400, "F": 500}
    }

    def get_round_points(row):
        name = row["tourney_name"]
        level = row["tourney_level"]
        rnd = row["round"]

        # ATP finals uses different points
        if name in ["Masters Cup", "Tour Finals", "ATP Finals"]:
            return POINTS_TABLE["FNL"].get(rnd, 0)

        return POINTS_TABLE.get(level, {}).get(rnd, np.nan)

    df["round_points"] = df.apply(get_round_points, axis=1)

    #add tournament countries
    add_tourney_countries(df)

    # home court features
    df["p1_home"] = (df["p1_ioc"] == df["tourney_country"]).astype(int)
    df["p2_home"] = (df["p2_ioc"] == df["tourney_country"]).astype(int)
    df["home_advantage_diff"] = df["p1_home"] - df["p2_home"]

    add_indoor_feature(df)


    return df
