import csv
import json
import pandas as pd
from sklearn import linear_model, model_selection, preprocessing
import numpy as np
import pickle


class ELO(object):
    w = 80
    c = 10
    d = 400
    init_rate = 1000
    k0 = 4
    lamda = 1.6
    elo_rate = {}
    diff_step = 10
    min_diff = -300
    max_diff = 310
    AD = []
    min_count = 15
    rate_diff_dist = {}
    rate_diff_prob = {}

    def __init__(self):
        self.setADList()

    def calc_origin_odds(self, h, d, a):
        m = 1 / h + 1 / d + 1 / a - 1
        h = 3 * h / (3 - m * h)
        d = 3 * d / (3 - m * d)
        a = 3 * a / (3 - m * a)

        return h, d, a

    def calc_origin_over_under(self, H, A):
        x, y = 1 / H, 1 / A
        c = x * y
        a = x + y - c - 1
        s2 = -(-4 * a * c) ** 0.5 / (2 * a)
        p = x / (s2 + x - (s2 * x))
        q = y / (s2 + y - (s2 * y))

        return 1 / p, 1 / q

    def calc_origin_odds_ratio(self, h, d, a):
        pi = np.pi
        x, y, z = 1 / h, 1 / d, 1 / a
        a = x + y + z - x * y - y * z - z * x + x * y * z - 1
        b = 0
        c = x * y + y * z + z * x - 3 * x * y * z
        d = 2 * x * y * z
        r, s, t = b / a, c / a, d / a
        u = s - r ** 2 / 3
        v = s * r / 3 - 2 * r ** 3 / 27 - t
        w = v ** 2 + 4 * u ** 3 / 27
        if w > 0:
            if abs(v + w ** 0.5) / 2 > 0:
                xx = abs(v + w ** 0.5) / 2
                yy = 0
            else:
                xx = abs(v - w ** 0.5) / 2
                yy = pi
        else:
            xx = (-u ** 3 / 27) ** 0.5
            yy = np.arccos(v / (2 * xx))

        zz = xx ** (1 / 3) * np.cos(yy / 3)
        aa = xx ** (1 / 3) * np.cos((yy + 2 * pi) / 3)
        ab = xx ** (1 / 3) * np.cos((yy + 4 * pi) / 3)
        ac = xx ** (1 / 3) * np.sin(yy / 3)
        ad = xx ** (1 / 3) * np.sin((yy + 2 * pi) / 3)
        ae = xx ** (1 / 3) * np.sin((yy + 4 * pi) / 3)
        af = - u / (3 * xx ** (1 / 3)) * np.cos(yy / 3)
        ag = - u / (3 * xx ** (1 / 3)) * np.cos((yy + 2 * pi) / 3)
        ah = - u / (3 * xx ** (1 / 3)) * np.cos((yy + 4 * pi) / 3)
        ai = u / (3 * xx ** (1 / 3)) * np.sin(yy / 3)
        aj = u / (3 * xx ** (1 / 3)) * np.sin((yy + 2 * pi) / 3)
        ak = u / (3 * xx ** (1 / 3)) * np.sin((yy + 4 * pi) / 3)
        al = zz + af
        am = aa + ag
        an = ab + ah
        ao = ac + ai
        ap = ad + aj
        aq = ae + ak
        if xx == 0:
            ar = -d ** (1 / 3) if d > 0 else (-d) ** (1 / 3)
            ass = -d ** (1 / 3) if d > 0 else (-d) ** (1 / 3)
            at = -d ** (1 / 3) if d > 0 else (-d) ** (1 / 3)
        else:
            ar = al - r / 3
            ass = am - r / 3
            at = an - r / 3

        au = 0 if xx == 0 else ao
        av = 0 if xx == 0 else ap
        aw = 0 if xx == 0 else aq

        d2 = (x / (ar + x - (ar * x)))
        e2 = (y / (ar + y - (ar * y)))
        f2 = (z / (ar + z - (ar * z)))
        h, d, a = 1 / d2, 1 / e2, 1 / f2

        return h, d, a

    def calc_elo_rate(self, h, a, home_goal, away_goal):
        e_h = 1 / (1 + self.c ** ((a - h - self.w) / self.d))
        e_a = 1 - e_h

        if home_goal > away_goal:
            a_h = 1
        elif home_goal == away_goal:
            a_h = 0.5
        else:
            a_h = 0

        a_a = 1 - a_h
        delta = abs(home_goal - away_goal)
        # k = self.k0 * ( 1 + delta) ** self.lamda
        k = 14

        h = h + k * (a_h - e_h)
        a = a + k * (a_a - e_a)

        return h, a

    def calc_elo_rate_odds(self, h, a, ph, pd, pa):
        e_h = 1 / (1 + self.c ** ((a - h - self.w) / self.d))
        e_a = 1 - e_h

        ph, pd, pa = self.calc_origin_odds_ratio(ph, pd, pa)

        a_h = 1 / ph + 0.5 * 1 / pd
        a_a = 1 - a_h

        k = 175

        h = h + k * (a_h - e_h)
        a = a + k * (a_a - e_a)

        return h, a

    def analysis_game_data(self, game_data):
        self.elo_rate = {}
        for game in game_data:
            home = game["HomeTeam"]
            away = game["AwayTeam"]
            home_goal = game["FTHG"]
            away_goal = game["FTAG"]

            game["HomeRate"] = None
            game["AwayRate"] = None

            try:
                ph = float(game["PSCH"])
                pd = float(game["PSCD"])
                pa = float(game["PSCA"])
            except:
                continue

            if home not in self.elo_rate.keys():
                self.elo_rate[home] = {
                    "Div": game["Div"],
                    "rate": self.init_rate,
                    "count": 0
                }
            if away not in self.elo_rate.keys():
                self.elo_rate[away] = {
                    "Div": game["Div"],
                    "rate": self.init_rate,
                    "count": 0
                }
            h, a = self.calc_elo_rate_odds(self.elo_rate[home]["rate"], self.elo_rate[away]["rate"], ph, pd, pa)

            self.elo_rate[home]["rate"] = h
            self.elo_rate[home]["Div"] = game["Div"]
            self.elo_rate[home]["count"] += 1
            self.elo_rate[away]["rate"] = a
            self.elo_rate[away]["Div"] = game["Div"]
            self.elo_rate[away]["count"] += 1

            game["HomeRate"] = h
            game["AwayRate"] = a

        return game_data, self.elo_rate

    def setADList(self):
        arr = []
        arr.append(np.arange(self.min_diff, -220, self.diff_step))
        arr.append(np.arange(-220, 200, 5))
        arr.append(np.arange(200, self.max_diff, self.diff_step))
        self.AD = np.concatenate(arr)

    def getIndex(self, val):
        n = len(self.AD)
        if self.AD[n - 1] < val:
            return n - 1

        if self.AD[0] >= val:
            return 0

        for i in range(n - 1):
            if self.AD[i] < val <= self.AD[i + 1]:
                return i

    def analysis_game_data_detail(self, game_data):
        self.elo_rate = {}
        checked_game_data = []
        f = open("./prediction_data/elo_rate.csv", 'w', newline='')
        writer = csv.writer(f)
        row = ["Div", "Date", "HomeTeam", "AwayTeam", "HomeRate", "AwayRate"]
        writer.writerow(row)
        for game in game_data:
            home = game["HomeTeam"]
            away = game["AwayTeam"]
            home_goal = game["FTHG"]
            away_goal = game["FTAG"]

            try:
                ph = float(game["PSCH"])
                pd = float(game["PSCD"])
                pa = float(game["PSCA"])
            except:
                continue

            if home not in self.elo_rate.keys():
                self.elo_rate[home] = {
                    "Div": game["Div"],
                    "rate": self.init_rate,
                    "count": 0
                }
            if away not in self.elo_rate.keys():
                self.elo_rate[away] = {
                    "Div": game["Div"],
                    "rate": self.init_rate,
                    "count": 0
                }

            if self.elo_rate[home]["count"] > self.min_count and self.elo_rate[away]["count"] > self.min_count:
                diff = self.elo_rate[home]["rate"] - self.elo_rate[away]["rate"]
                index = self.getIndex(diff)
                if index not in self.rate_diff_dist.keys():
                    self.rate_diff_dist[index] = {
                        "games": [],
                        "home": 0,
                        "away": 0,
                        "draw": 0,
                        "over2.5": 0,
                        "count": 0,
                    }
                self.rate_diff_dist[index]["games"].append(game)
                self.rate_diff_dist[index]["count"] += 1
                if home_goal > away_goal:
                    self.rate_diff_dist[index]["home"] += 1
                elif home_goal == away_goal:
                    self.rate_diff_dist[index]["draw"] += 1
                else:
                    self.rate_diff_dist[index]["away"] += 1

                if home_goal + away_goal > 2.5:
                    self.rate_diff_dist[index]["over2.5"] += 1

                if game['FTR']:
                    game["HomeRate"] = self.elo_rate[home]["rate"]
                    game["AwayRate"] = self.elo_rate[away]["rate"]
                    game["Over2.5"] = 1 if home_goal + away_goal > 2.5 else 0
                    checked_game_data.append(game)

            h, a = self.calc_elo_rate_odds(self.elo_rate[home]["rate"], self.elo_rate[away]["rate"], ph, pd, pa)

            self.elo_rate[home]["rate"] = h
            self.elo_rate[home]["Div"] = game["Div"]
            self.elo_rate[home]["count"] += 1
            self.elo_rate[away]["rate"] = a
            self.elo_rate[away]["Div"] = game["Div"]
            self.elo_rate[away]["count"] += 1
            row = [game["Div"], game["Date"], game["HomeTeam"], game["AwayTeam"], h, a]
            writer.writerow(row)

        f.close()

        self.logisticReg(checked_game_data)

        return

        for index in range(len(self.AD)):
            if index in self.rate_diff_dist.keys():
                one = self.rate_diff_dist[index]
                hda = one["home"] + one["draw"] + one["away"]
                self.rate_diff_prob[index] = {
                    "home": one["home"] / hda,
                    "draw": one["draw"] / hda,
                    "away": one["away"] / hda,
                    "over2.5": one["over2.5"] / hda,
                    "count": one["count"],
                    "AD": int(self.AD[index])
                }
            else:
                self.rate_diff_prob[index] = {
                    "home": 0.33,
                    "draw": 0.33,
                    "away": 0.33,
                    "over2.5": 0.5,
                    "count": 0,
                    "AD": int(self.AD[index])
                }

        with open(f"./prediction_data/elo_rate_AD.json", 'w') as outfile:
            json.dump(self.rate_diff_prob, outfile, indent=2)

    def readAD(self):
        data = []
        try:
            with open("./prediction_data/elo_rate_AD.json") as json_file:
                data = json.load(json_file)
        except:
            pass

        return data

    def logisticReg(self, game_data):
        learning_data = pd.DataFrame(game_data)
        home_teams = np.sort(learning_data['HomeTeam'].unique())
        away_teams = np.sort(learning_data['AwayTeam'].unique())
        leagues = np.sort(learning_data['Div'].unique())

        i = 0
        home_team_map = {}
        for team in home_teams:
            home_team_map[team] = i
            i += 1

        i = 0
        away_team_map = {}
        for team in away_teams:
            away_team_map[team] = i
            i += 1

        i = 0
        league_map = {}
        for league in leagues:
            league_map[league] = i
            i += 1

        learning_data['HomeTeam'] = learning_data['HomeTeam'].map(home_team_map)
        learning_data['AwayTeam'] = learning_data['AwayTeam'].map(away_team_map)
        learning_data['Div'] = learning_data['Div'].map(league_map)
        learning_data['H'] = learning_data['FTR'].apply(lambda x: 1 if x == 'H' else 0)
        learning_data['D'] = learning_data['FTR'].apply(lambda x: 1 if x == 'D' else 0)
        learning_data['A'] = learning_data['FTR'].apply(lambda x: 1 if x == 'A' else 0)

        features = learning_data[['Div', 'HomeTeam', 'HomeRate', 'AwayTeam', 'AwayRate']]

        home_results = learning_data['H']
        draw_results = learning_data['D']
        away_results = learning_data['A']
        over_results = learning_data['Over2.5']

        scaler = preprocessing.StandardScaler()

        train_features = scaler.fit_transform(features)

        home_model = linear_model.LogisticRegression()
        draw_model = linear_model.LogisticRegression()
        away_model = linear_model.LogisticRegression()
        over_model = linear_model.LogisticRegression()

        home_model.fit(train_features, home_results)
        draw_model.fit(train_features, draw_results)
        away_model.fit(train_features, away_results)
        over_model.fit(train_features, over_results)

        print(home_model.score(train_features, home_results))
        print(home_model.coef_)
        print(draw_model.score(train_features, draw_results))
        print(draw_model.coef_)
        print(away_model.score(train_features, away_results))
        print(away_model.coef_)
        print(over_model.score(train_features, over_results))
        print(over_model.coef_)

        map_data = {
            "home_team_map": home_team_map,
            "away_team_map": away_team_map,
            "league_map": league_map
        }

        pickle.dump(home_model, open("./prediction_data/home_model.sav", "wb"))
        pickle.dump(draw_model, open("./prediction_data/draw_model.sav", "wb"))
        pickle.dump(away_model, open("./prediction_data/away_model.sav", "wb"))
        pickle.dump(over_model, open("./prediction_data/over_model.sav", "wb"))
        pickle.dump(map_data, open("./prediction_data/map_data.sav", "wb"))
        pickle.dump(scaler, open("./prediction_data/scaler.sav", "wb"))

    def calcHDAProb(self, div, home_team, home_rate, away_team, away_rate):
        home_model = pickle.load(open("./prediction_data/home_model.sav", 'rb'))
        draw_model = pickle.load(open("./prediction_data/draw_model.sav", 'rb'))
        away_model = pickle.load(open("./prediction_data/away_model.sav", 'rb'))
        over_model = pickle.load(open("./prediction_data/over_model.sav", 'rb'))
        map_data = pickle.load(open("./prediction_data/map_data.sav", 'rb'))
        scaler = pickle.load(open("./prediction_data/scaler.sav", "rb"))

        league_map = map_data["league_map"]
        home_team_map = map_data["home_team_map"]
        away_team_map = map_data["away_team_map"]

        test = np.array([league_map[div], home_team_map[home_team], home_rate, away_team_map[away_team], away_rate])
        test = np.array([test, ])
        test_features = scaler.transform(test)

        home = home_model.predict_proba(test_features)
        draw = draw_model.predict_proba(test_features)
        away = away_model.predict_proba(test_features)
        over = over_model.predict_proba(test_features)

        ph = home[0][1] / (home[0][1] + draw[0][1] + away[0][1])
        pd = draw[0][1] / (home[0][1] + draw[0][1] + away[0][1])
        pa = away[0][1] / (home[0][1] + draw[0][1] + away[0][1])

        print(home_team, away_team, draw)
        print(home_team, away_team, ph, pd, pa, over[0][1])
        ph1 = 1 / (1 + self.c ** ((away_rate - home_rate - self.w) / self.d))
        pa1 = 1 - ph1
        print(home_team, away_team, ph1 * draw[0][0], draw[0][1], pa1 * draw[0][0])

        return ph, pd, pa, over[0][1]
        # return ph1 * draw[0][0], draw[0][1], pa1 * draw[0][0], over[0][1]

    def calcXGModel(self, game_data):
        pass

    def overUnderXG(self, game_data):
        team_xG = {}
        league_avg_home_goal_list = []
        league_avg_away_goal_list = []

        for game in game_data:
            league_avg_home_goal_list.append(game["FTHG"])
            league_avg_away_goal_list.append(game["FTAG"])

        league_avg_home_goal = np.average(np.array(league_avg_home_goal_list))
        league_avg_away_goal = np.average(np.array(league_avg_away_goal_list))

        for game in game_data:
            game["PCAHHWM"] = None
            game["PCAHAWM"] = None
            game["TotalGoal"] = game["FTHG"] + game["FTAG"]
            game["TotalXG"] = None
            game["HomeAdvance"] = None
            game["HomeXG"] = None
            game["AwayXG"] = None

            if game["Over"] and game["Under"]:
                game["Over"], game["Under"] = self.calc_origin_over_under(game["Over"], game["Under"])

            if not game['Total']:
                game['Total'] = 2.5

            if game["PCAHH"] and game["PCAHA"]:
                game["PCAHHWM"], game["PCAHAWM"] = self.calc_origin_over_under(game["PCAHH"], game["PCAHA"])

            AHCh = game["AHCh"]

            if AHCh == 0:
                xG = 1 / game["Over"] * game['Total'] * 2

                home_xG = 1 / game["PCAHHWM"] * xG
                away_xG = 1 / game["PCAHAWM"] * xG

                game["TotalXG"] = xG
                game["HomeAdvance"] = home_xG - away_xG
                game["HomeXG"] = home_xG
                game["AwayXG"] = away_xG

            else:
                xG = (1 / game["Over"] + 0.5) * game['Total']

                h = 1 / game["PCAHHWM"] * 2.
                a = 1 / game["PCAHAWM"] * 2.

                if game["PCAHHWM"] <= 2.:
                    if AHCh < 0.:
                        home_advance = - AHCh * h
                    elif AHCh == 0.:
                        home_advance = 2. - game["PCAHHWM"]
                    else:
                        home_advance = - AHCh * a
                else:
                    if AHCh < 0.:
                        home_advance = - AHCh * h
                    elif AHCh == 0.:
                        home_advance = - (2. - game["PCAHAWM"])
                    else:
                        home_advance = - AHCh * a

                home_xG = xG / 2. + home_advance / 2.
                away_xG = xG / 2. - home_advance / 2.
                game["TotalXG"] = xG
                game["HomeAdvance"] = home_advance
                game["HomeXG"] = home_xG
                game["AwayXG"] = away_xG

            if game["HomeTeam"] not in team_xG.keys():
                team_xG[game["HomeTeam"]] = {
                    "home":{
                        "score":[],
                        "conceded":[]
                    },
                    "away":{
                        "score":[],
                        "conceded":[]
                    },
                }

            if game["AwayTeam"] not in team_xG.keys():
                team_xG[game["AwayTeam"]] = {
                    "home":{
                        "score":[],
                        "conceded":[]
                    },
                    "away":{
                        "score":[],
                        "conceded":[]
                    },
                }
            team_xG[game["HomeTeam"]]["home"]["score"].append(home_xG)
            team_xG[game["HomeTeam"]]["home"]["conceded"].append(away_xG)
            team_xG[game["AwayTeam"]]["away"]["score"].append(away_xG)
            team_xG[game["AwayTeam"]]["away"]["conceded"].append(home_xG)

            team_xG[game["HomeTeam"]]["home_avg_goal"] = np.average(np.array(team_xG[game["HomeTeam"]]["home"]["score"]))
            team_xG[game["HomeTeam"]]["home_attack"] = np.average(np.array(team_xG[game["HomeTeam"]]["home"]["score"])) / league_avg_home_goal
            team_xG[game["HomeTeam"]]["home_avg_conceded"] = np.average(np.array(team_xG[game["HomeTeam"]]["home"]["conceded"]))
            team_xG[game["HomeTeam"]]["home_defence"] = np.average(np.array(team_xG[game["HomeTeam"]]["home"]["conceded"])) / league_avg_away_goal
            team_xG[game["AwayTeam"]]["away_avg_goal"] = np.average(np.array(team_xG[game["AwayTeam"]]["away"]["score"]))
            team_xG[game["AwayTeam"]]["away_attack"] = np.average(np.array(team_xG[game["AwayTeam"]]["away"]["score"])) / league_avg_away_goal
            team_xG[game["AwayTeam"]]["away_avg_conceded"] = np.average(np.array(team_xG[game["AwayTeam"]]["away"]["conceded"]))
            team_xG[game["AwayTeam"]]["away_defence"] = np.average(np.array(team_xG[game["AwayTeam"]]["away"]["conceded"])) / league_avg_home_goal

            game["HomeGoal"] = team_xG[game["HomeTeam"]]["home_attack"] * team_xG[game["AwayTeam"]]["away_defence"] * league_avg_home_goal
            game["AwayGoal"] = team_xG[game["AwayTeam"]]["away_attack"] * team_xG[game["HomeTeam"]]["home_defence"] * league_avg_away_goal
            game["HomeAttack"] = team_xG[game["HomeTeam"]]["home_attack"]
            game["HomeDefence"] = team_xG[game["HomeTeam"]]["home_defence"]
            game["AwayAttack"] = team_xG[game["AwayTeam"]]["away_attack"]
            game["AwayDefence"] = team_xG[game["AwayTeam"]]["away_defence"]

        # for key in team_xG.keys():
        #     team_xG[key]["home_avg_goal"] = np.average(np.array(team_xG[key]["home"]["score"]))
        #     team_xG[key]["home_attack"] = np.average(np.array(team_xG[key]["home"]["score"]))/league_avg_home_goal
        #     team_xG[key]["home_avg_conceded"] = np.average(np.array(team_xG[key]["home"]["conceded"]))
        #     team_xG[key]["home_defence"] = np.average(np.array(team_xG[key]["home"]["conceded"]))/league_avg_away_goal
        #     team_xG[key]["away_avg_goal"] = np.average(np.array(team_xG[key]["away"]["score"]))
        #     team_xG[key]["away_attack"] = np.average(np.array(team_xG[key]["away"]["score"]))/league_avg_away_goal
        #     team_xG[key]["away_avg_conceded"] = np.average(np.array(team_xG[key]["away"]["conceded"]))
        #     team_xG[key]["away_defence"] = np.average(np.array(team_xG[key]["away"]["conceded"]))/league_avg_home_goal
        # print(team_xG)


        league_avg = {
            "home_goal": league_avg_home_goal,
            "away_goal": league_avg_away_goal
        }

        return game_data, team_xG, league_avg
