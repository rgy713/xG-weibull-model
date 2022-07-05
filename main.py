import datetime
import difflib
import multiprocessing
import os
import threading
import time
import unicodedata

import numpy as np
import csv
from PySide2 import QtCore
import requests
import json
import winsound
from lxml import html

from ELO import ELO
from football_weibull import WeibullPrediction


class PiRatingSystem(QtCore.QObject):
    c = 3.
    b = 10.

    lamda = 0.091
    gamma = 0.61

    update_game_count = 10

    lamda_s = np.arange(0.000, 0.2, 0.002)
    gamma_s = np.arange(0.00, 1, 0.01)

    lamda_count = np.size(lamda_s)
    gamma_count = np.size(gamma_s)

    lamda_m = np.tile(lamda_s, (gamma_count, 1))
    gamma_m = np.tile(np.array([gamma_s]).T, lamda_count)

    delta = 2.5
    myu = 0.01
    phi = 1

    x_max = 2.1
    dx = 0.1
    rank_count = 42
    x_min = x_max - dx * (rank_count - 2)

    disc_min = 1
    disc_max = 40

    disc_level = disc_max - disc_min + 1

    max_goal = 7

    min_game_count = 0

    scrap_base_url = 'https://www.football-data.co.uk/mmz4281'
    league_list = ['E0', 'E1', 'E2', 'E3', 'EC', 'SC0', 'SC1', 'SC2', 'SC3', 'D1', 'D2', 'I1', 'I2',
                   'SP1', 'SP2', 'F1', 'F2', 'N1', 'B1', 'P1', 'T1', 'G1']
    data_directory = 'data'
    result_directory = 'result'
    COL_NAMES = ['Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Referee',
                 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'B365H', 'B365D', 'B365A',
                 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD',
                 'VCA',
                 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA', 'B365>2.5', 'B365<2.5', 'P>2.5', 'P<2.5', 'Max>2.5',
                 'Max<2.5', 'Avg>2.5', 'Avg<2.5', 'AHh', 'B365AHH', 'B365AHA', 'PAHH', 'PAHA', 'MaxAHH', 'MaxAHA',
                 'AvgAHH', 'AvgAHA', 'B365CH', 'B365CD', 'B365CA', 'BWCH', 'BWCD', 'BWCA', 'IWCH', 'IWCD', 'IWCA',
                 'PSCH', 'PSCD', 'PSCA', 'WHCH', 'WHCD', 'WHCA', 'VCCH', 'VCCD', 'VCCA', 'MaxCH', 'MaxCD', 'MaxCA',
                 'AvgCH', 'AvgCD', 'AvgCA', 'B365C>2.5', 'B365C<2.5', 'PC>2.5', 'PC<2.5', 'MaxC>2.5', 'MaxC<2.5',
                 'AvgC>2.5', 'AvgC<2.5', 'AHCh', 'B365CAHH', 'B365CAHA', 'PCAHH', 'PCAHA', 'MaxCAHH', 'MaxCAHA',
                 'AvgCAHH', 'AvgCAHA']

    NEED_COL_NAMES = ['Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                      'PSH', 'PSD', 'PSA', 'PSCH', 'PSCD', 'PSCA', 'PCAHH', 'PCAHA', 'AHh', 'AHCh', 'PC>2.5', 'PC<2.5']

    READ_COL_NAMES = ['Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                      'PSH', 'PSD', 'PSA', 'PSCH', 'PSCD', 'PSCA', 'PCAHH', 'PCAHA', 'AHh', 'AHCh', 'Over', 'Under', 'Total']

    COL_NAMES_PART = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']

    LEARNED_FILE_COL = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'PSH', 'PSD', 'PSA', 'PSCH', 'PSCD', 'PSCA', 'PCAHH', 'PCAHA', 'AHCh', 'Over', 'Under', 'Total', 'Season', 'PCAHHWM', 'PCAHAWM', 'TotalGoal', 'TotalXG', 'HomeAdvance', 'HomeXG', 'AwayXG', 'HomeGoal', 'AwayGoal', 'HomeAttack', 'HomeDefence', 'AwayAttack', 'AwayDefence']

    # data file name
    learning_data_file_name = 'learning_data.csv'
    learning_result_file_name = 'learning_history.csv'

    last_rating_file_name = "last_rating.csv"
    GH_GA_file_name = "GH_GA.csv"
    AD_P_result_file_name = "AD_P.csv"

    prediction_data_file_name = 'odds.csv'
    prediction_result_file_name = 'odds_prediction.csv'

    learning_data = None

    team_list_info = None
    rate_error = None
    all_game_count = None

    new_team_adjust = 5
    now_year = 2020

    game_AD_list = {}
    coef_AD = {}
    AD_data = []

    PSH_list = {}
    PSD_list = {}
    PSA_list = {}
    margin_list = {}
    is_read_excel = False
    is_run = False
    excel_data = []
    track_game = {}
    selected_teams = []
    excel_time_interval = 2
    track_time_interval = 10

    pushLog = QtCore.Signal(str)
    finished = QtCore.Signal(str)
    readExcel = QtCore.Signal()
    signalSelectedTeams = QtCore.Signal()
    league_teams = {}
    league_avg = {}

    def __init__(self):
        QtCore.QObject.__init__(self)

        self.WP = WeibullPrediction()

        self.getTeams()

        now = datetime.datetime.now()
        start_day = datetime.datetime.strptime(f"{now.year}-08-01", "%Y-%m-%d")
        if now > start_day:
            self.now_year = now.year + 1
        else:
            self.now_year = now.year

    def sendLog(self, msg):
        self.emit(QtCore.SIGNAL("pushLog(QString)"), msg)

    def convert_date(self, date):
        try:
            date_ = date.split("/")
            pre_year = "20" if int(date_[2]) < 50 else "19"
            date_[2] = date_[2] if len(date_[2]) == 4 else pre_year + date_[2]
            date_[1] = date_[1] if len(date_[1]) == 2 else "0" + date_[1]
            date_[0] = date_[0] if len(date_[0]) == 2 else "0" + date_[0]
            date = date_[2] + "-" + date_[1] + "-" + date_[0]
        except:
            pass
        return date

    def getOldGames(self, file_name):
        try:
            data_file = open(file_name, 'r')
        except:
            return []

        rows = csv.reader(data_file)
        old_games = []
        for row in rows:
            try:
                old_games.append(f"{row[1]}-{row[3]}-{row[4]}")
            except:
                continue

        return old_games

    def basic_league_scrapping(self, leagues):
        for league_name in leagues:
            directory = f"{self.data_directory}/{league_name}"
            self.sendLog(f"start scrapping {league_name}...")
            if not os.path.exists(directory):
                os.makedirs(directory)
            for year in range(self.now_year - 1, self.now_year):
                try:
                    season = f"{year}-{year + 1}"
                    data_file_name = f"{directory}/{season}.csv"
                    old_games = self.getOldGames(data_file_name)

                    url = self.scrap_base_url + f"/{str(year)[2:]}{str(year + 1)[2:]}/{league_name}.csv"

                    content = requests.get(url)

                    with open(data_file_name, 'a', newline='') as f:
                        writer = csv.writer(f)
                        reader = csv.reader(content.text.splitlines())
                        is_read = False
                        col_dict = {}
                        for row in reader:
                            try:
                                game_idx = f"{row[1]}-{row[3]}-{row[4]}"
                                if row[0] == league_name or row[0] == "Div":
                                    if row[0] == "Div":
                                        header = []
                                        for item in self.NEED_COL_NAMES:
                                            index = row.index(item)
                                            col_dict[item] = index
                                            if item == 'PC>2.5':
                                                item = 'Over'
                                            if item == 'PC<2.5':
                                                item = 'Under'
                                            header.append(item)
                                        header.append('Total')
                                        if game_idx not in old_games:
                                            writer.writerow(header)
                                    elif game_idx not in old_games:
                                        line = []
                                        for item in self.NEED_COL_NAMES:
                                            line.append(row[col_dict[item]])
                                        line.append(2.5)
                                        writer.writerow(line)
                                    is_read = True
                            except Exception as e:
                                print(e)
                                pass

                        if (not is_read) and len(old_games) == 0:
                            writer.writerow(self.READ_COL_NAMES)

                    self.sendLog(f"scraped {data_file_name}")

                except Exception as e:
                    print(e)
                    pass

        self.sendLog("success scrapping game data file.")

        return

    def fixtures_json(self):
        data = {}
        try:
            with open("./prediction_data/fixtures.json") as json_file:
                data = json.load(json_file)
        except:
            pass

        return data

    def normalize(self, s):
        def normalize_char(c):
            try:
                cname = unicodedata.name(c)
                cname = cname[:cname.index(' WITH')]
                return unicodedata.lookup(cname)
            except (ValueError, KeyError):
                return c

        return ''.join(normalize_char(c) for c in s)

    def scrape_fixtures(self, leagues):
        fixtures_json = self.fixtures_json()
        for league in leagues:
            self.sendLog(f"start scrape {league} fixtures")
            if league not in fixtures_json:
                self.sendLog(
                    f"_MSG_No new season data. Please add fixtures of new season in './data/{league}/fixtures.csv'")
                continue

            url = fixtures_json[league]
            fixtures = []
            response = requests.get(url)
            content = html.fromstring(response.text)
            tr_list = content.xpath('.//table[@class="schedule-table"]/tr')
            for tr in tr_list:
                try:
                    date = tr.xpath(".//td[1]/text()")[0]
                    home = tr.xpath(".//td[2]/a/text()")[0]
                    hm = tr.xpath(".//td[3]/strong/text()")[0]
                    away = tr.xpath(".//td[4]/a/text()")[0]

                    date_time = f"{date} {hm}"
                    one = {
                        "Div": league,
                        "DateTime": date_time,
                        "HomeTeam": self.normalize(home),
                        "AwayTeam": self.normalize(away)
                    }
                    fixtures.append(one)
                except:
                    pass

            data_file_name = f"data/{league}/fixtures.csv"
            old_game_idx = []
            new_fixtures = []
            now = datetime.datetime.now()

            try:
                data_file = open(data_file_name, 'r')
                rows = csv.reader(data_file)
                first = True

                for row in rows:
                    if first:
                        first = False
                        continue

                    one = {
                        "Div": row[0],
                        "DateTime": row[1],
                        "HomeTeam": row[2],
                        "AwayTeam": row[3]
                    }
                    game_date = datetime.datetime.strptime(self.time_convert_fixtures(row[1]), "%Y-%m-%d %H:%M")
                    if now < game_date:
                        new_fixtures.append(one)
                        old_game_idx.append(f"{row[1]}-{row[2]}-{row[3]}")
            except:
                pass

            for one_game in fixtures:
                game_date = datetime.datetime.strptime(self.time_convert_fixtures(one_game["DateTime"]),
                                                       "%Y-%m-%d %H:%M")
                idx = f"{one_game['DateTime']}-{one_game['HomeTeam']}-{one_game['AwayTeam']}"
                if now < game_date and idx not in old_game_idx:
                    new_fixtures.append(one_game)

            if len(new_fixtures) == 0:
                new_fixtures.append(
                    {
                        "Div": None,
                        "DateTime": None,
                        "HomeTeam": None,
                        "AwayTeam": None
                    }
                )

            self.saveCsv(new_fixtures, f"./data/{league}/fixtures.csv")
            self.sendLog(f"end scrape {league} fixtures")
        self.sendLog(f"success scrape fixtures")

    def time_convert_fixtures(self, t):
        t_ = datetime.datetime.strptime(t, "%d.%m.%Y %H:%M")
        offset = datetime.datetime.now() - datetime.datetime.utcnow()
        return (t_ + offset).strftime("%Y-%m-%d %H:%M")

    def get_fixtures(self, leagues):
        game_data = []
        for league in leagues:
            data_file_name = f"data/{league}/fixtures.csv"
            try:
                data_file = open(data_file_name, 'r')
            except:
                continue
            rows = csv.reader(data_file)
            first = True

            for row in rows:
                if first:
                    first = False
                    continue

                one = {
                    "Div": row[0],
                    "StartDate": self.time_convert_fixtures(row[1]),
                    "HomeTeam": row[2],
                    "AwayTeam": row[3]
                }

                game_data.append(one)

        return game_data

    def get_int(self, val):
        try:
            val = int(val)
        except:
            val = 0
        return val

    def get_float(self, val):
        try:
            val = float(val)
        except:
            val = 0
        return val

    def calc_RPS(self, p, e):
        """
        calculation RPS
        :param p: the predicted outcome
        :param e: the observed outcome
        :return:
        """
        r = len(p)
        rps = 0.
        for i in range(r - 1):
            rps += (np.sum(p[:i + 1]) - np.sum(e[:i + 1])) ** 2

        rps = rps / (r - 1)

        return rps

    def calc_pr(self, br, phi_c):
        myu_ = self.myu
        if phi_c < 0:
            myu_ = - self.myu

        pr = br + myu_ * (phi_c - self.phi) / (np.abs(phi_c - self.phi) ** self.delta)

        return pr

    def calc_P(self, GH, GA):
        P = np.zeros(3)
        for gh in range(self.max_goal + 1):
            for ga in range(self.max_goal + 1):
                p1 = 1 if gh > ga else 0
                p2 = 1 if gh == ga else 0
                p3 = 1 if gh < ga else 0

                P += np.array([p1, p2, p3]) * GH[gh] * GA[ga]

        return P

    def calc_over_under_performance(self, GD, phi_c_H, phi_c_A):
        # calculating continuous over/under-performance
        if GD > 0:
            if phi_c_H >= 0:
                phi_c_H += 1
            elif phi_c_H < 0:
                phi_c_H = 1
            if phi_c_A >= 0:
                phi_c_A = -1
            elif phi_c_A < 0:
                phi_c_A -= 1
        elif GD < 0:
            if phi_c_H >= 0:
                phi_c_H = -1
            elif phi_c_H < 0:
                phi_c_H -= 1
            if phi_c_A >= 0:
                phi_c_A += 1
            elif phi_c_A < 0:
                phi_c_A = 1

        return phi_c_H, phi_c_A

    def calc_AD(self, RD):
        if RD > self.x_max:
            AD = self.rank_count - 1
        elif RD < self.x_min:
            AD = 0
        else:
            AD = int(np.round((np.ceil(RD * 10) / 10 - self.x_min) / self.dx))

        return AD

    def predict_goal_diff(self, R_aH, R_bA):
        # step 1
        g_DH_hat = np.sign(R_aH) * (self.b ** (np.abs(R_aH) / self.c) - 1)
        g_DA_hat = np.sign(R_bA) * (self.b ** (np.abs(R_bA) / self.c) - 1)

        # step 2
        g_D_hat = g_DH_hat - g_DA_hat

        return g_D_hat

    def calc_rating(self, R_aH, R_aA, R_bH, R_bA, FTHG, FTAG):
        """
        pi-rating caculation function
        :param R_aH:
        :param R_aA:
        :param R_bH:
        :param R_bA:
        :param FTHG:
        :param FTAG:
        :return:
        """
        # step 1
        # step 2
        g_D_hat = self.predict_goal_diff(R_aH, R_bA)

        # step 3
        g_D = FTHG - FTAG

        # step 4
        e = np.abs(g_D - g_D_hat)

        # step 5
        psi = self.c * np.log10(1. + e)

        psi_H = np.sign(g_D - g_D_hat) * psi
        psi_A = np.sign(g_D_hat - g_D) * psi

        # step 6
        R_aH_hat = R_aH + psi_H * self.lamda
        R_aA_hat = R_aA + (R_aH_hat - R_aH) * self.gamma

        R_bA_hat = R_bA + psi_A * self.lamda
        R_bH_hat = R_bH + (R_bA_hat - R_bA) * self.gamma

        return R_aH_hat, R_aA_hat, R_bH_hat, R_bA_hat

    def read_GH_GA(self):
        GH = np.zeros((self.rank_count, self.max_goal + 1))
        GA = np.zeros((self.rank_count, self.max_goal + 1))

        GH_GA_file = open(f"{self.result_directory}/{self.GH_GA_file_name}", 'r')
        rows = csv.reader(GH_GA_file)

        first = True
        for row in rows:

            if first:
                first = False
                continue
            for i in range(self.max_goal + 1):
                GH[int(row[0]), i] = float(row[i + 1])
                GA[int(row[0]), i] = float(row[i + 1 + self.max_goal + 1])

        return GH, GA

    def write_last_rating(self, team_list):
        last_result_file = open(f"{self.result_directory}/{self.last_rating_file_name}", 'w', newline='')
        writer = csv.writer(last_result_file)
        writer.writerow(["Date", "Team", "R_H", "R_A", "phi_c", "GD", "Count"])

        for team in team_list.keys():
            team_info = team_list[team]
            writer.writerow(
                [team_info["Date"], team, team_info["R_H"], team_info["R_A"], team_info["phi_c"], team_info["GD"],
                 team_info["count"]])

        last_result_file.close()

    def read_last_rating(self, ):
        team_list = {}

        last_rating_file = open(f"{self.result_directory}/{self.last_rating_file_name}", 'r')
        rows = csv.reader(last_rating_file)

        first = True
        for row in rows:

            if first:
                first = False
                continue

            team_list[row[1]] = {
                "R_H": float(row[2]),
                "R_A": float(row[3]),
                "phi_c": int(row[4]),
                "GD": int(row[5]),
                "Date": row[0]
            }

        return team_list

    def calc_rating_matrix(self, R_aH, R_aA, R_bH, R_bA, FTHG, FTAG):
        # step 1
        g_DH_hat = np.sign(R_aH) * (self.b ** (np.abs(R_aH) / self.c) - 1)
        g_DA_hat = np.sign(R_bA) * (self.b ** (np.abs(R_bA) / self.c) - 1)

        # step 2
        g_D_hat = g_DH_hat - g_DA_hat

        # step 3
        g_D = FTHG - FTAG

        # step 4
        e = np.abs(g_D - g_D_hat)

        # step 5
        psi = self.c * np.log10(1. + e)

        psi_H = np.sign(g_D - g_D_hat) * psi
        psi_A = np.sign(g_D_hat - g_D) * psi

        # step 6
        R_aH_hat = R_aH + psi_H * self.lamda_m
        R_aA_hat = R_aA + (R_aH_hat - R_aH) * self.gamma_m

        R_bA_hat = R_bA + psi_A * self.lamda_m
        R_bH_hat = R_bH + (R_bA_hat - R_bA) * self.gamma_m

        return R_aH_hat, R_aA_hat, R_bH_hat, R_bA_hat, e

    def ADData(self, HDA="Home"):
        file = f"./prediction_data/AD_{HDA}.json"
        data = []
        try:
            with open(file) as json_file:
                data = json.load(json_file)
        except:
            pass

        return data

    def saveADData(self, data, HDA="Home"):
        with open(f"./prediction_data/AD_{HDA}.json", 'w') as outfile:
            json.dump(data, outfile, indent=2)

    def getAD(self, HDA="Home"):
        data = self.ADData(HDA)
        arr = []
        for one in data:
            arr.append(np.arange(one['start'], one['end'], one['step']))
        AD = np.concatenate(arr)

        return AD

    def getIndex(self, AD, val):
        n = len(AD)
        if AD[n - 1] < val:
            return n - 1

        if AD[0] >= val:
            return 0

        for i in range(n - 1):
            if AD[i] < val <= AD[i + 1]:
                return i

    def saveCsv(self, data, file_path):
        f = open(file_path, 'w', newline='')
        writer = csv.writer(f)
        first = True
        for one in data:
            if first:
                writer.writerow(list(one.keys()))
                first = False
            writer.writerow(list(one.values()))
        f.close()

    def parseGameData(self, AD, data, open_idx='PSH', close_idx='PSCH'):
        paser_data = {}
        for game in data:
            try:
                open_odds = float(game[open_idx])
            except:
                continue
            try:
                close_odds = float(game[close_idx])
            except:
                continue

            idx = self.getIndex(AD, open_odds)
            if idx in paser_data.keys():
                paser_data[idx].append(game)
            else:
                paser_data[idx] = [game]
        return paser_data

    def saveParseData(self, leagues, data, HDA="Home"):
        result_file = open(f"result\Top{len(leagues)}_Leagues_Parse_Data_{HDA}.csv", 'w', newline='')
        writer = csv.writer(result_file)
        writer.writerow(
            ["AD",
             "Count",
             "Interval",
             "BelowOpen%",
             "AboveOpen%",
             "DiffAverage",
             "Deviation",
             "Deviation%",
             "Home%",
             "Draw%",
             "Away%"
             ])

        for one in data:
            writer.writerow(
                [
                    one["AD"],
                    one["Count"],
                    one["Interval"],
                    one["BelowOpen%"],
                    one["AboveOpen%"],
                    one["DiffAverage"],
                    one["Deviation"],
                    one["Deviation%"],
                    one["Home"],
                    one["Draw"],
                    one["Away"],
                ]
            )

        result_file.close()

    def getADData(self, AD, parse_data, open_idx="PSH", close_idx="PSCH"):
        AD_data = []
        keys = sorted(parse_data.keys())
        for key in keys:
            n = len(parse_data[key])
            if n == 0:
                continue
            diff_open_close_list_home = []
            diff_open_close_list_away = []
            diff_open_close_list_draw = []
            open_odds_list_home = []
            open_odds_list_away = []
            open_odds_list_draw = []
            home_count = 0
            draw_count = 0
            away_count = 0
            for game in parse_data[key]:
                open_odds_home = float(game['PSH'])
                close_odds_home = float(game['PSCH'])
                open_odds_away = float(game['PSA'])
                close_odds_away = float(game['PSCA'])
                open_odds_draw = float(game['PSD'])
                close_odds_draw = float(game['PSCD'])
                diff_open_close_list_home.append(open_odds_home - close_odds_home)
                diff_open_close_list_away.append(open_odds_away - close_odds_away)
                diff_open_close_list_draw.append(open_odds_draw - close_odds_draw)
                open_odds_list_home.append(open_odds_home)
                open_odds_list_away.append(open_odds_away)
                open_odds_list_draw.append(open_odds_draw)
                if game["FTR"] == "H":
                    home_count += 1
                if game["FTR"] == "D":
                    draw_count += 1
                if game["FTR"] == "A":
                    away_count += 1

            psh_avg = np.average(diff_open_close_list_home)
            psa_avg = np.average(diff_open_close_list_away)
            psd_avg = np.average(diff_open_close_list_draw)
            psh_std = np.std(diff_open_close_list_home)
            psa_std = np.std(diff_open_close_list_away)
            psd_std = np.std(diff_open_close_list_draw)
            odds_avg_home = np.average(open_odds_list_home)
            odds_avg_away = np.average(open_odds_list_away)
            odds_avg_draw = np.average(open_odds_list_draw)
            interval = f"{round(AD[key - 1], 3)}~{round(AD[key], 3)}" if key > 0 else f"~{AD[key]}"
            # print(
            #     f"AD:{key}, Game Count:{n}, Interval:{interval}, Below Open Count:{below_open_count}, Above Open Count:{above_open_count}, Average:{psh_avg}, Deviation:{psh_std}, Deviation%,{psh_std / odds_avg * 100} Home:{home_count / n * 100}%, Draw:{draw_count / n * 100}%, Away:{away_count / n * 100}%")
            one = {
                "AD": key,
                "Count": n,
                "Interval": interval,
                "HomeDiffAverage": psh_avg,
                "HomeDeviation": psh_std,
                "HomeDeviation%": psh_std / odds_avg_home * 100,
                "DrawDiffAverage": psd_avg,
                "DrawDeviation": psd_std,
                "DrawDeviation%": psd_std / odds_avg_draw * 100,
                "AwayDiffAverage": psa_avg,
                "AwayDeviation": psa_std,
                "AwayDeviation%": psa_std / odds_avg_away * 100,
                "Home": home_count / n * 100,
                "Draw": draw_count / n * 100,
                "Away": away_count / n * 100
            }
            AD_data.append(one)
        return AD_data

    def getADRow(self, open_odds):
        ad = False
        for one_ad in self.AD_data:
            interval_arr = one_ad['Interval'].split('~')
            s_int = float(interval_arr[0]) if interval_arr[0] else 0
            e_int = float(interval_arr[1]) if interval_arr[1] else 100
            if s_int <= open_odds <= e_int:
                ad = one_ad
                break
        return ad

    def runAlertSound(self):
        winsound.PlaySound("./asset/alert.wav", winsound.SND_FILENAME)

    def getTeamInfo(self, league, team, start_year, end_year):
        game_data = []
        col_dict = {}
        for year in range(start_year, end_year + 1):
            season = f"{year}-{year + 1}"
            data_file_name = f"data/{league}/{season}.csv"
            try:
                data_file = open(data_file_name, 'r')
            except:
                continue
            rows = csv.reader(data_file)
            first = True

            for row in rows:
                if first:
                    for col in self.READ_COL_NAMES:
                        try:
                            col_idx = row.index(col)
                        except:
                            col_idx = None

                        col_dict[col] = col_idx
                    first = False
                    continue

                date = row[col_dict["Date"]]
                date_arr = date.split("/")
                try:
                    if len(date_arr[2]) == 2:
                        date = f"{date_arr[0]}/{date_arr[1]}/20{date_arr[2]}"
                except:
                    continue

                try:
                    one = {
                        "Div": row[col_dict["Div"]],
                        "Date": date,
                        "HomeTeam": row[col_dict["HomeTeam"]].lstrip(),
                        "AwayTeam": row[col_dict["AwayTeam"]].lstrip(),
                        "FTHG": int(row[col_dict["FTHG"]]),
                        "FTAG": int(row[col_dict["FTAG"]]),
                        "FTR": row[col_dict["FTR"]],
                        "PSH": row[col_dict["PSH"]] if col_dict["PSH"] else None,
                        "PSD": row[col_dict["PSD"]] if col_dict["PSD"] else None,
                        "PSA": row[col_dict["PSA"]] if col_dict["PSA"] else None,
                        "PSCH": row[col_dict["PSCH"]] if col_dict["PSCH"] else None,
                        "PSCD": row[col_dict["PSCD"]] if col_dict["PSCD"] else None,
                        "PSCA": row[col_dict["PSCA"]] if col_dict["PSCA"] else None,
                        "PCAHH": float(row[col_dict["PCAHH"]]) if col_dict["PCAHH"] else None,
                        "PCAHA": float(row[col_dict["PCAHA"]]) if col_dict["PCAHA"] else None,
                        "AHCh": float(row[col_dict["AHCh"]]) if col_dict["AHCh"] else None,
                        "Over": float(row[col_dict["Over"]]) if col_dict["Over"] else None,
                        "Under": float(row[col_dict["Under"]]) if col_dict["Under"] else None,
                        "Total": float(row[col_dict["Total"]]) if col_dict["Total"] else None,
                        "Season": season
                    }
                    if one["HomeTeam"] == team or one["AwayTeam"] == team:
                        game_data.append(one)
                except:
                    continue
        return game_data

    def getLeagueData(self, leagues, start_year, end_year):
        game_data = []
        col_dict = {}
        for league in leagues:
            for year in range(start_year, end_year + 1):
                season = f"{year}-{year + 1}"
                data_file_name = f"data/{league}/{season}.csv"
                try:
                    data_file = open(data_file_name, 'r')
                except:
                    continue
                rows = csv.reader(data_file)
                first = True

                for row in rows:
                    if first:
                        for col in self.READ_COL_NAMES:
                            try:
                                col_idx = row.index(col)
                            except:
                                col_idx = None

                            col_dict[col] = col_idx
                        first = False
                        continue

                    date = row[col_dict["Date"]]
                    date_arr = date.split("/")
                    try:
                        if len(date_arr[2]) == 2:
                            date = f"{date_arr[0]}/{date_arr[1]}/20{date_arr[2]}"
                    except:
                        continue

                    try:
                        one = {
                            "Div": row[col_dict["Div"]],
                            "Date": date,
                            "HomeTeam": row[col_dict["HomeTeam"]].lstrip(),
                            "AwayTeam": row[col_dict["AwayTeam"]].lstrip(),
                            "FTHG": int(row[col_dict["FTHG"]]),
                            "FTAG": int(row[col_dict["FTAG"]]),
                            "FTR": row[col_dict["FTR"]],
                            "PSH": row[col_dict["PSH"]] if col_dict["PSH"] else None,
                            "PSD": row[col_dict["PSD"]] if col_dict["PSD"] else None,
                            "PSA": row[col_dict["PSA"]] if col_dict["PSA"] else None,
                            "PSCH": row[col_dict["PSCH"]] if col_dict["PSCH"] else None,
                            "PSCD": row[col_dict["PSCD"]] if col_dict["PSCD"] else None,
                            "PSCA": row[col_dict["PSCA"]] if col_dict["PSCA"] else None,
                            "PCAHH": float(row[col_dict["PCAHH"]]) if col_dict["PCAHH"] else None,
                            "PCAHA": float(row[col_dict["PCAHA"]]) if col_dict["PCAHA"] else None,
                            "AHCh": float(row[col_dict["AHCh"]]) if col_dict["AHCh"] else None,
                            "Over": float(row[col_dict["Over"]]) if col_dict["Over"] else None,
                            "Under": float(row[col_dict["Under"]]) if col_dict["Under"] else None,
                            "Total": float(row[col_dict["Total"]]) if col_dict["Total"] else None,
                            "Season": season
                        }

                        game_data.append(one)
                    except:
                        continue
        return game_data

    def getAllLeagueData(self):
        game_data = []
        col_dict = {}
        for league in self.league_list:
            for year in range(2000, 2020):
                season = f"{year}-{year + 1}"
                data_file_name = f"data/{league}/{season}.csv"
                try:
                    data_file = open(data_file_name, 'r')
                except:
                    continue
                rows = csv.reader(data_file)
                first = True

                for row in rows:
                    if first:
                        for col in self.COL_NAMES_PART:
                            try:
                                col_idx = row.index(col)
                            except:
                                continue

                            col_dict[col] = col_idx
                        first = False
                        continue

                    date = row[col_dict["Date"]]
                    date_arr = date.split("/")
                    try:
                        if len(date_arr[2]) == 2:
                            date = f"{date_arr[0]}/{date_arr[1]}/20{date_arr[2]}"
                    except:
                        continue

                    try:
                        one = {
                            "Div": row[col_dict["Div"]],
                            "Date": date,
                            "HomeTeam": row[col_dict["HomeTeam"]].lstrip(),
                            "AwayTeam": row[col_dict["AwayTeam"]].lstrip(),
                            "FTHG": int(row[col_dict["FTHG"]]),
                            "FTAG": int(row[col_dict["FTAG"]]),
                            "Season": season
                        }

                        game_data.append(one)
                    except:
                        continue
        return game_data

    def getReadCsv(self, file_path):
        game_data = []
        col_dict = {}
        data_file = open(file_path, 'r')
        rows = csv.reader(data_file)
        first = True
        for row in rows:
            if first:
                for col in self.LEARNED_FILE_COL:
                    try:
                        col_idx = row.index(col)
                    except:
                        col_idx = None

                    col_dict[col] = col_idx
                first = False
                continue

            try:
                one = {
                    "Div": row[col_dict["Div"]],
                    "Date": row[col_dict["Date"]],
                    "HomeTeam": row[col_dict["HomeTeam"]].lstrip(),
                    "AwayTeam": row[col_dict["AwayTeam"]].lstrip(),
                    "FTHG": int(row[col_dict["FTHG"]]),
                    "FTAG": int(row[col_dict["FTAG"]]),
                    "FTR": row[col_dict["FTR"]],
                    "PSH": row[col_dict["PSH"]] if col_dict["PSH"] else None,
                    "PSD": row[col_dict["PSD"]] if col_dict["PSD"] else None,
                    "PSA": row[col_dict["PSA"]] if col_dict["PSA"] else None,
                    "PSCH": row[col_dict["PSCH"]] if col_dict["PSCH"] else None,
                    "PSCD": row[col_dict["PSCD"]] if col_dict["PSCD"] else None,
                    "PSCA": row[col_dict["PSCA"]] if col_dict["PSCA"] else None,
                    "PCAHH": float(row[col_dict["PCAHH"]]) if col_dict["PCAHH"] else None,
                    "PCAHA": float(row[col_dict["PCAHA"]]) if col_dict["PCAHA"] else None,
                    "AHCh": float(row[col_dict["AHCh"]]) if col_dict["AHCh"] else None,
                    "Over": float(row[col_dict["Over"]]) if col_dict["Over"] else None,
                    "Under": float(row[col_dict["Under"]]) if col_dict["Under"] else None,
                    "Total": float(row[col_dict["Total"]]) if col_dict["Total"] else None,
                    "Season": float(row[col_dict["Season"]]) if col_dict["Season"] else None,
                    "PCAHHWM": float(row[col_dict["PCAHHWM"]]) if col_dict["PCAHHWM"] else None,
                    "PCAHAWM": float(row[col_dict["PCAHAWM"]]) if col_dict["PCAHAWM"] else None,
                    "TotalGoal": float(row[col_dict["TotalGoal"]]) if col_dict["TotalGoal"] else None,
                    "TotalXG": float(row[col_dict["TotalXG"]]) if col_dict["TotalXG"] else None,
                    "HomeAdvance": float(row[col_dict["HomeAdvance"]]) if col_dict["HomeAdvance"] else None,
                    "HomeXG": float(row[col_dict["HomeXG"]]) if col_dict["HomeXG"] else None,
                    "AwayXG": float(row[col_dict["AwayXG"]]) if col_dict["AwayXG"] else None,
                    "HomeGoal": float(row[col_dict["HomeGoal"]]) if col_dict["HomeGoal"] else None,
                    "AwayGoal": float(row[col_dict["AwayGoal"]]) if col_dict["AwayGoal"] else None,
                    "HomeAttack": float(row[col_dict["HomeAttack"]]) if col_dict["HomeAttack"] else None,
                    "HomeDefence": float(row[col_dict["HomeDefence"]]) if col_dict["HomeDefence"] else None,
                    "AwayAttack": float(row[col_dict["AwayAttack"]]) if col_dict["AwayAttack"] else None,
                    "AwayDefence": float(row[col_dict["AwayDefence"]]) if col_dict["AwayDefence"] else None,
                }

                game_data.append(one)
            except:
                continue

        return game_data

    def runLearning(self, game_data, strength_file):

        self.WP.get_team_strength(game_data, strength_file)

        self.emit(QtCore.SIGNAL("finished(QString)"), strength_file)

        self.sendLog("End learning")


    def weibullLearning(self, game_data, strength_file):
        self.sendLog("Weibull learning start...")
        self.sendLog("Please wait 10~15 minutes")

        t = threading.Thread(name="Learning", target=self.runLearning, args=[game_data, strength_file])
        t.start()

    def match_name(self, name1, name2):
        b = difflib.SequenceMatcher(None, name1.lower(), name2.lower()).find_longest_match(0, len(name1), 0, len(name2)).size
        return b / min(len(name1),len(name2))

    def get_team_name(self, name, team_name_list, ignore_team=False):
        team_name_list = [item for item in team_name_list if item !=""]
        prob = np.array([self.match_name(name, team_name) for team_name in team_name_list])
        max_index = np.argmax(prob)
        max_value = np.max(prob)
        if ignore_team:
            name = team_name_list[max_index]
        else:
            if max_value > 0.2:
                name = team_name_list[max_index]
        return name

    def getFutureGames(self, league, start_year, end_year):
        if end_year < self.now_year - 1:
            game_data = self.getLeagueData([league], end_year + 1, end_year + 1)
            return game_data
        else:
            fixtures = self.get_fixtures([league])
            now_time = datetime.datetime.now()
            new_games = []
            for game in fixtures:
                start_date = datetime.datetime.strptime(game["StartDate"], "%Y-%m-%d %H:%M")

                if start_date < now_time:
                    continue

                home = self.get_team_name(game["HomeTeam"], self.league_teams[league])
                away = self.get_team_name(game["AwayTeam"], self.league_teams[league])

                new_games.append({
                    "Div": league,
                    "Date": game["StartDate"],
                    "HomeTeam": home,
                    "AwayTeam": away
                })

            # game_data = self.getLeagueData([league], end_year, end_year)
            # played_games = []
            # for game in game_data:
            #     played_games.append(f"{game['HomeTeam']}*{game['AwayTeam']}")
            #
            # team_list = self.league_teams[league]
            # new_games = []
            # for home_team in team_list:
            #     for away_team in team_list:
            #         if home_team == away_team or home_team == '' or away_team == '':
            #             continue
            #         home_away = f"{home_team}*{away_team}"
            #
            #         if home_away not in played_games:
            #             new_games.append({
            #                 "Div": league,
            #                 "Date": "",
            #                 "HomeTeam": home_team,
            #                 "AwayTeam": away_team
            #             })

            return new_games

    def getTeams(self):
        self.league_teams = {}
        for league in self.league_list:
            game_data = self.getLeagueData([league], self.now_year - 1, self.now_year - 1)

            teams = [""]

            for one in game_data:
                home = one["HomeTeam"]
                away = one["AwayTeam"]

                if home not in teams:
                    teams.append(home)
                if away not in teams:
                    teams.append(away)

            self.league_teams[league] = sorted(teams)

    def getSeasonTeams(self, league, season):
        game_data = self.getLeagueData([league], season, season)
        teams = []

        for one in game_data:
            home = one["HomeTeam"]
            away = one["AwayTeam"]

            if home not in teams:
                teams.append(home)
            if away not in teams:
                teams.append(away)

        return teams

    def readStrength(self, league, start_year, end_year):
        file = f"./prediction_data/strength_{league}_{start_year}_{end_year}.json"
        data = {}
        try:
            with open(file) as json_file:
                data = json.load(json_file)

            if end_year == self.now_year - 1:
                old_season_teams = self.getSeasonTeams(league, end_year - 1)
                for team in data["team_list"].keys():
                    if team not in old_season_teams:
                        data["team_list"][team]["New"] = "New"
                    else:
                        data["team_list"][team]["New"] = ""
            else:
                new_season_teams = self.getSeasonTeams(league, end_year + 1)
                for team in new_season_teams :
                    if team not in data["team_list"].keys():
                        data["team_list"][team] = {
                            "attack": 0,
                            "defence": 0,
                            "HomePlayed": 0,
                            "AwayPlayed": 0,
                            "New": "New"
                        }
                    else:
                        data["team_list"][team]["New"] = ""
        except:
            msg = "Please learn data"
            self.sendLog(f"_MSG_{msg}")

        return data

    def runPrediction_(self, game_data, team_rate):
        self.sendLog(f"prediction start ...")
        elo = ELO()
        rate_AD = elo.readAD()
        for game in game_data:
            if game["Div"] and game["HomeTeam"] and game["AwayTeam"]:
                home = game["HomeTeam"]
                away = game["AwayTeam"]
                home_rate = team_rate[home]["rate"]
                away_rate = team_rate[away]["rate"]

                diff = home_rate - away_rate

                index = elo.getIndex(diff)

                AD = rate_AD[f"{index}"]
                game["HomeRate"] = team_rate[home]["rate"]
                game["AwayRate"] = team_rate[away]["rate"]
                game["HomeOdds"] = 1 / AD["home"]
                game["DrawOdds"] = 1 / AD["draw"]
                game["AwayOdds"] = 1 / AD["away"]
                game["Over2.5"] = 1 / AD["over2.5"]
                game["Under2.5"] = 1 / (1 - AD["over2.5"])
                game["Home%"] = AD["home"] * 100
                game["Draw%"] = AD["draw"] * 100
                game["Away%"] = AD["away"] * 100
                game["Over2.5%"] = AD["over2.5"] * 100
                game["Under2.5%"] = 100 - game["Over2.5%"]

        self.sendLog(f"prediction end ...")
        return game_data

    def runPrediction(self, game_data, team_rate):
        self.sendLog(f"prediction start ...")
        for game in game_data:
            if game["Div"] and game["HomeTeam"] and game["AwayTeam"]:
                league_avg_home_goal = self.league_avg["home_goal"]
                league_avg_away_goal = self.league_avg["away_goal"]
                for game in game_data:
                    home_attack = team_rate[game["HomeTeam"]]["home_attack"]
                    home_defence = team_rate[game["HomeTeam"]]["home_defence"]
                    away_attcak = team_rate[game["AwayTeam"]]["away_attack"]
                    away_defence = team_rate[game["AwayTeam"]]["away_defence"]

                    game["HomeAttack"] = home_attack
                    game["HomeDefence"] = home_defence
                    game["AwayAttack"] = away_attcak
                    game["AwayDefence"] = away_defence
                    game["LeagueAvgHomeGoal"] = league_avg_home_goal
                    game["LeagueAvgAwayGoal"] = league_avg_away_goal
                    game["HomeGoal"] = home_attack * away_defence * league_avg_home_goal
                    game["AwayGoal"] = away_attcak * home_defence * league_avg_away_goal

        self.sendLog(f"prediction end ...")
        return game_data

    def runOldPrediction(self, game_data, team_rate):
        elo = ELO()
        betting_count = 0
        profit_open = 0.
        profit_close = 0.
        success_count = 0
        if len(game_data) == 0:
            msg = "No prediction game data"
            self.sendLog(f"_MSG_{msg}")
            return

        for game in game_data:
            game["OriginHome"], game["OriginDraw"], game["OriginAway"] = elo.calc_origin_odds_ratio(float(game["PSCH"]),
                                                                                               float(game["PSCD"]),
                                                                                               float(game["PSCA"]))
            game["HomeOdds"] = None
            game["DrawOdds"] = None
            game["AwayOdds"] = None
            game["Over2.5"] = None
            game["Under2.5"] = None
            game["Home%"] = None
            game["Draw%"] = None
            game["Away%"] = None
            game["Over2.5%"] = None
            game["Under2.5%"] = None
            game["HomeRate"] = None
            game["AwayRate"] = None
            # game["Betting"] = None
            # game["BettingOpenOdds"] = None
            # game["BettingCloseOdds"] = None
            # game["Profit(Open)"] = None
            # game["Profit(Close)"] = None

            if game["Div"] and game["HomeTeam"] and game["AwayTeam"]:
                if game["HomeTeam"] in team_rate.keys() and game["AwayTeam"] in team_rate.keys():
                    home = game["HomeTeam"]
                    away = game["AwayTeam"]
                    home_rate = team_rate[home]["rate"]
                    away_rate = team_rate[away]["rate"]

                    game["HomeRate"] = team_rate[home]["rate"]
                    game["AwayRate"] = team_rate[away]["rate"]

                    team_rate[home]["rate"], team_rate[away]["rate"] = elo.calc_elo_rate_odds(home_rate, away_rate, float(game["PSCH"]),
                                                                                               float(game["PSCD"]),
                                                                                               float(game["PSCA"]))

                    ph, pd, pa, over = elo.calcHDAProb(game['Div'], home, home_rate, away, away_rate)
                    game["HomeRate"] = team_rate[home]["rate"]
                    game["AwayRate"] = team_rate[away]["rate"]
                    game["HomeOdds"] = 1 / ph
                    game["DrawOdds"] = 1 / pd
                    game["AwayOdds"] = 1 / pa
                    game["Over2.5"] = 1 / over
                    game["Under2.5"] = 1 / (1 - over)
                    game["Home%"] = ph * 100
                    game["Draw%"] = pd * 100
                    game["Away%"] = pa * 100
                    game["Over2.5%"] = over * 100
                    game["Under2.5%"] = 100 - game["Over2.5%"]

                    # if game["Home%"] > game["Away%"]:
                    #     game["Betting"] = "Home"
                    #     game["BettingOpenOdds"] = float(game['PSH'])
                    #     game["BettingCloseOdds"] = float(game['PSCH'])
                    # else:
                    #     game["Betting"] = "Away"
                    #     game["BettingOpenOdds"] = float(game['PSA'])
                    #     game["BettingCloseOdds"] = float(game['PSCA'])
                    #
                    # betting_count += 1
                    # if game["Betting"][0] == game["FTR"]:
                    #     profit_open += (game["BettingOpenOdds"] - 1)
                    #     profit_close += (game["BettingCloseOdds"] - 1)
                    #     game["Profit(Open)"] = game["BettingOpenOdds"] - 1
                    #     game["Profit(Close)"] = game["BettingCloseOdds"] - 1
                    #     success_count += 1
                    # else:
                    #     profit_open -= 1
                    #     profit_close -= 1
                    #     game["Profit(Open)"] = -1
                    #     game["Profit(Close)"] = -1

        # if betting_count > 0:
        #     result = f"Profit(Open)={'%.3f' % profit_open}, Profit(Open)%={'%.3f' % (profit_open / betting_count * 100)}%, Profit(Close)={'%.3f' % profit_close}, Profit(Close)%={'%.3f' % (profit_close / betting_count * 100)}%, betting count={betting_count}, success count={success_count}"
        # else:
        #     result = ""

        self.sendLog("end")

        return game_data, ""

    def runMultiLearning(self, leagues, start_year, end_year):
        self.sendLog("learning start...")
        args = []
        for league in leagues:
            game_data = self.getLeagueData([league], start_year, end_year)
            args.append({
                "league": league,
                "game_data": game_data,
                "start_year": start_year,
                "end_year": end_year
            })

        pool = multiprocessing.Pool(processes=8)
        wp = WeibullPrediction()
        pool.map(wp.get_team_strength_multi, args)

        self.sendLog("learning end...")

    def getEloRate(self, league, start_year, end_year):

        game_data = self.getLeagueData([league], start_year, end_year)
        elo = ELO()

        game_data, team_rate = elo.analysis_game_data(game_data)

        return game_data, team_rate

    def allEloRate(self):
        game_data = self.getLeagueData(self.league_list, 2012, 2019)
        elo = ELO()

        elo.analysis_game_data_detail(game_data)

        self.sendLog("learning end")

    def runEloPrediction(self, games, team_rate):
        elo = ELO()

        for game in games:
            home = game["HomeTeam"]
            away = game["AwayTeam"]
            home_rate = team_rate[home]["rate"]
            away_rate = team_rate[away]["rate"]

            ph, pd, pa, over = elo.calcHDAProb(game['Div'], home, home_rate, away, away_rate)
            game["HomeRate"] = team_rate[home]["rate"]
            game["AwayRate"] = team_rate[away]["rate"]
            game["HomeOdds"] = 1 / ph
            game["DrawOdds"] = 1 / pd
            game["AwayOdds"] = 1 / pa
            game["Over2.5"] = 1 / over
            game["Under2.5"] = 1 / (1 - over)
            game["Home%"] = ph * 100
            game["Draw%"] = pd * 100
            game["Away%"] = pa * 100
            game["Over2.5%"] = over * 100
            game["Under2.5%"] = 100 - game["Over2.5%"]

        return games

    def learningXG(self, league, start_year, end_year):
        self.sendLog("xG learning start")
        game_data = self.getLeagueData([league], start_year, end_year)
        elo = ELO()
        game_data, team_xG, self.league_avg = elo.overUnderXG(game_data)
        self.sendLog("xG learning end")
        return game_data, team_xG

    def runOUPrediction(self, games, team_rate):

        league_avg_home_goal = self.league_avg["home_goal"]
        league_avg_away_goal = self.league_avg["away_goal"]

        for game in games:

            if not game["HomeTeam"] and not game["AwayTeam"]:
                continue

            home_attack = team_rate[game["HomeTeam"]]["home_attack"]
            home_defence = team_rate[game["HomeTeam"]]["home_defence"]
            away_attcak = team_rate[game["AwayTeam"]]["away_attack"]
            away_defence = team_rate[game["AwayTeam"]]["away_defence"]

            game["HomeAttack"] = home_attack
            game["HomeDefence"] = home_defence
            game["AwayAttack"] = away_attcak
            game["AwayDefence"] = away_defence
            game["LeagueAvgHomeGoal"] = league_avg_home_goal
            game["LeagueAvgAwayGoal"] = league_avg_away_goal
            game["HomeGoal"] = home_attack * away_defence * league_avg_home_goal
            game["AwayGoal"] = away_attcak * home_defence * league_avg_away_goal

        return games

    def runOUPrediction_weibull(self, games, team_rate, team_strength):

        league_avg_home_goal = self.league_avg["home_goal"]
        league_avg_away_goal = self.league_avg["away_goal"]

        for game in games:

            if not game["HomeTeam"] and not game["AwayTeam"]:
                continue

            prediction = self.WP.prediction(team_strength, game["HomeTeam"], game["AwayTeam"], game["Date"])
            # home_attack = team_rate[game["HomeTeam"]]["home_attack"]
            # home_defence = team_rate[game["HomeTeam"]]["home_defence"]
            # away_attcak = team_rate[game["AwayTeam"]]["away_attack"]
            # away_defence = team_rate[game["AwayTeam"]]["away_defence"]
            #
            # game["HomeAttack"] = home_attack
            # game["HomeDefence"] = home_defence
            # game["AwayAttack"] = away_attcak
            # game["AwayDefence"] = away_defence
            game["LeagueAvgHomeGoal"] = league_avg_home_goal
            game["LeagueAvgAwayGoal"] = league_avg_away_goal
            game["HomeXG"] = np.sum([v * i for i, v in enumerate(prediction["HomeXG"])])
            game["AwayXG"] = np.sum([v * i for i, v in enumerate(prediction["AwayXG"])])
            game["TotalXG"] = game["HomeXG"] + game["AwayXG"]
            game["HomeAttack"] = team_strength["team_list"][game["HomeTeam"]]["attack"]
            game["HomeDefence"] = team_strength["team_list"][game["HomeTeam"]]["defence"]
            game["AwayAttack"] = team_strength["team_list"][game["AwayTeam"]]["attack"]
            game["AwayDefence"] = team_strength["team_list"][game["AwayTeam"]]["defence"]
            game["HomeWin"] = prediction["H"]
            game["AwayWin"] = prediction["A"]
            game["Draw"] = prediction["D"]
            game[f"OVER{self.WP.over_under_str}"] = prediction[f"OVER{self.WP.over_under_str}"]
            game[f"UNDER{self.WP.over_under_str}"] = prediction[f"UNDER{self.WP.over_under_str}"]

            home_goal_dist = ", ".join(["%d:%.3f" % (i, v) for i, v in enumerate(prediction["HomeXG"])])
            away_goal_dist = ", ".join(["%d:%.3f" % (i, v) for i, v in enumerate(prediction["AwayXG"])])

            self.sendLog(f"Home: {game['HomeTeam']} : {home_goal_dist}")
            self.sendLog(f"Away: {game['AwayTeam']} : {away_goal_dist}")
        return games