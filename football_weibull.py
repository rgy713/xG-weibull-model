import json

import pandas as pd
import numpy as np
from scipy.special import gamma
from scipy.optimize import minimize

class WeibullPrediction(object):

    max_goals = 7
    over_under = 2.5
    alpha_dic = {}
    series_max = 30
    over_under_str = f"{over_under}"
    over_under = int(over_under + 1.5)

    def __init__(self):
        pass

    def alpha(self, x, j, c):
        key = f"{x}_{j}_{'{0:.12f}'.format(c)}"
        if key in self.alpha_dic.keys():
            return self.alpha_dic[key]
        else:
            if x == 0:
                self.alpha_dic[key] = gamma(c * j + 1) / gamma(j + 1)
                return self.alpha_dic[key]
            else:
                result = 0
                for m in range(x - 1, j):
                    result += self.alpha(x - 1, m, c) * gamma(c * j - c * m + 1) / gamma(j - m + 1)

                self.alpha_dic[key] = result
                return result

    def weibull_pmf(self, x_arr, shape, scale):

        def weibull_pmf_gen(x, shape, scale):
            result = 0.0
            for j in range(x, x + self.series_max):
                result += (-1) ** (x + j) * scale ** j * self.alpha(x, j, shape) / gamma(shape * j + 1)

            # if result == 0:
            #     print(x, shape, scale)

            return result

        if isinstance(x_arr, int) or  isinstance(x_arr, float) :
            return weibull_pmf_gen(int(x_arr), shape, scale)
        else:
            return [weibull_pmf_gen(x, shape, scale) for x in x_arr]


    def weibull_fit(self, x_data, y_data):
        init_vals = [1., 1.]
        options = {'disp': True, 'maxiter': 100}
        constraints = [{'type': 'ineq', 'fun': lambda x: x[0]}, {'type': 'ineq', 'fun': lambda x: x[1]}]

        def estimate_paramters(params):
            return np.linalg.norm(y_data - self.weibull_pmf(x_data, params[0], params[1]))

        opt_output = minimize(estimate_paramters, init_vals, options=options, constraints=constraints)

        print(opt_output)
        return opt_output.x

    def calc_means(self, team_strangth, homeTeam, awayTeam):
        return [np.exp(team_strangth["team_list"][homeTeam]["attack"] + team_strangth["team_list"][awayTeam]["defence"] + team_strangth["info"]['home_adv']),
                np.exp(team_strangth["team_list"][homeTeam]["defence"] + team_strangth["team_list"][awayTeam]["attack"])]


    def rho_correction(self, x, y, lambda_x, mu_y, rho):
        if x == 0 and y == 0:
            return 1 - (lambda_x * mu_y * rho)
        elif x == 0 and y == 1:
            return 1 + (lambda_x * rho)
        elif x == 1 and y == 0:
            return 1 + (mu_y * rho)
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1.0

    def dixon_coles_simulate_match(self, team_strength, homeTeam, awayTeam, date):
        team_avgs = self.calc_means(team_strength, homeTeam, awayTeam)
        # print(f"\nDate:{date},  Home:{homeTeam}, Away:{awayTeam}")
        team_pred = [[self.weibull_pmf(i, team_strength["info"]["home_shape"], team_avgs[0]) for i in range(0, self.max_goals + 1)], [self.weibull_pmf(i, team_strength["info"]["away_shape"], team_avgs[1]) for i in range(0, self.max_goals + 1)]]

        output_matrix = np.outer(np.array(team_pred[0]), np.array(team_pred[1]))
        correction_matrix = np.array([[self.rho_correction(home_goals, away_goals, team_avgs[0],
                                                      team_avgs[1], team_strength["info"]['rho']) for away_goals in range(2)]
                                      for home_goals in range(2)])
        output_matrix[:2, :2] = output_matrix[:2, :2] * correction_matrix
        return output_matrix

    def get_xg(self, team_strength, homeTeam, awayTeam):
        team_avgs = self.calc_means(team_strength, homeTeam, awayTeam)
        team_pred = [[self.weibull_pmf(i, team_strength["info"]["home_shape"], team_avgs[0]) for i in
                      range(0, self.max_goals + 1)],
                     [self.weibull_pmf(i, team_strength["info"]["away_shape"], team_avgs[1]) for i in
                      range(0, self.max_goals + 1)]]

        return team_pred

    def solve_parameters_decay(self, dataset, home_shape, away_shape, xi=0.001, debug=False, init_vals=None, **kwargs):
        teams = np.sort(dataset['HomeTeam'].unique())
        # check for no weirdness in dataset
        away_teams = np.sort(dataset['AwayTeam'].unique())
        if not np.array_equal(teams, away_teams):
            raise ValueError("something not right")
        n_teams = len(teams)
        options = {'disp': True, 'maxiter': 100}

        constraints = [
            {'type': 'eq', 'fun': lambda x: sum(x[: n_teams]) - n_teams},
            {'type': 'ineq', 'fun': lambda x: x[-2] * x[-2]}
        ]

        if init_vals is None:
            # random initialisation of model parameters
            init_vals = np.concatenate((np.random.uniform(0, 1, (n_teams)),  # attack strength
                                        np.random.uniform(0, -1, (n_teams)),  # defence strength
                                        np.array([0, 1.0])  # k (score correction), rho (home advantage)
                                        ))
        def copula(u, v, k):
            ret = -1 / k * np.log(1 + (np.exp(- k * u) - 1) * (np.exp(- k * v ) - 1) / (np.exp(-k) - 1))
            return ret



        def dc_log_like_decay(x, y, shape_x, shape_y, alpha_x, beta_x, alpha_y, beta_y, rho, k, t, xi=xi):
            lambda_x, mu_y = np.exp(alpha_x + beta_y + rho), np.exp(alpha_y + beta_x)
            # return np.log(np.exp(-xi * t)  * ( copula(weibull_pmf(x, shape_x, lambda_x), weibull_pmf(y, shape_y, mu_y), k)
            #                            -  copula(weibull_pmf(x - 1, shape_x, lambda_x)  if x > 1 else 0, weibull_pmf(y, shape_y, mu_y), k)
            #                            -  copula(weibull_pmf(x, shape_x, lambda_x), weibull_pmf(y - 1, shape_y, mu_y) if y > 1 else 0, k)
            #                            +  copula(weibull_pmf(x - 1, shape_x, lambda_x) if x > 1 else 0, weibull_pmf(y - 1, shape_y, mu_y) if y > 1 else 0, k)
            #                            ))

            return np.exp(-xi * t) * (np.log(self.rho_correction(x, y, lambda_x, mu_y, k)) +
                                  np.log(self.weibull_pmf(x, shape_x, lambda_x)) + np.log(self.weibull_pmf(y, shape_y, mu_y)))

        def estimate_paramters(params):
            score_coefs = dict(zip(teams, params[:n_teams]))
            defend_coefs = dict(zip(teams, params[n_teams:(2 * n_teams)]))
            k, rho = params[-2:]
            log_like = [
                dc_log_like_decay(row.FTHG, row.FTAG, home_shape, away_shape, score_coefs[row.HomeTeam], defend_coefs[row.HomeTeam],
                                  score_coefs[row.AwayTeam], defend_coefs[row.AwayTeam],
                                  rho, k, row.time_diff, xi=xi) for row in dataset.itertuples()]
            # log_like = -sum(log_like)
            # print(log_like)
            return -sum(log_like)

        opt_output = minimize(estimate_paramters, init_vals, options=options, constraints=constraints)
        if debug:
            # sort of hacky way to investigate the output of the optimisation process
            return opt_output
        else:
            result = dict(zip(["attack_" + team for team in teams] +
                            ["defence_" + team for team in teams] +
                            ['rho', 'home_adv'],
                            opt_output.x))
            for key in result.keys():
                print(f"{key}:{result[key]}")

            return teams, result


    def get_1x2_probs(self, match_score_matrix):
        result = dict({
                "H": np.sum(np.tril(match_score_matrix, -1)),
                "A": np.sum(np.triu(match_score_matrix, 1)),
                "D": np.sum(np.diag(match_score_matrix)),
                f"OVER{self.over_under_str}" : np.sum(np.tril(np.fliplr(match_score_matrix), np.shape(match_score_matrix)[0] - self.over_under)),
                f"UNDER{self.over_under_str}" : 1 - np.sum(np.tril(np.fliplr(match_score_matrix), np.shape(match_score_matrix)[0] - self.over_under))
            })
        # print("game prediction")
        # print(result)
        # max_points = np.unravel_index(np.argmax(match_score_matrix, axis=None), match_score_matrix.shape)
        # print(f"Home:{max_points[0]}, Away:{max_points[1]}")
        return result

    def calc_team_strength(self, train_dataset, home_shape, away_shape, xi):

        teams, team_strength = self.solve_parameters_decay(train_dataset, home_shape, away_shape, xi)
        strength = {}
        for team in teams:
            strength[team] = {
                "attack": team_strength["attack_" + team],
                "defence": team_strength["defence_" + team],
            }
        team_strength = {
            "team_list": strength,
            "info": {
                "rho": team_strength["rho"],
                "home_adv": team_strength["home_adv"]
            }
        }
        return team_strength

    def get_total_score_xi(self, game_data, xi):
        home_count, home_division = np.histogram(game_data[['FTHG']], bins=range(0, self.max_goals + 1))
        away_count, away_division = np.histogram(game_data[['FTAG']], bins=range(0, self.max_goals + 1))

        x_data = np.array(range(0, self.max_goals))
        home_goal_dist = home_count / np.sum(home_count)
        away_goal_dist = away_count / np.sum(away_count)

        home_fit_params = self.weibull_fit(x_data, home_goal_dist)
        # print(home_fit_params)
        away_fit_params = self.weibull_fit(x_data, away_goal_dist)
        # print(away_fit_params)

        home_shape = home_fit_params[0]
        away_shape = away_fit_params[0]

        for i in range(0, self.max_goals + 2):
            for j in range(0, self.series_max + self.max_goals + 2):
                self.alpha(i, j, home_shape)
                self.alpha(i, j, away_shape)

        team_stength = self.calc_team_strength(game_data, home_shape, away_shape, xi=xi)

        team_stength["info"]["home_shape"] = home_shape
        team_stength["info"]["away_shape"] = away_shape
        last_game = f"{game_data.Date.iat[-1].strftime('%d/%m/%Y')}*{game_data.HomeTeam.iat[-1]}*{game_data.AwayTeam.iat[-1]}"
        team_stength["info"]["last_game"] = last_game
        return team_stength

    def get_team_strength(self, game_data, strength_file):
        game_data = pd.DataFrame(game_data)
        game_data['Date'] = pd.to_datetime(game_data['Date'], format='%d/%m/%Y')
        game_data['time_diff'] = (max(game_data['Date']) - game_data['Date']).dt.days
        # game_data['HomeGoal'] = pd.to_numeric(game_data['HomeGoal'], downcast='integer')
        # game_data['AwayGoal'] = pd.to_numeric(game_data['AwayGoal'], downcast='integer')

        game_data = game_data[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'time_diff']]

        # game_data = game_data.rename(columns={'HomeGoal': 'FTHG', 'AwayGoal': 'FTAG'})

        home_team_count = game_data.groupby(['HomeTeam']).size()
        away_team_count = game_data.groupby(['AwayTeam']).size()

        xi_vals = 0.00650

        team_strength = self.get_total_score_xi(game_data, xi_vals)

        for key in home_team_count.keys():
            team_strength["team_list"][key]["HomePlayed"] = int(home_team_count[key])
        for key in away_team_count.keys():
            team_strength["team_list"][key]["AwayPlayed"] = int(away_team_count[key])

        with open(strength_file, 'w') as outfile:
            json.dump(team_strength, outfile, indent=2)

    def prediction(self, team_strength, home_team, away_team, game_date):
        result = self.get_1x2_probs(self.dixon_coles_simulate_match(team_strength, home_team, away_team, game_date))

        xg = self.get_xg(team_strength, home_team, away_team)

        result["HomeXG"] = xg[0]
        result["AwayXG"] = xg[1]

        return result

    def get_team_strength_multi(self, data):
        league = data["league"]
        game_data = data["game_data"]
        start_year = data["start_year"]
        end_year = data["end_year"]
        game_data = pd.DataFrame(game_data)
        game_data['Date'] = pd.to_datetime(game_data['Date'], format='%d/%m/%Y')
        game_data['time_diff'] = (max(game_data['Date']) - game_data['Date']).dt.days
        game_data = game_data[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'time_diff']]

        home_team_count = game_data.groupby(['HomeTeam']).size()
        away_team_count = game_data.groupby(['AwayTeam']).size()

        xi_vals = 0.00650

        team_strength = self.get_total_score_xi(game_data, xi_vals)

        for key in home_team_count.keys():
            team_strength["team_list"][key]["HomePlayed"] = int(home_team_count[key])
        for key in away_team_count.keys():
            team_strength["team_list"][key]["AwayPlayed"] = int(away_team_count[key])

        with open(f"./prediction_data/strength_{league}_{start_year}_{end_year}.json", 'w') as outfile:
            json.dump(team_strength, outfile, indent=2)
