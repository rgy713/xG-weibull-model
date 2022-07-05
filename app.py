import json
import os
import sys
import datetime

from PySide2 import QtCore, QtWidgets
from PySide2.QtCore import QModelIndex
from PySide2.QtGui import QBrush

from CheckableComboBox import CheckableComboBox
from ui_mainwindow import Ui_MainWindow
from main import PiRatingSystem
import threading

ONE_GAME = {
    "Div": "E0",
    "Date": datetime.datetime.now().strftime("%d/%m/%Y"),
    "HomeTeam": "",
    "AwayTeam": "",
    "HomeAttack": None,
    "HomeDefence": None,
    "AwayAttack": None,
    "AwayDefence": None,
    "LeagueAvgHomeGoal": None,
    "LeagueAvgAwayGoal": None,
    "HomeXG": None,
    "AwayXG": None,
    "TotalXG": None,
}


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            value = list(self._data[index.row()].values())[index.column()]

            if isinstance(value, float):
                return "%.3f" % value

            if isinstance(value, str):
                return '%s' % value

            return value

    def rowCount(self, index=QModelIndex()):
        return len(self._data)

    def columnCount(self, index=QModelIndex()):
        if len(self._data) > 0:
            return len(self._data[0].keys())
        else:
            return 0

    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole and len(self._data) > 0:
            if orientation == QtCore.Qt.Horizontal:
                return str(list(self._data[0].keys())[section])
            if orientation == QtCore.Qt.Vertical:
                return str(section)


class EditableTableModel(TableModel):

    def __init__(self, data):
        super(EditableTableModel, self).__init__(data)
        self._data = data

    def data(self, index, role):
        value = list(self._data[index.row()].values())[index.column()]
        key = list(self._data[index.row()].keys())[index.column()]
        if role == QtCore.Qt.DisplayRole:
            if isinstance(value, float):
                return "%.3f" % value

            if isinstance(value, str):
                return '%s' % value

            return value

        if role == QtCore.Qt.BackgroundRole and key in ['TeamName']:
            if self._data[index.row()]["New"] == "New":
                return QBrush(QtCore.Qt.blue)

    def setData(self, index, value, role: float = ...) -> bool:
        if role == QtCore.Qt.EditRole:
            key = list(self._data[index.row()].keys())[index.column()]
            if value:
                try:
                    value = float(value)
                except:
                    return False

                if (key == 'Attack' and value < 0) or (key == 'Defence' and value > 0):
                    return False

                self._data[index.row()][key] = value

            return True
        else:
            return False

    def flags(self, index) -> QtCore.Qt.ItemFlags:
        key = list(self._data[index.row()].keys())[index.column()]
        if key in ["Attack", "Defence"]:
            return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        else:
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable


class DateDelegate(QtWidgets.QStyledItemDelegate):
    def initStyleOption(self, option: QtWidgets.QStyleOptionViewItem, index: QtCore.QModelIndex):
        super(DateDelegate, self).initStyleOption(option, index)
        option.text = index.data().toString("dd/MM/yyyy")


class AssetDelegate(QtWidgets.QStyledItemDelegate):

    def paint(self, painter, option, index):
        if isinstance(self.parent(), QtWidgets.QAbstractItemView):
            self.parent().openPersistentEditor(index)
        QtWidgets.QStyledItemDelegate.paint(self, painter, option, index)

    def createEditor(self, parent, option, index):
        combobox = QtWidgets.QComboBox(parent)
        combobox.addItems(index.data(NewGameTableModel.ItemsRole))
        combobox.currentIndexChanged.connect(self.onCurrentIndexChanged)
        return combobox

    def onCurrentIndexChanged(self, ix):
        editor = self.sender()
        self.commitData.emit(editor)
        # self.closeEditor.emit(editor, QtWidgets.QAbstractItemDelegate.NoHint)

    def setEditorData(self, editor, index):
        ix = index.data(NewGameTableModel.ActiveRole)
        editor.setCurrentIndex(ix)

    def setModelData(self, editor, model, index):
        ix = editor.currentIndex()
        model.setData(index, ix, NewGameTableModel.ActiveRole)


class NewGameTableModel(TableModel):
    ItemsRole = QtCore.Qt.UserRole + 1
    ActiveRole = QtCore.Qt.UserRole + 2

    changeRecord = QtCore.Signal(int, str, str)

    def __init__(self, data, teams):
        super(NewGameTableModel, self).__init__(data)
        self._data = data
        self._teams = teams

    def data(self, index, role):
        value = list(self._data[index.row()].values())[index.column()]
        key = list(self._data[index.row()].keys())[index.column()]

        if role == NewGameTableModel.ItemsRole:
            return self._teams[self._data[index.row()]["Div"]]

        if role == NewGameTableModel.ActiveRole:
            if value:
                return self._teams[self._data[index.row()]["Div"]].index(value)
            else:
                return 0

        if role == QtCore.Qt.DisplayRole:
            if key == "Date":
                return QtCore.QDateTime.fromString(value, "dd/MM/yyyy")
            else:
                if isinstance(value, float):
                    return "%.3f" % value
                if isinstance(value, str):
                    return '%s' % value
                return value

        if role == QtCore.Qt.BackgroundRole and key == "Betting" and value in ['Home', 'Away', 'Draw']:
            if value == 'Home':
                return QBrush(QtCore.Qt.blue)
            elif value == 'Away':
                return QBrush(QtCore.Qt.cyan)
            elif value == 'Draw':
                return QBrush(QtCore.Qt.green)

    def setData(self, index, value, role: float = ...) -> bool:
        key = list(self._data[index.row()].keys())[index.column()]
        if role == QtCore.Qt.EditRole:
            if value:
                try:
                    value_check = str(value)

                    if key in ["PSH", "PSD", "PSA"]:
                        value_check = float(value)

                except:
                    return False

                self._data[index.row()][key] = value

                if key in ["PSH", "PSD", "PSA"]:
                    if value_check > 1 and self._data[index.row()]["PSH"] and self._data[index.row()]["PSD"] and \
                            self._data[index.row()]["PSA"]:
                        self.emit(QtCore.SIGNAL("changeRecord(int, QString, QString)"), index.row(), key, str(value))
                    else:
                        return False
                return True
            else:
                return False

        if role == NewGameTableModel.ActiveRole:
            self._data[index.row()][key] = self._teams[self._data[index.row()]["Div"]][value]
            if self._data[index.row()]["HomeTeam"] and self._data[index.row()]["AwayTeam"]:
                self.emit(QtCore.SIGNAL("changeRecord(int, QString, QString)"), index.row(), key,
                          self._teams[self._data[index.row()]["Div"]][value])
            return True

    def flags(self, index) -> QtCore.Qt.ItemFlags:
        key = list(self._data[index.row()].keys())[index.column()]
        if key in ["Date", "HomeTeam", "AwayTeam"]:
            return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        if key in ["PSH", "PSD", "PSA"]:
            return QtCore.Qt.ItemIsEditable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        else:
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def removeRow(self, row: int, index=QModelIndex()) -> bool:
        self.beginRemoveRows(index, row, row)
        self._data.pop(row)
        self.endRemoveRows()
        return True

    def insertRow(self, row: int, index=QModelIndex(), new_game=None) -> bool:
        self.beginInsertRows(index, row, row)
        self._data.insert(row, new_game)
        self.endInsertRows()
        return True


class ResultTableModel(TableModel):
    def __init__(self, data):
        super(ResultTableModel, self).__init__(data)
        self._data = data

    def data(self, index, role):
        value = list(self._data[index.row()].values())[index.column()]
        if role == QtCore.Qt.DisplayRole:
            if isinstance(value, float):
                return "%.3f" % value
            if isinstance(value, str):
                return '%s' % value

            return value
        if role == QtCore.Qt.BackgroundRole and value in ['Home', 'Away', 'Draw']:
            if value == 'Home' and self._data[index.row()]["FTR"] == 'H':
                return QBrush(QtCore.Qt.blue)
            elif value == 'Away' and self._data[index.row()]["FTR"] == 'A':
                return QBrush(QtCore.Qt.cyan)
            elif value == 'Draw' and self._data[index.row()]["FTR"] == 'D':
                return QBrush(QtCore.Qt.green)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.PRS = PiRatingSystem()
        self.learning_data = []
        self.prediction_data = []

        default_leagues = ['E0', 'SP1', 'F1', 'D1', 'I1', 'E1', 'F2', 'I2', 'SP2', 'D2', 'B1', 'N1']

        self.ui.cbxSelectLeague.addItems(self.PRS.league_list)

        self.seasons = [f"{year}-{year + 1}" for year in range(2019, self.PRS.now_year)]

        self.ui.cboStartSeason.addItems(self.seasons)
        self.ui.cboEndSeason.addItems(self.seasons)
        self.ui.cboStartSeason.setCurrentIndex(len(self.seasons) - 1)
        self.ui.cboEndSeason.setCurrentIndex(len(self.seasons) - 1)

        self.setLeagueTeam()

        self.newGameModel = None
        new_game = ONE_GAME.copy()
        new_game["Div"] = self.ui.cbxSelectLeague.currentText()
        self.viewNewGameTable([new_game], self.PRS.league_teams)

        self.connect(self.ui.btnStart, QtCore.SIGNAL("clicked()"), self.prediction)
        self.connect(self.PRS, QtCore.SIGNAL("pushLog(QString)"), self.printLog)
        self.connect(self.ui.btnScraping, QtCore.SIGNAL("clicked()"), self.runScraping)
        self.connect(self.ui.cbxSelectLeague, QtCore.SIGNAL("currentIndexChanged(int)"), self.changeLeague)
        self.connect(self.ui.cboStartSeason, QtCore.SIGNAL("currentIndexChanged(int)"), self.changeStartSeason)
        self.connect(self.ui.cboEndSeason, QtCore.SIGNAL("currentIndexChanged(int)"), self.changeEndSeason)
        self.connect(self.ui.btnViewLearningData, QtCore.SIGNAL("clicked()"), self.viewLearningData)
        self.connect(self.PRS, QtCore.SIGNAL("finished(QString)"), self.finished)
        self.connect(self.ui.btnAddGame, QtCore.SIGNAL("clicked()"), self.addGame)
        self.connect(self.ui.btnRemoveGame, QtCore.SIGNAL("clicked()"), self.removeGame)
        self.connect(self.ui.btnClearLog, QtCore.SIGNAL("clicked()"), self.clearLog)
        self.connect(self.ui.btnViewTeam, QtCore.SIGNAL("clicked()"), self.viewTeamInfo)

        self.haveLearning = False
        self.team_rate = {}

        self.learning_file = ''
        self.strength_file = ''
        self.team_strength = {}


    @QtCore.Slot()
    def clearLog(self):
        self.ui.logView.clear()

    def closeEvent(self, event):
        self.PRS.is_run = False

    @QtCore.Slot()
    def prediction(self):
        if not self.haveLearning:
            mb = QtWidgets.QMessageBox()
            mb.setText("please click 'Learning Data View'")
            mb.exec_()
            return
        league = self.ui.cbxSelectLeague.currentText()
        start = int(self.ui.cboStartSeason.currentText().split("-")[0])
        end = int(self.ui.cboEndSeason.currentText().split("-")[0])
        games = self.PRS.getFutureGames(league, start, end)
        team_count = len(self.PRS.league_teams[league]) - 1
        all_count = team_count * (team_count - 1)
        if end < self.PRS.now_year - 1:
            games, result = self.PRS.runOldPrediction(games, self.team_rate)
        else:
            games = self.PRS.runPrediction(games, self.team_rate)
            new_count = len(games)
            played_count = all_count - new_count
            result = f"All Games Count: {all_count}, Played Games Count: {played_count}, New Games Count: {new_count}"
            self.printLog(result)
        if len(games) > 0:
            self.PRS.saveCsv(games, f"./result/prediction_{league}_{start}_{end}.csv")

            tableModel = ResultTableModel(games)
            self.ui.tblPredictionGames.setModel(tableModel)
            self.ui.tblPredictionGames.show()

            self.ui.lblResult.setText(result)

    @QtCore.Slot(int, str, str)
    def changeNewGame(self, index, key, value):
        if len(self.team_rate) == 0:
            mb = QtWidgets.QMessageBox()
            mb.setText("please select learning data")
            mb.exec_()
            return

        self.printLog(f"new game added")
        games = self.newGameModel._data
        # games = self.PRS.runOUPrediction(games, self.team_rate, self.team_strength)
        games = self.PRS.runOUPrediction_weibull(games, self.team_rate, self.team_strength)

        if len(games) > 0:
            self.viewNewGameTable(games, self.PRS.league_teams)

    @QtCore.Slot(str)
    def printLog(self, msg):
        print(str(msg))
        if "_MSG_" in msg:
            mb = QtWidgets.QMessageBox()
            mb.setText(msg.split("_MSG_")[1])
            mb.exec_()
            return

        self.ui.logView.appendPlainText(msg)

    @QtCore.Slot()
    def runScraping(self):
        leagues = [self.ui.cbxSelectLeague.currentText()]
        t = threading.Thread(name="Scraping", target=self.PRS.basic_league_scrapping, args=[leagues])
        t.start()

        t = threading.Thread(name="Scraping_fixtures", target=self.PRS.scrape_fixtures, args=[leagues])
        t.start()

    @QtCore.Slot(int)
    def changeStartSeason(self, idx):
        endIdx = self.ui.cboEndSeason.currentIndex()
        if idx > endIdx:
            self.ui.cboEndSeason.setCurrentIndex(idx)

    @QtCore.Slot(int)
    def changeEndSeason(self, idx):
        startIdx = self.ui.cboStartSeason.currentIndex()
        if idx < startIdx:
            self.ui.cboStartSeason.setCurrentIndex(idx)

    @QtCore.Slot()
    def viewLearningData(self):
        # self.PRS.allEloRate()
        # return
        self.haveLearning = True
        league = self.ui.cbxSelectLeague.currentText()
        start = int(self.ui.cboStartSeason.currentText().split("-")[0])
        end = int(self.ui.cboEndSeason.currentText().split("-")[0])

        self.learning_data, self.team_rate = self.PRS.learningXG(league, start, end)

        self.learning_file = f"./result/learning_{league}_{start}_{end}.csv"
        self.strength_file = f"./result/strength_{league}_{start}_{end}.json"

        self.PRS.saveCsv(self.learning_data, self.learning_file)


        tableModel = TableModel(self.learning_data)
        self.ui.tblLearningGames.setModel(tableModel)
        self.ui.tblLearningGames.show()

        self.PRS.weibullLearning(self.learning_data, self.strength_file)

    def viewStrengthData(self, team_rate, team_strength):

        team_strength_list = []
        for key in team_rate.keys():
            team_strength_list.append({
                "Team": key,
                "HomeAvgXG": team_rate[key]["home_avg_goal"],
                # "HomeAttack": team_rate[key]["home_attack"],
                "HomeAvgConceded": team_rate[key]["home_avg_conceded"],
                # "HomeDefence": team_rate[key]["home_defence"],
                "AwayAvgXG": team_rate[key]["away_avg_goal"],
                # "AwayAttack": team_rate[key]["away_attack"],
                "AwayAvgConceded": team_rate[key]["away_avg_conceded"],
                # "AwayDefence": team_rate[key]["away_defence"],
                "Attack": team_strength["team_list"][key]["attack"],
                "Defence": team_strength["team_list"][key]["defence"],
            })

        league = self.ui.cbxSelectLeague.currentText()

        self.PRS.saveCsv(team_strength_list, f"./result/{league}_strength.csv")

        tableStrengthModel = EditableTableModel(team_strength_list)

        self.ui.tblStrengthView.setModel(tableStrengthModel)
        self.ui.tblStrengthView.show()

    def tr(self, text):
        return QtCore.QObject.tr(self, text)

    @QtCore.Slot(str)
    def finished(self, strength_file):
        json_file = open(strength_file)
        self.team_strength = json.load(json_file)

        self.viewStrengthData(self.team_rate, self.team_strength)


    @QtCore.Slot()
    def addGame(self):
        if not self.haveLearning:
            mb = QtWidgets.QMessageBox()
            mb.setText("please click 'Learning Data View'")
            mb.exec_()
            return
        indeces = self.ui.tblNewGames.selectionModel().selectedRows()
        new_game = ONE_GAME.copy()
        new_game["Div"] = self.ui.cbxSelectLeague.currentText()
        if len(indeces) == 1:
            for index in indeces:
                self.newGameModel.insertRow(index.row() + 1, new_game=new_game)
        else:
            self.newGameModel.insertRow(self.newGameModel.rowCount(), new_game=new_game)

    @QtCore.Slot()
    def removeGame(self):
        indeces = self.ui.tblNewGames.selectionModel().selectedRows()

        if len(indeces) == 0:
            mb = QtWidgets.QMessageBox()
            mb.setText("please select row")
            mb.exec_()
            return

        if len(indeces) == 1:
            for index in indeces:
                self.newGameModel.removeRow(index.row())

    def viewNewGameTable(self, game_data, league_teams):
        self.newGameModel = NewGameTableModel(game_data, league_teams)
        self.connect(self.newGameModel, QtCore.SIGNAL("changeRecord(int, QString, QString)"), self.changeNewGame)
        self.ui.tblNewGames.setModel(self.newGameModel)
        self.ui.tblNewGames.setItemDelegateForColumn(1, DateDelegate(self.ui.tblNewGames))
        self.ui.tblNewGames.setItemDelegateForColumn(2, AssetDelegate(self.ui.tblNewGames))
        self.ui.tblNewGames.setItemDelegateForColumn(3, AssetDelegate(self.ui.tblNewGames))
        self.ui.tblNewGames.show()

    @QtCore.Slot(int)
    def changeLeague(self):
        self.setLeagueTeam()

    def setLeagueTeam(self):
        league = self.ui.cbxSelectLeague.currentText()
        league_teams = self.PRS.league_teams[league]
        self.ui.cbxSelectTeam.clear()
        self.ui.cbxSelectTeam.addItems(league_teams)

    @QtCore.Slot()
    def viewTeamInfo(self):
        league = self.ui.cbxSelectLeague.currentText()
        team = self.ui.cbxSelectTeam.currentText()
        if not league:
            mb = QtWidgets.QMessageBox()
            mb.setText("Please select league name")
            mb.exec_()
            return

        if not team:
            mb = QtWidgets.QMessageBox()
            mb.setText("Please select team name")
            mb.exec_()
            return
        if not self.haveLearning:
            mb = QtWidgets.QMessageBox()
            mb.setText("Please click learning button.")
            mb.exec_()
            return

        team_data = []
        
        for game in self.learning_data:
            if team == game["HomeTeam"] or team == game["AwayTeam"]:
                team_data.append(game)
                
        file_path = f"result\\team_information_{league}_{team}.csv"
        self.PRS.saveCsv(team_data, file_path)
        os.system('start excel.exe "%s\\%s"' % (os.path.dirname(os.path.abspath(__file__)), file_path))

        return


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
