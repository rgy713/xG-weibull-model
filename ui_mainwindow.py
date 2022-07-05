# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 5.14.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient)
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.setWindowModality(Qt.NonModal)
        MainWindow.resize(1412, 1001)
        self.verticalLayout_2 = QVBoxLayout(MainWindow)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(-1, 5, -1, 5)
        self.btnViewLearningData = QPushButton(MainWindow)
        self.btnViewLearningData.setObjectName(u"btnViewLearningData")
        self.btnViewLearningData.setMinimumSize(QSize(0, 40))
        font = QFont()
        font.setPointSize(10)
        self.btnViewLearningData.setFont(font)

        self.horizontalLayout.addWidget(self.btnViewLearningData)

        self.label = QLabel(MainWindow)
        self.label.setObjectName(u"label")
        self.label.setMinimumSize(QSize(0, 0))
        self.label.setFont(font)
        self.label.setLayoutDirection(Qt.LeftToRight)
        self.label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)

        self.horizontalLayout.addWidget(self.label)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(5, -1, 5, -1)
        self.cbxSelectLeague = QComboBox(MainWindow)
        self.cbxSelectLeague.setObjectName(u"cbxSelectLeague")
        self.cbxSelectLeague.setFont(font)

        self.horizontalLayout_6.addWidget(self.cbxSelectLeague)


        self.horizontalLayout.addLayout(self.horizontalLayout_6)

        self.btnScraping = QPushButton(MainWindow)
        self.btnScraping.setObjectName(u"btnScraping")
        self.btnScraping.setMinimumSize(QSize(0, 40))
        self.btnScraping.setFont(font)

        self.horizontalLayout.addWidget(self.btnScraping)

        self.label_2 = QLabel(MainWindow)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMinimumSize(QSize(100, 0))
        self.label_2.setFont(font)
        self.label_2.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)

        self.horizontalLayout.addWidget(self.label_2)

        self.cboStartSeason = QComboBox(MainWindow)
        self.cboStartSeason.setObjectName(u"cboStartSeason")
        self.cboStartSeason.setMinimumSize(QSize(100, 30))

        self.horizontalLayout.addWidget(self.cboStartSeason)

        self.label_4 = QLabel(MainWindow)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout.addWidget(self.label_4)

        self.cboEndSeason = QComboBox(MainWindow)
        self.cboEndSeason.setObjectName(u"cboEndSeason")
        self.cboEndSeason.setMinimumSize(QSize(100, 30))

        self.horizontalLayout.addWidget(self.cboEndSeason)

        self.btnViewTeam = QPushButton(MainWindow)
        self.btnViewTeam.setObjectName(u"btnViewTeam")
        self.btnViewTeam.setMaximumSize(QSize(16777215, 30))
        font1 = QFont()
        font1.setPointSize(8)
        self.btnViewTeam.setFont(font1)

        self.horizontalLayout.addWidget(self.btnViewTeam)

        self.cbxSelectTeam = QComboBox(MainWindow)
        self.cbxSelectTeam.setObjectName(u"cbxSelectTeam")
        self.cbxSelectTeam.setMaximumSize(QSize(16777215, 16777215))

        self.horizontalLayout.addWidget(self.cbxSelectTeam)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer)


        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(-1, 5, -1, 5)
        self.logView = QPlainTextEdit(MainWindow)
        self.logView.setObjectName(u"logView")
        self.logView.setReadOnly(True)

        self.gridLayout.addWidget(self.logView, 3, 1, 1, 1)

        self.tblStrengthView = QTableView(MainWindow)
        self.tblStrengthView.setObjectName(u"tblStrengthView")

        self.gridLayout.addWidget(self.tblStrengthView, 0, 1, 1, 1)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.tblNewGames = QTableView(MainWindow)
        self.tblNewGames.setObjectName(u"tblNewGames")

        self.verticalLayout.addWidget(self.tblNewGames)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(-1, 0, -1, 0)
        self.btnStart = QPushButton(MainWindow)
        self.btnStart.setObjectName(u"btnStart")
        self.btnStart.setMinimumSize(QSize(0, 40))
        self.btnStart.setMaximumSize(QSize(100, 16777215))
        self.btnStart.setFont(font)

        self.horizontalLayout_7.addWidget(self.btnStart)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_3)


        self.verticalLayout.addLayout(self.horizontalLayout_7)

        self.tblPredictionGames = QTableView(MainWindow)
        self.tblPredictionGames.setObjectName(u"tblPredictionGames")
        self.tblPredictionGames.setMinimumSize(QSize(0, 250))

        self.verticalLayout.addWidget(self.tblPredictionGames)


        self.gridLayout.addLayout(self.verticalLayout, 3, 0, 1, 1)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setSpacing(7)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(-1, 0, -1, 0)
        self.tblLearningGames = QTableView(MainWindow)
        self.tblLearningGames.setObjectName(u"tblLearningGames")
        self.tblLearningGames.setMinimumSize(QSize(0, 250))

        self.horizontalLayout_5.addWidget(self.tblLearningGames)

        self.horizontalLayout_5.setStretch(0, 2)

        self.gridLayout.addLayout(self.horizontalLayout_5, 0, 0, 1, 1)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(-1, 5, -1, 5)
        self.btnAddGame = QPushButton(MainWindow)
        self.btnAddGame.setObjectName(u"btnAddGame")
        self.btnAddGame.setMinimumSize(QSize(0, 40))
        self.btnAddGame.setMaximumSize(QSize(200, 16777215))
        font2 = QFont()
        font2.setPointSize(11)
        self.btnAddGame.setFont(font2)

        self.horizontalLayout_2.addWidget(self.btnAddGame)

        self.btnRemoveGame = QPushButton(MainWindow)
        self.btnRemoveGame.setObjectName(u"btnRemoveGame")
        self.btnRemoveGame.setMinimumSize(QSize(0, 40))
        self.btnRemoveGame.setFont(font2)

        self.horizontalLayout_2.addWidget(self.btnRemoveGame)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer_4)


        self.gridLayout.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_5 = QLabel(MainWindow)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setFont(font)

        self.horizontalLayout_3.addWidget(self.label_5)

        self.btnClearLog = QPushButton(MainWindow)
        self.btnClearLog.setObjectName(u"btnClearLog")

        self.horizontalLayout_3.addWidget(self.btnClearLog)

        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_5)


        self.gridLayout.addLayout(self.horizontalLayout_3, 2, 1, 1, 1)

        self.gridLayout.setColumnStretch(0, 2)
        self.gridLayout.setColumnStretch(1, 1)

        self.verticalLayout_2.addLayout(self.gridLayout)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(-1, 5, -1, 5)
        self.label_7 = QLabel(MainWindow)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setFont(font2)

        self.horizontalLayout_4.addWidget(self.label_7)

        self.lblResult = QLabel(MainWindow)
        self.lblResult.setObjectName(u"lblResult")
        self.lblResult.setFont(font2)

        self.horizontalLayout_4.addWidget(self.lblResult)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_2)


        self.verticalLayout_2.addLayout(self.horizontalLayout_4)


        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"FootballAnalysis", None))
        self.btnViewLearningData.setText(QCoreApplication.translate("MainWindow", u"Learning Data View", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"League", None))
        self.btnScraping.setText(QCoreApplication.translate("MainWindow", u"scraping", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Learning Season", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"~", None))
        self.btnViewTeam.setText(QCoreApplication.translate("MainWindow", u"View Team", None))
        self.btnStart.setText(QCoreApplication.translate("MainWindow", u"Prediction", None))
        self.btnAddGame.setText(QCoreApplication.translate("MainWindow", u"Add", None))
        self.btnRemoveGame.setText(QCoreApplication.translate("MainWindow", u"Remove", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Log Viewer", None))
        self.btnClearLog.setText(QCoreApplication.translate("MainWindow", u"Clear", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Result:", None))
        self.lblResult.setText("")
    # retranslateUi

