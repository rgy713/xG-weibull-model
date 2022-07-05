# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'intervalEditor.ui'
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


class Ui_IntervalEditDialog(object):
    def setupUi(self, IntervalEditDialog):
        if not IntervalEditDialog.objectName():
            IntervalEditDialog.setObjectName(u"IntervalEditDialog")
        IntervalEditDialog.resize(472, 390)
        self.buttonBox = QDialogButtonBox(IntervalEditDialog)
        self.buttonBox.setObjectName(u"buttonBox")
        self.buttonBox.setGeometry(QRect(250, 350, 201, 32))
        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QDialogButtonBox.Cancel|QDialogButtonBox.Ok)
        self.tableView = QTableView(IntervalEditDialog)
        self.tableView.setObjectName(u"tableView")
        self.tableView.setGeometry(QRect(20, 10, 431, 321))
        self.btnAddInterval = QPushButton(IntervalEditDialog)
        self.btnAddInterval.setObjectName(u"btnAddInterval")
        self.btnAddInterval.setGeometry(QRect(20, 350, 93, 28))
        self.btnRemoveInterval = QPushButton(IntervalEditDialog)
        self.btnRemoveInterval.setObjectName(u"btnRemoveInterval")
        self.btnRemoveInterval.setGeometry(QRect(120, 350, 93, 28))

        self.retranslateUi(IntervalEditDialog)
        self.buttonBox.accepted.connect(IntervalEditDialog.accept)
        self.buttonBox.rejected.connect(IntervalEditDialog.reject)

        QMetaObject.connectSlotsByName(IntervalEditDialog)
    # setupUi

    def retranslateUi(self, IntervalEditDialog):
        IntervalEditDialog.setWindowTitle(QCoreApplication.translate("IntervalEditDialog", u"Interval Editor", None))
        self.btnAddInterval.setText(QCoreApplication.translate("IntervalEditDialog", u"Add", None))
        self.btnRemoveInterval.setText(QCoreApplication.translate("IntervalEditDialog", u"Remove", None))
    # retranslateUi

