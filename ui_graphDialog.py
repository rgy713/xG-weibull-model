# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'graphDialog.ui'
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


class Ui_GraphDialog(object):
    def setupUi(self, GraphDialog):
        if not GraphDialog.objectName():
            GraphDialog.setObjectName(u"GraphDialog")
        GraphDialog.resize(731, 511)
        self.verticalLayout_2 = QVBoxLayout(GraphDialog)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")

        self.verticalLayout_2.addLayout(self.verticalLayout)


        self.retranslateUi(GraphDialog)

        QMetaObject.connectSlotsByName(GraphDialog)
    # setupUi

    def retranslateUi(self, GraphDialog):
        GraphDialog.setWindowTitle(QCoreApplication.translate("GraphDialog", u"Dialog", None))
    # retranslateUi

