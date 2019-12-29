from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_OperatorWindow(object):
    def setupUi(self, OperatorWindow):
        OperatorWindow.setObjectName("Оператор")
        OperatorWindow.resize(1280, 800)
        self.windowLayout = QtWidgets.QVBoxLayout(OperatorWindow)
        self.windowLayout.setObjectName("windowLayout")
        self.y_layout = QtWidgets.QVBoxLayout()
        self.y_layout.setObjectName("y_layout")
        self.windowLayout.addLayout(self.y_layout)
        self.line = QtWidgets.QFrame(OperatorWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.line.sizePolicy().hasHeightForWidth())
        self.line.setSizePolicy(sizePolicy)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.windowLayout.addWidget(self.line)
        self.buttons_layout = QtWidgets.QHBoxLayout()
        self.buttons_layout.setObjectName("buttons_layout")
        self.start_button = QtWidgets.QPushButton(OperatorWindow)
        self.start_button.setObjectName("start_button")
        self.buttons_layout.addWidget(self.start_button)
        self.windowLayout.addLayout(self.buttons_layout)
        self.windowLayout.setStretch(0, 10)
        self.windowLayout.setStretch(2, 1)

        self.retranslateUi(OperatorWindow)
        self.start_button.pressed.connect(OperatorWindow.manipulate_timer)
        QtCore.QMetaObject.connectSlotsByName(OperatorWindow)

    def retranslateUi(self, OperatorWindow):
        _translate = QtCore.QCoreApplication.translate
        OperatorWindow.setWindowTitle(_translate("OperatorWindow", "Оператор"))
        self.start_button.setText(_translate("OperatorWindow", "РОЗПОЧАТИ ДІАГНОСТУВАННЯ"))

