from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.uic import loadUiType

from find_degrees import *
from solve_custom import SolveExpTh
from solve import Solve
from begin_window import Ui_Form

# form_class, base_class = loadUiType('bruteforce_window.ui')


class BruteForceWindow(QDialog, Ui_Form):
    update_degrees = pyqtSignal(int, int, int)

    def __init__(self, *args):
        super(BruteForceWindow, self).__init__(*args)
        self.setupUi(self)

    @staticmethod
    def launch(parent):
        dialog = BruteForceWindow(parent)
        dialog.params = parent._get_params()
        dialog.custom_struct = parent.custom_func_struct
        dialog.update_degrees.connect(parent.update_degrees)
        dialog.setWindowTitle("Polynomial's degree finder")
        dialog.show()

    @pyqtSlot()
    def triggered(self):
        self.low_edge  = [self.from_1.value(), self.from_2.value(), self.from_3.value()]
        self.high_edge = [self.to_1.value(), self.to_2.value(), self.to_3.value()]
        self.step = [self.st_1.value(), self.st_2.value(), self.st_3.value()]
        if self.custom_struct:
            solver = SolveExpTh(self.params)
        else:
            solver = Solve(self.params)
        p = [[i for i in range(self.low_edge[j], self.high_edge[j]+1, self.step[j])] for j in range(len(self.step))]
        best_deg = determine_deg(solver, p[0], p[1], p[2])
        bd = best_deg[0]
        self.res_1.setValue(bd[0])
        self.res_2.setValue(bd[1])
        self.res_3.setValue(bd[2])

        msgbox = QMessageBox()

        msgbox.setText('Best degrees:'+bd.__str__()+'.')
        msgbox.setInformativeText("Do you want to copy degrees in main window?")
        msgbox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msgbox.setDefaultButton(QMessageBox.Ok)
        ret = msgbox.exec_()
        if ret == QMessageBox.Ok:
            self.update_degrees.emit(bd[0],bd[1], bd[2])
            self.close()
        # result = QMessageBox.question(self, 'Long-time operation',
        #                               'Adjusting degrees lasts long. Do you want to perform it?',
        #                               QMessageBox.Ok | QMessageBox.No)
        # if result == QMessageBox.Ok:
        #     BruteForceWindow.launch(self)

        #self.update_degrees.emit(3,3,3)
        return

    def _process_bruteforce(self, lower, upper):
        pass

