import random
from solve import *
from solve_custom import SolveExpTh
from read_data import read_data
from operator_table import OperatorViewWindow

from PyQt5.QtCore import pyqtSlot, pyqtSignal, Qt
from PyQt5.QtGui import QTextDocument, QFont
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox, QTableWidgetItem
from PyQt5.uic import loadUiType

import os

START_FUEL = 47
TIME_DELTA = 10
reason = [u'Мала бортова напруга\n',u'Недостатній запас палива\n',u'Низький рівень заряду АБ\n']
#reason = [u'Малий прибуток; ',u'Недостатній запас ходу; ',u'Низький рівень заряду АБ; ']

def calculate_rdr_delta(ycurrent, yf,  yd):
    maxl = np.max(yf[:-1] - yf[1:])
    return (ycurrent - yd)/ maxl

def lblText(lbl, text):
    lbl.setText(str(text)[:10])
    return

def prob(x, xmax, xmin):
    # print(x, xmax, xmin)
    res = np.fabs((x - xmax) / (xmax - xmin))
    r = np.ma.array(res, mask=np.array(x >= xmax), fill_value=0)
    return r.filled()

def insert_data(tw, row, data):
    assert len(data) <= 8
    try:
        for i, d in enumerate(data):
            item = QTableWidgetItem(d)
            item.setTextAlignment(Qt.AlignHCenter)
            tw.setItem(row, i, item)
    except Exception as e:
        raise ('insert data in table' + str(e))


def classify_danger_rating(level):
    if 0 <= level <= 0.07:
        return 0, u"Безпечна ситуація"
    elif 0.07 < level <= 0.25:
        return 1, u"Нештатна ситуація по одному параметру"
    elif 0.25 < level <= 0.375:
        return 2, u"Нештатна ситуація по декількох параметрах"
    elif 0.375 < level <= 0.5:
        return 3, u"Спостерігається загроза аварії"
    elif 0.5 < level <= 0.625:
        return 4, u"Висока загроза аварії"
    elif 0.625 < level <= 0.75:
        return 5, u"Критична ситуація"
    elif 0.75 < level <= 0.875:
        return 6, u"Шанс уникнути аварії дуже малий"
    elif 0.875 < level:
        return 7, u"Аварія"


class SolverManager(object):
    Y_C = np.array([[11.7], [1], [12]])  # warning value
    Y_D = np.array([[10.5], [0.0], [10.5]])  # failure value

    def __init__(self, d):
        self.custom_struct = d['custom_struct']
        d['dimensions'][3] = 1
        if d['custom_struct']:
            self.solver = SolveExpTh(d)
        else:
            self.solver = Solve(d)
        self.first_launch = True
        self.batch_size = d['samples']
        self.forecast_size = d['pred_steps']
        self.operator_view = OperatorViewWindow(warn=self.Y_C, fail=self.Y_D, callback=self,
                                                descriptions=[u'боротова\ напруга', u'запас\ палива',
                                                              u'напруга\ в\ АБ'],
                                                tail=self.forecast_size)
        self.current_iter = 1
        self.y_influenced = None
        self.data_window = None
        self.tablewidget = d['tablewidget']
        self.reason = np.array([],dtype = str) # init warning reason
        self.lbl = d['lbl']

    def prepare(self, filename):
        self.time, self.data = read_data(filename)
        increment = self.time[-1] - self.time[-2]
        self.time = np.append(self.time, np.arange(1, 1 + self.forecast_size) * increment + self.time[-1])
        self.N_all_iter = len(self.time)
        self.operator_view.show()
        self.operator_view.status_bar.showMessage('Loaded successfully.', 1000)

    def start_machine(self):
        self.operator_view.start_process()

    def launch(self):
        if self.current_iter + self.batch_size < len(self.data):
            self.fit(self.current_iter, self.batch_size)
            self.current_iter += 1
        else:
            self.operator_view.timer.stop()

    def fit(self, shift, n):
        # print('fit', shift, n)

        self.data_window = self.data[shift:shift + n]
        self.solver.load_data(self.data_window[:, :-2])  # y2 and y3 not used
        self.solver.prepare()
        self.predict(shift, n)
        self.check_sensors_consistency()
        self.risk()  # really suspicious realisation
        self.y_influenced = [y * (1 - self.f) for y in self.y_forecasted]
        if self.first_launch:
            self.operator_view.initial_graphics_fill(real_values=self.data_window[:, -3:],
                                                     predicted_values=self.y_forecasted,
                                                     risk_values=self.y_influenced,
                                                     time_ticks=self.time[shift:shift + n + self.solver.pred_step])
            self.first_launch = False
        else:
            self.operator_view.update_graphics(self.data_window[-1, -3:], self.y_forecasted, self.y_influenced,
                                               self.time[shift + n - 1:shift + n + self.solver.pred_step])


        fuel = (13120 - self.time[shift] ) / 13120 * self.data_window[-1, -3:][1]
        self.y_current = np.array([self.solver.Y_[-1,0], fuel, self.solver.X_[0][-1,0]])

        # self.y_current = np.array([self.solver.Y_[-1,0], self.solver.X_[0][-1,3], self.solver.X_[1][-1,2]])
        self.rdr_calc()
        self.current_data()
        self.table_data_forecasted()

    def check_sensors_consistency(self):
        # mask_positive = np.array([True, True, True, True, True, True, False,
        #                           True, True, True, True, True, False,
        #                           False, True, True, True, False])
        mask_positive = np.array([False, False, False, False, False, False, False,
                                  False, False, ])
        result_positive = (self.data_window[:, :-3] < 0).max(axis=0) * mask_positive
        if result_positive.any():
            self.operator_view.status_bar.showMessage('Sensors {} have problems'.format(
                    str(np.where(result_positive)[0].tolist())), 1000)
        else:
            self.operator_view.status_bar.showMessage('OK',1000)



    def risk(self):
        self.p = prob(self.y_forecasted, self.Y_C, self.Y_D)
        # print('p', self.p)
        self.f = 1 - (1 - self.p[0, :]) * (1 - self.p[1, :]) * (1 - self.p[2, :])
        for i in range(len(self.f)):
            if self.f[i] > 1:
                self.f[i] = 1
        #calculate reason warning situation
        self.reason = np.array([],dtype = str)
        for i in range(self.forecast_size):
            string = ''
            for j in range(3):
                if self.p[j, i] > 0:
                    string = string + reason[j]
            if string == '':
                self.reason = np.append(self.reason, '-')
            else:
                self.reason = np.append(self.reason, string)
        assert len(self.reason) == self.forecast_size

        self.danger_rate = np.array([classify_danger_rating(i) for i in self.f])

    def predict(self, shift, n):
        fuel = np.array(list(
            [self.data[self.current_iter + self.batch_size + i][-2] + random.randrange(-1, 1) / 50
             for i in range(self.forecast_size)]
        ))
        voltage = np.array(list(
            [x[0] for x in self.solver.X_[0][-10:, 0].tolist()]
        ))
        self.y_forecasted = [self.solver.YF, fuel, voltage]
        return

    def table_data_forecasted(self):
        t = self.time[self.batch_size + self.current_iter: self.batch_size + self.current_iter + self.solver.pred_step]
        y1 = self.y_forecasted[0]
        y2 = self.y_forecasted[1]
        y3 = self.y_forecasted[2]
        state = self.danger_rate[:, 1]
        risk = self.f
        reason =self.reason
        rate = self.danger_rate[:, 0]
        data = np.array([t, y1, y2, y3, state, risk, reason, rate]).T
        assert data.shape == (self.solver.pred_step, 8)
        for i ,j in enumerate(data):
            insert_data(self.tablewidget, i, j)
        return

    def current_data(self):
        lblText(self.lbl['time'], self.time[self.batch_size + self.current_iter -1])
        lblText(self.lbl['y1'], self.y_current[0])
        lblText(self.lbl['y2'], self.y_current[1])
        lblText(self.lbl['y3'], self.y_current[2])
        lblText(self.lbl['rmr'], self.rdr)


    def rdr_calc(self):
        self.rdr = np.inf
        rdr = np.inf
        for i in range(3):
            if self.y_current[i]>=self.Y_C[i,0]:
                continue
            if self.y_current[i]<= self.Y_D  [i,0]:
                rdr = 0
                continue
            s = calculate_rdr_delta(self.y_current[i], self.y_influenced[i], self.Y_D[i,0])
            if s <= 0:
                continue
            t =(self.y_current[i] - self.Y_D[i,0])/s
            if t < rdr:
                rdr = t
        if rdr != np.inf:
            self.rdr = rdr

