# This is the UI window which can help user to category sketch with different quality
import sys
import os
from pyUI.select_quality import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from visualize_compare import superimpose, load_image
from csv_parser import CsvManager
from functools import partial


def cv2Qimg(img_arry):
    height, width, channel = img_arry.shape
    bytesPerLine = 3 * width
    qImg = QtGui.QImage(img_arry.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
    return qImg


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.root = 'C:/Users/tan/Desktop/data/'
        self.img_root = self.root + 'image/'
        self.pen_root = self.root + 'input/'
        self.skt_root = self.root + 'output/'

        self.name_list = os.listdir(self.img_root)
        self.name_list.sort()
        self.imgNum = len(self.name_list)

        self.fileManager = CsvManager('test.csv')
        self.index = self.fileManager.resume_index

        # If I finished the whole annotation process, the resume index should be the last one instead of last + 1
        if self.index == self.imgNum:
            self.index = self.imgNum - 1

        self.curr_quali = -1
        self.curr_compli = -1

        self.ui.nextBtn.clicked.connect(self.next_image)
        self.ui.lastBtn.clicked.connect(self.last_image)

        # ====================
        # quality buttons
        # ====================
        self.ui.firstBtn.clicked.connect(partial(self.set_quality, 1))
        self.ui.secondBtn.clicked.connect(partial(self.set_quality, 2))
        self.ui.thirdBtn.clicked.connect(partial(self.set_quality, 3))

        # ====================
        # complicity buttons
        # ====================
        self.ui.complexBtn.clicked.connect(partial(self.set_complicity, 1))
        self.ui.mediumBtn.clicked.connect(partial(self.set_complicity, 2))
        self.ui.simpleBtn.clicked.connect(partial(self.set_complicity, 3))

        self.ui.saveBtn.clicked.connect(self.save_file)

        # Initial
        self.fill_imgWin(self.index)
        self.setProcessBar()
        self.setCheckBox()

    # set process bar
    def setProcessBar(self):
        self.ui.progressBar.setValue(int(round((self.index/self.imgNum)*100)))

    # set checkbox using the current quality and complicity info
    def setCheckBox(self):
        # ==== quality =======
        if self.curr_quali == 1:
            self.ui.firstCB.setCheckState(True)
            self.ui.secondCB.setCheckState(False)
            self.ui.thirdCB.setCheckState(False)
        elif self.curr_quali == 2:
            self.ui.firstCB.setCheckState(False)
            self.ui.secondCB.setCheckState(True)
            self.ui.thirdCB.setCheckState(False)
        elif self.curr_quali == 3:
            self.ui.firstCB.setCheckState(False)
            self.ui.secondCB.setCheckState(False)
            self.ui.thirdCB.setCheckState(True)
        else:
            self.ui.firstCB.setCheckState(False)
            self.ui.secondCB.setCheckState(False)
            self.ui.thirdCB.setCheckState(False)

        # ==== complicity =======
        if self.curr_compli == 1:
            self.ui.firstCB_2.setCheckState(True)
            self.ui.secondCB_2.setCheckState(False)
            self.ui.thirdCB_2.setCheckState(False)
        elif self.curr_compli == 2:
            self.ui.firstCB_2.setCheckState(False)
            self.ui.secondCB_2.setCheckState(True)
            self.ui.thirdCB_2.setCheckState(False)
        elif self.curr_compli == 3:
            self.ui.firstCB_2.setCheckState(False)
            self.ui.secondCB_2.setCheckState(False)
            self.ui.thirdCB_2.setCheckState(True)
        else:
            self.ui.firstCB_2.setCheckState(False)
            self.ui.secondCB_2.setCheckState(False)
            self.ui.thirdCB_2.setCheckState(False)

    # keyboard shortcuts
    def keyPressEvent(self, event):
        if event.key() == 76:  # L
            self.next_image()
        elif event.key() == 75:  # K
            self.last_image()
        elif event.key() == 49:  # 1
            pass

    # fill image windows, usually be called after loading a new image
    def fill_imgWin(self, index):
        name = self.name_list[index]
        img = load_image(self.img_root + name)
        pen = load_image(self.pen_root + name)
        skt = load_image(self.skt_root + name)
        sup = superimpose(img, skt)
        self.ui.filenameTxt.setText(self.skt_root + name)

        self.ui.imageWin1.setPixmap(QtGui.QPixmap(cv2Qimg(img)))
        self.ui.imageWin2.setPixmap(QtGui.QPixmap(cv2Qimg(pen)))
        self.ui.imageWin3.setPixmap(QtGui.QPixmap(cv2Qimg(skt)))
        self.ui.imageWin4.setPixmap(QtGui.QPixmap(cv2Qimg(sup)))

    def next_image(self):
        if self.curr_quali == -1 or self.curr_compli == -1:
            print("You haven't finished your choice, please press the buttons first!!!")
        else:
            # 1. save works
            self.fileManager.set(self.index, self.name_list[self.index], self.curr_quali, self.curr_compli)

            print(self.index)
            # 2. prepare for the next shot
            self.index += 1
            if self.index == self.imgNum:
                self.index = self.imgNum - 1

            self.fill_imgWin(self.index)
            self.setProcessBar()

            if self.index < len(self.fileManager):  # if there exist record, load in
                self.curr_quali = self.fileManager.quali[self.index]
                self.curr_compli = self.fileManager.compli[self.index]
            else:
                self.curr_quali = -1
                self.curr_compli = -1

            self.setCheckBox()

    def last_image(self):
        # self.fileManager.set(self.index, self.name_list[self.index], self.curr_quali, self.curr_compli)

        self.index -= 1
        if self.index < 0:
            self.index = 0

        self.fill_imgWin(self.index)
        self.setProcessBar()

        if self.index < len(self.fileManager):  # if there exist record, load in
            self.curr_quali = self.fileManager.quali[self.index]
            self.curr_compli = self.fileManager.compli[self.index]
        else:
            self.curr_quali = -1
            self.curr_compli = -1

        self.setCheckBox()

    def set_quality(self, quality):
        self.curr_quali = quality
        self.setCheckBox()

    def set_complicity(self, comp):
        self.curr_compli = comp
        self.setCheckBox()

    def save_file(self):
        self.fileManager.write()


# launch window
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())



