# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Hsuan\Course\MS\DigitalSignal\Work4\Analysis_ui\Record_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MEanalysis(object):
    def setupUi(self, MEanalysis):
        MEanalysis.setObjectName("MEanalysis")
        MEanalysis.resize(600, 700)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MEanalysis.sizePolicy().hasHeightForWidth())
        MEanalysis.setSizePolicy(sizePolicy)
        MEanalysis.setMinimumSize(QtCore.QSize(600, 700))
        MEanalysis.setStyleSheet("background-color: rgb(63, 66, 56);")
        self.centralwidget = QtWidgets.QWidget(MEanalysis)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMinimumSize(QtCore.QSize(600, 700))
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Grid = QtWidgets.QGridLayout()
        self.Grid.setHorizontalSpacing(0)
        self.Grid.setVerticalSpacing(6)
        self.Grid.setObjectName("Grid")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(50)
        self.line_3.setFont(font)
        self.line_3.setStyleSheet("color: rgb(0, 0, 0);")
        self.line_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_3.setLineWidth(10)
        self.line_3.setMidLineWidth(5)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setObjectName("line_3")
        self.Grid.addWidget(self.line_3, 14, 0, 1, 6)
        self.horizontalFrame_2 = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalFrame_2.sizePolicy().hasHeightForWidth())
        self.horizontalFrame_2.setSizePolicy(sizePolicy)
        self.horizontalFrame_2.setMinimumSize(QtCore.QSize(360, 0))
        self.horizontalFrame_2.setObjectName("horizontalFrame_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalFrame_2)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.RecordTime_bar = QtWidgets.QProgressBar(self.horizontalFrame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.RecordTime_bar.sizePolicy().hasHeightForWidth())
        self.RecordTime_bar.setSizePolicy(sizePolicy)
        self.RecordTime_bar.setMinimumSize(QtCore.QSize(0, 15))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.RecordTime_bar.setFont(font)
        self.RecordTime_bar.setAutoFillBackground(False)
        self.RecordTime_bar.setStyleSheet("min-height: 15px;\n"
"max-height: 15px;\n"
"border-radius: 50px;\n"
"background-color: #6b705c;")
        self.RecordTime_bar.setMaximum(15)
        self.RecordTime_bar.setProperty("value", 7)
        self.RecordTime_bar.setAlignment(QtCore.Qt.AlignCenter)
        self.RecordTime_bar.setTextVisible(True)
        self.RecordTime_bar.setOrientation(QtCore.Qt.Horizontal)
        self.RecordTime_bar.setTextDirection(QtWidgets.QProgressBar.TopToBottom)
        self.RecordTime_bar.setFormat("")
        self.RecordTime_bar.setObjectName("RecordTime_bar")
        self.horizontalLayout_3.addWidget(self.RecordTime_bar)
        self.counter = QtWidgets.QLabel(self.horizontalFrame_2)
        self.counter.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.counter.sizePolicy().hasHeightForWidth())
        self.counter.setSizePolicy(sizePolicy)
        self.counter.setMinimumSize(QtCore.QSize(80, 0))
        self.counter.setMaximumSize(QtCore.QSize(80, 16777215))
        self.counter.setBaseSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.counter.setFont(font)
        self.counter.setStyleSheet("color: rgb(229, 229, 229);\n"
"")
        self.counter.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.counter.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.counter.setLineWidth(10)
        self.counter.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.counter.setObjectName("counter")
        self.horizontalLayout_3.addWidget(self.counter, 0, QtCore.Qt.AlignRight)
        self.Grid.addWidget(self.horizontalFrame_2, 9, 3, 1, 2)
        self.CNN_result = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.CNN_result.sizePolicy().hasHeightForWidth())
        self.CNN_result.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.CNN_result.setFont(font)
        self.CNN_result.setStyleSheet("color: #ccc5b9;\n"
"min-width: 180px;")
        self.CNN_result.setText("")
        self.CNN_result.setAlignment(QtCore.Qt.AlignCenter)
        self.CNN_result.setWordWrap(True)
        self.CNN_result.setObjectName("CNN_result")
        self.Grid.addWidget(self.CNN_result, 16, 3, 1, 1)
        self.clock = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clock.sizePolicy().hasHeightForWidth())
        self.clock.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(8)
        self.clock.setFont(font)
        self.clock.setStyleSheet("color: rgb(229, 229, 229);\n"
"min-width: 180px;")
        self.clock.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.clock.setObjectName("clock")
        self.Grid.addWidget(self.clock, 0, 4, 1, 1)
        self.SVMresult_text = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SVMresult_text.sizePolicy().hasHeightForWidth())
        self.SVMresult_text.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(12)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.SVMresult_text.setFont(font)
        self.SVMresult_text.setStyleSheet("color: rgb(182, 173, 144);\n"
"min-width: 180px;")
        self.SVMresult_text.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.SVMresult_text.setFrameShadow(QtWidgets.QFrame.Raised)
        self.SVMresult_text.setLineWidth(3)
        self.SVMresult_text.setAlignment(QtCore.Qt.AlignCenter)
        self.SVMresult_text.setObjectName("SVMresult_text")
        self.Grid.addWidget(self.SVMresult_text, 15, 4, 1, 1)
        self.Start = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Start.sizePolicy().hasHeightForWidth())
        self.Start.setSizePolicy(sizePolicy)
        self.Start.setMinimumSize(QtCore.QSize(204, 0))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.Start.setFont(font)
        self.Start.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.Start.setMouseTracking(False)
        self.Start.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.Start.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.Start.setStyleSheet("color: #ccc5b9;\n"
"background-color: #6b705c;\n"
"border-color: #333533;\n"
"border-width: 2px;\n"
"height: 30px;\n"
"padding: 10px;\n"
"border-style: solid;\n"
"border-radius: 13px;\n"
"min-width: 180px;")
        self.Start.setCheckable(True)
        self.Start.setAutoExclusive(True)
        self.Start.setAutoDefault(False)
        self.Start.setDefault(False)
        self.Start.setFlat(True)
        self.Start.setObjectName("Start")
        self.Grid.addWidget(self.Start, 11, 3, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMaximumSize(QtCore.QSize(160, 16777215))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(229, 229, 229);")
        self.label.setObjectName("label")
        self.Grid.addWidget(self.label, 0, 0, 1, 2)
        self.Info_text = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Info_text.sizePolicy().hasHeightForWidth())
        self.Info_text.setSizePolicy(sizePolicy)
        self.Info_text.setMaximumSize(QtCore.QSize(160, 16777215))
        font = QtGui.QFont()
        font.setFamily("Helvetica Condensed")
        font.setPointSize(10)
        self.Info_text.setFont(font)
        self.Info_text.setStyleSheet("color: rgb(229, 229, 229);")
        self.Info_text.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Info_text.setObjectName("Info_text")
        self.Grid.addWidget(self.Info_text, 20, 0, 1, 1)
        self.horizontalFrame = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalFrame.sizePolicy().hasHeightForWidth())
        self.horizontalFrame.setSizePolicy(sizePolicy)
        self.horizontalFrame.setMinimumSize(QtCore.QSize(360, 0))
        self.horizontalFrame.setObjectName("horizontalFrame")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalFrame)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.audio_name = QtWidgets.QLineEdit(self.horizontalFrame)
        self.audio_name.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.audio_name.sizePolicy().hasHeightForWidth())
        self.audio_name.setSizePolicy(sizePolicy)
        self.audio_name.setMinimumSize(QtCore.QSize(0, 25))
        self.audio_name.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        self.audio_name.setFont(font)
        self.audio_name.setStyleSheet("color: rgb(229, 229, 229);\n"
"")
        self.audio_name.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.audio_name.setDragEnabled(True)
        self.audio_name.setClearButtonEnabled(True)
        self.audio_name.setObjectName("audio_name")
        self.horizontalLayout_4.addWidget(self.audio_name)
        self.FileName_checkbox = QtWidgets.QCheckBox(self.horizontalFrame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.FileName_checkbox.sizePolicy().hasHeightForWidth())
        self.FileName_checkbox.setSizePolicy(sizePolicy)
        self.FileName_checkbox.setMinimumSize(QtCore.QSize(80, 0))
        self.FileName_checkbox.setMaximumSize(QtCore.QSize(80, 16777215))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.FileName_checkbox.setFont(font)
        self.FileName_checkbox.setStyleSheet("color: rgb(229, 229, 229);\n"
"")
        self.FileName_checkbox.setCheckable(True)
        self.FileName_checkbox.setChecked(False)
        self.FileName_checkbox.setTristate(False)
        self.FileName_checkbox.setObjectName("FileName_checkbox")
        self.horizontalLayout_4.addWidget(self.FileName_checkbox, 0, QtCore.Qt.AlignRight)
        self.Grid.addWidget(self.horizontalFrame, 6, 3, 1, 2)
        self.FileName_text = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.FileName_text.sizePolicy().hasHeightForWidth())
        self.FileName_text.setSizePolicy(sizePolicy)
        self.FileName_text.setMaximumSize(QtCore.QSize(160, 16777215))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.FileName_text.setFont(font)
        self.FileName_text.setStyleSheet("color: rgb(229, 229, 229);")
        self.FileName_text.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.FileName_text.setObjectName("FileName_text")
        self.Grid.addWidget(self.FileName_text, 6, 0, 1, 3)
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(50)
        font.setBold(True)
        font.setWeight(75)
        self.line_4.setFont(font)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_4.setLineWidth(10)
        self.line_4.setMidLineWidth(5)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setObjectName("line_4")
        self.Grid.addWidget(self.line_4, 18, 0, 1, 6)
        self.SVM_result = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SVM_result.sizePolicy().hasHeightForWidth())
        self.SVM_result.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.SVM_result.setFont(font)
        self.SVM_result.setStyleSheet("color: #ccc5b9;\n"
"min-width: 180px;")
        self.SVM_result.setText("")
        self.SVM_result.setAlignment(QtCore.Qt.AlignCenter)
        self.SVM_result.setWordWrap(True)
        self.SVM_result.setObjectName("SVM_result")
        self.Grid.addWidget(self.SVM_result, 16, 4, 1, 1)
        self.SVM_analyze = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SVM_analyze.sizePolicy().hasHeightForWidth())
        self.SVM_analyze.setSizePolicy(sizePolicy)
        self.SVM_analyze.setMinimumSize(QtCore.QSize(204, 0))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.SVM_analyze.setFont(font)
        self.SVM_analyze.setStyleSheet("color: #ccc5b9;\n"
"background-color: #6b705c;\n"
"border-color: #333533;\n"
"border-width: 2px;\n"
"height: 30px;\n"
"padding: 10px;\n"
"border-style: solid;\n"
"border-radius: 13px;\n"
"min-width: 180px;")
        self.SVM_analyze.setCheckable(True)
        self.SVM_analyze.setAutoExclusive(True)
        self.SVM_analyze.setObjectName("SVM_analyze")
        self.Grid.addWidget(self.SVM_analyze, 13, 4, 1, 1)
        self.Analysi_text = QtWidgets.QLabel(self.centralwidget)
        self.Analysi_text.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Analysi_text.sizePolicy().hasHeightForWidth())
        self.Analysi_text.setSizePolicy(sizePolicy)
        self.Analysi_text.setMaximumSize(QtCore.QSize(160, 16777215))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.Analysi_text.setFont(font)
        self.Analysi_text.setStyleSheet("color: rgb(229, 229, 229);")
        self.Analysi_text.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Analysi_text.setObjectName("Analysi_text")
        self.Grid.addWidget(self.Analysi_text, 13, 0, 1, 3)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(50)
        font.setBold(True)
        font.setWeight(75)
        self.line_2.setFont(font)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_2.setLineWidth(10)
        self.line_2.setMidLineWidth(5)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setObjectName("line_2")
        self.Grid.addWidget(self.line_2, 12, 0, 1, 6)
        self.horizontalFrame1 = QtWidgets.QFrame(self.centralwidget)
        self.horizontalFrame1.setMinimumSize(QtCore.QSize(360, 0))
        self.horizontalFrame1.setObjectName("horizontalFrame1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalFrame1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.FilePath = QtWidgets.QLineEdit(self.horizontalFrame1)
        self.FilePath.setMinimumSize(QtCore.QSize(0, 25))
        self.FilePath.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(11)
        self.FilePath.setFont(font)
        self.FilePath.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.FilePath.setStyleSheet("color: rgb(229, 229, 229);\n"
"")
        self.FilePath.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.FilePath.setReadOnly(True)
        self.FilePath.setObjectName("FilePath")
        self.horizontalLayout_2.addWidget(self.FilePath)
        self.browse = QtWidgets.QToolButton(self.horizontalFrame1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.browse.sizePolicy().hasHeightForWidth())
        self.browse.setSizePolicy(sizePolicy)
        self.browse.setMinimumSize(QtCore.QSize(80, 0))
        self.browse.setMaximumSize(QtCore.QSize(80, 16777215))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.browse.setFont(font)
        self.browse.setStyleSheet("color: rgb(229, 229, 229);\n"
"")
        self.browse.setObjectName("browse")
        self.horizontalLayout_2.addWidget(self.browse, 0, QtCore.Qt.AlignRight)
        self.Grid.addWidget(self.horizontalFrame1, 7, 3, 1, 2)
        self.FilePath_text = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.FilePath_text.sizePolicy().hasHeightForWidth())
        self.FilePath_text.setSizePolicy(sizePolicy)
        self.FilePath_text.setMaximumSize(QtCore.QSize(160, 16777215))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(13)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.FilePath_text.setFont(font)
        self.FilePath_text.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.FilePath_text.setStyleSheet("color: rgb(229, 229, 229);")
        self.FilePath_text.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.FilePath_text.setObjectName("FilePath_text")
        self.Grid.addWidget(self.FilePath_text, 7, 0, 1, 3)
        self.Results_text = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Results_text.sizePolicy().hasHeightForWidth())
        self.Results_text.setSizePolicy(sizePolicy)
        self.Results_text.setMaximumSize(QtCore.QSize(160, 16777215))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.Results_text.setFont(font)
        self.Results_text.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.Results_text.setStyleSheet("color: rgb(229, 229, 229);")
        self.Results_text.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.Results_text.setObjectName("Results_text")
        self.Grid.addWidget(self.Results_text, 15, 0, 2, 3)
        self.Stop = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Stop.sizePolicy().hasHeightForWidth())
        self.Stop.setSizePolicy(sizePolicy)
        self.Stop.setMinimumSize(QtCore.QSize(204, 0))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.Stop.setFont(font)
        self.Stop.setStyleSheet("color: #ccc5b9;\n"
"background-color: #6b705c;\n"
"border-color: #333533;\n"
"border-width: 2px;\n"
"height: 30px;\n"
"padding: 10px;\n"
"border-style: solid;\n"
"border-radius: 13px;\n"
"min-width: 180px;")
        self.Stop.setCheckable(True)
        self.Stop.setAutoExclusive(True)
        self.Stop.setObjectName("Stop")
        self.Grid.addWidget(self.Stop, 11, 4, 1, 1)
        self.line_1 = QtWidgets.QFrame(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(50)
        font.setBold(True)
        font.setWeight(75)
        self.line_1.setFont(font)
        self.line_1.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_1.setLineWidth(10)
        self.line_1.setMidLineWidth(5)
        self.line_1.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_1.setObjectName("line_1")
        self.Grid.addWidget(self.line_1, 2, 0, 1, 6)
        self.Title = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Title.sizePolicy().hasHeightForWidth())
        self.Title.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.Title.setFont(font)
        self.Title.setAutoFillBackground(False)
        self.Title.setStyleSheet("color: rgb(229, 229, 229);")
        self.Title.setLineWidth(3)
        self.Title.setAlignment(QtCore.Qt.AlignCenter)
        self.Title.setObjectName("Title")
        self.Grid.addWidget(self.Title, 1, 0, 1, 5)
        self.RecordingTime_text = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.RecordingTime_text.sizePolicy().hasHeightForWidth())
        self.RecordingTime_text.setSizePolicy(sizePolicy)
        self.RecordingTime_text.setMaximumSize(QtCore.QSize(160, 16777215))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.RecordingTime_text.setFont(font)
        self.RecordingTime_text.setTabletTracking(False)
        self.RecordingTime_text.setStyleSheet("color: rgb(229, 229, 229);")
        self.RecordingTime_text.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.RecordingTime_text.setObjectName("RecordingTime_text")
        self.Grid.addWidget(self.RecordingTime_text, 9, 0, 1, 3)
        self.CNNresult_text = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.CNNresult_text.sizePolicy().hasHeightForWidth())
        self.CNNresult_text.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(12)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.CNNresult_text.setFont(font)
        self.CNNresult_text.setStyleSheet("color: rgb(182, 173, 144);\n"
"min-width: 180px;")
        self.CNNresult_text.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.CNNresult_text.setFrameShadow(QtWidgets.QFrame.Raised)
        self.CNNresult_text.setLineWidth(3)
        self.CNNresult_text.setAlignment(QtCore.Qt.AlignCenter)
        self.CNNresult_text.setObjectName("CNNresult_text")
        self.Grid.addWidget(self.CNNresult_text, 15, 3, 1, 1)
        self.CNN_analyze = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.CNN_analyze.sizePolicy().hasHeightForWidth())
        self.CNN_analyze.setSizePolicy(sizePolicy)
        self.CNN_analyze.setMinimumSize(QtCore.QSize(204, 0))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.CNN_analyze.setFont(font)
        self.CNN_analyze.setStyleSheet("color: #ccc5b9;\n"
"background-color: #6b705c;\n"
"border-color: #333533;\n"
"border-width: 2px;\n"
"height: 30px;\n"
"padding: 10px;\n"
"border-style: solid;\n"
"border-radius: 13px;\n"
"min-width: 180px;")
        self.CNN_analyze.setCheckable(True)
        self.CNN_analyze.setAutoExclusive(True)
        self.CNN_analyze.setAutoDefault(False)
        self.CNN_analyze.setFlat(False)
        self.CNN_analyze.setObjectName("CNN_analyze")
        self.Grid.addWidget(self.CNN_analyze, 13, 3, 1, 1)
        self.Info = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Info.sizePolicy().hasHeightForWidth())
        self.Info.setSizePolicy(sizePolicy)
        self.Info.setMinimumSize(QtCore.QSize(360, 20))
        self.Info.setMaximumSize(QtCore.QSize(16777215, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(10)
        self.Info.setFont(font)
        self.Info.setStyleSheet("color: rgb(229, 229, 229);")
        self.Info.setFrame(False)
        self.Info.setReadOnly(True)
        self.Info.setPlaceholderText("")
        self.Info.setCursorMoveStyle(QtCore.Qt.LogicalMoveStyle)
        self.Info.setObjectName("Info")
        self.Grid.addWidget(self.Info, 20, 2, 1, 3)
        self.horizontalLayout.addLayout(self.Grid)
        MEanalysis.setCentralWidget(self.centralwidget)

        self.retranslateUi(MEanalysis)
        QtCore.QMetaObject.connectSlotsByName(MEanalysis)

    def retranslateUi(self, MEanalysis):
        _translate = QtCore.QCoreApplication.translate
        MEanalysis.setWindowTitle(_translate("MEanalysis", "Music Emotion Analyser"))
        self.counter.setText(_translate("MEanalysis", "00：00 s"))
        self.clock.setText(_translate("MEanalysis", "Date： 1999/04/30 17:01:00"))
        self.SVMresult_text.setText(_translate("MEanalysis", "<html><head/><body><p>   SVM Result</p></body></html>"))
        self.Start.setText(_translate("MEanalysis", "Start"))
        self.label.setText(_translate("MEanalysis", "Group 12"))
        self.Info_text.setText(_translate("MEanalysis", "Info. ：   "))
        self.audio_name.setText(_translate("MEanalysis", "temp_audio"))
        self.FileName_checkbox.setText(_translate("MEanalysis", "Enable"))
        self.FileName_text.setText(_translate("MEanalysis", "File Name(.wav)：  "))
        self.SVM_analyze.setText(_translate("MEanalysis", "SVM"))
        self.Analysi_text.setText(_translate("MEanalysis", "Analyze Method：  "))
        self.FilePath.setText(_translate("MEanalysis", ".../"))
        self.browse.setText(_translate("MEanalysis", "..."))
        self.FilePath_text.setText(_translate("MEanalysis", "File Path：  "))
        self.Results_text.setText(_translate("MEanalysis", "Results：  "))
        self.Stop.setText(_translate("MEanalysis", "Stop"))
        self.Title.setWhatsThis(_translate("MEanalysis", "<html><head/><body><p><br/></p></body></html>"))
        self.Title.setText(_translate("MEanalysis", "Music Emotion Analyser"))
        self.RecordingTime_text.setText(_translate("MEanalysis", "Recording Time：  "))
        self.CNNresult_text.setText(_translate("MEanalysis", "<html><head/><body><p>  CNN Result</p></body></html>"))
        self.CNN_analyze.setText(_translate("MEanalysis", "CNN"))