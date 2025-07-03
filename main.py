# -*- coding: utf-8 -*-
"""
@author: Connor J Newstead
"""

import os
from DCIUI import *
from dimensionalityreduction import Dimensionality_Reduction
from dataparser import DataParser
from dci import DCI
from Bayesotscript import DCIopt
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget, QFileDialog, QMessageBox, QLabel
from PyQt5 import QtGui
from PyQt5.QtCore import QPropertyAnimation, QSize, Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont
import io
from PIL import Image
 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

"""Class for offloading demanding processes to a separate worker thread so that the GUI remains responsive"""
class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)  # Optional for progress reporting
 
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
 
    def run(self):
        self.result = self.func(*self.args, **self.kwargs)
        self.finished.emit()

"""Class for creating widgets for Figures"""
class FigureWindow(QMainWindow):
    def __init__(self, fig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Figure Viewer")
        self.setMinimumSize(800, 600)
 
        self.canvas = FigureCanvas(fig)
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.canvas)
 
        self.setCentralWidget(central_widget)
        self.canvas.draw()
 
"""Main GUI functions are within this class, as well as connecting buttons/ GUI settings to external class functions like DCI"""
class MainWindow(QMainWindow):
    """All GUI updates must be completed in the main thread so pyqtSignals are created for their respective functions"""
    update_gui_signal = pyqtSignal(object)
    figure_ready = pyqtSignal(object)
    emb_and_hmap = pyqtSignal(object)
    hide_label_thread = pyqtSignal(object)
    show_label_thread = pyqtSignal(object)
    bayoptlabel_thread = pyqtSignal(object)
    loadwarning = pyqtSignal(object)
    dictwarning = pyqtSignal(object)
    clearcanvas = pyqtSignal(object)
    emitdciscore = pyqtSignal(object)
    
    def __init__(self):
        self.ui = Ui_MainWindow()
        super().__init__()
        self.ui.setupUi(self)
        """Connecting pyqtSignals to functions"""
        self.update_gui_signal.connect(self.create_canvas)
        self.figure_ready.connect(self.new_window_fig)
        self.hide_label_thread.connect(self.hidelabel)
        self.show_label_thread.connect(self.showlabel)
        self.bayoptlabel_thread.connect(self.bayoptlabel)
        self.loadwarning.connect(self.load_warning)
        self.dictwarning.connect(self.dict_warning)
        self.clearcanvas.connect(self.clear_canvas)
        self.emitdciscore.connect(self.emit_dci_score)
        
        """Preloading the loading animation and inserting it into a QLabel widget"""
        self.logo_label = QLabel()
        self.movie = QtGui.QMovie("Beeloading.gif")
        self.logo_label.setMovie(self.movie)
        self.movie.start()
        self.load_tag = QLabel("Generating/ Loading... Please wait")
        self.load_tag.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.load_tag.setFont(font)
        self.ui.loading_screen_layout.addWidget(self.load_tag)
        index = self.ui.loading_screen_layout.indexOf(self.logo_label)
        self.ui.loading_screen_layout.insertWidget(index+1, self.load_tag)
        self.logo_label.hide()
        self.ui.loading_screen_layout.addWidget(self.logo_label, alignment=Qt.AlignHCenter)
        
        """Initial left menu size"""
        self.collapsed_width = 55  # Width when collapsed
        self.expanded_width = 200  # Width when expanded
        self.ui.LeftMenuContainer.setMaximumWidth(self.collapsed_width)

        """Connecting mouse events/ tracking"""
        self.ui.LeftMenuContainer.setMouseTracking(True)  # Enable mouse tracking
        self.ui.LeftMenuContainer.enterEvent = self.expand_menu
        self.ui.LeftMenuContainer.leaveEvent = self.collapse_menu
        self.ui.ToolsMenu.setCurrentIndex(0)
 
        """Connecting buttons to changing gui pages"""
        self.ui.OpenButton.clicked.connect(lambda: self.ui.ToolsMenu.setCurrentIndex(0))
        self.ui.DCIButton.clicked.connect(lambda: self.ui.ToolsMenu.setCurrentIndex(1))
        self.ui.drbutton.clicked.connect(lambda: self.ui.ToolsMenu.setCurrentIndex(2))
        self.ui.dr_tsne.clicked.connect(lambda: self.ui.ToolsMenu.setCurrentIndex(3))
        self.ui.HelpButton.clicked.connect(lambda: self.ui.ToolsMenu.setCurrentIndex(4))
        
        """Connecting sliders to spin box updating"""
        self.ui.clustersamplesslider.valueChanged.connect(self.ui.clustersamplespercent.setValue)
        self.ui.clustersamplespercent.valueChanged.connect(self.ui.clustersamplesslider.setValue)   
        self.ui.horizontalSlider_2.valueChanged.connect(self.ui.randomsamples.setValue)
        self.ui.randomsamples.valueChanged.connect(self.ui.horizontalSlider_2.setValue)
        
        """Placeholder values"""
        self.canvas = None
        self.save_mat_path = None
        self.data = None
        self.embedded_data = None
        self.dataparser = None
        self.mask = None
        self.mzs = None
        self.dci_dict_savepath = None
        self.name = "Placeholder name"
        self.emb_name = "Placeholder name"
        self.sampling = "cluster"
        self.percent = 5
        self.num_pixels = None
        self.init = self.ui.initselectionbox.currentText()
        
        """Connecting the load data buttons to their respective functions"""
        self.ui.openmatbutton.clicked.connect(lambda: self.run_func_in_thread(self.load_datacube))
        self.ui.opendatacsvbutton.clicked.connect(lambda: self.run_func_in_thread(self.load_data_npy))
        self.ui.openmzscsv.clicked.connect(self.load_mzs_npy)
        self.ui.openmaskbutton.clicked.connect(self.load_mask_npy)
        self.ui.openembeddingnpy.clicked.connect(self.load_embedding_dict)
        self.ui.loadspectraldatabutton.clicked.connect(self.load_embedding_npy)
        
        """Connecting DCI, Bayesion opt and t-SNE functions to their buttons"""
        self.ui.rundcibutton.clicked.connect(lambda: self.run_func_in_thread(self.run_dci))
        self.ui.runoptbutton.clicked.connect(lambda: self.run_func_in_thread(self.bayesopt))
        self.ui.tsnecreateembbutton.clicked.connect(lambda: self.run_func_in_thread(self.run_tsne))
        
        """Connecting help buttons to their functions"""
        self.ui.whatdata_helpbtn.clicked.connect(self.help_whatdata)  
        self.ui.matbroken_helpbtn.clicked.connect(self.help_matbroken)
        self.ui.csvbroken_helpbtn.clicked.connect(self.help_csvbroken)
        self.ui.HowDCI_helpbtn.clicked.connect(self.help_howdci)
        self.ui.HowBayes_helpbtn.clicked.connect(self.help_howbayes)
        self.ui.Whatparams_helpbtn.clicked.connect(self.help_whatparams)
        self.ui.Whatoutputs_helpbtn.clicked.connect(self.help_whatoutputs)
        self.ui.loadingbeecheck.stateChanged.connect(self.loadingscreen_toggle)
        
    """Help buttons"""
    def help_whatdata(self):
        QMessageBox.information(self, "Information", r"A loaded .mat file should have been saved in v7.3. Additionally, within the .mat, data should be saved as data, mzs should be saved as spectralChannels, and the mask should be saved as pixelSelection within regionOfInterest saved as a struct. Otherwise, the relevant npy/csv files can be loaded. This GUI outputs tsne embedding dict.npy files compatible with the load embedding dict.npy button in the open tab, and should be loaded if possible. Alternatively, an embedding .npy/csv file can be loaded, and the embedding name will be the loaded filename.")
    def help_matbroken(self):
        QMessageBox.information(self, "Information", r".mat file should have been saved in v7.3. Within the .mat, data should be saved as data, mzs should be saved as spectralChannels, and the mask should be saved as pixelSelection within regionOfInterest saved as a struct.")
    def help_csvbroken(self):
        QMessageBox.information(self, "Information", r"The data npy/csv should be a 2D array of samples/pixels (rows) and features/mz (columns). Ensure there is no header or index. npy files are loaded with np.load.")
    def help_howdci(self):
        QMessageBox.information(self, "Information", r"First, ensure that either a .mat has been loaded, or all relevant npy/csv files in the open tab. Select one of two methods - the cluster method (which uses n% of data pixels as samples), or the random sampling method. All relevant output files will be saved in your chosen dir.")
    def help_howbayes(self):
        QMessageBox.information(self, "Information", r"First, ensure that either a .mat has been loaded, or all relevant npy/csv files in the open tab. Choose the desired maximum number of hyperparameters for the optimisation, in addition to the number of iterations.")                
    def help_whatparams(self):
        QMessageBox.information(self, "Information", r"The selection of perplexity and exaggeration under Bayesion Optimisation controls the maximum value of hyperparameter used in the optimisation process. The Initialisation method can be changed between random and pca. If default is chosen, random is selected.")
    def help_whatoutputs(self):
        QMessageBox.information(self, "Information", r"The DCI function will output a .npy dictionary of high and low dimensional cophenetic distance matrices, the selected samples, high and low dimensional linkage matrices, centroids of clusters (if applicable), and the DCI score which is the mutual information between cophenetic distance matrices. The Bayesian optimization function will output all created embedding dictionaries as a .npy, in addition to the DCI dictionaries for each embedding, and the optimisation figure.")
        
    """Change load_tag text"""
    def bayoptlabel(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.loading_screen)
        self.logo_label.show()
        self.load_tag.setText("Generating... Please wait... Live figures can be found in your working directory")
        self.load_tag.show()
    
    """Show the loading gif and text"""
    def showlabel(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.loading_screen)
        self.logo_label.show()
        self.load_tag.setText("Generating/ Loading... Please wait")
        self.load_tag.show()
    
    """Hide loading gif and text"""
    def hidelabel(self):
        self.logo_label.hide()
        if self.load_tag is not None:
            self.load_tag.hide()
    
    """Expand the left side menu when mouse moves in"""
    def expand_menu(self, event):
        # Animate expanding the menu
        self.ui.LeftMenuContainer.setMaximumWidth(200)
        self.animate_menu(self.expanded_width)
 
    """Collapse left menu when mouse moves out"""
    def collapse_menu(self, event):
        # Animate collapsing the menu
        self.anim1 = QPropertyAnimation(self.ui.LeftMenuContainer, b'maximumWidth')
        self.anim1.setDuration(400)
        self.anim1.setEndValue(self.collapsed_width)
        self.anim1.start()
        
    """The animation for the left menu expanding"""
    def animate_menu(self, target_width):
        # Animation for menu resizing
        self.anim = QPropertyAnimation(self.ui.LeftMenuContainer, b'size')
        self.anim.setDuration(400)
        current_size = self.ui.LeftMenuContainer.size()
        new_size = QSize(target_width, current_size.height())
        self.anim.setEndValue(new_size)
        self.anim.start()
        
    """Warning for not loading .mat or data"""
    def load_warning(self):
        QMessageBox.information(self, "Error", "You must upload an embedding, and either upload a .mat or all three of data, mask and mzs npy!")
        self.hidelabel()
        
    """Warning for not selecting a savepath"""
    def dict_warning(self):
        QMessageBox.information(self, "Error", "A dict savepath has not been selected")
        self.hidelabel()
        
    """Information popup informing of the DCI score"""
    def emit_dci_score(self):
        QMessageBox.information(self, "Information", f"DCI score: {self.dci_score}")
        
    """Function for loading a datacube .mat"""
    def load_datacube(self):
        path = self.save_mat_path
        if path.endswith(".mat"):
            self.show_label_thread.emit(None)
            self.dataparser = DataParser(mat_path=path)
            self.data = self.dataparser.data
            self.mask = self.dataparser.mask
            self.mzs = self.dataparser.mzs
            self.hide_label_thread.emit(None)
        if not path:
            self.loadwarning.emit(None)
            return
            
    """Function for loading a data.npy or csv"""
    def load_data_npy(self):
        path = self.datanpypath
        if not path:
            self.loadwarning.emit(None)
            return
        
        if path.endswith(".npy"):
            self.show_label_thread.emit(None)
            self.name = os.path.splitext(os.path.basename(path))[0]
            self.hide_label_thread.emit(None)
            return np.load(path, allow_pickle=True)
        if path.endswith(".csv"):
            self.show_label_thread.emit(None)
            self.name = os.path.splitext(os.path.basename(path))[0]
            self.hide_label_thread.emit(None)
            return np.genfromtxt(path, delimiter=',')
    
    """Function for loading a mask.npy or csv"""
    def load_mask_npy(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open mask .npy or .csv")
        if not path:
            return
        if path.endswith(".npy"):
            self.mask = np.load(path, allow_pickle=True)
            self.ui.openmaskedit.setText(path)
        if path.endswith(".csv"):
            self.mask = np.genfromtxt(path, delimiter=',')
            self.ui.openmaskedit.setText(path)
            print(self.mask)
            
    """Function for loading a mzs.npy or csv"""
    def load_mzs_npy(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open mzs .npy or .csv")
        if not path:
            return
        if path.endswith(".npy"):
            self.mzs = np.load(path, allow_pickle=True)
            self.ui.openmzsedit.setText(path)
        if path.endswith(".csv"):
            self.mzs = np.genfromtxt(path, delimiter=',')
            self.ui.openmzsedit.setText(path)
            print(self.mask)
        
    """Function for loading an embedding.npy or csv"""
    def load_embedding_npy(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open embedding .npy or .csv")
        if not path:
            return
        if path.endswith(".npy"):
            self.embedded_data = np.load(path, allow_pickle=True)
            self.emb_name = os.path.splitext(os.path.basename(path))[0]
            self.ui.loadmebeddingcsvedit.setText(path)
        if path.endswith(".csv"):
            self.embedded_data = np.genfromtxt(path, delimiter=',')
            self.emb_name = os.path.splitext(os.path.basename(path))[0]
            self.ui.loadmebeddingcsvedit.setText(path)
    
    """Function for loading an embedding dict compatible with the outputs of the Dimensionality_Reduction class"""
    def load_embedding_dict(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open embedding .npy")
        if not path:
            return
        if path.endswith(".npy"):
            self.embedding_parser = DataParser()
            self.embedding_parser.load_embedding(path)
            self.embedded_data = self.embedding_parser.embedded_data
            print(self.embedded_data)
            self.emb_name = self.embedding_parser.emb_name
            self.ui.embeddingnpyedit.setText(path)        
            
    """Function for setting a save path for the embedding dict"""
    def save_emb_dict(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save DCI dict .npy")
        if not path:
            return
        else:
            self.emb_dict_savepath = path
            
    """Function for setting a DCI dict save path"""
    def save_dci_dict_filename(self):
        path = QFileDialog.getExistingDirectory(self, "Save DCI data and figures")
        if not path:
            return
        else:
            self.dci_dict_savepath = path
    
    """Function to toggle the loading gif"""
    def loadingscreen_toggle(self):
        if self.ui.loadingbeecheck.isChecked():
            self.movie = QtGui.QMovie("Beeloading.gif")
            self.logo_label.setMovie(self.movie)
            self.movie.start()
            self.logo_label.hide()
        else:
            self.movie = QtGui.QMovie("spinner2.gif")
            self.logo_label.setMovie(self.movie)
            self.movie.start()
            self.logo_label.hide()
    
    """Function to convert a matplotlib figure to PIL image"""
    def fig_to_img(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.0)
        buf.seek(0)
        img = Image.open(buf)
        img = np.array(img)
        return img
            
    """Function to create a canvas widget and embed within the figure_page StackedWidget page"""
    def create_canvas(self, img):
        # Create the canvas
        self.ui.stackedWidget.setCurrentWidget(self.ui.figure_page)
        self.canvas = FigureCanvas(Figure(figsize=(3,3)))
        self.ui.figure_page_layout.addWidget(self.canvas)
        self.ax= self.canvas.figure.subplots()
        
        self.ax.clear()
        self.ax.imshow(img)
        self.ax.axis('off')
        
        self.canvas.draw()
    
    """Function to remove canvas widget"""
    def clear_canvas(self):
        self.ui.verticalLayout_10.removeWidget(self.canvas)
        self.canvas.setParent(None)
        self.canvas = None
    
    """Function to popout a new window for a figure"""
    def new_window_fig(self, fig):
        self.figure_window = FigureWindow(fig)
        self.figure_window.show()
        
    """Function to produce t-SNE embeddings using the Dimensionality_Reduction class"""
    def run_tsne(self):
        self.show_label_thread.emit(None)
        perplexity = self.ui.spinBox.value()
        exaggeration = self.ui.spinBox_2.value()
        components = self.ui.spinBox_3.value()
        distance = self.ui.comboBox_2.currentText()
        init = self.ui.comboBox_3.currentText()
        if init=='default':init='random'
        
        if self.dataparser is not None:
            self.show_label_thread.emit(None)
            dr_obj = Dimensionality_Reduction(data_parser = self.dataparser)
            embedding, emb_dict = dr_obj.tSNE_creation(perplexity = perplexity, exaggeration = exaggeration, 
                                                       n_components = components, metric = distance, init = init)

            self.dataparser.save_data(self.emb_dict_savepath, emb_dict)
            self.hide_label_thread.emit(None)
        
        if self.dataparser is None:
            self.show_label_thread.emit(None)
            self.dataparser = DataParser(name=self.name, data=self.data, mzs=self.mzs, mask=self.mask)
            dr_obj = Dimensionality_Reduction(data_parser = self.dataparser)
            embedding, emb_dict = dr_obj.tSNE_creation(perplexity = perplexity, exaggeration = exaggeration, 
                                                       n_components = components, metric = distance, init = init)
            self.dataparser.save_data(self.emb_dict_savepath, emb_dict)
            self.hide_label_thread.emit(None)
    
    """return a worker thread result back to the main thread"""
    def on_worker_finished(self):
        if hasattr(self.worker, "result") and self.worker.result is not None:
            self.data = self.worker.result
        else:
            return
    
    """Function to run DCI with the selected parameters, using the DCI class"""
    def run_dci(self):
            
        if self.ui.checkBox.isChecked():
            self.sampling = "cluster"
            self.percent = int(self.ui.clustersamplespercent.value())
            
        if self.ui.checkBox_2.isChecked():
            self.sampling = "random"
            self.num_pixels = int(self.ui.randomsamples.value())
            
        if self.dataparser is None:
            if self.dci_dict_savepath is None:
                self.dictwarning.emit(None)
                return
            if self.data is None:
                self.loadwarning.emit(None)
                return
            if self.embedded_data is None:
                self.loadwarning.emit(None)
                return
            self.show_label_thread.emit(None)
            self.dataparser = DataParser(name=self.name, data=self.data, mzs=self.mzs, mask=self.mask)
            dci_obj = DCI(self.embedded_data, data_parser=self.dataparser, emb_name=self.emb_name)
            
            dci, dci_dict, emb_img, fig = dci_obj.dci(sampling=self.sampling, percentage=self.percent, num_pixels=self.num_pixels) 
            path0 = self.dci_dict_savepath+"\\"+self.emb_name+" Embedding.png"
            path1 = self.dci_dict_savepath+"\\"+self.emb_name+" Highd_Low2_Coph_heatmap.png"
            plt.imsave(path0, emb_img)
            fig.savefig(path1)
            if self.embedded_data.shape[1]==3:
                self.update_gui_signal.emit(emb_img)
            self.figure_ready.emit(fig)
            path2 = self.dci_dict_savepath+"\\"+self.emb_name+" DCI dict"
            self.dataparser.save_data(path2, dci_dict)
            self.dci_score = dci_dict['DCI']
            self.emitdciscore.emit(None)
            self.hide_label_thread.emit(None)
            
        else:
            if self.dci_dict_savepath is None:
                self.dictwarning.emit(None)
                return
            if self.embedded_data is None:
                self.loadwarning.emit(None)
                return
            self.show_label_thread.emit(None)
            dci_obj = DCI(self.embedded_data, data_parser=self.dataparser, emb_name=self.emb_name)
            dci, dci_dict, emb_img, fig= dci_obj.dci(sampling=self.sampling, percentage=self.percent, num_pixels=self.num_pixels)
            path0 = self.dci_dict_savepath+"\\"+self.emb_name+" Embedding.png"
            path1 = self.dci_dict_savepath+"\\"+self.emb_name+" Highd_Low2_Coph_heatmap.png"
            plt.imsave(path0, emb_img)
            fig.savefig(path1)
            if self.embedded_data.shape[1]==3:
                self.update_gui_signal.emit(emb_img)
            self.figure_ready.emit(fig)
            path2 = self.dci_dict_savepath+"\\"+self.emb_name+" DCI dict"
            self.dataparser.save_data(path2, dci_dict)
            self.dci_score = dci_dict['DCI']
            self.emitdciscore.emit(None)
            self.hide_label_thread.emit(None)
            
    """Function to run other functions in another thread so the main thread containing the GUI remains responsive"""
    def run_func_in_thread(self, func):
        if func==self.run_dci:
            if self.canvas is not None:
                self.clear_canvas()
            self.save_dci_dict_filename()
            
        if func==self.run_tsne:
            if self.canvas is not None:
                self.clear_canvas()
            self.save_emb_dict()
            
        if func == self.load_datacube:
            self.save_mat_path, _ = QFileDialog.getOpenFileName(self, "Open datacube .mat")
            self.ui.openmatedit.setText(self.save_mat_path)
            self.clear_canvas
            
        if func == self.load_data_npy:
            self.datanpypath, _ = QFileDialog.getOpenFileName(self, "Open data .npy or .csv")
            self.ui.opendataedit.setText(self.datanpypath)
            self.clear_canvas
        
        if func == self.bayesopt:
            if self.canvas is not None:
                self.clear_canvas()
            self.save_dci_dict_filename()

        self.thread = QThread()
        self.worker = Worker(func)
        self.worker.moveToThread(self.thread)
     
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self.on_worker_finished)
        self.thread.start()
    
    """Function to run the Bayesion optimisation using the selected parameters, using the DCIopt class"""
    def bayesopt(self):

        if self.ui.checkBox.isChecked():
            self.sampling = "cluster"
            self.percent = int(self.ui.clustersamplespercent.value())
            
        if self.ui.checkBox_2.isChecked():
            self.sampling = "random"
            print(self.sampling)
            self.num_pixels = int(self.ui.randomsamples.value())
            
        if self.dataparser is None:
            if self.dci_dict_savepath is None:
                self.dictwarning.emit(None)
                return
            if self.data is None:
                self.loadwarning.emit(None)
                return
            self.bayoptlabel_thread.emit(None)
            self.dataparser = DataParser(name=self.name, data=self.data, mzs=self.mzs, mask=self.mask)
            algo = self.ui.comboBox.currentText()
            max_neighbours = self.ui.maxneighboursinput.value()
            max_exaggeration = self.ui.maxexaggerationinput.value()
            iters = self.ui.maxexaggerationinput_2.value()
            kwargs = {'algo':algo, 'neighbours':max_neighbours, 'exaggeration':max_exaggeration, 
                      'iters':iters, 'init':self.init, 'sampling':self.sampling, 'percent':self.percent, 'num_pixels':self.num_pixels}
            dciopt_obj = DCIopt(self.dataparser, self.dci_dict_savepath, kwargs)
            scores, best_params = dciopt_obj.runopt()
            score_path = self.dci_dict_savepath+"\\DCI score.csv"
            params_path = self.dci_dict_savepath+"\\Best params.txt"
            np.savetxt(score_path, scores, delimiter=',')
            np.savetxt(params_path, best_params, delimiter=',')
            self.hide_label_thread.emit(None)
            
        if self.dataparser is not None:
            if self.dci_dict_savepath is None:
                self.dictwarning.emit(None)
                return
            self.bayoptlabel_thread.emit(None)
            algo = self.ui.comboBox.currentText()
            max_neighbours = self.ui.maxneighboursinput.value()
            max_exaggeration = self.ui.maxexaggerationinput.value()
            iters = self.ui.maxexaggerationinput_2.value()
            kwargs = {'algo':algo, 'neighbours':max_neighbours, 'exaggeration':max_exaggeration, 
                      'iters':iters, 'init':self.init, 'sampling':self.sampling, 'percent':self.percent, 'num_pixels':self.num_pixels}
            dciopt_obj = DCIopt(self.dataparser, self.dci_dict_savepath, kwargs)
            scores, best_params = dciopt_obj.runopt()
            score_path = self.dci_dict_savepath+"\\DCI score.csv"
            params_path = self.dci_dict_savepath+"\\Best params.txt"
            np.savetxt(score_path, scores, delimiter=',')
            np.savetxt(params_path, best_params, delimiter=',')
            self.hide_label_thread.emit(None)
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())