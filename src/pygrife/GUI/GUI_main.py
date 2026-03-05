import json
import os
import sys
import time

# from .QtImageViewer import QtImageViewer
from pathlib import Path
from queue import Queue

import astropy.io.fits as pf
import numpy as np
import PyQt6
import qimage2ndarray
from astropy import wcs
from astropy.visualization import (
    AsinhStretch,
    LinearStretch,
    LogStretch,
    ManualInterval,
    SqrtStretch,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPalette, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .qt_utils import TerminalWindow, Worker, WriteStream
from .seg_map_viewer import FilesWindow, LineBrowse, SegMapViewer, Separator


class ExtractorGUI(SegMapViewer):
    def __init__(
        self,
        field_root: str = "nis-wfss",
        detection_filter: str = "ir",
        filters: list[str] = ["F115W", "F150W", "F200W"],
        new_dir_name: str | None = "ForcedExtractions",
        new_dir_path: str | os.PathLike | None = None,
        **kwargs,
    ):
        super().__init__(
            field_root=field_root,
            detection_filter=detection_filter,
            filters=filters,
            **kwargs,
        )

        self.new_dir_name = new_dir_name
        self.new_dir_path = new_dir_path

        self.layout_side.addWidget(Separator())
        self.extract_object_button = QPushButton("Extract Object", self)
        self.extract_object_button.clicked.connect(self.extraction_handler)
        self.layout_side.addWidget(self.extract_object_button)

        self.terminal_window = None
        self.extract_in_progress = False

        self.ge = None

    def open_files_window(self, event=None):
        if self.files_window is None:
            self.files_window = GrizliFilesWindow(self)
        self.files_window.show()

    def open_terminal_window(self, event=None):
        if self.terminal_window is None:
            self.terminal_window = TerminalWindow(self)
        self.terminal_window.show()

    def save_output(self, event=None):
        if not (hasattr(self, "seg_img_path") and hasattr(self, "seg_data")):
            print("No segmentation mask loaded.")
            return

        out_dir_path = self._setup_new_dir()

        with open(out_dir_path / "remapped_ids.json", "w") as f:
            json.dump(self.remapped_ids, f)

        with pf.open(self.seg_img_path) as seg_hdul:
            seg_hdul[0].data = self.seg_data[::-1, :]

            seg_hdul.writeto(out_dir_path / self.seg_img_path.name, overwrite=True)

    def receiver_fn(self, queue, progress_callback=None):
        while self.extract_in_progress:
            text = queue.get()
            progress_callback.emit(text)
        return

    def extraction_handler(self, event=None):
        self.extract_in_progress = True
        self.extract_object_button.setEnabled(False)
        self.open_terminal_window()

        queue = Queue()
        sys.stdout = WriteStream(queue)

        receive_worker = Worker(self.receiver_fn, queue)
        receive_worker.signals.progress.connect(self.terminal_window.append_text)
        self.threadpool.start(receive_worker)

        extract_worker = Worker(self.extract_object)
        # worker.signals.progress.connect(self.progress_fn)
        # worker.signals.result.connect(self.root.set_img)
        extract_worker.signals.finished.connect(self.finish_extractions)
        self.threadpool.start(extract_worker)

        # self.threadpool

    def finish_extractions(self):
        # print ("pls end")
        self.extract_in_progress = False
        self.extract_object_button.setEnabled(True)
        sys.stdout = sys.__stdout__

    def _setup_new_dir(self) -> os.PathLike:

        if self.new_dir_path is not None:
            out_dir_path = Path(self.new_dir_path)
        elif self.new_dir_name is not None:
            out_dir_path = Path(self.in_dir.parent) / self.new_dir_name
        else:
            raise NameError(
                "Either the path or name of the output directory must be supplied."
            )

        out_dir_path.mkdir(exist_ok=True, parents=True)
        return out_dir_path

    def extract_object(self, event=None, progress_callback=None):
        # import logging
        # root = logging.getLogger()
        # root.setLevel(logging.INFO)
        # fh = logging.FileHandler('debug.log')
        # fh.setLevel(logging.INFO)

        # old_stdout = sys.stdout    # in case you want to restore later
        # sys.stdout = fh.stream

        # root.addHandler(fh)
        print("Beginning extraction.")
        print(self.selected_ids)

        out_dir_path = self._setup_new_dir()

        if self.ge is None:
            self.ge = GrismExtractor(
                field_root=self.field_root,
                in_dir=self.in_dir,
                out_dir=out_dir_path,
            )
        self.ge.load_seg_img(self.seg_data[::-1, :])
        self.ge.regen_multiband_catalogue()
        if not hasattr(self.ge, "grp"):
            self.ge.load_contamination_maps()
        # self.ge.extract_sep()
        self.ge.extract_spectra(self.selected_ids)
        # sys.stdout = old_stdout
        return


class GrizliFilesWindow(FilesWindow):
    def __init__(self, root):
        super().__init__(root)

        self.in_dir_line = LineBrowse(
            parent=self, is_dir=True, root_name="in_dir"
        )
        self.sub_layout.insertRow(0, "Input Directory", self.in_dir_line)

    def select_from_directory(self, event=None):
        if self.in_dir_line.line.text() is not (None or ""):
            self._load_from_dir(self.in_dir_line.line.text())
            return

        if self.root.in_dir is not None:
            init = str(self.root.in_dir)
        elif self.recent_dir is not None:
            init = str(self.recent_dir)
        else:
            init = str(Path.home())

        dir_name = QFileDialog.getExistingDirectory(self, "Open directory", init)

        if dir_name:
            self._load_from_dir(dir_name)
            self.in_dir_line.line.setText(str(dir_name))

    def _load_from_dir(self, dir_name):
        super()._load_from_dir(dir_name)
        self.in_dir_line.line.setText(str(dir_name))


def run_GUI(name: str = "extractor", **kwargs):
    if name.lower() == "extractor":
        GUI = ExtractorGUI
    elif name.lower() == "viewer":
        GUI = SegMapViewer
    else:
        raise NameError("Please specify one of `extractor` or `viewer`.")

    app = QApplication(sys.argv)
    window = GUI(**kwargs)
    window.showMaximized()
    sys.exit(app.exec())
