import os
import sys
import time
from pathlib import Path

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
from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal, pyqtSlot
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
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .qt_utils import QtImageViewer, Worker, WorkerSignals


class SegMapViewer(QMainWindow):
    """
    A Qt-based viewer for segmentation maps.

    This composites together an RGB image from three separate images, and
    overlays a segmentation map on top, which can be modified through the supplied tools.

    Parameters
    ----------
    QMainWindow : _type_
        _description_
    """

    def __init__(
        self,
        field_root: str = "nis-wfss",
        detection_filter: str = "ir",
        filters: list[str] = ["F115W", "F150W", "F200W"],
        **kwargs,
    ):
        super(SegMapViewer, self).__init__()

        self.threadpool = QThreadPool()

        self.filters = filters
        self.field_root = field_root
        self.detection_filter = detection_filter

        self.setWindowTitle("Object Selection")

        self.layout_h = QHBoxLayout()

        self.left_toolbar = QWidget(self)
        self.left_toolbar.setFixedWidth(175)
        self.layout_side = QVBoxLayout(self.left_toolbar)
        self.layout_side.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.in_dir = None
        self.seg_img_path = None
        self.stack_img_path = None
        self.r_img_path = None
        self.g_img_path = None
        self.b_img_path = None

        self.files_window = None
        dir_sel_button = QPushButton("Select Files", self)
        dir_sel_button.clicked.connect(self.open_files_window)
        self.layout_side.addWidget(dir_sel_button)
        self.layout_side.addWidget(Separator())

        self.seg_text = QLabel(f"\n\n", self)
        self.layout_side.addWidget(self.seg_text)
        self.layout_side.addWidget(Separator())

        # Image stretch
        self.stretch = SqrtStretch()
        stretch_box = QComboBox()
        stretch_box.addItems(["Logarithmic", "Square Root", "Asinh", "Linear"])
        stretch_box.setCurrentText("Square Root")
        stretch_box.currentTextChanged.connect(self.stretch_update)

        stretch_label = QLabel("Stretch:", self)
        self.layout_side.addWidget(stretch_label)
        self.layout_side.addWidget(stretch_box)

        # Interval
        self.interval = ManualInterval(0, 1)
        interval_box = QComboBox()
        self.interval_keys = ["minmax", "99.9%", "99.8%", "99.5%", "99%", "98%", "95%"]
        interval_box.addItems(self.interval_keys)
        interval_box.setCurrentText("99.8%")
        interval_box.currentTextChanged.connect(self.interval_update)

        interval_label = QLabel("Interval:", self)
        self.layout_side.addWidget(interval_label)
        self.layout_side.addWidget(interval_box)

        # Opacity
        self.opacity = 0
        self.opacity_box = QComboBox()
        self.opacity_box.addItems(["100%", "90%", "75%", "50%", "25%", "0%"])
        self.opacity_box.setCurrentText("0%")
        self.opacity_box.currentTextChanged.connect(self.opacity_update)

        opacity_label = QLabel("Opacity:", self)
        self.layout_side.addWidget(opacity_label)
        self.layout_side.addWidget(self.opacity_box)

        self.selected_ids = []
        self.remapped_ids = {}

        self.invert_box = QCheckBox("Invert image")
        self.invert_box.stateChanged.connect(self.opacity_update)
        p = QPalette(self.invert_box.palette())
        p.setColor(
            QPalette.ColorGroup.Active,
            QPalette.ColorRole.Base,
            QColor(90, 90, 90),
        )
        self.invert_box.setPalette(p)
        self.layout_side.addWidget(self.invert_box)

        self.layout_side.addWidget(QLabel("Background colour:", self))

        self.bkg_frm = QFrame(self)
        self.bkg_frm.mousePressEvent = self.choose_background_colour
        self.bkg_frm.setMinimumHeight(50)
        self.bkg_frm.bkg_col = QColor("#787878")
        self.bkg_frm.setStyleSheet(
            f"QWidget {{ background-color: {self.bkg_frm.bkg_col.name()} }}"
        )
        self.layout_side.addWidget(self.bkg_frm)

        combine_button = QPushButton("Combine Selection", self)
        combine_button.clicked.connect(self.combine_ids)
        save_button = QPushButton("Save Map", self)
        save_button.clicked.connect(self.save_output)
        self.layout_side.addWidget(Separator())
        self.layout_side.addWidget(combine_button)
        self.layout_side.addWidget(save_button)

        self.progress_bar = QProgressBar(self)
        self.progress_label = QLabel("", self)
        self.progress_title = QLabel("Loading:", self)
        self.layout_side.addWidget(Separator())
        self.layout_side.addWidget(self.progress_title)
        self.layout_side.addWidget(self.progress_bar)
        self.layout_side.addWidget(self.progress_label)
        self.progress_bar.setHidden(True)
        self.progress_label.setHidden(True)

        self.layout_h.addWidget(self.left_toolbar)

        self.viewer = QtImageViewer()
        self.viewer.aspectRatioMode = Qt.AspectRatioMode.KeepAspectRatio
        self.viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.viewer.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.viewer.regionZoomButton = (
            Qt.MouseButton.LeftButton
        )  # set to None to disable
        self.viewer.zoomOutButton = Qt.MouseButton.RightButton  # set to None to disable
        self.viewer.wheelZoomFactor = 1.1  # Set to None or 1 to disable

        self.viewer.setMouseTracking(True)
        self.viewer.mousePositionOnImageChanged.connect(self.seg_text_update)
        self.viewer.leftMouseButtonReleased.connect(self.click_location)

        self.current_seg_id = 0

        self.viewer.panButton = Qt.MouseButton.MiddleButton
        self.layout_h.addWidget(self.viewer)

        widget = QWidget()
        widget.setLayout(self.layout_h)
        self.setCentralWidget(widget)

        self.progress_popup = None

        self.loader_fn(**kwargs)

    def open_files_window(self, event=None):
        if self.files_window is None:
            self.files_window = FilesWindow(self)
        self.files_window.show()

    def click_location(self, modifiers, x, y):
        if not hasattr(self, "img_array"):
            return
        x = int(x)
        y = int(y)
        seg_id = self.seg_data[y, x]

        if seg_id in self.selected_ids:
            self.selected_ids.remove(seg_id)
        elif (len(self.selected_ids) > 0) & (
            modifiers == Qt.KeyboardModifier.ControlModifier
        ):
            if seg_id == 0:
                pass
            elif self.selected_ids == [0]:
                self.selected_ids = [seg_id]
            else:
                self.selected_ids.append(seg_id)
        else:
            self.selected_ids = [seg_id]

        if self.selected_ids == []:
            self.selected_ids = [0]

        self.highlight_section(self.selected_ids)

    def highlight_section(self, seg_id):
        if not hasattr(self, "img_array"):
            return

        if seg_id == [0]:
            seg_map = np.zeros_like(self.seg_data, dtype="uint8")
        else:
            seg_map = np.isin(self.seg_data, seg_id).astype("uint8") * 255
        seg_plot = self.seg_q.copy()
        seg_plot.setAlphaChannel(
            QImage(
                seg_map,
                seg_map.shape[1],
                seg_map.shape[0],
                seg_map.strides[0],
                QImage.Format.Format_Indexed8,
            )
        )
        pixmap = QPixmap.fromImage(seg_plot)

        if not hasattr(self.viewer, "overlay"):
            self.viewer.overlay = self.viewer.scene.addPixmap(pixmap)
        else:
            self.viewer.overlay.setPixmap(pixmap)

    def seg_text_update(self, pos):
        if not hasattr(self, "img_array"):
            return

        x = int(pos.x())
        y = int(self.seg_data.shape[0] - pos.y() - 1) if pos.y() != -1 else -1
        # print (pos.y(), self.seg_data.shape)
        seg_id = self.seg_data[int(pos.y()), x]

        # shape = self.seg_data.shape()
        # print (shape)

        self.seg_text.setText(
            f"{wcs.utils.pixel_to_skycoord(x, y, self.wcs).to_string(precision=6)}\n"
            f"x={x: <6}y={y: <6}\nID={seg_id}"
        )

    def interval_update(self, value):
        self.interval = ManualInterval(
            self.interval_dict[value][0], self.interval_dict[value][1]
        )
        self.reload_image()

    def stretch_update(self, value):
        match value:
            case "Linear":
                self.stretch = LinearStretch()
            case "Asinh":
                self.stretch = AsinhStretch()
            case "Square Root":
                self.stretch = SqrtStretch()
            case "Logarithmic":
                self.stretch = LogStretch()
        self.reload_image()

    def opacity_update(self, event=None):
        print("step 1")
        self.opacity = float(self.opacity_box.currentText().split("%")[0]) / 100

        print("step 2")

        if not hasattr(self, "img_array"):
            print("returning")
            return

        if self.invert_box.checkState() == Qt.CheckState.Checked:
            print("checked")
            self.img_array[:, :, -1] = (
                np.clip(
                    self.opacity * self.seg_mask + self.opacity_mask, a_min=0, a_max=1
                )
                * 255
            ).astype("uint8")
        else:
            print("unchecked")
            self.img_array[:, :, -1] = (
                np.clip(
                    self.opacity * self.opacity_mask + self.seg_mask, a_min=0, a_max=1
                )
                * 255
            ).astype("uint8")

        print("almost there")
        q_img = QImage(
            self.img_array,
            self.img_array.shape[1],
            self.img_array.shape[0],
            self.img_array.strides[0],
            QImage.Format.Format_RGBA8888,
        )
        self.viewer.setImage(q_img)

    def reload_image(self):
        if not hasattr(self, "img_array"):
            return

        self.img_array[:, :, :-1] = (
            self.stretch(self.interval(self.data_array[:, :, :-1])) * 255
        ).astype("uint8")
        q_img = QImage(
            self.img_array,
            self.img_array.shape[1],
            self.img_array.shape[0],
            self.img_array.strides[0],
            QImage.Format.Format_RGBA8888,
        )
        self.viewer.setImage(q_img)

    def loader_fn(self, **kwargs):

        worker = Worker(
            self.load_image,
            **kwargs,
        )
        self.progress_bar.setValue(0)
        self.progress_label.setText("")
        self.progress_bar.setHidden(False)
        self.progress_label.setHidden(False)
        worker.signals.progress.connect(self.update_progress)
        worker.signals.result.connect(self.set_img)
        # worker.signals.finished.connect(self.cleanup_load)
        self.threadpool.start(worker)

    def update_progress(self, value, text):
        print(text)
        self.progress_bar.setValue(value)
        self.progress_label.setText(text)

    def test_path(
        self, key: str, value: str | os.PathLike | None, is_dir: bool = False
    ) -> bool:

        if value is None:
            return False

        try:
            test_path = Path(value)
            if is_dir and test_path.is_dir():
                setattr(self, key, test_path)
            elif (not is_dir) and test_path.is_file():
                setattr(self, key, test_path)
            return True
        except:
            return False

    def load_from_dir(self):

        if self.in_dir is not None:
            test_dir = Path(self.in_dir)
        else:
            test_dir = Path.home()

        print (test_dir)

        if self.seg_img_path is None:
            try:
                seg_img = [
                    *test_dir.glob(
                        f"{self.field_root}-{self.detection_filter}_seg.fits"
                    )
                ][0]
                self.seg_img_path = seg_img
            except:
                print("Segmentation map not found.")

        if self.stack_img_path is None:
            try:
                stack_img = [
                    *test_dir.glob(
                        f"{self.field_root}-{self.detection_filter}_dr[zc]_sci.fits"
                    )
                ][0]
                self.stack_img_path = stack_img
            except:
                print("Stacked image not found.")

        for f, l in zip(self.filters, ["b_img_path", "g_img_path", "r_img_path"]):
            if getattr(self, l) is None:
                try:
                    filt_img = [
                        *test_dir.glob(f"{self.field_root}-{f.lower()}_dr[zc]_sci.fits")
                    ][0]
                    setattr(self, l, filt_img)
                except:
                    print(f"Filter {f} image not found.")

        return (
            self.seg_img_path,
            self.stack_img_path,
            self.b_img_path,
            self.g_img_path,
            self.r_img_path,
        )

    def load_image(
        self,
        progress_callback=None,
        # in_dir: str | os.PathLike | None = None,
        # seg_img_path: str | os.PathLike | None = None,
        # r_img_path: str | os.PathLike | None = None,
        # g_img_path: str | os.PathLike | None = None,
        # b_img_path: str | os.PathLike | None = None,
        **kwargs,
    ):
        progress_callback.emit(10, "Locating files...")

        if "in_dir" in kwargs:
            self.test_path("in_dir", kwargs.pop("in_dir"), is_dir=True)
        for k, v in kwargs.items():
            self.test_path(k, v, is_dir=False)

        if (self.seg_img_path is None) or (
            None in [self.r_img_path, self.g_img_path, self.b_img_path]
        ):
            self.load_from_dir()

        t1 = time.time()
        # print ("Reading images...", end="\r")

        if self.seg_img_path is None:
            progress_callback.emit(100, "No files found.")
            return

        progress_callback.emit(25, "Locating files... DONE")
        progress_callback.emit(25, "Reading images...")

        with pf.open(self.seg_img_path) as hdul_seg:
            self.seg_mask = (hdul_seg[0].data > 0).astype("uint8")[::-1, :]
            self.opacity_mask = 1 - self.seg_mask
            self.seg_data = hdul_seg[0].data[::-1, :]
            self.seg_q = qimage2ndarray.array2qimage(
                np.stack(
                    [
                        np.zeros_like(self.seg_mask),
                        np.ones_like(self.seg_mask),
                        np.zeros_like(self.seg_mask),
                        self.seg_mask * 0.5,
                    ],
                    axis=-1,
                ),
                True,
            )

            self.wcs = wcs.WCS(hdul_seg[0].header)

        self.data_array = np.zeros((self.seg_mask.shape[0], self.seg_mask.shape[1], 4))
        self.overlap_mask = np.zeros_like(self.seg_mask, dtype="bool")

        self.data_array[:, :, -1] = self.seg_mask

        for i, filt_img_path in enumerate(
            [self.r_img_path, self.g_img_path, self.b_img_path]
        ):
            try:
                with pf.open(filt_img_path) as hdul_sci:
                    self.data_array[:, :, i] = hdul_sci[0].data[::-1, :]
                try:
                    # print (filt_img_path)
                    # wht_path = filt_img_path.replace("sci", "wht")
                    wht_path = filt_img_path.with_name(
                        filt_img_path.name.replace("sci", "wht")
                    )
                    # print (wht_path)
                    with pf.open(wht_path) as hdul_wht:
                        self.overlap_mask = self.overlap_mask | (
                            hdul_wht[0].data[::-1, :] > 0
                        )
                except:
                    print("Weight file not found.")
                    pass
            except:
                pass

        if np.sum(self.overlap_mask) != 0:
            self.opacity_mask[~self.overlap_mask] = 0
        img_array = self.data_array.copy()
        # print ("Reading images... DONE", time.time()-t1)
        progress_callback.emit(50, "Reading images... DONE")

        # print ("Computing intervals...", end="\r")
        progress_callback.emit(50, "Computing intervals...")
        self.interval_dict = {}
        percentiles = []
        for interval in self.interval_keys:
            match interval:
                case "minmax":
                    percentiles.append(100)
                case _:
                    percentiles.append(float(interval.split("%")[0]))
        lims = self.calc_interval_limits(percentiles)
        for k, l in zip(self.interval_keys, lims):
            self.interval_dict[k] = l

        self.interval = ManualInterval(
            self.interval_dict["99.8%"][0], self.interval_dict["99.8%"][1]
        )
        # print ("Computing intervals... DONE", time.time()-t1)
        progress_callback.emit(75, "Computing intervals... DONE")

        # print ("Formatting image for display...", end="\r")
        progress_callback.emit(75, "Formatting image for display...")
        img_array[:, :, :-1] = self.stretch(self.interval(img_array[:, :, :-1]))
        self.img_array = (img_array * 255).astype("uint8")

        self.opacity = float(self.opacity_box.currentText().split("%")[0]) / 100

        if self.invert_box.checkState() == Qt.CheckState.Checked:
            self.img_array[:, :, -1] = (
                np.clip(
                    self.opacity * self.seg_mask + self.opacity_mask, a_min=0, a_max=1
                )
                * 255
            )
        else:
            self.img_array[:, :, -1] = (
                np.clip(
                    self.opacity * self.opacity_mask + self.seg_mask, a_min=0, a_max=1
                )
                * 255
            )

        self.q_img = QImage(
            self.img_array,
            self.img_array.shape[1],
            self.img_array.shape[0],
            self.img_array.strides[0],
            QImage.Format.Format_RGBA8888,
        )
        # self.viewer.setImage(self.q_img)
        # self.viewer.setBackgroundBrush(QColor(120,120,120))
        progress_callback.emit(100, "Formatting image for display... DONE")
        # print ("Formatting image for display... DONE", time.time()-t1)
        # print (f"Completed in {(time.time()-t1):.2f}s")
        progress_callback.emit(100, f"Completed in {(time.time()-t1):.2f}s")

        return self.q_img

    def set_img(self, img):
        if img is not None:
            self.viewer.setImage(img)
            self.viewer.setBackgroundBrush(self.bkg_frm.bkg_col)

    def calc_interval_limits(self, percentiles):
        all_p = []
        for p in percentiles:
            lower_percent = (100 - p) * 0.5
            upper_percent = 100 - lower_percent
            all_p.extend([lower_percent, upper_percent])
        limits = np.percentile(self.data_array[:, :, :-1].ravel(), all_p)

        res = [*zip(limits[::2], limits[1::2])]
        return res

    def choose_background_colour(self, event=None):
        col = QColorDialog.getColor(self.bkg_frm.bkg_col)

        if col.isValid():
            self.bkg_frm.bkg_col = col
            self.bkg_frm.setStyleSheet(
                f"QWidget {{ background-color: {self.bkg_frm.bkg_col.name()} }}"
            )
            self.viewer.setBackgroundBrush(col)

    def combine_ids(self, event=None):
        if not hasattr(self, "seg_data"):
            print("No segmentation mask to modify.")
            return

        current_selection = np.isin(self.seg_data, self.selected_ids)

        self.seg_data[current_selection] = np.nanmin(self.selected_ids)

        selected = [int(i) for i in self.selected_ids]

        existing_entries = [k for k in self.remapped_ids.keys() if k in selected]

        for e in existing_entries:
            selected.extend(self.remapped_ids[e])
            del self.remapped_ids[e]

        self.remapped_ids[int(np.nanmin(selected))] = selected

    def save_output(self, event=None):
        if not (hasattr(self, "seg_img_path") and hasattr(self, "seg_data")):
            print("No segmentation mask loaded.")
            return

        backup_path = self.seg_img_path.parent / f"{self.seg_img_path.stem}_backup.fits"
        with pf.open(self.seg_img_path) as seg_hdul:
            if not backup_path.is_file():
                seg_hdul.writeto(backup_path)

            seg_hdul[0].data = self.seg_data[::-1, :]

            seg_hdul.writeto(
                self.seg_img_path.parent / f"{self.seg_img_path.stem}_modified.fits"
            )


class Separator(QFrame):
    def __init__(self):
        super(QFrame, self).__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setLineWidth(3)


class LineBrowse(QWidget):
    clicked = pyqtSignal()

    def __init__(
        self,
        parent,
        is_dir: bool = False,
        root_name: str = "",
    ):

        super().__init__()
        self.is_dir = is_dir

        h_layout = QHBoxLayout()
        h_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(h_layout)

        self.line = QLineEdit(self)
        h_layout.addWidget(self.line)

        self.browse_button = QPushButton("Browse", self)
        self.browse_button.clicked.connect(self.browse_filesystem)
        h_layout.addWidget(self.browse_button)

        if root_name != "" and getattr(parent.root, root_name, None) is not None:
            self.line.setText(str(getattr(parent.root, root_name, None)))

    def browse_filesystem(self, QMouseEvent):
        if (self.line.text is None or self.line.text == "") and (
            self.parent().recent_dir is None
        ):
            init = str(Path.home())
        else:
            init = self.line.text()

        if self.is_dir:
            f = QFileDialog.getExistingDirectory(self, "Select File", init)

            if f:
                self.line.setText(f)
                self.parent().recent_dir = f
        else:
            f, _ = QFileDialog.getOpenFileName(
                self, "Select File", init, "FITS files (*.fits)"
            )

            if f:
                self.line.setText(f)
                self.parent().recent_dir = Path(f).parent


class FilesWindow(QWidget):
    def __init__(self, root):
        super().__init__()
        self.root = root

        self.v_layout = QVBoxLayout()

        self.recent_dir = None

        dir_sel_button = QPushButton("Fill From Directory", self)
        dir_sel_button.clicked.connect(self.select_from_directory)

        self.sub_layout = QFormLayout()
        self.sub_layout.addRow(dir_sel_button)
        self.sub_layout.addRow(Separator())
        self.seg_line = LineBrowse(self, root_name="seg_img_path")
        self.sub_layout.addRow("Segmentation Map:", self.seg_line)
        self.sub_layout.addRow(Separator())
        self.stack_line = LineBrowse(self, root_name="stack_img_path")
        self.sub_layout.addRow("Stacked Image:", self.stack_line)
        self.sub_layout.addRow(Separator())
        self.b_line = LineBrowse(self, root_name="b_img_path")
        self.sub_layout.addRow("Blue:", self.b_line)
        self.g_line = LineBrowse(self, root_name="g_img_path")
        self.sub_layout.addRow("Green:", self.g_line)
        self.r_line = LineBrowse(self, root_name="g_img_path")
        self.sub_layout.addRow("Red:", self.r_line)
        self.v_layout.addLayout(self.sub_layout)

        self.load_all_button = QPushButton("Load Images", self)
        self.load_all_button.clicked.connect(self.load_all)
        self.v_layout.addWidget(Separator())
        self.v_layout.addWidget(self.load_all_button)

        self.progress_label = QLabel(f"Test", self)
        self.v_layout.addWidget(self.progress_label)
        self.progress_label.setHidden(True)

        self.setLayout(self.v_layout)
        self.setMinimumWidth(540)

    def select_from_directory(self, event=None):
        if self.root.in_dir is not None:
            init = str(self.root.in_dir)
        elif self.recent_dir is not None:
            init = str(self.recent_dir)
        else:
            init = str(Path.home())

        dir_name = QFileDialog.getExistingDirectory(self, "Open directory", init)

        if dir_name:
            self._load_from_dir(dir_name)

    def _load_from_dir(self, dir_name):
        self.root.in_dir = Path(dir_name)
        seg, stack, b, g, r = self.root.load_from_dir()
        if seg is not None:
            self.seg_line.line.setText(str(seg))
        if stack is not None:
            self.stack_line.line.setText(str(stack))
        if b is not None:
            self.b_line.line.setText(str(b))
        if g is not None:
            self.g_line.line.setText(str(g))
        if r is not None:
            self.r_line.line.setText(str(r))

    def load_all(self):
        self.load_all_button.setEnabled(False)
        self.progress_label.setHidden(False)
        worker = Worker(
            self.root.load_image,
            in_dir=self.root.in_dir,
            seg_img_path=self.seg_line.line.text(),
            r_img_path=self.r_line.line.text(),
            g_img_path=self.g_line.line.text(),
            b_img_path=self.b_line.line.text(),
        )
        worker.signals.progress.connect(self.progress_fn)
        worker.signals.result.connect(self.root.set_img)
        worker.signals.finished.connect(self.cleanup_load)
        self.root.threadpool.start(worker)

    def progress_fn(self, value, text):
        self.root.update_progress(value, text)
        self.progress_label.setText(f"{value}% - {text}")

    def cleanup_load(self):
        self.load_all_button.setEnabled(True)

    def printText(self):
        print("This works")
