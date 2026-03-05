"""Organise the forced extraction tool, and hook into grizli methods."""

import glob
import math
import multiprocessing
import os
import pickle
import shutil
import time
import warnings
from collections.abc import Iterable
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
from packaging.version import Version, parse
from pathlib import Path
from typing import Any, Optional

import astropy
import astropy.io.fits as pf
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.wcs import WCS
from tqdm import tqdm

if parse(version("sep")) < Version("1.4.0"):
    warnings.warn(RuntimeWarning("Modifying the object catalogue requires SEP>=1.4.0."))
    HAS_SEP = False
else:
    HAS_SEP = True

# grizli_dir = Path(
#     "/Path/to/out/dir" # Point this to the output directory
# ) / "grizli"

# grizli_dir.mkdir(exist_ok=True)
# (grizli_dir / "CONF").mkdir(exist_ok=True)
# (grizli_dir / "templates").mkdir(exist_ok=True)
# (grizli_dir / "iref").mkdir(exist_ok=True)
# (grizli_dir / "iref").mkdir(exist_ok=True)
# os.environ["GRIZLI"] = str(grizli_dir)
# os.environ["iref"] = str(grizli_dir / "iref")
# os.environ["jref"] = str(grizli_dir / "jref")
# cwd = Path.cwd()
# os.chdir(os.path.join(grizli.GRIZLI_PATH, 'CONF'))

# grizli.utils.fetch_default_calibs()
# grizli.utils.fetch_config_files(get_jwst=True)
# if not os.path.exists("GR150C.F115W.221215.conf"):
#     os.system(
#         'wget "https://zenodo.org/record/7628094/files/niriss_config_221215.tar.gz?download=1"'
#         " -O niriss_config_221215.tar.gz"
#     )
#     os.system("tar xzvf niriss_config_221215.tar.gz")
# if not os.path.exists("niriss_sens_221215.tar.gz"):
#     os.system(
#         'wget "https://zenodo.org/record/7628094/files/niriss_sens_221215.tar.gz" -O'
#         " niriss_sens_221215.tar.gz"
#     )
#     os.system("tar xzvf niriss_sens_221215.tar.gz")
# os.chdir(cwd)
# eazy.fetch_eazy_photoz()

# try:
#     grizli.utils.symlink_templates(force=True)
# except:
#     pass

try:
    Path(os.getenv("GRIZLI")).is_dir()
    Path(os.getenv("iref")).is_dir()
    Path(os.getenv("jref")).is_dir()
except:
    warnings.warn(
        RuntimeWarning(
            "Either the grizli environment variables are not set correctly, "
            "or the directories they point to do not yet exist. "
            "Check that the environment is correctly configured "
            "(https://grizli.readthedocs.io/en/latest/grizli/install.html), "
            "or edit and uncomment the lines above."
        )
    )

import grizli
from grizli import fitting, jwst_utils, model, multifit, prep, utils
from grizli.pipeline import auto_script, photoz

from .grizli_functions import FLT_fns, catalogue_fns

multifit._loadFLT = FLT_fns.load_and_mod_FLT
model.GrismFLT.transform_JWST_WFSS = FLT_fns.mod_transform_JWST_WFSS
model.GrismFLT.compute_model_orders = FLT_fns.mod_compute_model_orders
model.BeamCutout.init_from_input = FLT_fns.init_from_input_multispec


class GrismExtractor:
    """
    A class for extracting additional objects from processed grism data.

    This class enables the extraction of objects from arbitrary regions
    of existing slitless spectroscopic exposures. This allows the
    modification of both catalogues and contamination maps previously
    derived using `grizli`.

    Parameters
    ----------
    field_root : str
        The root name of the catalogue and processed images, e.g.
        ``'{field_root}-ir.cat.fits'``.
    in_dir : str or os.PathLike
        The input directory, containing all the original (unmodified)
        grism files, mosaics, and the segmentation map. Often named
        ``'Prep/'``.
    out_dir : str or os.PathLike
        The output directory, where all the modified files will be
        saved. Any relevant files are copied from ``in_dir`` to
        ``out_dir``, if they do not already exist.
    seg_path : str or os.PathLike, optional
        The path pointing to the original segmentation map. If not
        supplied, this will have to be loaded later.

    Attributes
    ----------
    catalogue : `astropy.table.Table`
        The multiband catalogue used by `grizli`.
    field_root : str
        The root name of the catalogue and processed images.
    grp : `grizli.multifit.GroupFLT`
        The container for multiple grism exposures.
    in_dir : os.PathLike
        The input directory.
    out_dir : os.PathLike
        The output directory, where all the modified files are saved.
    seg_hdr : `astropy.io.fits.Header`
        The header of the current segmentation map.
    seg_map : array-like
        The current segmentation map.
    seg_name : str
        The name of the current segmentation map.
    seg_wcs : `astropy.wcs.WCS`
        The WCS of the current segmentation map.
    """

    def __init__(
        self,
        field_root: str,
        in_dir: str | os.PathLike,
        out_dir: str | os.PathLike,
        seg_path: str | os.PathLike | None = None,
    ):
        """
        Initialise the extractor object, and make copies of the
        relevant files.
        """

        self.field_root = field_root
        self.in_dir = Path(in_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)

        copy_patterns = ["*FLT.fits", "*FLT.pkl", "*01.wcs.fits"]
        for pattern in copy_patterns:
            orig_files = self.in_dir.glob(pattern)
            for o in orig_files:
                out_path = self.out_dir / o.name
                if not out_path.is_file():
                    utils.log_comment(
                        utils.LOGFILE,
                        f"Copying file {o}",
                        verbose=True,
                    )
                    shutil.copy(
                        src=o,
                        dst=self.out_dir / o.name,
                        follow_symlinks=True,
                    )

        link_patterns = ["*dr[zc]_sci.fits", "*dr[zc]_wht.fits"]
        for pattern in link_patterns:
            orig_files = self.in_dir.glob(self.field_root + pattern)
            for o in orig_files:
                out_path = self.out_dir / o.name
                try:
                    out_path.symlink_to(o)
                except:
                    utils.log_comment(
                        utils.LOGFILE,
                        f"File {out_path.name} exists already.",
                        verbose=True,
                    )

        if seg_path is not None:
            self.load_orig_seg_map(seg_path)

    def load_orig_seg_map(
        self, seg_path: str | os.PathLike, ext: int | str = 0
    ) -> None:
        """
        Load a segmentation map into memory.

        The stored name and WCS information are also updated.

        Parameters
        ----------
        seg_path : str or os.PathLike
            The path pointing to the segmentation map.
        ext : int or str, optional
            The number or name of the HDUList extension containing the
            segmentation data, by default 0.
        """

        seg_path = Path(seg_path)

        self.seg_name = seg_path.name

        with pf.open(seg_path) as hdul:
            if parse(version("numpy")) < Version("2.0.0"):
                self.seg_map = hdul[ext].data.byteswap().newbyteorder()
            else:
                self.seg_map = hdul[ext].data.astype(
                    hdul[ext].data.dtype.newbyteorder("=")
                )
            self.seg_hdr = hdul[ext].header
            self.seg_wcs = WCS(self.seg_hdr)

    def regen_multiband_catalogue(self, **kwargs) -> astropy.table.Table:
        """
        Recreate the `grizli` catalogue for the current segmentation map.

        Parameters
        ----------
        **kwargs : dict, optional
            Keyword arguments are passed through to
            `~pygrife.grizli_functions.catalogue_fns.regen_multiband_catalogue`.

        Returns
        -------
        `~astropy.table.Table`
            The multiband catalogue.
        """

        if not HAS_SEP:
            raise ImportError("Modifying the object catalogue requires SEP>=1.4.0.")

        utils.log_comment(
            utils.LOGFILE,
            "Regenerating multiband catalogue...",
            verbose=True,
            show_date=True,
        )

        for p in self.out_dir.glob("*phot_apcorr.fits"):
            p.unlink()

        kwargs["get_all_filters"] = kwargs.get("get_all_filters", True)

        now = datetime.now(timezone.utc).strftime(r"%Y%m%dT%H%M%SZ")

        try:
            end_string = self.seg_name.split("-")[-1].split(".")[0]
            test = datetime.strptime(end_string, r"%Y%m%dT%H%M%SZ")
            if type(test) == datetime:
                self.seg_name = self.seg_name.replace(end_string, now)
        except:
            self.seg_name = self.seg_name.replace(".fits", f"-{now}.fits")

        seg_out_path = self.out_dir / self.seg_name

        self.catalogue = catalogue_fns.regen_multiband_catalogue(
            self.field_root,
            seg_image=self.seg_map,
            in_dir=self.out_dir,
            out_dir=self.out_dir,
            seg_out_path=seg_out_path,
            **kwargs,
        )

        utils.log_comment(
            utils.LOGFILE,
            f"Multiband catalogue complete. Segmentation map saved to {self.seg_name}",
            verbose=True,
            show_date=True,
        )
        return self.catalogue

    def load_grism_files(
        self,
        grism_files: npt.ArrayLike | None = None,
        detection_filter: str = "ir",
        pad: int | tuple[int, int] = 800,
        cpu_count: int = 4,
        catalog_path: str | os.PathLike | None = None,
        seg_path: str | os.PathLike | None = None,
        **kwargs,
    ) -> grizli.multifit.GroupFLT:
        """
        Load an existing set of grism files into memory.

        Load (or reload) a set of ``"*GrismFLT"`` files into memory, keeping
        track of the current segmentation map, and the map used to
        generate the contamination model.

        Parameters
        ----------
        grism_files : array-like, optional
            An explicit list of the grism files to use. By default, all
            ``"*GrismFLT.fits"`` files in the output directory will be used.
        detection_filter : str, optional
            The filter image used for the source detection, by default
            ``"ir"``. This is used to locate the catalogue.
        pad : int or tuple[int, int], optional
            The padding in pixels, allowing modelling of sources outside
            the detector field of view. If a tuple is supplied, this is
            taken as ``(pady, padx)``. Defaults to 800pix in both axes.
        cpu_count : int, optional
            If < 0, load files serially. If > 0, load files in ``cpu_count``
            parallel processes. Use all available cores if
            ``cpu_count=0``. Defaults to 4 processes.
        catalog_path : str or os.Pathlike, optional
            The path of the catalogue to use when loading the grism files,
            Defaults to ``"{self.field_root}-{detection_filter}.cat.fits"``.
        seg_path : str or os.Pathlike, optional
            The path of the segmentation map to use when loading the grism
            files, if different to ``self.seg_map``.
            Defaults to ``"{self.field_root}-{detection_filter}_seg.fits"``.
        **kwargs : dict, optional
            Any other keyword arguments, passed through to
            `~grizli.multifit.GroupFLT()`.

        Returns
        -------
        ~grizli.multifit.GroupFLT
            The container for multiple grism exposures.

        Notes
        -----
        Be careful with the ``cpu_count`` - the memory footprint per
        process is extremely high (e.g. with a 6P/8E CPU, and 32GB RAM, I
        typically limit this to <=6 cores).
        """

        if grism_files is None:
            grism_files = [str(p) for p in self.out_dir.glob("*GrismFLT.fits")]
        else:
            grism_files = np.atleast_1d(grism_files)
        if len(grism_files) == 0:
            raise Exception("No grism files found.")

        utils.log_comment(
            utils.LOGFILE,
            f"Loading {len(grism_files)} grism files, grizli"
            f" version={grizli.__version__}",
            verbose=True,
            show_date=True,
        )

        if catalog_path is not None:
            if kwargs.get("catalog") is not None:
                raise Exception(
                    "Only one of `catalog` and `catalog_path` can be supplied."
                )
            catalog_path = Path(catalog_path)
        else:
            catalog_path = Path(
                self.out_dir / f"{self.field_root}-{detection_filter}.cat.fits"
            )
        if not catalog_path.is_file():
            raise FileNotFoundError(
                f"Catalogue file not found at the specified location: {catalog_path}."
            )

        if seg_path is not None:
            if kwargs.get("seg_file") is not None:
                raise Exception(
                    "Only one of `seg_path` and `seg_file` can be supplied."
                )
            seg_path = Path(seg_path)
        elif hasattr(self, "seg_name"):
            seg_path = self.out_dir / self.seg_name
        else:
            seg_path = Path(
                self.out_dir / f"{self.field_root}-{detection_filter}_seg.fits"
            )
        if not Path(seg_path).is_file():
            raise FileNotFoundError(
                f"Segmentation map not found at the specified location: {seg_path}."
            )

        self.grp = multifit.GroupFLT(
            cpu_count=cpu_count,
            grism_files=grism_files,
            pad=pad,
            seg_file=str(seg_path),
            catalog=str(catalog_path),
            **kwargs,
        )

        return self.grp

    def match_objects(
        self,
        targets: astropy.table.Table,
        column_names: dict | None = None,
        return_all: bool = False,
    ) -> npt.NDArray | tuple[npt.NDArray, astropy.table.Table, astropy.table.Table]:
        """
        Match a table of targets against the existing catalogue.

        Parameters
        ----------
        targets : astropy.table.Table
            The targets to match.
        column_names : dict | None, optional
            A mapping of old and new column names for the targets table. By
            default ``None``.
        return_all : bool, optional
            Return the object table and all failed matches. By default
            ``False``, in which case only the matched object IDs will be
            returned.

        Returns
        -------
        npt.NDArray | tuple[npt.NDArray, astropy.table.Table, astropy.table.Table]
            Either an array of object IDs, or also the table of matched
            objects, and the table of failed objects.
        """

        if not hasattr(self, "grp"):
            raise Exception(
                "GrismFLT files not loaded. Run `load_grism_files()' first."
            )

        if column_names is not None:
            for k, v in column_names.items():
                targets.rename_column(v, k)

        targets.rename_columns(targets.colnames, [c.lower() for c in targets.colnames])

        print(targets)
        idx, dr = self.grp.catalog.match_to_catalog_sky(targets)
        indices = []
        failed_indices = []
        for i, n in enumerate(idx):
            if dr[i].value > 1.0:
                print(f"{targets['id'][i]} not matched: separation = {dr[i]:0.2f}")
                failed_indices.append(i)
                continue
            else:
                indices.append(i)

        obj_ids = np.asarray(self.grp.catalog["NUMBER"][idx][indices])
        failed_objs = targets[failed_indices]
        if not return_all:
            return obj_ids
        else:
            obj = Table()
            obj["id"] = obj_ids
            obj["ra"] = targets["ra"][indices]
            obj["dec"] = targets["dec"][indices]
            obj["idx"] = idx[indices]
            obj["dr"] = dr[indices]  # .to(u.mas)
            obj["dr"].format = "0.2f"
            obj["name"] = targets["id"][indices]
            for n in targets.colnames:
                if n not in ["id", "ra", "dec"]:
                    try:
                        obj[n] = targets[n][indices]
                    except:
                        pass
            return obj_ids, obj, failed_objs

    def extract_spectra(
        self,
        obj_id_list: npt.ArrayLike,
        z_range: npt.ArrayLike = [0.0, 0.5],
        fit_kwargs: dict[str, Any] | None = None,
        beams_kwargs: dict[str, Any] | None = None,
        multibeam_kwargs: dict[str, Any] | None = None,
        spectrum_1d: npt.ArrayLike | None = None,
        is_cgs: bool = True,
        trim_sensitivity : bool = False,
        fit_trace_shift : bool = False
    ) -> None:
        """
        Perform a full extraction of the specified objects.

        Parameters
        ----------
        obj_id_list : array-like
            The object ids in the segmentation map which will be
            extracted.
        z_range : array-like, optional
            The redshift range to consider for the extraction, by default
            0 < z < 0.5.
        fit_kwargs : dict, optional
            Keyword arguments to pass to
            `~grizli.pipeline.auto_script.generate_fit_params`.
        beams_kwargs : dict, optional
            Keyword arguments to pass to
            `~grizli.multifit.GroupFLT.get_beams`.
        multibeam_kwargs : dict, optional
            Keyword arguments to pass to `~grizli.multifit.MultiBeam`.
        spectrum_1d : [``wavelengths``, ``flux``], optional
            The flux spectrum and corresponding wavelengths of the object
            in the model. By default, this is calculated automatically
            from the stored ``object_dispersers``.
        is_cgs : bool, optional
            The flux units of ``spectrum_1d[1]`` are cgs f_lambda flux
            densities, rather than normalised in the detection band, by
            default True.
        trim_sensitivity : bool, optional
        """

        if not hasattr(self, "grp"):
            raise Exception(
                "GrismFLT files not loaded. Run `load_grism_files()' first."
            )
        if (
            hasattr(self, "seg_name")
            and Path(self.grp.FLTs[0].seg_file).name != self.seg_name
        ):
            raise Exception(
                f"The current segmentation map ({self.seg_name}) does not match the"
                " name stored in the GrismFLT files"
                f" ({Path(self.grp.FLTs[0].seg_file).name}). Run"
                " `load_grism_files()' before extracting any spectra, or load"
                " the correct segmentation map."
            )

        utils.log_comment(
            utils.LOGFILE,
            f"Generating fit parameters.",
            verbose=True,
            show_date=True,
        )
        os.chdir(self.out_dir)

        if beams_kwargs is None:
            beams_kwargs = {}
        beams_kwargs["size"] = beams_kwargs.get("size", -1)
        beams_kwargs["min_mask"] = beams_kwargs.get("min_mask", 0.0)
        beams_kwargs["min_sens"] = beams_kwargs.get("min_sens", 0.0)
        beams_kwargs["show_exception"] = beams_kwargs.get("show_exception", True)

        if multibeam_kwargs is None:
            multibeam_kwargs = {}
        multibeam_kwargs["fcontam"] = multibeam_kwargs.get("fcontam", 0.1)
        multibeam_kwargs["min_mask"] = multibeam_kwargs.get("min_mask", 0.0)
        multibeam_kwargs["min_sens"] = multibeam_kwargs.get("min_sens", 0.0)

        pline = {
            "kernel": "square",
            "pixfrac": 1.0,
            "pixscale": 0.03,
            "size": 8,
            "wcs": None,
        }
        if fit_kwargs is None:
            fit_kwargs = {}
        fit_kwargs["pline"] = fit_kwargs.get("pline", pline)
        fit_kwargs["field_root"] = fit_kwargs.get("field_root", self.field_root)
        fit_kwargs["min_sens"] = fit_kwargs.get("min_sens", 0.0)
        fit_kwargs["min_mask"] = fit_kwargs.get("min_mask", 0.0)

        args = auto_script.generate_fit_params(
            **fit_kwargs,
            include_photometry=False,
            use_phot_obj=False,
        )  # set both of these to True to include photometry in fitting

        obj_id_arr = np.atleast_1d(obj_id_list).flatten()

        for obj_id in tqdm(obj_id_arr):
            beams = FLT_fns.get_beams_with_spectrum(
                self.grp, obj_id, spectrum_1d=spectrum_1d, is_cgs=is_cgs, **beams_kwargs
            )

            mb = multifit.MultiBeam(
                beams, group_name=self.field_root, **multibeam_kwargs
            )
            if fit_trace_shift:
                mb.fit_trace_shift()

            # print(dir(mb))
            # print(mb.wavef.shape)
            # print(mb.scif.shape)

            # print(dir(mb.beams[0]))
            # print(Path.cwd())
            mb.write_master_fits()

            if trim_sensitivity:
                (Path.cwd() / f"{self.field_root}_{obj_id:05}.beams.fits").rename(
                    Path.cwd() / f"{self.field_root}_{obj_id:05}.orig_beams.fits",
                )
                for i, b_i in enumerate(mb.beams[:]):
                    # print (dir(b_i.grism))
                    # print ((b_i.grism.mdrizsky))
                    # print ((b_i.grism.pupil))
                    match b_i.grism.pupil:
                        case "F115W":
                            lam_l, lam_h = 10130, 12830
                        case "F150W":
                            lam_l, lam_h = 13300, 16710
                        case "F200W":
                            lam_l, lam_h = 17510, 22260
                    waves = b_i.wavef.reshape(b_i.sh)
                    # mb.beams[i].grism.data["SCI"][(waves < lam_l) | (waves > lam_h)] = 0
                    mb.beams[i].grism.data["ERR"][(waves < lam_l) | (waves > lam_h)] = 0
                    # mb.beams[i]
                    # plt.imshow(b_i.scif.reshape(b_i.sh))
                    # plt.show()
                    # scif[

                    # ]
                    # exit()
                mb.write_master_fits()

            _ = fitting.run_all_parallel(
                obj_id,
                zr=z_range,
                verbose=True,
                get_output_data=True,
            )
        utils.log_comment(
            utils.LOGFILE,
            f"Finished extracting spectra.",
            verbose=True,
            show_date=True,
        )

    def refine_contam_model_with_fits(
        self,
        spectrum: str = "full",
        max_chinu: int | float = 5,
        fit_files: list[str] | list[os.PathLike] | None = None,
        mag_limit=25,
        get_beams=None
    ) -> bool:
        """
        Refine existing contamination models.

        Refine the full-field grism models with the best fit spectra from
        individual extractions. [Modified version of a grizli function]

        Parameters
        ----------
        spectrum : str, optional
            The component of the best-fit spectrum to use, either
            ``"full"`` or ``"continuum"``.
        max_chinu : int or float, optional
            The maximum reduced chi-squared value of the fit to accept,
            in order to refine the contamination model with the resulting
            spectrum, by default 5.
        fit_files : list[str] or list[os.PathLike] or None, optional
            An explicit list of the best-fit files to use. By default, all
            ``*full.fits`` files in the current directory will be used.

        Returns
        -------
        bool
            Returns False if the contamination maps are not modified.
        """

        if fit_files is None:
            fit_files = glob.glob("*full.fits")
            fit_files.sort()
        fit_files_arr = np.atleast_1d(np.asarray(fit_files))
        N = fit_files_arr.shape[0]
        if N == 0:
            return False

        msg = "Refine model ({0}/{1}): {2} / skip (chinu={3:.1f}, dof={4})"

        for i, file in enumerate(fit_files_arr):
            try:
                hdu = pf.open(file)
                o_id = hdu[0].header["ID"]

                fith = hdu["ZFIT_STACK"].header
                chinu = fith["CHIMIN"] / fith["DOF"]
                if (chinu > max_chinu) | (fith["DOF"] < 10):
                    print(msg.format(i, N, file, chinu, fith["DOF"]))
                    continue

                sp = utils.GTable(hdu["TEMPL"].data)

                wave = np.asarray(sp["wave"], dtype=float)  # .byteswap()
                flux = np.asarray(sp[spectrum], dtype=float)  # .byteswap()
                for flt in self.grp.FLTs:
                    if int(o_id) not in flt.object_dispersers:
                        old_obj_ids = np.unique(flt.orig_seg[flt.seg == o_id])
                        old_obj_ids = old_obj_ids.ravel()[
                            np.flatnonzero(old_obj_ids)
                        ].astype(int)
                self.grp.compute_single_model(
                    int(o_id),
                    mag=mag_limit,
                    size=-1,
                    store=False,
                    spectrum_1d=[wave, flux],
                    is_cgs=True,
                    get_beams=get_beams,
                    in_place=True,
                )
                print("Refine model ({0}/{1}): {2}".format(i, N, file))
            except Exception as e:
                print("Refine model ({0}/{1}): {2} / failed {3}".format(i, N, file, e))

        for f in self.grp.FLTs:
            f.orig_seg = f.seg
            f.orig_seg_file = f.seg_file

        return True

    def _create_circular_mask(
        self, x_c: float, y_c: float, radius: float
    ) -> npt.NDArray[np.bool_]:
        """
        Create a boolean mask of all elements in the segmentation map
        within a specified distance of a point.

        Parameters
        ----------
        x_c : float
            The x-coordinate of the reference point.
        y_c : float
            The y-coordinate of the reference point.
        radius : float
            The maximum radius allowed.

        Returns
        -------
        ndarray, bool
            The mask, where elements are ``True`` if the distance to
            ``(x_c, y_c)`` is less than or equal to ``radius``.
        """

        Y, X = np.ogrid[: self.seg_map.shape[0], : self.seg_map.shape[1]]

        sqrd_dist = (X - x_c) ** 2 + (Y - y_c) ** 2

        mask = sqrd_dist <= radius**2
        return mask

    def _process_coords_radii(
        self,
        outer_radius: astropy.units.Quantity | npt.ArrayLike,
        inner_radius: astropy.units.Quantity | npt.ArrayLike,
        centre: astropy.coordinates.SkyCoord | None,
        **kwargs,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Process the coordinate and radii input.

        The outputs are formatted to work with
        `~GrismExtractor.add_circ_obj()` and
        `~GrismExtractor.add_segment_obj()`, and include only values
        inside the segmentation map footprint.

        Parameters
        ----------
        outer_radius : `astropy.units.Quantity` or array-like
            The outer radius of the aperture.
        inner_radius : `astropy.units.Quantity` or array-like
            If non-zero, an annulus will be extracted instead of
            an aperture.
        centre : `astropy.coordinates.SkyCoord`
            The centre of the aperture.
        **kwargs : dict, optional
            Any inputs accepted by `~astropy.coordinates.SkyCoord`, if
            `centre` is None.

        Returns
        -------
        (xs, ys, radii)
            A tuple of arrays.
        """

        if centre is None:
            try:
                centre = SkyCoord(**kwargs)
            except Exception as e:
                raise Exception(
                    f"Could not parse supplied arguments as on-sky coordinates: {e}"
                )

        centre = np.atleast_1d(centre).flatten()
        outer_radius = np.atleast_1d(outer_radius).flatten()
        inner_radius = np.atleast_1d(inner_radius).flatten()

        for r in [inner_radius, outer_radius]:
            if (r.shape[0] > 1) and (r.shape[0] != centre.shape[0]):
                raise ValueError(
                    f"Size of inputs do not match. {centre.shape[0]}"
                    f" coordinates and {r.shape[0]} radii have been supplied."
                )

        if (inner_radius.shape[0] == 1) and (centre.shape[0] > 1):
            try:
                inner_radius = (
                    np.tile(inner_radius.value, centre.shape[0]) * inner_radius.unit
                )
            except:
                inner_radius = np.tile(inner_radius, centre.shape[0])
        if (outer_radius.shape[0] == 1) and (centre.shape[0] > 1):
            try:
                outer_radius = (
                    np.tile(outer_radius.value, centre.shape[0]) * outer_radius.unit
                )
            except:
                outer_radius = np.tile(outer_radius, centre.shape[0])

        for i, o in zip(inner_radius, outer_radius):
            if i >= o:
                raise ValueError("Inner radius cannot be greater than outer radius.")

        contained = np.asarray([self.seg_wcs.footprint_contains(c) for c in centre])
        if not all(contained):
            warnings.warn(
                "The following coordinates are outside the segmentation map footprint,"
                f" and will be skipped: {centre[~contained].to_string()}",
            )
        xs, ys = centre[contained].to_pixel(self.seg_wcs)

        inner_radius, outer_radius = inner_radius[contained], outer_radius[contained]

        radii = np.zeros((xs.shape[0], 2))
        pix_scale = (
            np.nanmean([d.value for d in self.seg_wcs.proj_plane_pixel_scales()])
            * self.seg_wcs.proj_plane_pixel_scales()[0].unit
            / u.pix
        )
        for n in range(xs.shape[0]):
            for i, r in enumerate([inner_radius[n], outer_radius[n]]):
                if isinstance(r, u.Quantity):
                    if u.get_physical_type(r) == "angle":
                        radii[n, i] = (r / pix_scale).to(u.pix).value
                    else:
                        radii[n, i] = r.value
                else:
                    radii[n, i] = r

        return xs, ys, radii

    def add_circ_obj(
        self,
        radius: astropy.units.Quantity | npt.ArrayLike = 3 * u.arcsec,
        inner_radius: astropy.units.Quantity | npt.ArrayLike = 0,
        centre: astropy.coordinates.SkyCoord | None = None,
        init_id: int | None = None,
        **kwargs,
    ) -> npt.NDArray:
        """
        Add one or more circular objects to the segmentation map.

        Parameters
        ----------
        radius : `astropy.units.Quantity` or array-like, optional
            The outer radius of the aperture, by default 3 arcseconds.
        inner_radius : `astropy.units.Quantity` or array-like, optional
            If specified, an annulus will be extracted instead of
            an aperture.
        centre : `astropy.coordinates.SkyCoord`, optional
            The centre of the aperture.
        init_id : int, optional
            The ID to assign to the object. If multiple coordinates are
            given, the ID will increase in integer steps from ``init_id``.
            By default, ``init_id`` will start at ``max(seg_map)+1``.
        **kwargs : dict, optional
            Any inputs accepted by `~astropy.coordinates.SkyCoord`, if
            ``centre`` is None.

        Returns
        -------
        ndarray
            The ID(s) corresponding to the new object(s) in the
            segmentation map.
        """

        if not hasattr(self, "seg_map"):
            raise AttributeError("Segmentation map not set.")

        xs, ys, radii = self._process_coords_radii(
            radius, inner_radius, centre, **kwargs
        )

        if xs.shape[0] == 0:
            warnings.warn("No valid coordinates given, so no new objects added.")

        if init_id is None:
            init_id = np.nanmax(self.seg_map) + 1
        new_obj_ids = init_id + np.arange(xs.shape[0])

        test_in = np.isin(new_obj_ids, self.seg_map)
        if any(test_in):
            warnings.warn(
                f"Object IDs {new_obj_ids[test_in]} exist already in the segmentation"
                " map, and will be merged with the new object(s)."
            )
        for new_id, x_c, y_c, rads in zip(new_obj_ids, xs, ys, radii):
            mask = self._create_circular_mask(x_c, y_c, rads[1])
            if rads[0] != 0:
                mask[self._create_circular_mask(x_c, y_c, rads[0])] = 0
            self.seg_map[mask] = new_id

        return np.asarray(new_obj_ids)

    def add_sector_obj(
        self,
        radius: astropy.units.Quantity | npt.ArrayLike = 3 * u.arcsec,
        inner_radius: astropy.units.Quantity | npt.ArrayLike = 0,
        centre: astropy.coordinates.SkyCoord = None,
        segments: int = 4,
        angle: astropy.units.Quantity | float = 0,
        init_id: int | None = None,
        **kwargs,
    ) -> npt.NDArray:
        """
        Add one or more sector objects to the segmentation map.

        The new objects are obtained by dividing the requested
        circular objects in a specified number of sectors.

        Parameters
        ----------
        radius : `astropy.units.Quantity` or array-like, optional
            The outer radius of the aperture, by default 3 arcseconds.
        inner_radius : `~astropy.units.Quantity` or array-like, optional
            If specified, an annulus will be extracted instead of
            an aperture.
        centre : `astropy.coordinates.SkyCoord`, optional
            The centre of the aperture.
        segments : int, optional
            The number of sectors into which each object will be divided,
            by default 4.
        angle : `astropy.units.Quantity` or float, optional
            The position angle offset, by default 0.
        init_id : int, optional
            The ID to assign to the object. If multiple coordinates are
            given, the ID will increase in integer steps from ``init_id``.
            By default, ``init_id`` will start at ``max(seg_map)+1``.
        **kwargs : dict, optional
            Any inputs accepted by `~astropy.coordinates.SkyCoord`, if
            ``centre`` is ``None``.

        Returns
        -------
        ndarray
            The ID(s) corresponding to the new object(s) in the
            segmentation map.
        """

        if not hasattr(self, "seg_map"):
            raise AttributeError("Segmentation map not set.")

        xs, ys, radii = self._process_coords_radii(
            radius, inner_radius, centre, **kwargs
        )

        if xs.shape[0] == 0:
            warnings.warn("No valid coordinates given, so no new objects added.")

        if init_id is None:
            init_id = np.nanmax(self.seg_map) + 1
        potential_obj_ids = init_id + np.arange(0, radii.shape[0], int(segments))

        test_in = np.isin(potential_obj_ids, self.seg_map)
        if any(test_in):
            warnings.warn(
                f"Object IDs {potential_obj_ids[test_in]} exist already in the "
                "segmentation map, and will be merged with the new object(s)."
            )

        used_ids = []
        for pot_id, x_c, y_c, rads in zip(potential_obj_ids, xs, ys, radii):

            y, x = np.indices(self.seg_map.shape)

            if isinstance(angle, u.Quantity):
                angle = angle.to(u.deg).value
            angle_arr = (np.rad2deg(np.arctan2(x_c - x, y - y_c)) - angle) % 360.0

            mask = self._create_circular_mask(x_c, y_c, rads[1])
            if rads[0] != 0:
                mask[self._create_circular_mask(x_c, y_c, rads[0])] = 0

            for s in np.arange(segments):
                self.seg_map[
                    np.where(
                        (mask)
                        & (angle_arr >= s * 360 / segments)
                        & (angle_arr < (s + 1) * 360 / segments)
                    )
                ] = (pot_id + s)
                used_ids.append(pot_id + s)

        return np.asarray(used_ids)

    def add_reg_obj(
        self,
        reg_path: str | os.PathLike,
        format: str | None = None,
        reg_wcs: astropy.wcs.WCS | None = None,
        init_id: int | None = None,
    ) -> int:
        """
        Add a new object to the segmentation map from a regions file.

        Parameters
        ----------
        reg_path : str or os.PathLike
            The path pointing to the region file.
        format : str or None, optional
            The file format specifier. If None, the format is
            automatically inferred from the file extension.
        reg_wcs : `astropy.wcs.WCS` or None, optional
            The WCS to use to convert pixels to world coordinates.
            By default, the segmentation map WCS will be used.
        init_id : int, optional
            The ID to assign to the object. If multiple coordinates are
            given, the ID will increase in integer steps from ``init_id``.
            By default, ``init_id`` will start at ``max(seg_map)+1``.

        Returns
        -------
        int
            The ID corresponding to the new object in the
            segmentation map.

        Raises
        ------
        AttributeError
            If the segmentation map has not been loaded.
        ImportError
            If the regions package is not available.
        Exception
            If the supplied regions file cannot be read.
        """

        if not hasattr(self, "seg_map"):
            raise AttributeError("Segmentation map not set.")

        try:
            from regions import Regions, SkyRegion
        except:
            raise ImportError(
                "Astropy Regions is required to import region files.Please read"
                " https://astropy-regions.readthedocs.io for more information."
            )

        try:
            region = Regions.read(reg_path, format=format).regions[0]
        except Exception as e:
            raise Exception(
                f"Could not parse regions file, with the following error: {e}"
            )

        if issubclass(type(region), SkyRegion):
            pixel_region = region.to_pixel(self.seg_wcs if reg_wcs is None else reg_wcs)
        else:
            pixel_region = region

        mask = pixel_region.to_mask(mode="subpixels")

        matched_mask = mask.to_image(self.seg_map.shape) > 0.5

        if np.nansum(matched_mask) == 0:
            warnings.warn(
                "The specified region lies outside the segmentation map "
                "footprint. No new objects added."
            )

        if init_id is None:
            init_id = np.nanmax(self.seg_map) + 1
        else:
            test_in = np.isin(init_id, self.seg_map)
            if test_in:
                warnings.warn(
                    f"Object IDs {init_id} exist already in the "
                    "segmentation map, and will be merged with the new object(s)."
                )

        self.seg_map[matched_mask] = init_id

        return np.asarray(init_id)
