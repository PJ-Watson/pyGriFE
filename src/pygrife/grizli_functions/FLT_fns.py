"""Grizli functions used to load grism files and extract beam cutouts."""

import os
import traceback
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from grizli import grismconf, model, utils
from grizli.model import GrismDisperser


def init_from_input_multispec(self, flt, beam, conf=None, get_slice_header=True):
    """
    Initialize a `grizli.model.BeamCutout` object from data objects.

    The original function was taken from
    `~grizli.model.BeamCutout.init_from_input`, and has been modified to
    allow for a cutout overlapping multiple existing models in the
    contamination map (including multiple different spectra).

    Parameters
    ----------
    flt : `grizli.model.GrismFLT`
        Parent FLT frame.

    beam : `grizli.model.GrismDisperser`
        Object and spectral order to consider.

    conf : `grizli.grismconf.aXeConf`
        Pre-computed configuration file. If not specified will regenerate
        based on header parameters, which might be necessary for
        multiprocessing parallelization and pickling.

    get_slice_header : bool
        Get full header of the sliced data. Costs some overhead so can
        be skipped if full header information isn't required.
    """

    self.id = beam.id
    if conf is None:
        conf = grismconf.load_grism_config(flt.conf_file)

    # print(f"Reached here, key={self.id}")
    self.beam = model.GrismDisperser(
        id=beam.id,
        direct=beam.direct * 1,
        segmentation=beam.seg * 1,
        origin=beam.origin,
        pad=beam.pad,
        grow=beam.grow,
        beam=beam.beam,
        conf=conf,
        xcenter=beam.xcenter,
        ycenter=beam.ycenter,
        fwcpos=flt.grism.fwcpos,
        MW_EBV=flt.grism.MW_EBV,
    )

    if hasattr(beam, "psf_params"):
        self.beam.x_init_epsf(
            psf_params=beam.psf_params, psf_filter=beam.psf_filter, yoff=beam.psf_yoff
        )
        beam.x_init_epsf(
            psf_params=beam.psf_params, psf_filter=beam.psf_filter, yoff=beam.psf_yoff
        )

    slx_thumb = slice(self.beam.origin[1], self.beam.origin[1] + self.beam.sh[1])

    sly_thumb = slice(self.beam.origin[0], self.beam.origin[0] + self.beam.sh[0])

    if (
        hasattr(beam, "old_obj_ids")
        and len(np.atleast_1d(beam.old_obj_ids).flatten()) > 0
    ):
        # print(beam.old_obj_ids)
        seg_copy = deepcopy(beam.seg)
        beam.seg = deepcopy(flt.orig_seg[sly_thumb, slx_thumb])
        beam.seg[seg_copy != self.id] = 0
        beam.seg_ids = beam.old_obj_ids
        self.model = np.zeros_like(self.beam.model)
        self.modelf = np.zeros_like(self.beam.modelf)

        for o in beam.old_obj_ids:
            try:
                spec = flt.object_dispersers[o][1]
                cgs = flt.object_dispersers[o][0]
            except:
                print(f"{o} not in object dispersers")
                continue

            if cgs:
                scale = np.nansum(beam.direct[beam.seg == o]) / np.nansum(
                    flt.direct.data["REF"][flt.orig_seg == o]
                )
            else:
                scale = 1.0

            if hasattr(self.beam, "psf"):
                result = beam.compute_model_psf(
                    id=o,
                    in_place=False,
                    spectrum_1d=spec,
                    is_cgs=cgs,
                    scale=scale,
                )
            else:
                result = beam.compute_model(
                    id=o,
                    in_place=False,
                    spectrum_1d=spec,
                    is_cgs=cgs,
                    scale=scale,
                )

            self.modelf += result
            self.model += result.reshape(beam.sh_beam)

        self.beam.model = self.model
        self.beam.modelf = self.modelf

        # import matplotlib.pyplot as plt
        # plt.imshow(self.beam.model)
        # plt.show()

    elif beam.spectrum_1d is None:
        self.compute_model()
    else:
        self.compute_model(spectrum_1d=beam.spectrum_1d, is_cgs=beam.is_cgs)

    self.direct = flt.direct.get_slice(
        slx_thumb, sly_thumb, get_slice_header=get_slice_header
    )
    self.grism = flt.grism.get_slice(
        self.beam.slx_parent,
        self.beam.sly_parent,
        get_slice_header=get_slice_header,
    )

    self.contam = flt.model[self.beam.sly_parent, self.beam.slx_parent] * 1

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(3,1, sharex=True, sharey=True)
    # axs[0].imshow(np.log(self.contam), cmap="plasma", origin="lower")

    self.contam -= self.beam.model

    # axs[1].imshow(np.log(self.contam), cmap="plasma", origin="lower")
    # # plt.show()

    # # print (self.grism.data)
    # # print (dir(self.grism.data))

    # # import matplotlib.pyplot as plt
    # axs[2].imshow(np.log(self.grism.data["SCI"]-self.contam), cmap="plasma", origin="lower")
    # plt.show()


def get_beams_with_spectrum(
    grp,
    id,
    size=10,
    center_rd=None,
    beam_id="A",
    min_overlap=0.1,
    min_valid_pix=10,
    min_mask=0.01,
    min_sens=0.08,
    mask_resid=True,
    get_slice_header=True,
    show_exception=False,
    spectrum_1d=None,
    is_cgs=False,
):
    """
    Extract 2D spectra "beams" from the GroupFLT exposures.

    The original function was taken from
    `~grizli.multifit.GroupFLT.get_beams`, and has been modified by:

        - Allowing a 1D spectrum to be passed through to ``beam.model``.
        - Identifying the overlapping object IDs in the previous
          contamination model and segmentation map (if it exists), and
          passing them through to `~grizli.model.BeamCutout`.

    Parameters
    ----------
    grp : `grizli.multifit.GroupFLT`
        The container for multiple grism exposures.

    id : int
        Catalog ID of the object to extract.

    size : int
        Half-size of the 2D spectrum to extract, along cross-dispersion
        axis.

    center_rd : (float, float), optional
        Extract based on RA/Dec rather than catalog ID.

    beam_id : str
        Name of the order to extract.

    min_overlap : float
        Fraction of the spectrum along wavelength axis that has one
        or more valid pixels.

    min_valid_pix : int
        Minimum number of valid pixels (``beam.fit_mask == True``) in 2D
        spectrum.

    min_mask : float
        Minimum factor relative to the maximum pixel value of the flat
        f-lambda model where the 2D cutout data are considered good.
        Passed through to `~grizli.model.BeamCutout`.

    min_sens : float
        See `~grizli.model.BeamCutout`.

    mask_resid : bool
        See `~grizli.model.BeamCutout`.

    get_slice_header : bool
        Passed to `~grizli.model.BeamCutout`.

    show_exception : bool
        Show the exception if the beam cannot be loaded from an FLT file.
        False by default.

    spectrum_1d : [wavelengths, flux], optional
        The flux spectrum and corresponding wavelengths of the object
        in the model. By default, this is calculated automatically
        from the stored ``object_dispersers``.

    is_cgs : bool, optional
        The flux units of ``spectrum_1d[1]`` are cgs f_lambda flux
        densities, rather than normalised in the detection band, by
        default ``False``.

    Returns
    -------
    list
        List of `~grizli.model.BeamCutout` objects.
    """
    beams = grp.compute_single_model(
        id,
        center_rd=center_rd,
        size=size,
        store=False,
        get_beams=[beam_id],
    )

    out_beams = []
    for flt, beam in zip(grp.FLTs, beams):
        try:
            beam_in = beam[beam_id]

            if spectrum_1d is not None:
                beam_in.spectrum_1d = spectrum_1d
                beam_in.is_cgs = is_cgs
            elif hasattr(flt, "orig_seg"):
                old_obj_ids = np.unique(flt.orig_seg[flt.seg == id])
                beam_in.old_obj_ids = old_obj_ids.ravel()[
                    np.flatnonzero(old_obj_ids)
                ].astype(int)

            out_beam = model.BeamCutout(
                flt=flt,
                beam=beam_in,
                conf=flt.conf,
                min_mask=min_mask,
                min_sens=min_sens,
                mask_resid=mask_resid,
                get_slice_header=get_slice_header,
            )
        except:
            # print('Except: get_beams')
            if show_exception:
                utils.log_exception(utils.LOGFILE, traceback)

            continue

        valid = out_beam.grism["SCI"] != 0
        valid &= out_beam.fit_mask.reshape(out_beam.sh)
        hasdata = (valid.sum(axis=0) > 0).sum()
        if hasdata * 1.0 / out_beam.model.shape[1] < min_overlap:
            continue

        # Empty direct image?
        if out_beam.beam.total_flux == 0:
            continue

        if out_beam.fit_mask.sum() < min_valid_pix:
            continue

        out_beams.append(out_beam)

    return out_beams


def load_and_mod_FLT(
    grism_file,
    sci_extn,
    direct_file,
    pad,
    ref_file,
    ref_ext,
    seg_file,
    verbose,
    catalog,
    ix,
    use_jwst_crds,
):
    """
    Helper function for loading `~grizli.model.GrismFLT` objects.

    Originally taken from `grizli.multifit._loadFLT`, modified to allow
    for storing an original segmentation map, used for the contamination
    model, and an updated segmentation map, used for current object
    extraction. Docstring compiled by PJW for consistency.

    Parameters
    ----------
    grism_file : str
        The processed grism file, typically in the form
        ``"*GrismFLT.fits"``.
    sci_extn : int
        Science extension to extract from ``grism_file``. For WFC3/IR this
        can only be 1, though for the two-chip instruments WFC3/UVIS and
        ACS/WFC3 this can be 1 or 2.
    direct_file : str
        A direct exposure, corresponding to the grism exposure. If this is
        empty, ``ref_file`` will be used instead.
    pad : int or tuple[int, int]
        The padding in pixels, allowing modelling of sources outside
        the detector field of view. If a tuple is supplied, this is
        taken as ``(pady, padx)``.
    ref_file : str or None
        Undistorted reference image filename, e.g., a drizzled mosaic
        covering the area around a given grism exposure.
    ref_ext : int
        The FITS extension of the reference file containing the image.
    seg_file : str or None
        Segmentation image filename. If this is supplied, and it does not
        match the saved segmentation image, the returned
        `~grizli.model.GrismFLT` object will have two additional
        attributes, ``orig_seg_file``, and ``orig_seg``.
    verbose : bool
        Print verbose information.
    catalog : str
        Catalog filename assocated with ``seg_file``.  These are typically
        generated with "SExtractor", but the source of the files
        themselves isn't critical.
    ix : int
        The index of the returned `~grizli.model.GrismFLT` object in the
        parent `~grizli.multifit.GroupFLT` object.
    use_jwst_crds : bool
        Use CRDS ``specwcs`` reference files for JWST instruments.

    Returns
    -------
    `grizli.model.GrismFLT`
        The instantiated grism object, with all attributes loaded.
    """

    import time

    try:
        import cPickle as pickle
    except:
        # Python 3
        import pickle

    new_root = ".{0:02d}.GrismFLT.fits".format(sci_extn)
    save_file = grism_file.replace("_flt.fits", new_root)
    save_file = save_file.replace("_flc.fits", new_root)
    save_file = save_file.replace("_cmb.fits", new_root)
    save_file = save_file.replace("_rate.fits", new_root)
    save_file = save_file.replace("_elec.fits", new_root)

    if (save_file == grism_file) & ("GrismFLT" not in grism_file):
        # couldn't build new filename based on the extensions
        # so just insert at the end
        save_file = grism_file.replace(".fits", new_root)

    if (grism_file.find("_") < 0) & ("GrismFLT" not in grism_file):
        save_file = "xxxxxxxxxxxxxxxxxxx"

    if os.path.exists(save_file) & ("GrismFLT" in save_file):
        print("Load {0}!".format(save_file))

        fp = open(save_file.replace("GrismFLT.fits", "GrismFLT.pkl"), "rb")
        flt = pickle.load(fp)
        fp.close()

        flt.conf_file = f"{os.environ['GRIZLI']}/CONF/{flt.conf_file.split('/')[-1]}"

        status = flt.load_from_fits(save_file)

        if seg_file is not None:
            if hasattr(flt, "seg_file") and str(flt.seg_file) != str(seg_file):
                flt.orig_seg_file = deepcopy(flt.seg_file)
                flt.orig_seg = deepcopy(flt.seg)

            flt.process_seg_file(seg_file)

    else:
        flt = model.GrismFLT(
            grism_file=grism_file,
            sci_extn=sci_extn,
            direct_file=direct_file,
            pad=pad,
            ref_file=ref_file,
            ref_ext=ref_ext,
            seg_file=seg_file,
            shrink_segimage=True,
            verbose=verbose,
            use_jwst_crds=use_jwst_crds,
        )

    if flt.direct.wcs.wcs.has_pc():
        for obj in [flt.grism, flt.direct]:
            obj.get_wcs()

    if catalog is not None:
        flt.catalog = flt.blot_catalog(
            catalog, sextractor=("X_WORLD" in catalog.colnames)
        )
        flt.catalog_file = catalog

    else:
        flt.catalog = None

    if flt.grism.instrument in ["NIRCAM"]:
        flt.apply_POM()

    if flt.grism.instrument in ["NIRISS", "NIRCAM"]:
        flt.transform_JWST_WFSS()

    if hasattr(flt, "conf"):
        delattr(flt, "conf")

    return flt  # , out_cat


def mod_transform_JWST_WFSS(self, verbose=True):
    """
    Rotate data & wcs so that spectra are increasing to +x.

    This function was modified from
    `~grizli.model.GrismFLT.transform_JWST_WFSS`, so that any stored
    original segmentation map (``self.orig_seg``) would also be rotated.
    Docstring compiled by PJW for consistency.

    Parameters
    ----------
    verbose : bool
        Print verbose information.

    Returns
    -------
    bool
        Returns ``True`` if the transformation completed.
    """

    if self.grism.instrument not in ["NIRCAM", "NIRISS"]:
        return True

    if self.grism.instrument == "NIRISS":
        if self.grism.filter == "GR150C":
            rot = 2
        else:
            rot = -1

    elif self.grism.instrument in ["NIRCAM", "NIRCAMA"]:
        if self.grism.module == "A":
            #  Module A
            if self.grism.pupil == "GRISMC":
                rot = 1
            else:
                # Do nothing, A+GRISMR disperses to +x
                return True
        else:
            # Module B
            if self.grism.pupil == "GRISMC":
                rot = 1
            else:
                rot = 2

    elif self.grism.instrument == "NIRCAMB":
        if self.grism.pupil == "GRISMC":
            rot = 1
        else:
            rot = 2

    if self.is_rotated:
        rot *= -1

    self.is_rotated = not self.is_rotated
    if verbose:
        print("Transform JWST WFSS: flip={0}".format(self.is_rotated))

    # Compute new CRPIX coordinates
    # center = np.array(self.grism.sh)/2.+0.5
    # crpix = self.grism.wcs.wcs.crpix
    #
    # rad = np.deg2rad(-90*rot)
    # mat = np.zeros((2, 2))
    # mat[0, :] = np.array([np.cos(rad), -np.sin(rad)])
    # mat[1, :] = np.array([np.sin(rad), np.cos(rad)])
    #
    # crpix_new = np.dot(mat, crpix-center)+center

    # Full rotated SIP header
    orig_header = utils.to_header(self.grism.wcs, relax=True)
    hrot, wrot, desc = utils.sip_rot90(orig_header, rot)

    for obj in [self.grism, self.direct]:
        for k in hrot:
            obj.header[k] = hrot[k]

        # obj.header['CRPIX1'] = crpix_new[0]
        # obj.header['CRPIX2'] = crpix_new[1]
        #
        # # Get rotated CD
        # out_wcs = utils.transform_wcs(obj.wcs, translation=[0., 0.], rotation=rad, scale=1.)
        # new_cd = out_wcs.wcs.cd
        #
        # for i in range(2):
        #     for j in range(2):
        #         obj.header['CD{0}_{1}'.format(i+1, j+1)] = new_cd[i, j]

        # Update wcs
        obj.get_wcs()
        if obj.wcs.wcs.has_pc():
            obj.get_wcs()

        # Rotate data
        for k in obj.data.keys():
            if obj.data[k] is not None:
                obj.data[k] = np.rot90(obj.data[k], rot)

    # Rotate segmentation image
    self.seg = np.rot90(self.seg, rot)
    if hasattr(self, "orig_seg"):
        self.orig_seg = np.rot90(self.orig_seg, rot)
    self.model = np.rot90(self.model, rot)

    # print('xx Rotate images {0}'.format(rot))

    if self.catalog is not None:
        # print('xx Rotate catalog {0}'.format(rot))
        self.catalog = self.blot_catalog(
            self.catalog, sextractor=("X_WORLD" in self.catalog.colnames)
        )
    return True


def mod_compute_model_orders(
    self,
    id=0,
    x=None,
    y=None,
    size=10,
    mag=-1,
    spectrum_1d=None,
    is_cgs=False,
    compute_size=False,
    max_size=None,
    min_size=26,
    store=True,
    in_place=True,
    get_beams=None,
    psf_params=None,
    verbose=True,
):
    """
    Compute the dispersed spectrum for a given object id.

    The original function was taken from
    `~grizli.model.GrismFLT.compute_model_orders`, and has been modified
    to allow for a model computation overlapping multiple existing models
    in the contamination map (including multiple different spectra).

    Parameters
    ----------
    id : int
        Object ID number to match in the segmentation image.

    x, y : float
        Center of the cutout to extract.

    size : int
        Radius of the cutout to extract.  The cutout is equivalent to::

            xc, yc = int(x), int(y)
            thumb = self.direct.data['SCI'][yc-size:yc+size, xc-size:xc+size]

    mag : float
        Specified object magnitude, which will be compared to the
        "MMAG_EXTRACT_[BEAM]" parameters in `self.conf` to decide if the
        object is bright enough to compute the higher spectral orders.
        The default value of -1 means that all orders listed in
        `self.conf.beams` will be computed.

    spectrum_1d : None or [`~numpy.array`, `~numpy.array`]
        Template 1D spectrum to convolve with the grism disperser.  If
        None, assumes trivial spectrum flat in f_lambda flux densities.
        Otherwise, the template is taken to be::

            wavelength, flux = spectrum_1d

    is_cgs : bool
        Flux units of `spectrum_1d[1]` are cgs f_lambda flux densities,
        rather than normalized in the detection band.

    compute_size : bool
        Ignore `x`, `y`, and `size` and compute the extent of the
        segmentation polygon directly using
        `~grizli.utils_c.disperse.compute_segmentation_limits`.

    max_size : int or None
        Enforce a maximum size of the cutout when using ``compute_size``.

    min_size : int or None
        Enforce a minimum size of the cutout when using ``compute_size``.

    store : bool
        If True, then store the computed beams in the
        `~grizli.grismconf.OrderedDict` ``self.object_dispersers[id]``.

        If many objects are computed, this can be memory intensive. To
        save memory, set to ``False`` and then the function just stores
        the input template spectrum (``spectrum_1d``) and the beams will
        have to be recomputed if necessary.

    in_place : bool
        If ``True``, add the computed spectral orders into ``self.model``.
        Otherwise, make a clean array with only the orders of the given
        object.

    get_beams : list or None
        Spectral orders to retrieve with names as defined in the
        configuration files, e.g., ``['A']`` generally for the +1st order
        of HST grisms.  If ``None``, then get all orders listed in the
        ``beams`` attribute of the `~grizli.grismconf.aXeConf`
        configuration object.

    psf_params : list
        Optional parameters for generating an `~grizli.utils.EffectivePSF`
        object for the spatial morphology.

    verbose : bool
        Print verbose information.

    Returns
    -------
    bool or `numpy.array`
        If ``in_place`` is ``True``, return status of ``True`` if
        everything goes OK. The computed spectral orders are stored in
        place in ``self.model``.

        Returns ``False`` if the specified ``id`` is not found in the
        segmentation array independent of ``in_place``.

        If ``in_place`` is ``False``, return a full array including the
        model for the single object.
    """
    from grizli.utils_c import disperse

    old_spectrum_1d = None
    old_cgs = None
    if id in self.object_dispersers:
        object_in_model = True
        beams = self.object_dispersers[id]

        out = self.object_dispersers[id]

        # Handle pre 0.3.0-7 formats
        if len(out) == 3:
            old_cgs, old_spectrum_1d, beams = out
        else:
            old_cgs, old_spectrum_1d = out
            beams = None

    else:
        object_in_model = False
        beams = None

    if self.direct.data["REF"] is None:
        ext = "SCI"
    else:
        ext = "REF"

    # set up the beams to extract
    if get_beams is None:
        beam_names = self.conf.beams
    else:
        beam_names = get_beams

    # Did we initialize the PSF model this call?
    INIT_PSF_NOW = False

    # Do we need to compute the dispersed beams?
    if beams is None:
        # Use catalog
        xcat = ycat = None
        if self.catalog is not None:
            ix = self.catalog["id"] == id
            # print(len(self.catalog))
            if ix.sum() == 0:
                if verbose:
                    print(f"ID {id} not found in segmentation image")
                return False

            if hasattr(self.catalog["x_flt"][ix][0], "unit"):
                xcat = self.catalog["x_flt"][ix][0].value - 1
                ycat = self.catalog["y_flt"][ix][0].value - 1
            else:
                xcat = self.catalog["x_flt"][ix][0] - 1
                ycat = self.catalog["y_flt"][ix][0] - 1

            # print '!!! X, Y: ', xcat, ycat, self.direct.origin, size

            # use x, y if defined
            if x is not None:
                xcat = x
            if y is not None:
                ycat = y

        if (compute_size) | (x is None) | (y is None) | (size is None):
            # Get the array indices of the segmentation region
            out = disperse.compute_segmentation_limits(
                self.seg, id, self.direct.data[ext], self.direct.sh
            )

            ymin, ymax, y, xmin, xmax, x, area, segm_flux = out
            if (area == 0) | ~np.isfinite(x) | ~np.isfinite(y):
                if verbose:
                    print("ID {0:d} not found in segmentation image".format(id))
                return False

            # Object won't disperse spectrum onto the grism image
            if (
                (ymax < self.pad[0] - 5)
                | (ymin > self.direct.sh[0] - self.pad[0] + 5)
                | (ymin == 0)
                | (ymax == self.direct.sh[0])
                | (xmin == 0)
                | (xmax == self.direct.sh[1])
            ):
                return True

            if compute_size:
                try:
                    size = int(
                        np.ceil(np.max([x - xmin, xmax - x, y - ymin, ymax - y]))
                    )
                except ValueError:
                    return False

                size += 4

                # Enforce minimum size
                # size = np.maximum(size, 16)
                size = np.maximum(size, min_size)

                # To do: enforce a larger minimum cutout size for grisms
                # that need it, e.g., UVIS/G280L

                # maximum size
                if max_size is not None:
                    size = np.min([size, max_size])

                # Avoid problems at the array edges
                size = np.min([size, int(x) - 2, int(y) - 2])

                if size < 4:
                    return True

        # Thumbnails
        # print '!! X, Y: ', x, y, self.direct.origin, size

        if xcat is not None:
            xc, yc = int(np.round(xcat)) + 1, int(np.round(ycat)) + 1
            xcenter = xcat - (xc - 1)
            ycenter = ycat - (yc - 1)
        else:
            xc, yc = int(np.round(x)) + 1, int(np.round(y)) + 1
            xcenter = x - (xc - 1)
            ycenter = y - (yc - 1)

        origin = [yc - size + self.direct.origin[0], xc - size + self.direct.origin[1]]

        thumb = self.direct.data[ext][yc - size : yc + size, xc - size : xc + size]
        seg_thumb = self.seg[yc - size : yc + size, xc - size : xc + size]

        # Test that the id is actually in the thumbnail
        test = disperse.compute_segmentation_limits(
            seg_thumb, id, thumb, np.array(thumb.shape)
        )
        if test[-2] == 0:
            if verbose:
                print(f"ID {id} not found in segmentation image")
            return False

        # # Get precomputed dispersers
        # beams, old_spectrum_1d, old_cgs = None, None, False
        # if object_in_model:
        #     out = self.object_dispersers[id]
        #
        #     # Handle pre 0.3.0-7 formats
        #     if len(out) == 3:
        #         old_cgs, old_spectrum_1d, old_beams = out
        #     else:
        #         old_cgs, old_spectrum_1d = out
        #         old_beams = None
        #
        #     # Pull out just the requested beams
        #     if old_beams is not None:
        #         beams = OrderedDict()
        #         for b in beam_names:
        #             beams[b] = old_beams[b]
        #
        # if beams is None:

        # Compute spectral orders ("beams")
        beams = OrderedDict()

        for b in beam_names:
            # Only compute order if bright enough
            if mag > self.conf.conf_dict["MMAG_EXTRACT_{0}".format(b)]:
                continue

            try:
                beam = GrismDisperser(
                    id=id,
                    direct=thumb,
                    segmentation=seg_thumb,
                    xcenter=xcenter,
                    ycenter=ycenter,
                    origin=origin,
                    pad=self.pad,
                    grow=self.grism.grow,
                    beam=b,
                    conf=self.conf,
                    fwcpos=self.grism.fwcpos,
                    MW_EBV=self.grism.MW_EBV,
                )
            except:
                utils.log_exception(utils.LOGFILE, traceback)

                continue

            # Set PSF model if necessary
            if psf_params is not None:
                store = True
                INIT_PSF_NOW = True
                if self.direct.ref_filter is None:
                    psf_filter = self.direct.filter
                else:
                    psf_filter = self.direct.ref_filter

                beam.x_init_epsf(
                    flat_sensitivity=False,
                    psf_params=psf_params,
                    psf_filter=psf_filter,
                    yoff=0.0,
                )

            beams[b] = beam

    # Compute old model
    if hasattr(self, "orig_seg"):
        print("Contains old segmentation map.")
        for b in beams:
            beam = beams[b]

            # Derive, assuming code section above has not run
            slx_thumb = slice(beam.origin[1], beam.origin[1] + beam.sh[1])

            sly_thumb = slice(beam.origin[0], beam.origin[0] + beam.sh[0])
            seg_copy = deepcopy(beam.seg)
            beam.seg = deepcopy(self.orig_seg[sly_thumb, slx_thumb])
            beam.seg[seg_copy != beam.id] = 0
            old_obj_ids = np.unique(beam.seg[seg_copy == beam.id])
            old_obj_ids = old_obj_ids.ravel()[np.flatnonzero(old_obj_ids)].astype(int)
            seg_ids_copy = deepcopy(beam.seg_ids)
            beam.seg_ids = old_obj_ids

            print(f"old_obj_ids: {old_obj_ids}")

            new_modelf = np.zeros_like(beam.modelf)
            for i, o in enumerate(old_obj_ids):
                try:
                    if o == id:
                        spec = old_spectrum_1d
                        cgs = old_cgs
                    else:
                        spec = self.object_dispersers[o][1]
                        cgs = self.object_dispersers[o][0]

                    if cgs:
                        scale = np.nansum(beam.direct[beam.seg == o]) / np.nansum(
                            self.direct.data["REF"][self.orig_seg == o]
                        )
                    else:
                        scale = 1.0

                    if hasattr(beam, "psf") & (not INIT_PSF_NOW):
                        store = True
                        result = beam.compute_model_psf(
                            id=o,
                            in_place=False,
                            spectrum_1d=spec,
                            is_cgs=cgs,
                            scale=scale,
                        )
                    else:
                        result = beam.compute_model(
                            id=o,
                            in_place=False,
                            spectrum_1d=spec,
                            is_cgs=cgs,
                            scale=scale,
                        )

                    new_modelf += result
                except Exception as e:
                    print(f"Error in model computation: {e}")

            beam.modelf = new_modelf
            beam.model = new_modelf.reshape(beam.sh_beam)
            beam.seg = seg_copy
            beam.seg_ids = seg_ids_copy
            beam.id = id

            # import matplotlib.pyplot as plt
            # plt.imshow(beam.model)
            # plt.show()

            object_in_model = True

    elif object_in_model:
        for b in beams:
            beam = beams[b]
            if hasattr(beam, "psf") & (not INIT_PSF_NOW):
                store = True
                beam.compute_model_psf(spectrum_1d=old_spectrum_1d, is_cgs=old_cgs)
            else:
                beam.compute_model(spectrum_1d=old_spectrum_1d, is_cgs=old_cgs)

    if get_beams:
        out_beams = OrderedDict()
        for b in beam_names:
            out_beams[b] = beams[b]
        return out_beams

    if in_place:
        # Update the internal model attribute
        output = self.model

        if store:
            # Save the computed beams
            self.object_dispersers[id] = is_cgs, spectrum_1d, beams
        else:
            # Just save the model spectrum (or empty spectrum)
            self.object_dispersers[id] = is_cgs, spectrum_1d, None
    else:
        # Create a fresh array
        output = np.zeros_like(self.model)

    # if in_place:
    #     ### Update the internal model attribute
    #     output = self.model
    # else:
    #     ### Create a fresh array
    #     output = np.zeros_like(self.model)

    # Set PSF model if necessary
    if psf_params is not None:
        if self.direct.ref_filter is None:
            psf_filter = self.direct.filter
        else:
            psf_filter = self.direct.ref_filter

    # Loop through orders and add to the full model array, in-place or
    # a separate image
    for b in beams:
        beam = beams[b]
        print("HAS ORIG SEG?", self.orig_seg)
        # Subtract previously-added model
        if object_in_model & in_place:
            print("SUBTRACTING OLD MODEL")
            beam.add_to_full_image(-beam.model, output)

        # Update PSF params
        # if psf_params is not None:
        #     skip_init_psf = False
        #     if hasattr(beam, 'psf_params'):
        #         skip_init_psf |= np.product(np.isclose(beam.psf_params, psf_params)) > 0
        #
        #     if not skip_init_psf:
        #         beam.x_init_epsf(flat_sensitivity=False, psf_params=psf_params, psf_filter=psf_filter, yoff=0.06)

        # Compute model
        if hasattr(beam, "psf"):
            beam.compute_model_psf(spectrum_1d=spectrum_1d, is_cgs=is_cgs)
        else:
            beam.compute_model(spectrum_1d=spectrum_1d, is_cgs=is_cgs)

        # Add in new model
        beam.add_to_full_image(beam.model, output)

    if in_place:
        return True
    else:
        return beams, output
