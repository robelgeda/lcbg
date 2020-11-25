import warnings

import numpy as np

from astropy.modeling import models, fitting, functional_models, Parameter, custom_model
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.utils.exceptions import AstropyWarning

from matplotlib import pyplot as plt

from photutils import aperture_photometry, CircularAperture, CircularAnnulus, EllipticalAnnulus, EllipticalAperture

import ipywidgets as widgets
from IPython.display import display

from .utils import cutout
from .segmentation import segm_mask, masked_segm_image
from .fitting import plot_fit, fit_model, model_subtract, fit_plane, model_to_image
from .segmentation import get_source_e,  get_source_theta, get_source_position


def plot_apertures(image=None, apertures=[], vmin=None, vmax=None, color='white', lw=1.5):
    if image is not None:
        plt.imshow(image, cmap='Greys_r', vmin=vmin, vmax=vmax)

    for aperture in apertures:
        aperture.plot(axes=plt.gca(), color=color, lw=lw)

    if image or apertures:
        plt.title('Apertures')


def flux_to_abmag(flux, header):
    """Convert HST flux to AB Mag"""

    PHOTFLAM = header['PHOTFLAM']
    PHOTZPT = header['PHOTZPT']
    PHOTPLAM = header['PHOTPLAM']

    STMAG_ZPT = (-2.5 * np.log10(PHOTFLAM)) + PHOTZPT
    ABMAG_ZPT = STMAG_ZPT - (5. * np.log10(PHOTPLAM)) + 18.692

    return -2.5 * np.log10(flux) + ABMAG_ZPT


def order_cat(cat, key='area', reverse=True):
    """
    Sort a catalog by largest area and return the argsort
    Parameters
    ----------
    cat : `SourceCatalog` instance
        A `SourceCatalog` instance containing the properties of each
        source.
    key : string
        Key to sort
    reverse : bool
        Reverse sorting order. Default is `True` to place largest values on top.

    Returns
    -------
    output : list
        A list of catalog indices ordered by largest area
    """
    table = cat.to_table()[key]
    order_all = table.argsort()
    if reverse:
        return list(reversed(order_all))
    return order_all


def radial_elliptical_aperture(position, r, e=1., theta=0.):
    """
    Helper function given a radius, elongation and theta,
    will make an elliptical aperture.

    Parameters
    ----------
    position : tuple
        (x, y) coords for center of aperture
    r : int or float
        Semi-major radius of the aperture
    e : float
        Elongation
    theta : float
        Orientation in rad

    Returns
    -------
    EllipticalAperture
    """
    a, b = r, r / e
    return EllipticalAperture(position, a, b, theta=theta)


def radial_elliptical_annulus(position, r, dr, e=1., theta=0.):
    """
    Helper function given a radius, elongation and theta,
    will make an elliptical aperture.

    Parameters
    ----------
    position : tuple
        (x, y) coords for center of aperture
    r : int or float
        Semi-major radius of the inner ring
    dr : int or float
        Thickness of annulus (outer ring = r + dr).
    e : float
        Elongation
    theta : float
        Orientation in rad

    Returns
    -------
    EllipticalAnnulus
    """

    a_in, b_in = r, r / e
    a_out, b_out = r + dr, (r + dr) / e

    return EllipticalAnnulus(position, a_in, a_out, b_out, theta=theta)


def calculate_photometic_density(r_list, flux_list, e=1., theta=0.):
    """ Compute value between radii """
    density = []

    last_flux = 0
    last_area = 0
    for r, flux in zip(r_list, flux_list):
        aperture = radial_elliptical_aperture((0, 0), r, e=e, theta=theta)
        area = aperture.area
        density.append((flux - last_flux) / (area - last_area))
        last_area, last_flux = area, flux

    return density


def make_radius_list(max_pix, n):
    """Make an array of radii of size n up to max_pix"""
    r_list = [x * int(max_pix/n) for x in range(1, n+1)]
    return np.array(r_list)


def photometry_step(position, r_list, image, error=None, mask=None, e=1., theta=0.,
                    plot=False, vmin=0, vmax=None, method='exact'):
    """
    Given a position, a list of radii and the shape of apertures, calculate the
    photometry of the target in the image.

    Parameters
    ----------
    position : tuple
        (x, y) position in pixels
    r_list : list
        A list of radii for apertures.
    image : 2D array
        Image to preform photometry on.
    error : 2D array
        Error map of the image.
    mask : 2D array
        Boolean array with True meaning that pixel is unmasked.
    e : float
        Elongation
    theta : float
        Orientation in rad
    plot : bool
        Plot the target and apertures.
    vmin : int
        Min value for plot
    vmax : int
        Max value for plot
    method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture on
            the pixel grid.  Not all options are available for all
            aperture types.  Note that the more precise methods are
            generally slower.  The following methods are available:

                * ``'exact'`` (default):
                  The the exact fractional overlap of the aperture and
                  each pixel is calculated.  The returned mask will
                  contain values between 0 and 1.

                * ``'center'``:
                  A pixel is considered to be entirely in or out of the
                  aperture depending on whether its center is in or out
                  of the aperture.  The returned mask will contain
                  values only of 0 (out) and 1 (in).

                * ``'subpixel'``
                  A pixel is divided into subpixels (see the
                  ``subpixels`` keyword), each of which are considered
                  to be entirely in or out of the aperture depending on
                  whether its center is in or out of the aperture.  If
                  ``subpixels=1``, this method is equivalent to
                  ``'center'``.  The returned mask will contain values
                  between 0 and 1.


    Returns
    -------
    photometry, aperture_area, error: array
        Returns photometry, aperture area (unmasked pixels) and error at each radius.
    """

    photometry_arr = []
    error_arr = []
    area_arr = []

    if plot:
        plt.imshow(image, vmin=vmin, vmax=image.mean() * 10 if vmax is None else vmax)

    mask = ~mask if mask is not None else None
    for i, r in enumerate(r_list):
        aperture = radial_elliptical_aperture(position, r, e=e, theta=theta)

        photometric_value, photometric_err = aperture.do_photometry(data=image, error=error, mask=mask, method=method)
        aperture_area, aperture_area_err = aperture.do_photometry(data=np.ones_like(image), error=None,
                                                                  mask=mask, method=method)

        aperture_area = float(np.round(aperture_area, 6))
        photometric_value = float(np.round(photometric_value, 6))
        photometric_err = float(np.round(photometric_err, 6)) if photometric_err.size > 0 else np.nan

        photometric_value
        if np.isnan(photometric_value):
            raise Exception("Nan photometric_value")

        if plot:
            aperture.plot(plt.gca(), color='w', alpha=0.5)

        photometry_arr.append(photometric_value)
        area_arr.append(aperture_area)
        error_arr.append(photometric_err)

    return np.array(photometry_arr), np.array(area_arr), np.array(error_arr)


def object_photometry(obj, image, segm_deblend, r_list, error=None, sigma=3.0, mean_sub=False, method='exact',
                      mask_background=False, plot=False, vmin=0, vmax=None, cutout_size=None):
    """

    method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture on
            the pixel grid.  Not all options are available for all
            aperture types.  Note that the more precise methods are
            generally slower.  The following methods are available:

                * ``'exact'`` (default):
                  The the exact fractional overlap of the aperture and
                  each pixel is calculated.  The returned mask will
                  contain values between 0 and 1.

                * ``'center'``:
                  A pixel is considered to be entirely in or out of the
                  aperture depending on whether its center is in or out
                  of the aperture.  The returned mask will contain
                  values only of 0 (out) and 1 (in).

                * ``'subpixel'``
                  A pixel is divided into subpixels (see the
                  ``subpixels`` keyword), each of which are considered
                  to be entirely in or out of the aperture depending on
                  whether its center is in or out of the aperture.  If
                  ``subpixels=1``, this method is equivalent to
                  ``'center'``.  The returned mask will contain values
                  between 0 and 1.
        """
    if plot:
        print(obj.id)
        fig, ax = plt.subplots(1, 2, figsize=[24, 12])

    # Get object geometry
    # -------------------
    position = get_source_position(obj)
    e = get_source_e(obj)
    theta = get_source_theta(obj)

    if cutout_size is None:
        cutout_size = np.ceil(max(r_list) * 3)

    if cutout_size % 2 == 1:
        cutout_size += 1

    # Error cutout
    # ------------
    masked_err = None
    if error is not None:
        masked_err = Cutout2D(error, position, cutout_size, mode='partial', fill_value=np.nan).data

    # Image Cutout
    # ------------
    full_masked_image = masked_segm_image(obj, image, segm_deblend, fill=np.nan, mask_background=mask_background).data
    masked_nan_image = Cutout2D(full_masked_image, position, cutout_size, mode='partial', fill_value=np.nan)
    masked_image = masked_nan_image.data

    # Cutout for Stats
    # ----------------
    # This cutout has all sources masked
    stats_cutout_size = cutout_size  # max(obj.bbox.ixmax - obj.bbox.ixmin, obj.bbox.iymax - obj.bbox.iymin) * 2
    full_bg_image = masked_segm_image(0, image, segm_deblend, fill=np.nan, mask_background=False).data
    masked_stats_image = Cutout2D(full_bg_image, position, stats_cutout_size, mode='partial', fill_value=np.nan).data

    # Subtract Mean Plane
    # -------------------
    if mean_sub:
        if len(np.where(~np.isnan(masked_stats_image))[0]) > 10:
            with warnings.catch_warnings():
                mean, median, std = sigma_clipped_stats(masked_stats_image, sigma=sigma,
                                                        mask=np.isnan(masked_stats_image.data))

                warnings.simplefilter('ignore', AstropyWarning)
                fitted_model, _ = fit_plane(sigma_clip(masked_stats_image, sigma))
                masked_image -= model_to_image(cutout_size // 2, cutout_size // 2, cutout_size, fitted_model)
        elif plot:
            print("mean_sub: Not enough datapoints, did not subtract.")

    # Make mask
    # ---------
    mask = np.ones_like(masked_image)
    mask[np.where(np.isnan(masked_image))] = 0
    mask = mask.astype(bool)

    position = np.array(masked_image.data.shape) / 2.

    if plot:
        plt.sca(ax[0])

    photometry_arr, area_arr, error_arr = photometry_step(position, r_list, masked_image, error=masked_err, mask=mask,
                                                          e=e, theta=theta, plot=plot, vmin=vmin, vmax=vmax,
                                                          method=method)

    if plot:
        plt.sca(ax[1])
        plt.plot(r_list, photometry_arr, c='black', linewidth=3)
        for r in r_list:
            plt.axvline(r, alpha=0.5, c='r')
        plt.show()

        r = max(r_list)
        fig, ax = plt.subplots(1, 1, figsize=[24, 6])
        plt.plot(masked_image[:, int(position[0])], c='black', linewidth=3)
        plt.axhline(0, c='black')
        # plt.axhline(noise_sigma, c='b')
        plt.axvline(position[0], linestyle='--')
        plt.axvline(position[0] + r, alpha=0.5, c='r')
        plt.axvline(position[0] - r, alpha=0.5, c='r')
        plt.xlabel("Slice Along Y [pix]")
        plt.ylabel("Flux")

        fig, ax = plt.subplots(1, 1, figsize=[24, 6])

        plt.plot(masked_image[int(position[1]), :], c='black', linewidth=3)
        plt.axhline(0, c='black')
        # plt.axhline(noise_sigma, c='b')
        plt.axvline(position[0], linestyle='--')
        plt.axvline(position[0] + r, alpha=0.5, c='r')
        plt.axvline(position[0] - r, alpha=0.5, c='r')
        plt.xlabel("Slice Along X [pix]")
        plt.ylabel("Flux")

    return photometry_arr, area_arr, error_arr


"""
def photometry_step(position, r_list, image, error=None, mask=None, e=1., theta=0.,
                    plot=False, vmin=0, vmax=None):

    aperture_photometry_row = []
    aperture_area_row = []

    if plot:
        plt.imshow(image, vmin=vmin, vmax=image.mean() * 10 if vmax is None else vmax)

    for i, r in enumerate(r_list):
        aperture = radial_elliptical_aperture(position, r, e=e, theta=theta)

        photometric_value, photometric_err = aperture.do_photometry(data=image, error=error, mask=~mask)
        aperture_area, aperture_area_err = aperture.do_photometry(data=np.ones_like(image), error=None, mask=~mask)

        aperture_area = float(np.round(aperture_area, 6))
        photometric_value = float(np.round(photometric_value, 6))

        if np.isnan(photometric_value):
            raise Exception("Nan photometric_value")

        if plot:
            aperture.plot(plt.gca(), color='w', alpha=0.5)

        aperture_photometry_row.append(photometric_value)
        aperture_area_row.append(aperture_area)

    return aperture_photometry_row, aperture_area_row


def object_photometry(obj, image, segm_deblend, r_list, error=None, sigma=3.0, mean_sub=False,
                      mask_background=False, plot=False, vmin=0, vmax=None):
    if plot:
        print(obj.id)
        fig, ax = plt.subplots(1, 2, figsize=[24, 12])

    # Get object geometry
    position = get_source_position(obj)
    e = get_source_e(obj)
    theta = get_source_theta(obj)

    cutout_size = max(r_list) * 3

    # mask cutout
    mask = segm_mask(obj, segm_deblend, mask_background=mask_background)
    mask = Cutout2D(mask, position, cutout_size, mode='partial', fill_value=False).data

    # Error cutout
    masked_err = None
    if error is not None:
        masked_err = Cutout2D(error, position, cutout_size, mode='partial', fill_value=np.nan).data

    # Estimate mean in cutout
    masked_nan_image = masked_segm_image(obj, image, segm_deblend, fill=np.nan, mask_background=mask_background)
    masked_nan_image = Cutout2D(masked_nan_image.data, position, cutout_size, mode='partial', fill_value=np.nan)
    mean, median, std = sigma_clipped_stats(masked_nan_image.data, sigma=sigma, mask=np.isnan(masked_nan_image.data))

    # Make coutuout
    masked_image = masked_nan_image.data
    if mean_sub:
        masked_image -= mean

    position = np.array(masked_image.data.shape) / 2.

    if plot:
        plt.sca(ax[0])

    aperture_photometry_row, a_list = photometry_step(position, r_list, masked_image, error=masked_err, mask=mask,
                                                      e=e, theta=theta, plot=plot, vmin=vmin, vmax=vmax)

    if plot:
        plt.sca(ax[1])
        plt.plot(r_list, aperture_photometry_row, c='black', linewidth=3)
        for r in r_list:
            plt.axvline(r, alpha=0.5, c='r')
        plt.show()

        r = max(r_list)
        fig, ax = plt.subplots(1, 1, figsize=[24, 6])
        plt.plot(masked_image[:, int(position[0])], c='black', linewidth=3)
        plt.axhline(0, c='black')
        # plt.axhline(noise_sigma, c='b')
        plt.axvline(position[0], linestyle='--')
        plt.axvline(position[0] + r, alpha=0.5, c='r')
        plt.axvline(position[0] - r, alpha=0.5, c='r')
        plt.xlabel("Slice Along Y [pix]")
        plt.ylabel("Flux")

        fig, ax = plt.subplots(1, 1, figsize=[24, 6])

        plt.plot(masked_image[int(position[1]), :], c='black', linewidth=3)
        plt.axhline(0, c='black')
        # plt.axhline(noise_sigma, c='b')
        plt.axvline(position[0], linestyle='--')
        plt.axvline(position[0] + r, alpha=0.5, c='r')
        plt.axvline(position[0] - r, alpha=0.5, c='r')
        plt.xlabel("Slice Along X [pix]")
        plt.ylabel("Flux")

    return aperture_photometry_row, a_list
"""

def remove_fitted_sources(image, image_residual, cat,
                          r_inner_mult=1,
                          r_outter_mult=7,
                          r_output_mult=20,
                          index_start=None, index_end=None,
                          n_models=2,
                          show_progress=True,
                          print_fit=False,
                          show_fit=True):
    """
    Fits and removes bright sersic profiles from image.

    Parameters
    ----------
    image : array
        Input image to remove sources from. This image should be noise subtracted
    image_residual : array
        Image resulting from subtracting the input image by all the catalog sources.
    cat : `SourceCatalog` instance
        A `SourceCatalog` instance containing the properties of each
        source.
    r_inner_mult : int
        Multiples the object radius to give the radius of the object's core. Should be >= 1.
    r_outter_mult : int
        Multiples the object radius to give the radius of the outer components. Should be >= 1.
    r_output_mult : int
        Multiples the object radius to give the radius of the final cutout that is subtracted from the image.
        Should be >= 1.
    index_start : int
        Index of the first source in the area sorted catalog. So if this is set to 0,
        it will start removing at the object with the largest area in the catalog.
    index_end : int
        Index of last source in the area sorted catalog.
    n_models : int
        Number of models to use for fitting
        1 = Sersic2D
        2 = Sersic2D_core + Sersic2D_outter
        3 = Sersic2D_core + Sersic2D_outter + models.Gaussian2D
        all models are centered around the same pixel (the core).
    show_progress : bool
        Show jupyter notebook progress bar
    print_fit : bool
        Print fit params
    show_fit : bool
        Plot fit

    Returns
    -------
    subtracted_image : array
        An image with the fitted sources subtracted from it.

    subtracted_image_residual : array
        The input residual image with the fitted sources subtracted from it.

    fitted_sources : array
        An image showing only the fitted sources from the models.
    """

    order = order_cat(cat)

    order = order[index_start:index_end]

    if show_progress:
        pb = widgets.IntProgress(
            value=0,
            min=0,
            max=len(order),
            step=1,
            description='Loading:',
            bar_style='',
            orientation='horizontal'
        )

        display(pb)

    # Copy image for output
    image_copy = image.copy()
    image_residual_copy = image_residual.copy()
    image_zero = np.zeros_like(image_copy)
    photometric_sums = {}

    for i, cat_index in enumerate(order):
        if show_fit or print_fit:
            print(index_start + i, cat_index)

        if show_progress:
            # Update progress bar
            pb.value = i + 1
            pb.description = "{}/{}".format(pb.value, len(order))

        # Load object and aperture
        obj = cat[cat_index]

        # Estimate center of bobject
        cut = obj.segment.make_cutout(image, masked_array=True)
        cy, cx = np.unravel_index(cut.argmax(), cut.shape)

        x = obj.segment.bbox.ixmin + cx
        y = obj.segment.bbox.iymin + cy

        # Estimate other parts
        theta = obj.orientation.to(u.rad).value
        ellip = (1 - obj.semiminor_axis_sigma.value / obj.semimajor_axis_sigma.value)
        amp = cut.data.max()

        # Define fitting radius
        r_object = int(np.round(obj.semimajor_axis_sigma.value))  # inner cutout radius

        r_inner = r_object * r_inner_mult

        r_outter = r_object * r_outter_mult

        # Make images to fit
        target_zoom = cutout(image_copy, x, y, r_inner)
        target = cutout(image_copy, x, y, r_outter)
        target_residual = cutout(image_residual_copy, x, y, r_outter)

        # Fit inner core
        # --------------

        # Find center of zoomed cutout image
        y_0, x_0 = np.array(target_zoom.shape) // 2

        # Make inner model
        xy_slack = 10  # x and y value range / 2

        model_1 = models.Sersic2D(
            amplitude=amp,
            n=2,
            r_eff=r_inner,
            ellip=ellip,
            theta=theta,
            x_0=x_0,
            y_0=y_0,
            # fixed={'theta':True, 'ellip':True},
            bounds={
                'amplitude': (0, None),
                'r_eff': (0, None),
                'n': (0, 6),
                'ellip': (0, 1),
                'theta': (0, 2 * np.pi),
                'x_0': (x_0 - xy_slack, x_0 + xy_slack),
                'y_0': (y_0 - xy_slack, y_0 + xy_slack),
            })

        # Fit core model to zoomed cutout to tighten first guess
        model_1, fit = fit_model(target_zoom, model_1, maxiter=10000, epsilon=1e-40)

        # Fix the center x and y of all models to match the fitted guess
        """
        for pn in model.param_names:
            model.fixed[pn] = True            
        """

        model_1.fixed.update({
            'x_0': True,
            'y_0': True,
        })

        del model_1.bounds['x_0']
        del model_1.bounds['y_0']

        # Fit glow around target
        # ----------------------

        # Remove old center and add new center
        y_0, x_0 = np.array(target_zoom.shape) // 2

        model_1.x_0 -= x_0
        model_1.y_0 -= y_0

        y_0, x_0 = np.array(target.shape) // 2

        model_1.x_0 += x_0
        model_1.y_0 += y_0

        # Setup second model for fitting
        xy_slack = 5

        model_2 = models.Sersic2D(
            amplitude=0,
            n=0.1,
            r_eff=model_1.r_eff * 3,
            ellip=ellip,
            theta=theta,
            x_0=model_1.x_0,
            y_0=model_1.y_0,
            fixed={'x_0': True, 'y_0': True, 'theta': True, 'ellip': True},
            bounds={
                'amplitude': (0, None),
                'r_eff': (0, None),
                'n': (0, 2),
                'ellip': (0, 1),
                'theta': (0, 2 * np.pi),
            })

        # Fit second model to the residual of the image (image with segmented area masked)
        model_2, fits = fit_model(target_residual, model_2, maxiter=10000, epsilon=1e-40)

        # PSF Models
        # ----------

        # Setup PSF image and fit it to image of target to estimate param guess

        # Normal sources
        model_3 = models.Gaussian2D(
            amplitude=amp,
            x_mean=model_1.x_0,
            y_mean=model_1.y_0,
            x_stddev=model_1.r_eff,
            y_stddev=model_1.r_eff,
            fixed={'x_mean': True, 'y_mean': True, }
        )

        model_3, fits = fit_model(target, model_3, maxiter=10000, epsilon=1e-40)

        # Final Fit
        # ---------

        # Combine models
        if n_models == 1:
            model = model_1
        elif n_models == 2:
            model = model_1 + model_2
        elif n_models == 3:
            model = model_1 + model_2 + model_3
        else:
            raise Exception("n_models should be 1, 2 or 3")

        # Fit combined model
        model, fit = fit_model(target, model, maxiter=10000, epsilon=1e-40)
        model, fit = fit_model(target, model, maxiter=10000, epsilon=1e-40)

        # Make center of model the center of model_1
        model.x_0 = model_1.x_0
        model.y_0 = model_1.y_0

        # Plot and Print
        # --------------
        if show_fit:
            fig, ax = plot_fit(target, model)  # , vmin=vmin, vmax=vmax)
            plt.show()

        if print_fit:
            print("\n".join([str(j) for j in zip(model.param_names, model.parameters)]) + "\n" * 3)

        # Subtract from image
        # -------------------

        if fit.fit_info['ierr'] > 4:
            # Fit failed
            continue

        # Make new image
        size = r_inner * r_output_mult
        y_arange, x_arange = np.mgrid[
                             int(model.y_0.value) - size:int(model.y_0.value) + size,
                             int(model.x_0.value) - size:int(model.x_0.value) + size, ]
        model_image = model(x_arange, y_arange)

        # Subtract cutout from main copy image
        image_copy = model_subtract(image_copy, np.array(model_image), x, y)
        image_copy = np.clip(image_copy, 0, image_copy.max())
        #image_copy[y - 8:y + 8, x - 8:x + 8] = np.nan

        # Add cutout to main cutout only image
        image_zero = model_subtract(image_zero, -1 * np.array(model_image), x, y)

        # Subtract cutout second component from residual image
        image_residual_copy = model_subtract(image_residual_copy, np.array(model_image), x, y)
        image_residual_copy = np.clip(image_residual_copy, 0, image_residual_copy.max())

        # Photometry
        photometric_sums[obj.label] = model_image.sum()

    return image_copy, image_residual_copy, image_zero, photometric_sums