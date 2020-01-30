from astropy.io import fits
import numpy as np
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from scipy.io import readsav
import datetime
import os
import numba
from sunpy.coordinates.ephemeris import get_earth, get_horizons_coord
from sunpy.coordinates import frames
from astropy.time import Time
import glob
import sunpy.io
import sunpy.map
from skimage.exposure import equalize_adapthist
from scipy import ndimage
import matplotlib.dates as mdates
# reads IDL .sav files and returns date in format for interpoaltion (date_sec) and plotting (dates) as well as elongation


def read_sav(filepath):
    print('read_sav')

    sav_file = readsav(filepath)

    track = sav_file['track']

    date_time = track['track_date']
    elon = track['elon']
    #elon = track['track_y']

    date_time_dec = []

    for i in range(len(date_time[0])):
        date_time_dec.append(date_time[0][i].decode('utf-8'))

    dates = [datetime.datetime.strptime(date_time_dec[i], '%d-%b-%Y %H:%M:%S.%f') for i in range(len(date_time[0]))]
    #dates = [datetime.datetime.strptime(date_time_dec[i], '%Y-%m-%dT%H:%M:%S%f') for i in range(len(date_time[0]))]
    date_time_obj = []

    for i in range(len(date_time[0])):
        date_time_obj.append(datetime.datetime.strptime(date_time_dec[i], '%d-%b-%Y %H:%M:%S.%f') - datetime.datetime(1970, 1, 1))
        #date_time_obj.append(datetime.datetime.strptime(date_time_dec[i], '%Y-%m-%dT%H:%M:%S%f') - datetime.datetime(1970, 1, 1))

    date_sec = [date_time_obj[i].total_seconds() for i in range(len(date_time_obj))]

    return date_sec, elon, dates

#######################################################################################################################################

# direct conversion of hi_delsq.pro for IDL

@numba.njit
def hi_delsq(dat):

    # creates a del^2 field of an image

    least = 4

    nx = dat.shape[0]
    dhj = dat.copy()
    ctr = dat.copy()

    # forward addressing

    d0 = dat[0:nx - 1, :] - dat[1:nx, :]
    bmap = np.isfinite(d0)
    d0_new = np.where(np.isfinite(d0), d0, 0)

    dhj[0:nx - 1, :] = dhj[0:nx - 1, :] + d0_new
    ctr[0:nx - 1, :] = ctr[0:nx - 1, :] + bmap

    # backward addressing

    d0 = dat[1:nx, :] - dat[0:nx - 1, :]

    bmap = np.isfinite(d0)
    d0_new = np.where(np.isfinite(d0), d0, 0)

    dhj[1:nx, :] = dhj[1:nx, :] + d0_new
    ctr[1:nx, :] = ctr[1:nx, :] + bmap

    # downward addressing

    d0 = dat[:, 0:nx - 1] - dat[:, 1:nx]
    bmap = np.isfinite(d0)
    d0_new = np.where(np.isfinite(d0), d0, 0)

    dhj[:, 0:nx - 1] = dhj[:, 0:nx - 1] + d0_new
    ctr[:, 0:nx - 1] = ctr[:, 0:nx - 1] + bmap

    # upward addressing

    d0 = dat[:, 1:nx] - dat[:, 0:nx - 1]

    bmap = np.isfinite(d0)
    d0_new = np.where(np.isfinite(d0), d0, 0)

    dhj[:, 1:nx] = dhj[:, 1:nx] + d0_new
    ctr[:, 1:nx] = ctr[:, 1:nx] + bmap

    dhj = np.where(ctr < least, np.nan, dhj)

    # return field
    return dhj

#######################################################################################################################################

# direct conversion of hi_index_peakpix.pro for IDL

@numba.njit
def hi_index_peakpix(ddat, thresh):

    # finds indices of pixels whose absolute values are above a threshold
    # also finds pixels whose neighbours have absoulte values which are above a threshold

    ny = ddat.shape[1]
    nx = ddat.shape[0]

    ind = []

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):

            if abs(ddat[i, j]) > thresh:

                ind.append(np.array([i, j]))

            else:

                tst1 = abs(ddat[i, j+1]) > thresh

                tst2 = abs(ddat[i, j-1]) > thresh

                tst3 = abs(ddat[i+1, j]) > thresh

                tst4 = abs(ddat[i-1, j]) > thresh

                if tst1 or tst2 or tst3 or tst4:
                    ind.append(np.array([i, j]))
    return ind

#######################################################################################################################################

# direct conversion of hi_sor2.pro for IDL

@numba.njit
def hi_sor2(dat, ddat, indices):

    # solves del^2 = f(x,y) using a sucessive over-relaxation method
    # reconstructs pixels identified by hi_index_peakpix

    # w = overrelaxation parameter
    # nits = number of iterations

    w = 1.5
    nits = 100
    npts = len(indices)
    recon = dat.copy()

    for n in range(1, nits+1):
        for k in range(npts):
            dum = 0.25 * (recon[indices[k][0], indices[k][1] - 1] + recon[indices[k][0] - 1, indices[k][1]] +
                          recon[indices[k][0] + 1, indices[k][1]] + recon[indices[k][0], indices[k][1] + 1] -
                          ddat[indices[k][0], indices[k][1]])

            dum2 = w * dum + (1 - w) * recon[indices[k][0], indices[k][1]]
            recon[indices[k][0], indices[k][1]] = dum2

    return recon

#######################################################################################################################################

# detects and removes saturated pixels
# direct conversion from hi_remove_saturation.pro for IDL


def hi_remove_saturation(data, header):

    # threshold value before pixel is considered saturated
    sat_lim = 1000

    # number of pixels in a column before column is considered saturated
    nsaturated = 6

    n_im = header['imgseq']+1
    imsum = header['summed']

    dsatval = sat_lim*n_im*(2**(imsum-1))**2

    ind = np.where(data > dsatval)

    # if a pixel has a value greater than the dsatval, begin check to test if column is saturated

    if ind[0].size > 0:

        # all pixels are set to zero, except ones exceeding the saturation limit

        mask = data * 0
        ans = data.copy()
        mask[ind] = 1

        # pixels are summed up column-wise
        # where nsaturated is exceeded, values are replaced by nan

        colmask = np.sum(mask, 0)
        ii = np.where(colmask > nsaturated)

        if ii[0].size > 0:
            ans[:, ii] = np.nan

        else:
            ans = data.copy()

    else:
        ans = data.copy()

    # interpolate_replace_nans takes care of replacing saturated columns (now full of nan)

    kernel = Gaussian2DKernel(x_stddev=1, x_size=31., y_size=31.)
    fixed_image = interpolate_replace_nans(ans, kernel)

    # the fixed image with saturated columns replaced by interpolated ones is returend

    return fixed_image

#######################################################################################################################################

# direct conversion of scc_sebip.pro for IDL
# determines what has happened in terms of on-board image processing (division by 2, 4, 8, ...)
# corrects on-board image processing
# returns path to corrected .fits files


def scc_sebip(fit):
    data, header = fits.getdata(fit, header=True)
    ip_raw = header['IP_00_19']

    while len(ip_raw) < 60:
        ip_raw = ' ' + ip_raw

    ip_bytes = bytearray(ip_raw, encoding='ascii')
    ip_arr = np.array(ip_bytes)
    ip_reform = ip_arr.reshape(-1, 3).transpose()

    ip_temp = []
    ip = []

    for i in range(ip_reform.shape[1]):
        ip_temp.append(ip_reform[:, i].tostring())

    for i in range(len(ip_temp)):
        ip.append(ip_temp[i].decode('ascii'))

    cnt = ip.count('117')

    if cnt == 1:
        print('cnt = 1')
        ind = np.where(ip == '117')
        ip = ip[3 * ind:]
        while len(ip) < 60:
            ip = ip.append('  0')

    cnt1 = ip.count('  1')
    cnt2 = ip.count('  2')
    cntspw = ip.count(' 16')

    if cntspw == 0:
        cntspw = ip.count(' 17')

    cnt50 = ip.count(' 50')
    cnt53 = ip.count(' 53')
    cnt82 = ip.count(' 82')
    cnt83 = ip.count(' 83')
    cnt84 = ip.count(' 84')
    cnt85 = ip.count(' 85')
    cnt86 = ip.count(' 86')
    cnt87 = ip.count(' 87')
    cnt88 = ip.count(' 88')
    cnt118 = ip.count('118')

    if header['DIV2CORR']:
        cnt1 = cnt1 - 1

    if cnt1 < 0:
        cnt1 = 0

    if cnt1 > 0:
        data = data * (2.0 ** cnt1)
        print('Correted for divide by 2 x {}'.format(cnt1))
    if cnt2 > 0:
        data = data * (2.0 ** cnt2)
        print('Correted for square root x {}'.format(cnt2))
    if cntspw > 0:
        data = data * (64.0 ** cntspw)
        print('Correted for HI SPW divide by 64 x {}'.format(cntspw))
    if cnt50 > 0:
        data = data * (4.0 ** cnt50)
        print('Correted for divide by 4 x {}'.format(cnt50))
    if cnt53 > 0 and header['ipsum'] > 0:
        data = data * (4.0 ** cnt53)
        print('Correted for divide by 4 x {}'.format(cnt53))
    if cnt82 > 0:
        data = data * (2.0 ** cnt82)
        print('Correted for divide by 2 x {}'.format(cnt82))
    if cnt83 > 0:
        data = data * (4.0 ** cnt83)
        print('Correted for divide by 4 x {}'.format(cnt83))
    if cnt84 > 0:
        data = data * (8.0 ** cnt84)
        print('Correted for divide by 8 x {}'.format(cnt84))
    if cnt85 > 0:
        data = data * (16.0 ** cnt85)
        print('Correted for divide by 16 x {}'.format(cnt85))
    if cnt86 > 0:
        data = data * (32.0 ** cnt86)
        print('Correted for divide by 32 x {}'.format(cnt86))
    if cnt87 > 0:
        data = data * (64.0 ** cnt87)
        print('Correted for divide by 64 x {}'.format(cnt87))
    if cnt88 > 0:
        data = data * (128.0 ** cnt88)
        print('Correted for divide by 128 x {}'.format(cnt88))
    if cnt118 > 0:
        data = data * (3.0 ** cnt118)
        print('Correted for divide by 3 x {}'.format(cnt118))

    print('------------------------------------------------------')

    return data

#######################################################################################################################################


def rej_out(dat_arr, m):
    return dat_arr[abs(dat_arr - np.mean(dat_arr)) < m * np.std(dat_arr)]

#######################################################################################################################################


def bin_elong(noel, res2, tdi, tdata):
    for j in range(noel):
        if len(res2[j][0]) > 1:
            tmp_new = rej_out(tdi[res2[j][0]], 2)
            tmp1 = np.median(tmp_new)
            tdata[j] = tmp1

        elif len(res2[j][0]) == 1:
            tdata[j] = tdi[res2[j][0]]

        else:
            tdata[j] = np.nan
    return tdata

#######################################################################################################################################


def get_earth_pos(time, ftpsc):
    earth = [get_earth(t) for t in time]
    sta = [get_horizons_coord('STEREO-A', t) for t in time]

    pos_earth = [earth[i].transform_to(frames.Helioprojective(observer=sta[i])) for i in range(len(earth))]

    earth_lon = [pos_earth[i].Tx.radian for i in range(len(pos_earth))]
    earth_lat = [pos_earth[i].Ty.radian for i in range(len(pos_earth))]

    elongearth = np.arccos(np.cos(earth_lat) * np.cos(earth_lon)) * 180 / np.pi
    paearth = np.sin(earth_lat) / np.sin(elongearth)

    if ftpsc == 'A':
        paearth = np.arccos(paearth) * 180 / np.pi

    if ftpsc == 'B':
        paearth = (2 * np.pi - np.arccos(paearth)) * 180 / np.pi

    return paearth

#######################################################################################################################################


def hi_img(start, ftpsc, instrument, startel):
    cadence = 120.0
    maxgap = -3.5

    file = open('config.txt', 'r')
    path = file.readlines()
    path = path[0].splitlines()[0]

    red_path = path + start + '_' + ftpsc + '_red/'
    files = []

    for file in sorted(glob.glob(red_path + '*' + instrument + '*' + '*.fts')):
        files.append(file)

    data = np.array([fits.getdata(files[i]) for i in range(len(files))])
    header = [fits.getheader(files[i]) for i in range(len(files))]

    missing = np.array([header[i]['NMISSING'] for i in range(len(header))])
    mis = np.array(np.where(missing > 0))

    if np.size(mis) > 0:
        for i in mis:
            data[i, :, :] = np.nan

    time = [header[i]['DATE-OBS'] for i in range(len(header))]
    time_obj = [Time(time[i], format='isot', scale='utc') for i in range(len(time))]

    nandata = np.full_like(data[0], np.nan)

    r_dif = [nandata]

    for i in range(1, len(time)):
        tdiff = (time_obj[i] - time_obj[i - 1]).sec / 60
        if (tdiff <= -maxgap * cadence) & (tdiff > 0.5 * cadence):
            r_dif.append(data[i] - data[i - 1])
        else:
            r_dif.append(nandata)

    dif_map = [sunpy.map.Map(r_dif[k], header[k]) for k in range(len(files))]

    wc = []

    for i in range(len(dif_map)):
        wc.append(dif_map[i].wcs)

    w = np.shape(dif_map[0].data)[0]
    h = np.shape(dif_map[0].data)[1]

    p = [[y, x] for x in range(w) for y in range(h)]

    coords = np.zeros((len(dif_map), len(p), 2))

    # pixel coordinates are converted to world coordinates

    for i in range(len(dif_map)):
        coords[i, :, :] = wc[i].all_pix2world(p, 0)  # ??????? NOT SURE 0/1 (0 at first)

    # convert to radians

    coords = coords * np.pi / 180

    hpclat = coords[:, :, 1]
    hpclon = coords[:, :, 0]

    elongs = np.arccos(np.cos(hpclat) * np.cos(hpclon))

    pas = np.sin(hpclat) / np.sin(elongs)
    elongs = elongs * 180 / np.pi

    if ftpsc == 'A':
        pas = np.arccos(pas) * 180 / np.pi

    if ftpsc == 'B':
        pas = (2 * np.pi - np.arccos(pas)) * 180 / np.pi

    paearth = get_earth_pos(time, ftpsc)

    if instrument == 'hi_1':
        # maxel = np.max(elongs[-1,:,255])
        noelongs = 360
        elongint = startel / noelongs
        elongst = np.arange(0, startel, elongint)
        elongen = elongst + elongint
        y = np.arange(0, startel + elongint, elongint)
        endel = y[-1]

    if instrument == 'hi_2':
        # maxel = np.max(elongs[-1,:,255])
        noelongs = 180
        elongint = (90 - startel) / noelongs
        elongst = np.arange(startel, 90, elongint)
        elongen = elongst + elongint
        y = np.arange(startel, 90 + elongint, elongint)
        endel = y[-1]

    tmpdata = np.arange(noelongs) * np.nan
    patol = 3.

    result = [np.where((pas[i] >= paearth[i] - patol) & (pas[i] <= paearth[i] + patol)) for i in range(len(pas))]

    tmpelongs = [elongs[i][result[i][:]] for i in range(len(elongs))]
    re_dif = np.reshape(r_dif, (len(r_dif), 256 * 256))
    tmpdiff = [re_dif[i][result[i][:]] for i in range(len(re_dif))]

    result2 = [[np.where((tmpelongs[i] >= elongst[j] - elongint) & (tmpelongs[i] < elongen[j] + elongint)
                         & np.isfinite(tmpdiff[i])) for j in range(len(elongst))] for i in range(len(tmpelongs))]

    result2 = np.array(result2)

    t_data = np.zeros((len(tmpdiff), noelongs))

    for i in range(len(tmpdiff)):
        t_data[i][:] = bin_elong(len(elongst), result2[i], tmpdiff[i], tmpdata)

    zrange = [-12000, 12000]

    img = np.empty((len(t_data), noelongs,))
    img[:] = np.nan

    for i in range(len(t_data)):
        img[i][:] = (t_data[i] - zrange[0]) / (zrange[1] - zrange[0])

    nn = equalize_adapthist(img, kernel_size=len(t_data) / 2)
    # nn = np.nan_to_num(img)

    nimg = np.where(nn >= 0, nn, 0)
    nimg = np.where(nimg <= 1, nimg, 0)
    nimg = np.where(nimg == 0, np.nan, nimg)

    den = ndimage.uniform_filter(nimg)

    nanp = np.where(np.isnan(nimg))
    den[nanp] = np.nan

    imsh_im = nimg.T
    den_im = den.T

    time_t = [datetime.datetime.strptime(day, '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y/%m/%d %H:%M:%S.%f') for day in time]

    x_lims = mdates.datestr2num(time_t)

    dt = (np.max(x_lims) - np.min(x_lims)) / (imsh_im.shape[1] - 1)
    delon = (np.max(y) - np.min(y)) / (noelongs - 1)

    return x_lims, dt, delon, y, imsh_im, den_im, endel

