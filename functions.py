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
import math
from astropy import wcs


# reads IDL .sav files and returns date in format for interpoaltion (date_sec) and plotting (dates) as well as elongation


def read_sav(filepath):
    """Reads IDL .sav files and returns date in format for interpoaltion (date_sec) and plotting (dates) as well as elongation."""

    sav_file = readsav(filepath)

    track = sav_file['track']

    date_time = track['track_date']
    elon = track['elon']
    # elon = track['track_y']

    date_time_dec = []

    for i in range(len(date_time[0])):
        date_time_dec.append(date_time[0][i].decode('utf-8'))

    dates = [datetime.datetime.strptime(date_time_dec[i], '%d-%b-%Y %H:%M:%S.%f') for i in range(len(date_time[0]))]
    # dates = [datetime.datetime.strptime(date_time_dec[i], '%Y-%m-%dT%H:%M:%S%f') for i in range(len(date_time[0]))]
    date_time_obj = []

    for i in range(len(date_time[0])):
        date_time_obj.append(datetime.datetime.strptime(date_time_dec[i], '%d-%b-%Y %H:%M:%S.%f') - datetime.datetime(1970, 1, 1))
        # date_time_obj.append(datetime.datetime.strptime(date_time_dec[i], '%Y-%m-%dT%H:%M:%S%f') - datetime.datetime(1970, 1, 1))

    date_sec = [date_time_obj[i].total_seconds() for i in range(len(date_time_obj))]

    return date_sec, elon, dates


#######################################################################################################################################


@numba.njit
def hi_delsq(dat):
    """Direct conversion of hi_delsq.pro for IDL.
    Returns the delta^2 field of an image which is later used for computing the starfield."""

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


@numba.njit
def hi_index_peakpix(ddat, thresh):
    """Direct conversion of hi_index_peakpix.pro for IDL.
    Returns indices of pixels and their neighbours whose absolute values are above a given threshold."""

    ny = ddat.shape[1]
    nx = ddat.shape[0]

    ind = []

    for j in range(1, ny - 1):
        for i in range(1, nx - 1):

            if abs(ddat[i, j]) > thresh:

                ind.append(np.array([i, j]))

            else:

                tst1 = abs(ddat[i, j + 1]) > thresh

                tst2 = abs(ddat[i, j - 1]) > thresh

                tst3 = abs(ddat[i + 1, j]) > thresh

                tst4 = abs(ddat[i - 1, j]) > thresh

                if tst1 or tst2 or tst3 or tst4:
                    ind.append(np.array([i, j]))
    return ind


#######################################################################################################################################

# direct conversion of hi_sor2.pro for IDL

@numba.njit
def hi_sor2(dat, ddat, indices):
    """Direct conversion of hi_sor2.pro for IDL. Solves delta^2 = f(x,y) using a sucessive over-relaxation method.
    Reconstructs pixels identified by hi_index_peakpix.
    Takes image data (dat), delta^2 field (ddat) and indices returned by hi_index_peakpix as input."""

    # w = overrelaxation parameter
    # nits = number of iterations

    w = 1.5
    nits = 100
    npts = len(indices)
    recon = dat.copy()

    for n in range(1, nits + 1):
        for k in range(npts):
            dum = 0.25 * (recon[indices[k][0], indices[k][1] - 1] + recon[indices[k][0] - 1, indices[k][1]] +
                          recon[indices[k][0] + 1, indices[k][1]] + recon[indices[k][0], indices[k][1] + 1] -
                          ddat[indices[k][0], indices[k][1]])

            dum2 = w * dum + (1 - w) * recon[indices[k][0], indices[k][1]]
            recon[indices[k][0], indices[k][1]] = dum2

    return recon


#######################################################################################################################################


def hi_remove_saturation(data, header):
    """Direct conversion of hi_remove_saturation.pro for IDL.
    Detects and masks saturated pixels with nan. Takes image data and header as input. Returns fixed image."""

    # threshold value before pixel is considered saturated
    sat_lim = 1000

    # number of pixels in a column before column is considered saturated
    nsaturated = 6

    n_im = header['imgseq'] + 1
    imsum = header['summed']

    dsatval = sat_lim * n_im * (2 ** (imsum - 1)) ** 2

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

    # kernel = Gaussian2DKernel(x_stddev=1, x_size=31., y_size=31.)
    # fixed_image = np.nan_to_num(ans)

    # the fixed image with saturated columns replaced by interpolated ones is returend

    return ans


#######################################################################################################################################


def scc_sebip(data, header):
    """Direct conversion of scc_sebip.pro for IDL.
    Determines what has happened in terms of on-board sebip binning and corrects it.
    Takes image data and header as input and returns fixed image."""

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
    """Removes values from an array that deviate from the mean by more than m standard deviations."""
    return dat_arr[abs(dat_arr - np.mean(dat_arr)) < m * np.std(dat_arr)]


#######################################################################################################################################


def bin_elong(noel, res2, tdi, tdata):
    """Takes number of steps in elongation, """

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
    # This is what doesnt work if you ever want to try STB
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


def hi_img(start, ftpsc, instrument, startel, bflag):

    if bflag == 'beacon':
        cadence = 120.0
        maxgap = -3.5

    if bflag == 'science':
        cadence = 40.0
        maxgap = -3.5

    file = open('config.txt', 'r')
    path = file.readlines()
    path = path[0].splitlines()[0]

    red_path = path + start + '_' + ftpsc + '_' + bflag + '_red/'
    files = []

    if bflag == 'science':
        for file in sorted(glob.glob(red_path + '*s4' + instrument + '*.fts')):
            files.append(file)

    if bflag == 'beacon':
        for file in sorted(glob.glob(red_path + '*s7' + instrument + '*.fts')):
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

    r_dif = []
    nan_ind = []
    nan_dat = []

    for i in range(1, len(time)):
        tdiff = (time_obj[i] - time_obj[i - 1]).sec / 60

        if (tdiff <= -maxgap * cadence) & (tdiff > 0.5 * cadence):
            r_dif.append(data[i] - data[i - 1])
            flg = 1

        else:
            r_dif.append(nandata)
            flg = 0

        ap_ind = np.round(tdiff / cadence, 0)
        ap_ind = int(ap_ind)

        if ap_ind > 1:
            nan_ind.append(i)

            if flg:
                nan_dat.append(ap_ind)

            if not flg:
                nan_dat.append(ap_ind-1)

    # r_dif = np.nan_to_num(r_dif)

    dif_map = [sunpy.map.Map(r_dif[k], header[k + 1]) for k in range(0, len(files) - 1)]

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

    if instrument == 'h1':
        noelongs = 360
        elongint = startel / noelongs
        elongst = np.arange(0, startel, elongint)
        elongen = elongst + elongint
        y = np.arange(0, startel + elongint, elongint)
        endel = y[-1]

    if instrument == 'h2':
        mask = np.array([get_smask(ftpsc, header[i], path, time[i]) for i in range(1, len(header))])
        r_dif = np.array(r_dif)
        r_dif[mask == 0] = np.nan
        noelongs = 180
        elongint = (90 - startel) / noelongs
        elongst = np.arange(startel, 90, elongint)
        elongen = elongst + elongint
        y = np.arange(startel, 90 + elongint, elongint)
        endel = y[-1]

    tmpdata = np.arange(noelongs) * np.nan
    patol = 5.

    result = [np.where((pas[i] >= paearth[i] - patol) & (pas[i] <= paearth[i] + patol)) for i in range(len(pas))]
    tmpelongs = [elongs[i][result[i][:]] for i in range(len(elongs))]
    re_dif = np.reshape(r_dif, (len(r_dif), header[0]['NAXIS1'] * header[0]['NAXIS2']))
    tmpdiff = [re_dif[i][result[i][:]] for i in range(len(re_dif))]
    result2 = [[np.where((tmpelongs[i] >= elongst[j] - elongint) & (tmpelongs[i] < elongen[j] + elongint)
                         & np.isfinite(tmpdiff[i])) for j in range(len(elongst))] for i in range(len(tmpelongs))]

    result2 = np.array(result2)
    t_data = np.zeros((len(tmpdiff), noelongs))

    for i in range(len(tmpdiff)):
        t_data[i][:] = bin_elong(len(elongst), result2[i], tmpdiff[i], tmpdata)

    n = np.zeros(noelongs)
    n[:] = np.nan

    up_tdata = t_data.tolist()

    k = 0

    nan_ind.reverse()
    nan_dat.reverse()
    for i in nan_ind:
        for j in range(nan_dat[k]):
            up_tdata.insert(i - 1, n)
        k = k + 1

    if bflag == 'beacon':
        zrange = [-200, 200]

    if bflag == 'science':
        zrange = [-200, 200]

    img = np.empty((len(up_tdata), noelongs,))
    img[:] = np.nan
    up_tdata = np.array(up_tdata)

    for i in range(len(up_tdata)):
        img[i][:] = (up_tdata[i] - zrange[0]) / (zrange[1] - zrange[0])

    nn = equalize_adapthist(img, kernel_size=len(up_tdata) / 2)
    nn = np.nan_to_num(nn)

    nimg = np.where(nn >= 0, nn, 0)
    nimg = np.where(nimg <= 1, nimg, 0)
    nimg = np.where(nimg == 0, np.nan, nimg)

    den = ndimage.uniform_filter(nimg)

    nanp = np.where(np.isnan(nimg))
    den[nanp] = np.nan

    imsh_im = nimg.T
    den_im = den.T

    imsh_im = np.nan_to_num(imsh_im)
    den_im = np.nan_to_num(den_im)
    time_t = [datetime.datetime.strptime(day, '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y/%m/%d %H:%M:%S.%f') for day in time]

    x_lims = mdates.datestr2num(time_t)

    dt = (np.max(x_lims) - np.min(x_lims)) / (imsh_im.shape[1] - 1)
    delon = (np.max(y) - np.min(y)) / (noelongs - 1)

    return x_lims, dt, delon, y, imsh_im, den_im, endel, path


#######################################################################################################################################


def get_smask(ftpsc, header, path, time):
    if ftpsc == 'A':
        filename = 'hi2A_mask.fts'
        xy = [1, 51]

    if ftpsc == 'B':
        filename = 'hi2B_mask.fts'
        xy = [129, 79]

    calpath = path + 'calibration/' + filename
    time = datetime.datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f')

    smask, hdr = fits.getdata(calpath, header=True)

    fullm = np.zeros((2176, 2176))

    x1 = 2048 - np.shape(fullm[xy[1] - 1:, xy[0] - 1:])[0]
    y1 = 2048 - np.shape(fullm[xy[1] - 1:, xy[0] - 1:])[1]

    fullm[xy[0] - 1:y1, xy[1] - 1:x1] = smask

    tc = datetime.datetime(2015, 5, 19)

    flag = time > tc

    if flag:
        fullm = np.rot90(fullm)
        fullm = np.rot90(fullm)

    mask = rebin(fullm[header['r1row'] - 1:header['r2row'], header['r1col'] - 1:header['r2col']],
                 [header['NAXIS2'], header['NAXIS1']])
    mask = np.where(mask < 1, 0, 1)

    return mask


#######################################################################################################################################


def rebin(a, shape):
    sh = int(shape[0]), int(a.shape[0]) // int(shape[0]), int(shape[1]), int(a.shape[1]) // int(shape[1])
    return a.reshape(sh).mean(-1).mean(1)


#######################################################################################################################################

def hi_desmear(data, header):
    time = header['DATE-OBS']

    time = datetime.datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f')
    tc = datetime.datetime(2015, 5, 19)

    if time > tc:
        post_conj = 1

    else:
        post_conj = 0

    if header['dstart1'] < 1 or (header['naxis1'] == header['naxis2']):
        image = data

    else:
        image = data[header['dstart2'] - 1:header['dstop1'], header['dstart1'] - 1:header['dstop1']]

    clearest = 0.70
    exp_eff = header['EXPTIME'] + header['n_images'] * (clearest - header['CLEARTIM'] + header['RO_DELAY'])

    dataWeight = header['n_images'] * ((2 ** (header['ipsum'] - 1)))

    inverted = 0

    if header['rectify'] == 'T':
        if header['OBSRVTRY'] == 'STEREO_B':
            inverted = 1
        if header['OBSRVTRY'] == 'STEREO_A' and post_conj == 1:
            inverted = 1

    if inverted == 1:

        n = header['NAXIS2']

        ab = np.zeros((n, n))
        ab[:] = dataWeight * header['line_ro']
        bel = np.zeros((n, n))
        bel[:] = dataWeight * header['line_clr']

        fixup = np.triu(ab) + np.tril(bel)

        for i in range(0, n):
            fixup[i, i] = exp_eff

    else:

        n = header['NAXIS2']

        ab = np.zeros((n, n))
        ab[:] = dataWeight * header['line_clr']
        bel = np.zeros((n, n))
        bel[:] = dataWeight * header['line_ro']

        fixup = np.triu(ab) + np.tril(bel)

        for i in range(0, n):
            fixup[i, i] = exp_eff

    fixup = np.linalg.inv(fixup)
    image = fixup @ image

    if header['dstart1'] < 1 or (header['naxis1'] == header['naxis2']):
        img = image

    else:
        img = image[header['dstart2'] - 1:header['dstop1'], header['dstart1'] - 1:header['dstop1']]

    return img



#######################################################################################################################################


def get_calimg(instr, ftpsc, path, header):
    if instr == 'hi_1':

        if header['summed'] == 1:
            cal_version = '20061129_flatfld_raw_h1' + ftpsc.lower() + '.fts'

        else:
            cal_version = '20100421_flatfld_sum_h1' + ftpsc.lower() + '.fts'
            sumflg = 1

    if instr == 'hi_2':

        if header['summed'] == 1:
            cal_version = '20150701_flatfld_raw_h2' + ftpsc.lower() + '.fts'

        else:
            cal_version = '20150701_flatfld_sum_h2' + ftpsc.lower() + '.fts'
            sumflg = 1

    calpath = path + 'calibration/' + cal_version
    cal_image, cal_hdr = fits.getdata(calpath, header=True)
    try:
        cal_p1col = cal_hdr['P1COL']

    except KeyError:
        cal_p1col = 0

    if cal_p1col < 1:
        if sumflg:

            x1 = 25
            x2 = 1048
            y1 = 0
            y2 = 1023

        else:

            x1 = 50
            x2 = 2047 + 50
            y1 = 0
            y2 = 2047

        cal = cal_image[y1:y2 + 1, x1:x2 + 1]

    else:
        cal = cal_image

    time = cal_hdr['DATE']

    time = datetime.datetime.strptime(time, '%Y-%m-%d')

    if (header['RECTIFY'] == 'T') and (cal_hdr['RECTIFY'] == 'F'):
        cal = secchi_rectify(cal, cal_hdr, ftpsc, instr, flg)

    if sumflg:
        if header['summed'] < 2:
            hdr_sum = 1

        else:
            hdr_sum = 2 ** (header['summed'] - 2)

    else:
        hdr_sum = 2 ** (header['summed'] - 1)

    s = np.shape(cal)
    cal = rebin(cal, [s[1] / hdr_sum, s[0] / hdr_sum])

    tc = datetime.datetime(2015, 5, 19)

    flag_hdr = time > tc

    if flag_hdr:
        cal = np.rot90(cal)
        cal = np.rot90(cal)

    return cal


#######################################################################################################################################


def secchi_rectify(cal, scch, ftpsc, instr, flg):
    stch = scch

    stch['RECTIFY'] = 'T'

    if ftpsc == 'B':
        b = np.rot90(cal)
        b = np.rot90(b)
        stch['r1row'] = 2176 - scch['p2row'] + 1
        stch['r2row'] = 2176 - scch['p1row'] + 1
        stch['r1col'] = 2176 - scch['p2col'] + 1
        stch['r2col'] = 2176 - scch['p1col'] + 1

        stch['crpix1'] = scch['naxis1'] - scch['crpix1'] + 1
        stch['crpix2'] = scch['naxis2'] - scch['crpix2'] + 1
        stch['naxis1'] = scch['naxis1']
        stch['naxis2'] = scch['naxis2']
        stch['rectrota'] = 2
        rotcmt = 'rotate 180 deg CCW'

        stch['dstart1'] = (79 - stch['r1col'] + 1) > 1
        stch['dstop1'] = stch['dstart1'] - 1 + ((stch['r2col'] - stch['r1col'] + 1) < 2048)

        stch['dstart2'] = (129 - stch['r1row'] + 1) > 1
        stch['dstop2'] = stch['dstart2'] - 1 + ((stch['r2row'] - stch['r1row'] + 1) < 2048)

    if (ftpsc == 'A') and flag == 1:
        b = np.rot90(cal)
        b = np.rot90(b)
        stch['r1row'] = 2176 - scch['p2row'] + 1
        stch['r2row'] = 2176 - scch['p1row'] + 1
        stch['r1col'] = 2176 - scch['p2col'] + 1
        stch['r2col'] = 2176 - scch['p1col'] + 1

        stch['crpix1'] = scch['naxis1'] - scch['crpix1'] + 1
        stch['crpix2'] = scch['naxis2'] - scch['crpix2'] + 1
        stch['naxis1'] = scch['naxis1']
        stch['naxis2'] = scch['naxis2']
        stch['rectrota'] = 2
        rotcmt = 'rotate 180 deg CCW'

        stch['dstart1'] = (79 - stch['r1col'] + 1) > 1
        stch['dstop1'] = stch['dstart1'] - 1 + ((stch['r2col'] - stch['r1col'] + 1) < 2048)

        stch['dstart2'] = (129 - stch['r1row'] + 1) > 1
        stch['dstop2'] = stch['dstart2'] - 1 + ((stch['r2row'] - stch['r1row'] + 1) < 2048)

    if (ftpsc == 'A') and flag == 0:
        b = cal
        stch['r1row'] = scch['p1row']
        stch['r2row'] = scch['p2row']
        stch['r1col'] = scch['p1col']
        stch['r2col'] = scch['p2col']

        stch['rectrota'] = 0

    if stch['r1col'] < 1:
        stch['r2col'] = stch['r2col'] + np.abs(stch['r1col']) + 1
        stch['r1col'] = 1

    if stch['r1row'] < 1:
        stch['r2row'] = stch['r2row'] + np.abs(stch['r1row']) + 1
        stch['r1row'] = 1

    xden = 2 ** (scch['ipsum'] + scch['sumcol'] - 2)
    yden = 2 ** (scch['ipsum'] + scch['sumrow'] - 2)

    stch['dstart1'] = math.modf(math.ceil(stch['dstart1'] / xden))[1] > 1
    stch['dstart2'] = math.modf(math.ceil(stch['dstart2'] / yden))[1] > 1
    stch['dstop1'] = math.modf(stch['dstop1'] / xden)[1]
    stch['dstop2'] = math.modf(stch['dstop2'] / yden)[1]

    if (stch['naxis1'] > 0 and stch['naxis2'] > 0):
        wcoord = wcs.WCS(stch)

        xycen = wcoord.wcs_pix2world([(stch['naxis1'] - 1.) / 2., (stch['naxis2'] - 1.) / 2.], 0)
        stch['xcen'] = xycen[0]
        stch['ycen'] = xycen[1]

    scch = stch

    return scch


#######################################################################################################################################

def get_biasmean(header):
    bias = header['BIASMEAN']
    ipsum = header['IPSUM']

    if ('103' in header['IP_00_19']) or (' 37' in header['IP_00_19']) or (' 38' in header['IP_00_19']):
        bias = 0

    elif header['OFFSETCR'] > 0:
        bias = 0

    else:

        bias = bias - (bias / header['n_images'])

        if ipsum > 1:
            bias = bias * ((2 ** (ipsum - 1)) ** 2)

    return bias


#######################################################################################################################################

def hi_fill_missing(data, header):
    if header['NMISSING'] == 0:
        data = data

    if header['NMISSING'] > 0:

        if len(header['MISSLIST']) < 1:
            print('Mismatch between nmissing and misslist.')
            data = data

        else:
            fields = scc_get_missing(header)
            data[fields] = np.nan

    return data


#######################################################################################################################################

def scc_get_missing(header):
    base = 34

    lenstr = len(header['MISSLIST'])

    if not lenstr % 2:
        misslisgt = ' ' + header['MISSLIST']
        lenstr = lenstr + 1

    else:
        misslist = header['MISSLIST']

    dex = np.arange(max(int(lenstr / 2), 1)) * 2


#######################################################################################################################################

def scc_img_trim(fit):
    im, header = fits.getdata(fit, header=True)

    x1 = header['DSTART1'] - 1
    x2 = header['DSTOP1'] - 1
    y1 = header['DSTART2'] - 1
    y2 = header['DSTOP2'] - 1

    img = im[y1:y2 + 1, x1:x2 + 1]

    s = np.shape(img)

    if (header['NAXIS1'] != s[0]) or (header['NAXIS2'] != s[1]):
        print('Removing under- and overscan...')

        hdrsum = 2 ** (header['SUMMED'] - 1)

        header['R1COL'] = header['R1COL'] + (x1 * hdrsum)
        header['R2COL'] = header['R1COL'] + (s[0] * hdrsum) - 1
        header['R1ROW'] = header['R1ROW'] + (y1 * hdrsum)
        header['R2ROW'] = header['R1ROW'] + (s[1] * hdrsum) - 1

        header['DSTART1'] = 1
        header['DSTOP1'] = s[0]
        header['DSTART2'] = 1
        header['DSTOP2'] = s[1]

        header['NAXIS1'] = s[0]
        header['NAXIS2'] = s[1]

        header['CRPIX1'] = header['CRPIX1'] - x1
        header['CRPIX1A'] = header['CRPIX1A'] - x1

        header['CRPIX2'] = header['CRPIX2'] - y1
        header['CRPIX2A'] = header['CRPIX2A'] - y1

        wcoord = wcs.WCS(header)
        xycen = wcoord.wcs_pix2world((header['naxis1'] - 1.) / 2., (header['naxis2'] - 1.) / 2., 0)
        header['xcen'] = np.round(xycen[0], 0)
        header['ycen'] = np.round(xycen[1], 0)

    return img

