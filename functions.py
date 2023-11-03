import numpy as np
import numba
from astropy.time import Time
from astropy.io import fits
import glob
import matplotlib.dates as mdates
from matplotlib import cm
from matplotlib.colors import ListedColormap
import math
from astropy import wcs
import requests
from bs4 import BeautifulSoup
import os
import pandas as pd
import datetime
from itertools import repeat
import matplotlib.pyplot as plt
import pickle
import cv2
import sys
from scipy.ndimage import shift
import warnings
from matplotlib import image
import stat
from scipy.interpolate import NearestNDInterpolator
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
from requests.adapters import HTTPAdapter, Retry
import psutil
from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from functools import partial
import traceback
import logging
from skimage import exposure
import subprocess
import scipy as sp
from matplotlib.ticker import MultipleLocator
from collections import Counter
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.coordinates import Helioprojective
from sunpy.coordinates.ephemeris import get_horizons_coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from skimage.transform import resize
warnings.filterwarnings("ignore")

#######################################################################################################################################

def limit_cpu():
    """
    Is called when starting a new multiprocessing pool. Decreases priority of processes to limit total CPU usage. 
    """
    p = psutil.Process(os.getpid())
    # set to lowest priority
    p.nice(19)

#######################################################################################################################################


def listfd(input_url, extension):
    """
    Provides list of urls and corresponding file names to download.

    @param input_url: URL of STEREO-HI image files
    @param extension: File ending of STEREO-HI image files
    @return: List of URLs and corresponding filenames to be downloaded
    """

    disable_warnings(InsecureRequestWarning)

    
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)

    session.mount('http://', adapter)
    session.mount('https://', adapter)

    output_urls = []

    page = session.get(input_url).text
    #page = requests.get(input_url, verify=False).text

    soup = BeautifulSoup(page, 'html.parser')
    url_found = [input_url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(extension)]
    filename = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith(extension)]

    for i in range(len(filename)):
        output_urls.append((filename[i], url_found[i]))

    return output_urls


#######################################################################################################################################

def fetch_url(path, entry):
    """
    Downloads URLs specified by listfd.

    @param path: Path where downloaded files are to be saved
    @param entry: Combination of filename and URL of downloaded file
    """
    filename, uri = entry

    if not os.path.exists(path + '/' + filename):
        r = requests.get(uri, allow_redirects=True)
        open(path + '/' + filename, 'wb').write(r.content)
#######################################################################################################################################

def check_calfiles(path):
    """
    Checks if SSW IDL HI calibration files are present - creates appropriate directory and downloads them if not.

    @param path: Path in which calibration files are located/should be located
    """
    url_cal = "https://soho.nascom.nasa.gov/solarsoft/stereo/secchi/calibration/"

    if not os.path.exists(path + 'calibration/'):
        print('Checking calibration files...')

        try:
            os.makedirs(path + 'calibration/')
            uri = listfd(url_cal, '.fts')
       
            for entry in uri:
                fetch_url(path + 'calibration', entry)
            
            return
            
        except KeyboardInterrupt:
            return
       
        except Exception as e:
            logging.error(traceback.format_exc())
            sys.exit()
    else:
        return

#######################################################################################################################################

def check_pointfiles(path):
    """
    Checks if SSW IDL HI calibration files are present - creates appropriate directory and downloads them if not.

    @param path: Path in which calibration files are located/should be located
    """
    url_point = "https://soho.nascom.nasa.gov/solarsoft/stereo/secchi/data/hi/"

    
    print('Checking pointing files...')

    if not os.path.exists(path + 'data/hi/'):
        os.makedirs(path + 'data/hi/')

    try:
        uri = listfd(url_point, '.fts')
          
        for entry in uri:
            if not os.path.isfile(path + 'data/hi/' + entry[0]):
                fetch_url(path + 'data/hi', entry)
            else:
                pass
        return
            
    except KeyboardInterrupt:
        return
       
    except Exception as e:
        logging.error(traceback.format_exc())
        sys.exit()


#######################################################################################################################################   
def download_files(start, duration, save_path, ftpsc, instrument, bflag, silent):
    """
    Downloads STEREO images from NASA pub directory

    @param start: Beginning date (DDMMYYYY) of files to be downloaded
    @param duration: Timespan for files to be downloaded (in days)
    @param save_path: Path for saving downloaded files
    @param ftpsc: Spacecraft (STEREO-A/STEREO-B) for which to download files
    @param instrument: Instrument (HI-1/HI-2) for which to download files
    @param bflag: Data type (science/beacon) for which to download files
    @param silent: Run in silent mode
    """
    fitsfil = []

    bg_dur = 7
    date = datetime.datetime.strptime(start, '%Y%m%d') - datetime.timedelta(days=bg_dur+1)
    
    if ftpsc == 'A':
        sc = 'ahead'

    if ftpsc == 'B':
        sc = 'behind'

    if instrument == 'hi1hi2':
        instrument = ['hi_1', 'hi_2']

    elif instrument == 'hi_1':
        instrument = ['hi_1']

    elif instrument == 'hi_2':
        instrument = ['hi_2']

    else:

        print('Invalid instrument specification. Exiting...')
        sys.exit()

    datelist = pd.date_range(date, periods=duration+bg_dur+1).tolist()
    datelist_int = [str(datelist[i].year) + datelist[i].strftime('%m') + datelist[i].strftime('%d') for i in range(len(datelist))]

    if not silent:
        print('Fetching files...')

    for ins in instrument:
        for date in datelist_int:

            if bflag == 'beacon':

                url = 'https://stereo-ssc.nascom.nasa.gov/pub/beacon/' + sc + '/secchi/img/' + ins + '/' + str(date)

            else:

                url = 'https://stereo-ssc.nascom.nasa.gov/pub/ins_data/secchi/L0/' + sc[0] + '/img/' + ins + '/' + str(date)

            if bflag == 'beacon':
                path_flg = 'beacon'
                path_dir = save_path + 'stereo' + sc[0] + '/' + path_flg + '/secchi/img/' + ins + '/' + str(date)

                if ins == 'hi_1':
                    if sc == 'ahead':
                        ext = 's7h1A.fts'
                    if sc == 'behind':
                        ext = 's7h1B.fts'

                if ins == 'hi_2':
                    if sc == 'ahead':
                        ext = 's7h2A.fts'
                    if sc == 'behind':
                        ext = 's7h2B.fts'

            if bflag == 'science':

                path_flg = 'L0'
                path_dir = save_path + 'stereo' + sc[0] + '/secchi/' + path_flg + '/img/' + ins + '/' + str(date)

                if ins == 'hi_1':
                    if sc == 'ahead':
                        ext = 's4h1A.fts'
                    if sc == 'behind':
                        ext = 's4h1B.fts'

                if ins == 'hi_2':
                    if sc == 'ahead':
                        ext = 's4h2A.fts'
                    if sc == 'behind':
                        ext = 's4h2B.fts'
            
            if not os.path.exists(path_dir):
              os.makedirs(path_dir)
              flag = True
              
            else:
              if not os.listdir(path_dir):
                flag = True
              else:
                flag = False
              
            num_cpus = cpu_count()

            pool = Pool(int(num_cpus/2), limit_cpu)

            if flag:
                urls = listfd(url, ext)
                inputs = zip(repeat(path_dir), urls)

                try:
                    results = pool.starmap(fetch_url, inputs, chunksize=5)

                except ValueError:
                    continue
                    
            pool.close()
            pool.join()
      
#######################################################################################################################################

def hi_remove_saturation(data, header):
    """Direct conversion of hi_remove_saturation.pro for IDL.
    Detects and masks saturated pixels with nan. Takes image data and header as input. Returns fixed image.
    @param data: Data of .fits file
    @param header: Header of .fits file
    @return: Data with oversaturated columns removed"""

    # threshold value before pixel is considered saturated
    sat_lim = 14000

    # number of pixels in a column before column is considered saturated
    nsaturated = 5
    
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
        ii = np.array(np.where(colmask > nsaturated))
        
        if len(ii) > 0:
            ans[:, ii] = np.nan #np.nanmedian(data)
        else:
            ans = data.copy()

    else:
        ans = data.copy()

    return ans


#######################################################################################################################################

def hi_remove_saturation_rdif(data):
    """Direct conversion of hi_remove_saturation.pro for IDL.
    Detects and masks saturated pixels with nan. Takes image data and header as input. Returns fixed image.
    @param data: Data of .fits file
    @param header: Header of .fits file
    @return: Data with oversaturated columns removed"""

    # number of pixels in a column before column is considered saturated
    nsaturated = 3

    dsatval = np.nanmedian(data) + 2 * np.std(data)

    ind = np.where(np.abs(data) > dsatval)

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
            ans[:, ii] = np.nan  # np.nanmedian(data)

        else:
            ans = data.copy()

    else:
        ans = data.copy()

    return ans

#######################################################################################################################################

def remove_bad_col(data):
    """
    Removes saturated columns from .fits files.

    @param data: Data of .fits file
    @return: Data with saturated columns removed
    """
    nsaturated = 3
    dsatval = 0.95

    ind = np.where(np.abs(data) > dsatval)
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
            ans[:, ii] = np.nanmedian(data)

        else:
            ans = data.copy()

    else:
        ans = data.copy()

    return ans


#######################################################################################################################################

def scc_sebip(data, header, silent):
    """Direct conversion of scc_sebip.pro for IDL.
    Determines what has happened in terms of on-board sebip binning and corrects it.
    Takes image data and header as input and returns fixed image.
    @param data: Data of .fits file
    @param header: Header of .fits file
    @param silent: Run in silent mode
    @return: Data corrected for on-board binning"""

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
        if not silent:
            print('Correted for divide by 2 x {}'.format(cnt1))

    if cnt2 > 0:
        data = data * (2.0 ** cnt2)
        if not silent:
            print('Correted for square root x {}'.format(cnt2))
    if cntspw > 0:
        data = data * (64.0 ** cntspw)
        if not silent:
            print('Correted for HI SPW divide by 64 x {}'.format(cntspw))
    if cnt50 > 0:
        data = data * (4.0 ** cnt50)
        if not silent:
            print('Correted for divide by 4 x {}'.format(cnt50))
    if cnt53 > 0 and header['ipsum'] > 0:
        data = data * (4.0 ** cnt53)
        if not silent:
            print('Correted for divide by 4 x {}'.format(cnt53))
    if cnt82 > 0:
        data = data * (2.0 ** cnt82)
        if not silent:
            print('Correted for divide by 2 x {}'.format(cnt82))
    if cnt83 > 0:
        data = data * (4.0 ** cnt83)
        if not silent:
            print('Correted for divide by 4 x {}'.format(cnt83))
    if cnt84 > 0:
        data = data * (8.0 ** cnt84)
        if not silent:
            print('Correted for divide by 8 x {}'.format(cnt84))
    if cnt85 > 0:
        data = data * (16.0 ** cnt85)
        if not silent:
            print('Correted for divide by 16 x {}'.format(cnt85))
    if cnt86 > 0:
        data = data * (32.0 ** cnt86)
        if not silent:
            print('Correted for divide by 32 x {}'.format(cnt86))
    if cnt87 > 0:
        data = data * (64.0 ** cnt87)
        if not silent:
            print('Correted for divide by 64 x {}'.format(cnt87))
    if cnt88 > 0:
        data = data * (128.0 ** cnt88)
        if not silent:
            print('Correted for divide by 128 x {}'.format(cnt88))
    if cnt118 > 0:
        data = data * (3.0 ** cnt118)
        if not silent:
            print('Correted for divide by 3 x {}'.format(cnt118))

    if not silent:
        print('------------------------------------------------------')

    return data


#######################################################################################################################################

def get_smask(ftpsc, header, timehdr, calpath):
    """
    Conversion of get_smask.pro for IDL. Returns smooth mask array. Checks common block before opening mask file.
    Saves mask file to common block and re-scales the mask array for summing.

    @param ftpsc: Spacecraft (STEREO-A/STEREO-B)
    @param header: Header of .fits file
    @param timehdr: Timestamps of HI images
    @param calpath: Path for calibration files
    @return: Smooth mask array
    """
    if ftpsc == 'A':
        filename = 'hi2A_mask.fts'
        xy = [1, 51]

    if ftpsc == 'B':
        filename = 'hi2B_mask.fts'
        xy = [129, 79]

    calpath = calpath + filename
    timehdr = datetime.datetime.strptime(timehdr, '%Y-%m-%dT%H:%M:%S.%f')

    hdul_smask = fits.open(calpath)

    fullm = np.zeros((2176, 2176))

    x1 = 2048 - np.shape(fullm[xy[1] - 1:, xy[0] - 1:])[0]
    y1 = 2048 - np.shape(fullm[xy[1] - 1:, xy[0] - 1:])[1]

    if ftpsc == 'A':
        zeszc = np.shape(fullm[xy[0] - 1:y1, xy[1] - 1:x1])
        fullm[xy[0] - 1:y1, xy[1] - 1:x1] = hdul_smask[0].data
    if ftpsc == 'B':
        zeszc = np.shape(fullm[xy[0] - 1:, xy[1] - 1:x1])
        fullm[xy[0] - 1:, xy[1] - 1:x1] = hdul_smask[0].data

    tc = datetime.datetime(2015, 5, 19)

    flag = timehdr > tc

    if flag:
        fullm = np.rot90(fullm)
        fullm = np.rot90(fullm)

    mask = rebin(fullm[header['r1row'] - 1:header['r2row'], header['r1col'] - 1:header['r2col']],
                 [header['NAXIS2'], header['NAXIS1']])
    mask = np.where(mask < 1, 0, 1)

    hdul_smask.close()

    return mask


#######################################################################################################################################


def rebin(a, shape_arr):
    """
    Reshapes array to given dimensions.

    @param a: Array to be reshaped
    @param shape_arr: Dimensions for reshaping
    @return: Reshaped array
    """
    sh = int(shape_arr[0]), int(a.shape[0]) // int(shape_arr[0]), int(shape_arr[1]), int(a.shape[1]) // int(shape_arr[1])
    return a.reshape(sh).mean(-1).mean(1)


#######################################################################################################################################

#@numba.njit()
def hi_desmear(data, header_int, header_flt, header_str):
    """
    Conversion of hi_desmear.pro for IDL. Removes smear caused by no shutter. First compute the effective exposure time
    [time the ccd was static, generate matrix with this in diagonal, the clear time below and the read time above;
    invert and multiply by the image.

    @param data: Data of .fits file
    @param header_int: Array containing dstart, dstop, naxis, n_images and a flag for post-conjecture
    derived from the .fits header
    @param header_flt: Array containing exptime, cleartim, ro_delay, ipsum, line_ro, line_clr derived from .fits header
    @param header_str: Array containing rectify, obsrvtry derived from .fits header
    @return: Array corrected for shutterless camera
    """

    dstart1, dstart2, dstop1, dstop2, naxis1, naxis2, n_images, post_conj = header_int

    exptime, cleartim, ro_delay, ipsum, line_ro, line_clr = header_flt

    rectify, obsrvtry = header_str

    if dstart1 <= int(1) or naxis1 == naxis2:
        image = data.copy()

    else:
        image = data[dstart2 - 1:dstop1, dstart1 - 1:dstop1]

    clearest = 0.70
    exp_eff = exptime + float(n_images) * (clearest - cleartim + ro_delay)

    dataweight = float(n_images) * (2. ** (ipsum - 1.))

    inverted = 0.0

    if rectify:
        if not obsrvtry:
            inverted = 1
        if obsrvtry:
            if post_conj:
                inverted = 1

    if inverted == 1:

        n = naxis2

        ab = np.zeros((n, n))
        ab[:] = dataweight * line_ro
        bel = np.zeros((n, n))
        bel[:] = dataweight * line_clr

        fixup = np.triu(ab) + np.tril(bel)

        for i in range(0, n):
            fixup[i, i] = exp_eff

    else:

        n = naxis2

        ab = np.zeros((n, n))
        ab[:] = dataweight * line_clr
        bel = np.zeros((n, n))
        bel[:] = dataweight * line_ro

        fixup = np.triu(ab) + np.tril(bel)

        for i in range(0, n):
            fixup[i, i] = exp_eff

    fixup = np.linalg.inv(fixup)
    fixup = np.ascontiguousarray(fixup)

    image = np.ascontiguousarray(image)

    image = fixup @ image

    if dstart1 < 1 or (naxis1 == naxis2):
        img = image.copy()

    else:
        img = image[dstart2 - 1:dstop2, dstart1 - 1:dstop1]

    return img


#######################################################################################################################################


def get_calimg(instr, ftpsc, header, calpath, post_conj, silent):
    """
    Conversion of get_calimg.pro for IDL. Returns calibration correction array. Checks common block before opening
    calibration file. Saves calibration file to common block. Trims calibration array for under/over scan.
    Re-scales calibration array for summing.

    @param instr: STEREO-HI instrument (HI-1/HI-2)
    @param ftpsc: Spacecraft (STEREO-A/STEREO-B)
    @param header: Header of .fits file
    @param calpath: Path to calibration files
    @param post_conj: Indicates whether spacecraft is pre or post conjecture
    @param silent: Run on silent mode (True or False)
    @return: Array to correct for calibration
    """
    if instr == 'hi_1':

        if header['summed'] == 1:
            cal_version = '20061129_flatfld_raw_h1' + ftpsc.lower() + '.fts'
            sumflg = 0
        else:
            cal_version = '20100421_flatfld_sum_h1' + ftpsc.lower() + '.fts'
            sumflg = 1

    if instr == 'hi_2':

        if header['summed'] == 1:
            cal_version = '20150701_flatfld_raw_h2' + ftpsc.lower() + '.fts'
            sumflg = 0
        else:
            cal_version = '20150701_flatfld_sum_h2' + ftpsc.lower() + '.fts'
            sumflg = 1

    calpath = calpath + cal_version

    hdul_cal = fits.open(calpath)
    
    try:
        hdul_cal[0].header['RECTIFY']

    except KeyError:
        hdul_cal[0].header['RECTIFY'] = False
        hdul_cal[0].header['P1ROW'] = 0
        hdul_cal[0].header['P1COL'] = 0
        hdul_cal[0].header['P2ROW'] = 0
        hdul_cal[0].header['P2COL'] = 0
        hdul_cal[0].header['CRPIX1'] = 0
        hdul_cal[0].header['CRPIX2'] = 0
        hdul_cal[0].header['DSTART1'] = 0
        hdul_cal[0].header['DSTOP1'] = 0
        hdul_cal[0].header['DSTART2'] = 0
        hdul_cal[0].header['DSTOP2'] = 0
        hdul_cal[0].header['IPSUM'] = 0
        hdul_cal[0].header['SUMCOL'] = 1
        hdul_cal[0].header['SUMROW'] = 1
        
    if (hdul_cal[0].header['P1COL'] < 1) and (hdul_cal[0].header['RECTIFY'] == False):
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

        cal = hdul_cal[0].data[y1:y2 + 1, x1:x2 + 1]

    else:
        cal = hdul_cal[0].data
    
    if (header['RECTIFY'] == True) and (hdul_cal[0].header['RECTIFY'] == False):
        cal = secchi_rectify(cal, hdul_cal[0].header, calpath, silent)

    if sumflg:
        if header['summed'] <= 2:
            hdr_sum = 1

        else:
            hdr_sum = 2 ** (header['summed'] - 2)

    else:
        hdr_sum = 2 ** (header['summed'] - 1)

    s = np.shape(cal)

    cal = resize(cal, (int(s[1] / hdr_sum), int(s[0] / hdr_sum)))

    if post_conj:
        cal = np.rot90(cal, k=2)

    hdul_cal.close()

    return cal


#######################################################################################################################################

def get_calfac(header, timehdr):
    """
    Conversion of get_calfac.pro for IDL. Returns calibration factor for a given image.
    If the images was SEB IP summed then the program corrects the calibration factor.

    @param header: Header of .fits file
    @param timehdr: Timestamps of images
    @return: Calibration factor for a given image
    """
    if header['DETECTOR'] == 'HI1':

        if header['OBSRVTRY'] == 'STEREO_A':
            years = (timehdr - datetime.datetime(2009, 1, 1)).total_seconds() / 3600 / 24 / 365.25
            calfac = 3.63e-13
            annualchange = 0.000910

        if header['OBSRVTRY'] == 'STEREO_B':
            years = (timehdr - datetime.datetime(2007, 1, 1)).total_seconds() / 3600 / 24 / 365.25
            calfac = 3.55e-13
            annualchange = 0.001503

        if years < 0:
            years = 0

        calfac = calfac / (1 - annualchange * years)

    if header['DETECTOR'] == 'HI2':

        years = (timehdr - datetime.datetime(2000, 12, 31)).total_seconds() / 3600 / 24 / 365.25

        if header['OBSRVTRY'] == 'STEREO_A':
            calfac = 4.411e-14 + 7.099e-17 * years

        if header['OBSRVTRY'] == 'STEREO_B':
            calfac = 4.293e-14 + 3.014e-17 * years

    header['CALFAC'] = calfac

    if header['IPSUM'] > 1 and calfac != 1.0:
        divfactor = (2 ** (header['IPSUM'] - 1)) ** 2
        sumcount = header['IPSUM'] - 1
        header['IPSUM'] = 1

        calfac = calfac / divfactor

    if (header['POLAR'] == 1001) and (header['SEB_PROG'] != 'DOUBLE'):
        calfac = 2 * calfac

    return calfac


#######################################################################################################################################

def scc_hi_diffuse(header, ipsum):
    """
    Conversion of scc_hi_diffuse.pro for IDL. Compute correction for diffuse sources arrising from changes
    in the solid angle in the optics. In the mapping of the optics the area of sky viewed is not equal off axis.

    @param header: Header of .fits file
    @param ipsum: Allows override of header ipsum value for use in L1 and beyond images
    @return: Correction factor for given image
    """
    summing = 2 ** (ipsum - 1)

    # if header['ravg'] > 0:
    #  mu = header['pv2_1']
    #  cdelt = header['cdelt1']*np.pi/180

    if header['detector'] == 'HI1':

        if header['OBSRVTRY'] == 'STEREO_A':
            mu = 0.102422
            cdelt = 35.96382 / 3600 * np.pi / 180 * summing

        if header['OBSRVTRY'] == 'STEREO_B':
            mu = 0.095092
            cdelt = 35.89977 / 3600 * np.pi / 180 * summing

    if header['detector'] == 'HI2':

        if header['OBSRVTRY'] == 'STEREO_A':
            mu = 0.785486
            cdelt = 130.03175 / 3600 * np.pi / 180 * summing

        if header['OBSRVTRY'] == 'STEREO_B':
            mu = 0.68886
            cdelt = 129.80319 / 3600 * np.pi / 180 * summing

    pixelSize = 0.0135 * summing
    fp = pixelSize / cdelt

    x = np.arange(header['naxis1']) - header['crpix1'] + header['dstart1']
    x = np.array([x for i in range(header['naxis1'])])

    y = np.arange(header['naxis2']) - header['crpix2'] + header['dstart2']
    y = np.transpose(y)
    y = np.array([y for i in range(header['naxis1'])])

    r = np.sqrt(x * x + y * y) * pixelSize

    gamma = fp * (mu + 1.0) / r
    cosalpha1 = (-1.0 * mu + gamma * np.sqrt(1.0 - mu * mu + gamma * gamma)) / (1.0 + gamma * gamma)

    correct = ((mu + 1.0) ** 2 * (mu * cosalpha1 + 1.0)) / ((mu + cosalpha1) ** 3)

    return correct


#######################################################################################################################################

def secchi_rectify(a, scch, calpath, silent, overwrite=False):
    """
    Conversion of secchi_rectify.pro for IDL. Function procedure to rectify the CCD image,
    put solar north to the top of the image. Rotates an image so that ecliptic north is up and modifies coordinate
    keywords accordingly.

    @param a: Flatfield for given image
    @param scch: Header of flatfield
    @param calpath: Path to correct flatfield
    @param silent: Run on silent mode
    @param overwrite: Overwrite calibration image with rectified image
    @return: Rectified image
    """
    
    info="$Id: secchi_rectify.pro,v 1.29 2023/08/14 17:50:07 secchia Exp $"
    histinfo = info[:-2]

    time_cal = datetime.datetime.strptime(scch['date'], '%Y-%m-%d')
    
    cal_post_conj = 1 if datetime.datetime.strptime('2015-07-01', '%Y-%m-%d') < time_cal < datetime.datetime.strptime('2023-08-12', '%Y-%m-%d') else 0
    
    if scch['rectify'] == True:
        if not silent:
            print("RECTIFY=T -- Returning with no changes")
        return a
    
    else:
        
        stch = scch.copy()
        
        stch['rectify'] = True
        
        if scch['OBSRVTRY'] == 'STEREO_A' and cal_post_conj == 0:
            if scch['detector'] == 'HI1' or scch['detector'] == 'HI2':
                b = a.copy()
                stch['r1row'] = scch['p1row']
                stch['r2row'] = scch['p2row']
                stch['r1col'] = scch['p1col']
                stch['r2col'] = scch['p2col']
                stch['rectrota'] = 0
                rotcmt = 'no rotation necessary'
                
        elif scch['OBSRVTRY'] == 'STEREO_B' and cal_post_conj == 0:
            if scch['detector'] == 'HI1' or scch['detector'] == 'HI2':
                b = np.rot90(a, k=2)
                stch['r1row'] = 2176 - scch['p2row'] + 1
                stch['r2row'] = 2176 - scch['p1row'] + 1
                stch['r1col'] = 2176 - scch['p2col'] + 1
                stch['r2col'] = 2176 - scch['p1col'] + 1
                stch['crpix1'] = scch['naxis1'] - scch['crpix1'] + 1
                stch['crpix2'] = scch['naxis2'] - scch['crpix2'] + 1
                stch['naxis1'], stch['naxis2'] = scch['naxis1'], scch['naxis2']
                stch['rectrota'] = 2
                rotcmt = 'rotate 180 deg CCW'
                stch['dstart1'] = (79 - stch['r1col'] + 1) > 1
                stch['dstop1'] = stch['dstart1'] - 1 + ((stch['r2col'] - stch['r1col'] + 1) < 2048)
                stch['dstart2'] = (129 - stch['r1row'] + 1) > 1
                stch['dstop2'] = stch['dstart2'] - 1 + ((stch['r2row'] - stch['r1row'] + 1) < 2048)
                
        elif scch['OBSRVTRY'] == 'STEREO_A' and cal_post_conj == 1:
            if scch['detector'] == 'HI1' or scch['detector'] == 'HI2':
                b = np.rot90(a, k=2)
                stch['r1row'] = 2176 - scch['p2row'] + 1
                stch['r2row'] = 2176 - scch['p1row'] + 1
                stch['r1col'] = 2176 - scch['p2col'] + 1
                stch['r2col'] = 2176 - scch['p1col'] + 1
                stch['crpix1'] = scch['naxis1'] - scch['crpix1'] + 1
                stch['crpix2'] = scch['naxis2'] - scch['crpix2'] + 1
                stch['naxis1'], stch['naxis2'] = scch['naxis1'], scch['naxis2']
                stch['rectrota'] = 2
                rotcmt = 'rotate 180 deg CCW'
                stch['dstart1'] = (79 - stch['r1col'] + 1) > 1
                stch['dstop1'] = stch['dstart1'] - 1 + ((stch['r2col'] - stch['r1col'] + 1) < 2048)
                stch['dstart2'] = (129 - stch['r1row'] + 1) > 1
                stch['dstop2'] = stch['dstart2'] - 1 + ((stch['r2row'] - stch['r1row'] + 1) < 2048)
                
        elif scch['OBSRVTRY'] == 'STEREO_B' and cal_post_conj == 1:
            print('Case of ST-B with cal_post_conj=True not implemented. Exiting...')
            sys.exit()

        if stch['r1col'] < 1:
            stch['r2col'] += np.abs(stch['r1col']) + 1
            stch['r1col'] = 1
        if stch['r1row'] < 1:
            stch['r2row'] += np.abs(stch['r1row']) + 1
            stch['r1row'] = 1
    
        xden = 2 ** (scch['ipsum'] + scch['sumcol'] - 2)
        yden = 2 ** (scch['ipsum'] + scch['sumrow'] - 2)
    
        stch['dstart1'] = max(int(np.ceil(float(stch['dstart1']) / xden)), 1)
        stch['dstart2'] = max(int(np.ceil(float(stch['dstart2']) / yden)), 1)
        stch['dstop1'] = int(float(stch['dstop1']) / xden)
        stch['dstop2'] = int(float(stch['dstop2']) / yden)
        
        if stch['naxis1'] > 0 and stch['naxis2'] > 0:
            
            try:
                del stch['CDELT1']
                del stch['CDELT2']
            except KeyError:
                pass
                
            wcoord = wcs.WCS(stch)
            xcen, ycen = wcoord.all_pix2world((stch['naxis1'] - 1.) / 2., (stch['naxis2'] - 1.) / 2., 1)

            #stch['xcen'] = (stch['naxis1'] - 1.) / 2.-float(xcen)
            #stch['ycen'] = (stch['naxis2'] - 1.) / 2.-float(ycen)

            stch['xcen'] = 0
            stch['ycen'] = 0
        
        scch['NAXIS1'] = stch['naxis1']
        scch['NAXIS2'] = stch['naxis2']
        scch['R1COL'] = stch['r1col']
        scch['R2COL'] = stch['r2col']
        scch['R1ROW'] = stch['r1row']
        scch['R2ROW'] = stch['r2row']
        scch['SUMROW'] = stch['sumrow']
        scch['SUMCOL'] = stch['sumcol']
        scch['RECTIFY'] = stch['rectify']
        scch['CRPIX1'] = stch['crpix1']
        scch['CRPIX2'] = stch['crpix2']
        scch['XCEN'] = stch['xcen']
        scch['YCEN'] = stch['ycen']
        scch['CRPIX1A'] = stch['crpix1']
        scch['CRPIX2A'] = stch['crpix2']
        scch['DSTART1'] = stch['dstart1']
        scch['DSTART2'] = stch['dstart2']
        scch['DSTOP1'] = stch['dstop1']
        scch['DSTOP2'] = stch['dstop2']
        scch['HISTORY'] = histinfo
        scch['RECTROTA'] = (stch['rectrota'], rotcmt)
        plt.imshow(b)
        if overwrite==True:
            fits.writeto(calpath, b, scch, overwrite=True)
    
        #if not silent:
            #print(f"Rectification applied to {calpath}: {rotcmt}")
        
        return b


#######################################################################################################################################

def get_biasmean(header):
    """
    Conversion of get_biasmean.pro for IDL. Returns mean bias for a give image.

    @param header: Header of .fits file
    @return: Bias to be subtracted from the image
    """
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
    """
    Conversion of fill_missing.pro for IDL. Set missing block values sensibly.

    @param data: Data from .fits file
    @param header:Header of .fits file
    @return: Corrected image
    """
    if header['NMISSING'] == 0:
        data = data

    if header['NMISSING'] > 0:

        if len(header['MISSLIST']) < 1:
            print('Mismatch between nmissing and misslist.')
            data = data

        else:
            #fields = scc_get_missing(header)
            data = np.where(data==0, np.nanmedian(data), data)
            #data[fields] = np.nanmedian(data)

    #header['bunit'] = 'DN/s'

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

    print('This function has not been properly implemented')

    return np.array([])
#######################################################################################################################################

def scc_img_trim(im, header):
    """
    Conversion of scc_img_trim.pro for IDL. Returns rectified images with under/over scan areas removed.
    The program returns the imaging area of the CCD. If the image has not been rectified such that ecliptic north
    is up then the image is rectified.

    @param im: Selected image
    @param header: Header of .fits file
    @return: Rectified image with under-/overscan removed
    """
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

#######################################################################################################################################

# @numba.njit()
def cadence_corr(tdiff, cadence):
    """
    Accounts for missing images in STEREO-HI data. If the gap between images is larger than the cadence,
    a block of nan is appended to the data. Further correction of missing images is done directly
    during creation of the Jplot.

    @param tdiff: Array of time differences between subsequent images
    @param cadence: Cadence of respective instrument
    @return:Blocks of nan data and indices at which to insert them.
    """
    nan_ind = []
    nan_dat = []

    # time difference is divided by cadence and rounded
    # if the time difference is more than one standard cadence, np.nan must be appended to account for time gaps

    for i in range(len(tdiff)):

        ap_ind = np.round(tdiff[i] / cadence, 0)
        ap_ind = int(ap_ind)

        if ap_ind > 1:
            nan_ind.append(int(i))
            nan_dat.append(int(ap_ind)-1)

    return nan_dat, nan_ind


#######################################################################################################################################
def create_rdif(time_obj, maxgap, cadence, data, hdul, wcoord, bflag, ins):
    """
    Creates running difference images from reduced HI data.
    The preceding image is subtracted from an image if their times are within a certain maximum time gap.
    For science data, processing images into running difference images also removes the starfield.

    @param time_obj: Array of timestamps of images
    @param maxgap: Maximum gap between images
    @param cadence: Cadence of HI instrument
    @param data: Image data
    @param hdul: Header of .fits file
    @param wcoord: World coordinate system extracted from .fits header
    @param bflag: Science or beacon images
    @param ins: HI instrument (HI-1/HI-2)
    @return: Array of running difference images and set of indices where there are time gaps
    """

    # Array of np.nan shaped like image data to replace missing timesteps
    nandata = np.full_like(data[0], np.nan)

    r_dif = []

    data = np.float32(data)

    # Define kernel size fo median filtering
    if bflag == 'science':

        if ins == 'hi_1':
            kernel = 5

        if ins == 'hi_2':
            kernel = 3

    if bflag == 'beacon':
        kernel = 3

    indices = []
    ims = []

    for i in range(1, len(data)):

        # Get center value for preceding and following image
        crval = [hdul[i - 1][0].header['crval1a'], hdul[i - 1][0].header['crval2a']]

        center = [hdul[i][0].header['crpix1'] - 1, hdul[i][0].header['crpix2'] - 1]

        center2 = wcoord[i].all_world2pix(crval[0], crval[1], 0)

        # Determine shift between preceding and following image
        xshift = center2[0] - center[0]
        yshift = center2[1] - center[1]
        
        shiftarr = [yshift, xshift]

        # Shift image by calculated shift and interpolate using spline interpolation
        ims.append(shift(data[i - 1], shiftarr, mode='nearest'))

        # produce regular running difference images if time difference is within maxgap * cadence

        j_ind = []

        # time difference is taken for every time object, not just the preceding one
        for j in range(len(data) - 1):
            time_i_j = (time_obj[i] - time_obj[i - j]).sec / 60

            # indices of time objects that fit the criteria are appended to the list
            # criteria: time difference smaller than -maxgap * cadence and time difference larger than cadence
            if (np.round(time_i_j) <= -maxgap * cadence) & (np.round(time_i_j) >= cadence):# & (np.mod(np.round(time_i_j), cadence) == 0):

                j_ind.append(i - j)

        # if no adequate preceding image is found, append array of np.nan to the running difference list
        # criteria: time diffference is larger than -maxgap * cadence or time difference is smaller than cadence

        if np.round((time_obj[i] - time_obj[i - 1]).sec / 60) > -maxgap * cadence:
            # indices.append(i)
            r_dif.append(nandata)

        if np.round((time_obj[i] - time_obj[i - 1]).sec / 60) < cadence:
            r_dif.append(nandata)

        # for appropriate time differences, create running differene images and apply median filter
        if len(j_ind) >= 1:
            j = j_ind[0]

            indices.append(i)

            if bflag == 'science':
                ndat = np.float32(data[i] - ims[j])
                ndat = cv2.medianBlur(ndat, kernel)
                r_dif.append(cv2.medianBlur(ndat, kernel))

            if bflag == 'beacon':
                ndat = np.float32(data[i] - ims[j])
                r_dif.append(cv2.medianBlur(ndat, kernel))

    return r_dif, indices


#######################################################################################################################################

def get_map_xrange(hdul):
    """
    Conversion of get_map_xrange.pro for IDL. Extract min/max X-coordinate of map. Coordinates correspond to the pixel center.

    @param hdul: Header of .fits file
    @return: x-axis extent, x-axis center, x-axis width of pixel in spacecraft coordinates as an array
    """
    nx = [hdul[i][0].header['NAXIS1'] for i in range(len(hdul))]
    xc = [hdul[i][0].header['CRVAL1'] for i in range(len(hdul))]
    dx = [hdul[i][0].header['CDELT1'] for i in range(len(hdul))]

    # xmin = np.nanmin(xc - dx * (nx - 1)/2)
    # xmax = np.nanmax(xc + dx * (nx - 1)/2)

    # x_range = [xmin, xmax]

    return nx, xc, dx


#######################################################################################################################################

def get_map_yrange(hdul):
    """
    Conversion of get_map_yrange.pro for IDL. Extract min/max Y-coordinate of map. Coordinates correspond to the pixel center.

    @param hdul: Header of .fits file
    @return: y-axis extent, y-axis center, y-axis width of pixel in spacecraft coordinates as an array
    """
    ny = [hdul[i][0].header['NAXIS2'] for i in range(len(hdul))]
    yc = [hdul[i][0].header['CRVAL2'] for i in range(len(hdul))]
    dy = [hdul[i][0].header['CDELT2'] for i in range(len(hdul))]

    # xmin = np.nanmin(xc - dx * (nx - 1)/2)
    # xmax = np.nanmax(xc + dx * (nx - 1)/2)

    # x_range = [xmin, xmax]

    return ny, yc, dy
    
#######################################################################################################################################

def get_bkgd(path, ftpsc, start, bflag, ins):
    """
    Creates weekly minimum background image for STEREO-HI data.
    
    """
    
    background = []
    
    bg_dur = 7
    
    
    date = datetime.datetime.strptime(start, '%Y%m%d') - datetime.timedelta(days=bg_dur) 
    interv = np.arange(bg_dur+1)
    
    datelist = [datetime.datetime.strftime(date + datetime.timedelta(days=int(i)), '%Y%m%d') for i in interv]  
    red_path = path + 'reduced/data/' + ftpsc + '/'

    red_paths = []
    red_files = []

    for k, dates in enumerate(datelist):
        red_paths.append(red_path + str(dates) + '/' + bflag + '/' + ins + '/*.fts')

        if k == 0:
            red_files.append(sorted(glob.glob(red_path + str(dates) + '/' + bflag + '/' + ins + '/*.fts'))[-1])

        else:
            red_files.extend(sorted(glob.glob(red_path + str(dates) + '/' + bflag + '/' + ins + '/*.fts')))

    data = []

    for i in range(len(red_files)):
        file = fits.open(red_files[i])
        data.append(file[0].data.copy())
        file.close()
        
    data = np.array(data)

    nan_mask = np.array([np.isnan(data[i]) for i in range(len(data))])
    
    for i in range(len(data)):
        data[i][nan_mask[i]] = np.array(np.interp(np.flatnonzero(nan_mask[i]), np.flatnonzero(~nan_mask[i]), data[i][~nan_mask[i]]))

    for j, file in enumerate(red_files):
        if start in file:
            index = j-1
            break
        
    bgkd = []

    for i in range(data.shape[0]-index):
        bkgd_arr = np.nanmin(data[i:index+i], axis=0)
        bgkd.append(bkgd_arr)

    bgkd = np.array(bgkd)
        
    return bgkd
    
#######################################################################################################################################
def running_difference(start, bkgd, path, datpath, ftpsc, ins, bflag, silent, save_img):
    """
    Creates running difference images from reduced STEREO-HI data and saves them to the specified location.

    @param start: First date (DDMMYYYY) for which to create running difference images
    @param bkgd: Minimum weekly/monthly background returned by get_bkgd function.
    @param path: The path where all reduced images, running difference images and J-Maps are saved
    @param datpath: Path to STEREO-HI calibration files
    @param ftpsc: Spacecraft (A/B)
    @param instrument: STEREO-HI instrument (HI-1/HI-2)
    @param bflag: Science or beacon data
    @param silent: Run in silent mode
    @param save_img: Save running difference images as .pngs
    """
    if not silent:
        print('-------------------')
        print('RUNNING DIFFERENCE')
        print('-------------------')
    
    # Initialize date as datetime, get day before start date as datetime
    date = datetime.datetime.strptime(start, '%Y%m%d')
    prev_date = date - datetime.timedelta(days=1)
    prev_date = datetime.datetime.strftime(prev_date, '%Y%m%d')

    bkgd_arr = bkgd.copy()
    
    # Get paths to files one day before start time
    prev_path = path + 'reduced/data/' + ftpsc + '/' + prev_date + '/' + bflag + '/' + ins + '/'

    files = []

    # Append files from day before start to list
    # If no files exist, start running difference images with first file of chosen start date

    if os.path.exists(prev_path):

        for file in sorted(glob.glob(prev_path + '*.fts')):
            files.append(file)

        try:
            files = [files[-1]]

        except IndexError:
            files = []

    calpath = datpath + 'calibration/'

    # tc = datetime.datetime(2015, 7, 1)

    # cadence of instruments in minutes
    if bflag == 'beacon':
        cadence = 120.0

    if bflag == 'science':
        if ins == 'hi_1':
            cadence = 40.0
        
        if ins == 'hi_2':
            cadence = 120.0

    # define maximum gap between consecutive images
    # if gap > maxgap, no running difference image is produced, timestep is filled with np.nan instead

    maxgap = -3.5

    redpath = path + 'reduced/data/' + ftpsc + '/' + start + '/' + bflag + '/' + ins + '/'

    # read in .fits files

    if not silent:
        print('Getting files...')

    for file in sorted(glob.glob(redpath + '*.fts')):
        files.append(file)

    # get times and headers from .fits files

    hdul = [fits.open(files[i]) for i in range(len(files))]
    tcomp = datetime.datetime.strptime(hdul[0][0].header['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f')
    time = [hdul[i][0].header['DATE-END'] for i in range(len(hdul))]
    wcoord = [wcs.WCS(files[i], key='A') for i in range(len(files))]

    crval = [hdul[i][0].header['crval1'] for i in range(len(hdul))]

    if ftpsc == 'A':    
        post_conj = [int(np.sign(crval[i])) for i in range(len(crval))]

    if ftpsc == 'B':    
        post_conj = [int(-1*np.sign(crval[i])) for i in range(len(crval))]
        
    if len(set(post_conj)) == 1:
    
        post_conj = post_conj[0]

        if post_conj == -1:
            post_conj = False
        if post_conj == 1:
            post_conj = True

    else:
        print('Invalid CRVAL in header. Exiting...')
        sys.exit()

    if not post_conj:

        if ftpsc == 'A':
            orig = 'upper'

        if ftpsc == 'B':
            orig = 'lower'

    if post_conj:

        if ftpsc == 'A':
            orig = 'lower'

        if ftpsc == 'B':
            orig = 'upper'
            
    if not silent: 
        print('Reading data...')

    # times are converted to objects

    time_obj = [Time(time[i], format='isot', scale='utc') for i in range(len(time))]

    data = np.array([hdul[i][0].data for i in range(len(files))])

    #Subtract coronal background from images
    
    nan_mask = np.array([np.isnan(data[i]) for i in range(len(data))])
    
    for i in range(len(data)):
        data[i][nan_mask[i]] = np.array(np.interp(np.flatnonzero(nan_mask[i]), np.flatnonzero(~nan_mask[i]), data[i][~nan_mask[i]]))

    # if bflag == 'beacon':
    #     min_arr = np.nanmin(data, axis=0)

    data = data - bkgd_arr
    
    if not silent:
        print('Replacing missing values...')
            
    exp = np.array([hdul[i][0].header['EXPTIME'] for i in range(len(hdul))])

    # Masked data is replaced with image median
    data = np.array(data)
    
    if ins == 'hi_2':
        
        mask = np.array([get_smask(ftpsc, hdul[i][0].header, time[i], calpath) for i in range(0, len(data))])
    
        data[mask == 0] = np.nanmedian(data)

    if not silent:
        print('Creating running difference images...')

    # Creation of running difference images
    
    r_dif, ind = create_rdif(time_obj, maxgap, cadence, data, hdul, wcoord, bflag, ins)
    r_dif = np.array(r_dif)
    
    if bflag == 'science':
        vmin = -1e-13
        vmax = 1e-13

    if bflag == 'beacon':
        vmin = np.nanmedian(r_dif) - np.std(r_dif)
        vmax = np.nanmedian(r_dif) + np.std(r_dif)

    if save_img:
    
        if not silent:
            print('Saving running difference images as png...')

        savepath = path + 'running_difference/pngs/' + ftpsc + '/' + start + '/' + bflag + '/' + ins + '/'

        if not os.path.exists(savepath):
            os.makedirs(savepath)

        else:
            oldfiles = glob.glob(os.path.join(savepath, "*.png"))
            
            for fil in oldfiles:
                os.remove(fil)               

        for i in np.array(ind)-1:

            fig, ax = plt.subplots(figsize=(1.024, 1.024), dpi=100, frameon=False)
            fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.axis('off')
            #image = (r_dif[i] - r_dif[i].min()) / (r_dif[i].max() - r_dif[i].min())
            ax.imshow(r_dif[i], vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', origin=orig)
            plt.savefig(savepath + files[i+1].rpartition('/')[2][0:21] + '.png', dpi=1000)
            plt.close()
    
    if not silent:
        print('Saving image as pickle...')

    savepath = path + 'running_difference/data/' + ftpsc + '/' + start + '/' + bflag + '/' + ins + '/'

    names = [files[i].rpartition('/')[2] for i in ind]

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    else:
        oldfiles = glob.glob(os.path.join(savepath, "*.pkl"))
                
        for fil in oldfiles:
            os.remove(fil)
            
        oldfiles = glob.glob(os.path.join(savepath, "*.fts"))
                
        for fil in oldfiles:
            os.remove(fil)
    
    header = [hdul[i][0].header for i in ind]
    r_dif = np.array([r_dif[i] for i in np.array(ind)-1])
    
    for i in range(len(r_dif)):
        fits.writeto(savepath + names[i], np.array(r_dif[i]), header[i])
#######################################################################################################################################

def reduced_nobg(start, bkgd, path, datpath, ftpsc, instrument, bflag, silent):
    """
    Creates running difference images from reduced STEREO-HI data and saves them to the specified location.

    @param start: First date (DDMMYYYY) for which to create running difference images
    @param bkgd: Minimum weekly/monthly background returned by get_bkgd function. Only applied to science data.
    @param path: The path where all reduced images, running difference images and J-Maps are saved
    @param datpath: Path to STEREO-HI calibration files
    @param ftpsc: Spacecraft (A/B)
    @param instrument: STEREO-HI instrument (HI-1/HI-2)
    @param bflag: Science or beacon data
    @param silent: Run in silent mode
    """
    if not silent:
        print('-------------------')
        print('BACKGROUND SUBTRACTION')
        print('-------------------')
    
    # Initialize date as datetime, get day before start date as datetime
    date = datetime.datetime.strptime(start, '%Y%m%d')
    prev_date = date - datetime.timedelta(days=1)
    prev_date = datetime.datetime.strftime(prev_date, '%Y%m%d')

    f = 0

    if instrument == 'hi1hi2':
        instrument = ['hi_1', 'hi_2']

    if instrument == 'hi_1':
        instrument = ['hi_1']

    if instrument == 'hi_2':
        instrument = ['hi_2']

    for ins in instrument:
        bkgd_arr = bkgd[f]
        
        # Get paths to files one day before start time
        prev_path = path + 'reduced/data/' + ftpsc + '/' + prev_date + '/' + bflag + '/' + ins + '/'
    
        files = []
    
        # Append files from day before start to list
        # If no files exist, start running difference images with first file of chosen start date
    
        if os.path.exists(prev_path):
    
            for file in sorted(glob.glob(prev_path + '*.fts')):
                files.append(file)
    
            try:
                files = [files[-1]]
    
            except IndexError:
                files = []
    
        calpath = datpath + 'calibration/'
    
        redpath = path + 'reduced/data/' + ftpsc + '/' + start + '/' + bflag + '/' + ins + '/'
    
        # read in .fits files
    
        if not silent:
            print('Getting files...')
    
        for file in sorted(glob.glob(redpath + '*.fts')):
            files.append(file)
    
        # get times and headers from .fits files

        hdul = [fits.open(files[i]) for i in range(len(files))]
    
        if not silent: 
            print('Reading data...')    
    
        data = np.array([hdul[i][0].data for i in range(len(files))])
    
        #Subtract coronal background from images
        
        nan_mask = np.array([np.isnan(data[i]) for i in range(len(data))])
        
        for i in range(len(data)):
            data[i][nan_mask[i]] = np.array(np.interp(np.flatnonzero(nan_mask[i]), np.flatnonzero(~nan_mask[i]), data[i][~nan_mask[i]]))

        data = data - bkgd_arr
        
        if not silent:
            print('Replacing missing values...')
                
    
        # Masked data is replaced with image median
        data = np.array(data)
        
        if ins == 'hi_2':
            
            mask = np.array([get_smask(ftpsc, hdul[i][0].header, time[i], calpath) for i in range(0, len(data))])
        
            data[mask == 0] = np.nanmedian(data)

        savepath = path + 'reduced/data_nobg/' + ftpsc + '/' + start + '/' + bflag + '/' + ins + '/'
    
        if not os.path.exists(savepath):
            os.makedirs(savepath)
    
        else:
                
            oldfiles = glob.glob(os.path.join(savepath, "*.fts"))
                    
            for fil in oldfiles:
                os.remove(fil)
        
        
        for i in range(len(files)):
            name = files[i].rpartition('/')[2]

            fits.writeto(savepath + name, np.array(data[i]), hdul[i][0].header)
        
        f = f+1

#######################################################################################################################################

def ecliptic_cut(data, header, bflag, ftpsc, mode='rotate'):

    
    if bflag == 'science':
        
        xsize = 1024
        ysize = 1024
         
    if bflag == 'beacon':
        
        xsize = 256
        ysize = 256
        
    x = np.linspace(0, xsize-1, xsize)
    y = np.linspace(ysize-1, 0, ysize)

    xv, yv = np.meshgrid(x, y)

    wcoord = [wcs.WCS(header[i]) for i in range(len(header))]

    dat = [header[i]['DATE-END'] for i in range(len(header))]
    earth = [get_body_heliographic_stonyhurst('earth', dat[i]) for i in [0, -1]]

    if ftpsc == 'A': 
        stereo = get_horizons_coord('STEREO-A', [dat[0], dat[-1]])

    if ftpsc == 'B':
        stereo = get_horizons_coord('STEREO-B', [dat[0], dat[-1]])

    e_hpc = [SkyCoord(earth[i]).transform_to(Helioprojective(observer=stereo[i])) for i in range(len(earth))]
       
    e_x = np.array([e_hpc[i].Tx.to(u.deg).value for i in range(len(e_hpc))])*np.pi/180
    e_y = np.array([e_hpc[i].Ty.to(u.deg).value for i in range(len(e_hpc))])*np.pi/180

    e_x_interp = np.linspace(e_x[0], e_x[1], len(dat))
    e_y_interp = np.linspace(e_y[0], e_y[1], len(dat))

    e_pa = np.arctan2(-np.cos(e_y_interp)*np.sin(e_x_interp), np.sin(e_y_interp))
    
    dif_cut = []
    elongation = []
    
    for i in range(len(wcoord)):
        
        thetax, thetay = wcoord[i].all_pix2world(xv, yv, 0)
        
        tx = thetax*np.pi/180
        ty = thetay*np.pi/180
        
        pa_reg = np.arctan2(-np.cos(ty)*np.sin(tx), np.sin(ty))
        elon_reg = np.arctan2(np.sqrt((np.cos(ty)**2)*(np.sin(tx)**2)+(np.sin(ty)**2)), np.cos(ty)*np.cos(tx))
        
        delta_pa = e_pa[i]

        e_val = [(delta_pa)-1*np.pi/180, (delta_pa)+1*np.pi/180]

        data_mask = np.where((pa_reg > min(e_val)) & (pa_reg < max(e_val)), data[i], np.nan)

        data_med = np.nanmedian(data_mask, 0)
        dif_cut.append(data_med)

        elon_mask = np.where((pa_reg > min(e_val)) & (pa_reg < max(e_val)), elon_reg, np.nan)
        
        elongation_max = np.nanmax(elon_mask*180/np.pi)
        elongation_min = np.nanmin(elon_mask*180/np.pi)
        elongation.append(elongation_min)
        elongation.append(elongation_max)

    dif_cut = np.array(dif_cut)
    elongation = np.array(elongation)

    return dif_cut, elongation
    
#######################################################################################################################################
def make_jplot(start, duration, path, datpath, ftpsc, instrument, bflag, save_path, path_flg, silent):
    """
    Creates Jplot from running difference images. Method similar to create_jplot_tam.pro written in IDL by Tanja Amerstorfer.
    Middle slice of each running difference is cut out, strips are aligned, time-gaps are filled with nan.

    @param start: Start date of Jplot
    @param duration: Length of Jplot (in days)
    @param path: The path where all reduced images, running difference images and J-Maps are saved
    @param datpath: Path to STEREO-HI calibration files
    @param ftpsc: Spacecraft (A/B)
    @param instrument: STEREO-HI instrument (HI-1/HI-2)
    @param bflag: Science or beacon data
    @param save_path: Path pointing towards downloaded STEREO .fits files
    @param path_flg: Specifies path for downloaded files, depending on wether science or beacon data is used
    @param silent: Run in silent mode
    """
    if not silent:
        print('-------------------')
        print('JPLOT')
        print('-------------------')

    date = datetime.datetime.strptime(start, '%Y%m%d')

    interv = np.arange(duration)

    datelst = [datetime.datetime.strftime(date + datetime.timedelta(days=int(interv[i])), '%Y%m%d') for i in interv]
    savepaths_h1 = [path + 'running_difference/data/' + ftpsc + '/' + datelst[i] + '/' + bflag + '/hi_1/' for i in interv]
    savepaths_h2 = [path + 'running_difference/data/' + ftpsc + '/' + datelst[i] + '/' + bflag + '/hi_2/' for i in interv]
    
    files_h1 = []
    files_h2 = []    
    
    for savepath in savepaths_h1:
        for file in sorted(glob.glob(savepath + '*.fts')):
            files_h1.append(file)
            
    for savepath in savepaths_h2:
        for file in sorted(glob.glob(savepath + '*.fts')):
            files_h2.append(file)


    # get times and headers from .fits files
    rdif_h1 = []
    header_h1 = []
    
    for i in range(len(files_h1)):
        file = fits.open(files_h1[i])
        rdif_h1.append(file[0].data.copy())
        header_h1.append(file[0].header.copy())
        file.close()
        
    rdif_h1 = np.array(rdif_h1)
    
    rdif_h2 = []
    header_h2 = []
    
    for i in range(len(files_h2)):
        file = fits.open(files_h2[i])
        rdif_h2.append(file[0].data.copy())
        header_h2.append(file[0].header.copy())
        file.close()
        
    rdif_h2 = np.array(rdif_h2)

    if bflag == 'beacon':
        cadence_h1 = 120.0
        cadence_h2 = 120.0

    if bflag == 'science':
        cadence_h1 = 40.0
        cadence_h2 = 120.0

    
    # define maximum gap between consecutive images
    # if gap > maxgap, no running difference image is produced, timestep is filled with np.nan instead

    maxgap = 3.5

    if not silent:
        print('Getting data...')

    time_h1_arr = [header_h1[i]['DATE-END'] for i in range(len(header_h1))]
    datetime_h1 = [datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f') for t in time_h1_arr]

    time_h2_arr = [header_h2[i]['DATE-END'] for i in range(len(header_h2))]
    datetime_h2 = [datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f') for t in time_h2_arr]
    # Determine if STEREO spacecraft are pre- or post-conjunction

    if ftpsc == 'A':
        sc = 'ahead'

    if ftpsc == 'B':
        sc = 'behind'
    
    crval1 = [header_h1[i]['crval1'] for i in range(len(header_h1))]

    if ftpsc == 'A':    
        post_conj = [int(np.sign(crval1[i])) for i in range(len(crval1))]

    if ftpsc == 'B':    
        post_conj = [int(-1*np.sign(crval1[i])) for i in range(len(crval1))]

    if len(set(post_conj)) == 1:
        post_conj = post_conj[0]

        if post_conj == -1:
            post_conj = False
        if post_conj == 1:
            post_conj = True

    else:
        print('Invalid dates. Exiting...')
        sys.exit()

    if not silent:
        print('Making ecliptic cut...')

    dif_med_h1, elongation_h1 = ecliptic_cut(rdif_h1, header_h1, bflag, ftpsc)
    dif_med_h2, elongation_h2 = ecliptic_cut(rdif_h2, header_h2, bflag, ftpsc)

    dif_med_h1 = np.where(np.isnan(dif_med_h1), np.nanmedian(dif_med_h1), dif_med_h1)
    dif_med_h2 = np.where(np.isnan(dif_med_h2), np.nanmedian(dif_med_h2), dif_med_h2)
    
    elongation_h1 = np.abs(elongation_h1)
    elongation_h2 = np.abs(elongation_h2)
    
    # Choose widths for ecliptic cut

    if bflag == 'science':
        pix = 32

    if bflag == 'beacon':
        pix = 8

    time_mdates_h1 = mdates.date2num(datetime_h1)
    time_mdates_h2 = mdates.date2num(datetime_h2)

    if not silent:
        print('Calculating elongation...')

    if not post_conj:

        if ftpsc == 'A':
            tflag = 0

        if ftpsc == 'B':
            tflag = 1

    if post_conj:

        if ftpsc == 'A':
            tflag = 1

        if ftpsc == 'B':
            tflag = 0

    if tflag:
        orig = 'lower'

    if not tflag:
        orig = 'upper'


    jmap_h1_interp = np.array(dif_med_h1).transpose()
    jmap_h2_interp = np.array(dif_med_h2).transpose()

    # Contrast stretching
    p2, p98 = np.nanpercentile(jmap_h1_interp, (2, 98))
    img_rescale_h1 = exposure.rescale_intensity(jmap_h1_interp, in_range=(p2, p98))

    # Contrast stretching
    p2, p98 = np.nanpercentile(jmap_h2_interp, (2, 98))
    img_rescale_h2 = exposure.rescale_intensity(jmap_h2_interp, in_range=(p2, p98))

    gap_ind_h1 = []
    
    for i in range(1, len(time_mdates_h1)):
        
        delta_t1 = np.round((time_mdates_h1[i]-time_mdates_h1[i-1])*24*60, 1)

        if delta_t1 > cadence_h1:
            gap_ind_h1 += int(delta_t1/cadence_h1 - 1) * [i]

    gap_ind_h2 = []
    
    for i in range(1, len(time_mdates_h2)):
        
        delta_t2 = np.round((time_mdates_h2[i]-time_mdates_h2[i-1])*24*60, 1)

        if delta_t2 > cadence_h2:
            gap_ind_h2 += int(delta_t2/cadence_h2 - 1) * [i]
            
    if len(gap_ind_h1) > 0:
        gap_ind_h1 = sorted(gap_ind_h1, reverse=True)
        img_rescale_h1 = np.insert(img_rescale_h1, gap_ind_h1, np.nan, axis=1)

    if len(gap_ind_h2) > 0:
        gap_ind_h2 = sorted(gap_ind_h2, reverse=True)
        img_rescale_h2 = np.insert(img_rescale_h2, gap_ind_h2, np.nan, axis=1)

    # Save images separately and together
    img_rescale_h1 = np.where(np.isnan(img_rescale_h1), np.nanmedian(img_rescale_h1), img_rescale_h1)
    img_rescale_h2 = np.where(np.isnan(img_rescale_h2), np.nanmedian(img_rescale_h2), img_rescale_h2)
    
    vmin_h1 = np.nanmedian(img_rescale_h1) - 2 * np.nanstd(img_rescale_h1)
    vmax_h1 = np.nanmedian(img_rescale_h1) + 2 * np.nanstd(img_rescale_h1)

    vmin_h2 = np.nanmedian(img_rescale_h2) - 2 * np.nanstd(img_rescale_h2)
    vmax_h2 = np.nanmedian(img_rescale_h2) + 2 * np.nanstd(img_rescale_h2)

    savepath_h1 = path + 'jplot/' + ftpsc + '/' + bflag + '/hi_1/' + str(start[0:4]) + '/'
    savepath_h2 = path + 'jplot/' + ftpsc + '/' + bflag + '/hi_2/' + str(start[0:4]) + '/'
    savepath_h1h2 = path + 'jplot/' + ftpsc + '/' + bflag + '/hi1hi2/' + str(start[0:4]) + '/'

    if not os.path.exists(savepath_h1):
        os.makedirs(savepath_h1)

    if not os.path.exists(savepath_h2):
        os.makedirs(savepath_h2)

    if not os.path.exists(savepath_h1h2):
        os.makedirs(savepath_h1h2)

    time_file_comb = datetime.datetime.strftime(min(np.nanmin(datetime_h1), np.nanmin(datetime_h2)), '%Y%m%d_%H%M%S')

    with open(savepath_h1 + 'jplot_hi1_' + start + '_' + time_file_comb + '_UT_' + ftpsc + '_' + bflag[0] + '.pkl', 'wb') as f:
        pickle.dump([img_rescale_h1, orig], f)

    with open(savepath_h2 + 'jplot_hi2_' + start + '_' + time_file_comb + 'UT_' + ftpsc + '_' + bflag[0] + '.pkl', 'wb') as f:
        pickle.dump([img_rescale_h2, orig], f)

    elongations = [np.nanmin(elongation_h1), np.nanmax(elongation_h1), np.nanmin(elongation_h2), np.nanmax(elongation_h2)]


    fig, ax = plt.subplots(figsize=(10,5), sharex=True, sharey=True)

    plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 24)))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
    plt.gca().xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    
    plt.gca().yaxis.set_minor_locator(MultipleLocator(2))

    ax.xaxis_date()
    
    # helcats=True
    
    # if helcats:
    #     helcats_data = []
        
    #     helcats_data.append(['TRACK_NO', 'DATE', 'ELON', 'PA', 'SC'])
        
    #     helcats_file = glob.glob("HELCATS/HCME_"+ftpsc+"__"+start+"_*.txt")[0]
        
    #     with open(helcats_file, mode="r") as f:
    #         for line in f:
    #             helcats_data.append(line.split())
        
    #     helcats_data = pd.DataFrame(helcats_data[1:], columns=helcats_data[0])
    #     helcats_time = mdates.datestr2num(helcats_data['DATE'])
    #     helcats_elon = helcats_data['ELON'].values
    #     helcats_elon = helcats_elon.astype(np.float)
    
    #     ax.scatter(helcats_time, helcats_elon, marker='+', facecolor='r', linewidths=.5)

    dt_h1 = (cadence_h1/60)/24
        
    dt_h2 = (cadence_h2/60)/24
    
    dy_h1, = np.abs(np.diff(elongations[:2])/(len(elongation_h1)-1))
    dy_h2, = np.abs(np.diff(elongations[2:])/(len(elongation_h2)-1))

    ax.imshow(img_rescale_h1, cmap='gray', aspect='auto', interpolation='none', vmin=vmin_h1, vmax=vmax_h1, origin=orig, extent=[time_mdates_h1[0], time_mdates_h1[-1], elongations[0], elongations[1]])    
    ax.imshow(img_rescale_h2, cmap='gray', aspect='auto', interpolation='none', vmin=vmin_h2, vmax=vmax_h2, origin=orig, extent=[time_mdates_h2[0], time_mdates_h2[-1], elongations[2], elongations[3]]) 

    ax.set_title(start + ' STEREO-' + ftpsc)
    
    plt.ylim(elongations[0], 80)

    plt.xlabel('Date (d/m/y)')
    plt.ylabel('Elongation (°)')

    if not os.path.exists(savepath_h1h2 + 'pub/'):
        os.makedirs(savepath_h1h2 + 'pub/')
        
    bbi = 'tight'
    pi = 0.5        

    plt.savefig(savepath_h1h2 + 'pub/' + 'jplot_' + instrument + '_' + start + '_' + time_file_comb + 'UT_' + ftpsc + '_' + bflag[0] + '.png', bbox_inches=bbi, pad_inches=pi)

    if instrument == 'hi_1':

        fig, ax = plt.subplots(figsize=(10,5), sharex=True, sharey=True)

        plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 24)))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
        plt.gca().xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
        
        plt.gca().yaxis.set_minor_locator(MultipleLocator(2))

        ax.xaxis_date()

        ax.imshow(img_rescale_h1, cmap='gray', aspect='auto', interpolation='none', vmin=vmin_h1, vmax=vmax_h1, origin=orig, extent=[time_mdates_h1[0], time_mdates_h1[-1], elongations[0], elongations[1]])    

        ax.set_title(start + ' STEREO-' + ftpsc)
        
        plt.ylim(elongations[0], 80)

        plt.xlabel('Date (d/m/y)')
        plt.ylabel('Elongation (°)')

        if not os.path.exists(savepath_h1 + 'pub/'):
            os.makedirs(savepath_h1 + 'pub/')
            
        bbi = 'tight'
        pi = 0.5

        plt.ylim(elongations[0], elongations[1])

        plt.savefig(savepath_h1 + 'pub/' + 'jplot_' + instrument + '_' + start + '_' + time_file_comb + 'UT_' + ftpsc + '_' + bflag[0] + '.png', bbox_inches=bbi, pad_inches=pi)

    if instrument == 'hi_2':

        fig, ax = plt.subplots(figsize=(10,5), sharex=True, sharey=True)

        plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 24)))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
        plt.gca().xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
        
        plt.gca().yaxis.set_minor_locator(MultipleLocator(2))

        ax.xaxis_date()

        ax.imshow(img_rescale_h2, cmap='gray', aspect='auto', interpolation='none', vmin=vmin_h2, vmax=vmax_h2, origin=orig, extent=[time_mdates_h2[0], time_mdates_h2[-1], elongations[2], elongations[3]])    

        ax.set_title(start + ' STEREO-' + ftpsc)
        

        plt.xlabel('Date (d/m/y)')
        plt.ylabel('Elongation (°)')

        if not os.path.exists(savepath_h2 + 'pub/'):
            os.makedirs(savepath_h2 + 'pub/')
            
        bbi = 'tight'
        pi = 0.5        

        plt.ylim(elongations[2], elongations[3])

        plt.savefig(savepath_h2 + 'pub/' + 'jplot_' + instrument + '_' + start + '_' + time_file_comb + 'UT_' + ftpsc + '_' + bflag[0] + '.png', bbox_inches=bbi, pad_inches=pi)

    savepath = path + 'jplot/' + ftpsc + '/' + bflag + '/' + instrument + '/' + str(start[0:4]) + '/params/'

    if not os.path.exists(savepath):
       os.makedirs(savepath)
    
    with open(savepath + 'jplot_' + instrument + '_' + start + '_' + time_file_comb + 'UT_' + ftpsc + '_' + bflag[0] + '_params.pkl', 'wb') as f:
       pickle.dump([datetime_h1[0], datetime_h1[-1], datetime_h2[0], datetime_h2[-1], elongations[0], elongations[1], elongations[2], elongations[3]], f)
        
#######################################################################################################################################

def hi_fix_pointing(header, point_path, ftpsc, ins, post_conj, silent_point):
    """
    Conversion of fix_pointing.pro for IDL. To read in the pointing information from the appropriate  pnt_HI??_yyyy-mm-dd_fix_mu_fov.fts file and update the
    supplied HI index with the best fit pointing information and optical parameters calculated by the minimisation
    process of Brown, Bewsher and Eyles (2008).

    @param header: Header of .fits file
    @param point_path: Path of pointing calibration files
    @param ftpsc: STEREO Spacecraft (A/B)
    @param ins: STEREO-HI instrument (HI-1/HI-2)
    @param post_conj: Is the spacecraft pre- or post conjunction (2014)
    @param silent_point: Run in silent mode
    """
    extra = 0
    silent_point=True
    hi_nominal = 1 #I changed this

    if ins == 'hi_1':
        ins = 'HI1'

    elif ins == 'hi_2':
        ins = 'HI2'

    else:
        print('Not a valid instrument, must be hi_1 or hi_2. Exiting...')
        sys.exit()

    try:
        header.rename_keyword('DATE-AVG', 'DATE_AVG')

    except ValueError:
        if not silent_point:
            print('Header information already corrected')

    hdr_date = header['DATE_AVG']
    hdr_date = hdr_date[0:10]

    rtmp = 5.

    point_file = 'pnt_' + ins + ftpsc + '_' + hdr_date + '_' + 'fix_mu_fov.fts'
    fle = point_path + point_file
    
    if os.path.isfile(fle):

        if not silent_point:
            print(('Reading {}...').format(point_file))
    
        hdul_point = fits.open(fle)

        for i in range(1, len(hdul_point)):
            extdate = hdul_point[i].header['extname']
            fledate = hdul_point[i].header['filename'][0:16]

            if (header['DATE_AVG'] == extdate) or (header['filename'][0:16] == fledate):
                ec = i
                break

        if (header['DATE_AVG'] == extdate) or (header['filename'][0:16] == fledate):

            stcravg = hdul_point[ec].header['ravg']
            stcnst1 = hdul_point[ec].header['nst1']

            if header['naxis1'] != 0:

                sumdif = np.round(header['cdelt1'] / hdul_point[ec].header['cdelt1'])

            else:
                sumdif = 1

            if stcnst1 < 20:

                if not silent_point:
                    print('Subfield presumed')
                    print('Using calibrated fixed instrument offsets')

                hi_calib_point(header, post_conj, hi_nominal)
                header['ravg'] = -894.

            else:

                if (stcravg < rtmp) & (stcravg > 0.):
                    header['crval1a'] = hdul_point[ec].header['crval1a']
                    header['crval2a'] = hdul_point[ec].header['crval2a']
                    header['pc1_1a'] = hdul_point[ec].header['pc1_1a']
                    header['pc1_2a'] = hdul_point[ec].header['pc1_2a']
                    header['pc2_1a'] = hdul_point[ec].header['pc2_1a']
                    header['pc2_2a'] = hdul_point[ec].header['pc2_2a']
                    header['cdelt1a'] = hdul_point[ec].header['cdelt1a'] * sumdif
                    header['cdelt2a'] = hdul_point[ec].header['cdelt2a'] * sumdif
                    header['pv2_1a'] = hdul_point[ec].header['pv2_1a']
                    header['crval1'] = hdul_point[ec].header['crval1']
                    header['crval2'] = hdul_point[ec].header['crval2']
                    header['pc1_1'] = hdul_point[ec].header['pc1_1']
                    header['pc1_2'] = hdul_point[ec].header['pc1_2']
                    header['pc2_1'] = hdul_point[ec].header['pc2_1']
                    header['pc2_2'] = hdul_point[ec].header['pc2_2']
                    header['cdelt1'] = hdul_point[ec].header['cdelt1'] * sumdif
                    header['cdelt2'] = hdul_point[ec].header['cdelt2'] * sumdif
                    header['pv2_1'] = hdul_point[ec].header['pv2_1']
                    header['xcen'] = hdul_point[ec].header['xcen']
                    header['ycen'] = hdul_point[ec].header['ycen']
                    header['crota'] = hdul_point[ec].header['crota']
                    header['ins_x0'] = hdul_point[ec].header['ins_x0']
                    header['ins_y0'] = hdul_point[ec].header['ins_y0']
                    header['ins_r0'] = hdul_point[ec].header['ins_r0']
                    header['ravg'] = hdul_point[ec].header['ravg']

                else:
                    if not silent_point:
                        print('R_avg does not meet criteria')
                        print('Using calibrated fixed instrument offsets')

                    hi_calib_point(header, post_conj, hi_nominal)
                    header['ravg'] = -883.

        else:
            if not silent_point:
                print(('No pointing calibration file found for file {}').format(point_file))
                print('Using calibrated fixed instrument offsets')

            hi_calib_point(header, post_conj, hi_nominal)
            header['ravg'] = -882.

    if not os.path.isfile(fle):
        if not silent_point:
            print(('No pointing calibration file found for file {}').format(point_file))
            print('Using calibrated fixed instrument offsets')

        hi_calib_point(header, post_conj, hi_nominal)
        header['ravg'] = -881.
#######################################################################################################################################

def hi_calib_point(header, post_conj, hi_nominal):
    """
    Conversion of hi_calib_pointing.pro for IDL.

    @param header: Header of .fits file
    @param post_conj: Is the spacecraft pre- or post conjunction (2014)
    @param hi_nominal: Retrieve nominal pointing values at launch (propagated to get_hi_params)
    """
    extra = 0

    roll = hi_calib_roll(header, 'gei', extra, post_conj, hi_nominal)

    header['pc1_1a'] = np.cos(roll * np.pi / 180.)
    header['pc1_2a'] = -np.sin(roll * np.pi / 180.)
    header['pc2_1a'] = np.sin(roll * np.pi / 180.)
    header['pc2_2a'] = np.cos(roll * np.pi / 180.)

    roll = hi_calib_roll(header, 'hpc', extra, post_conj, hi_nominal)

    header['crota'] = -roll
    header['pc1_1'] = np.cos(roll * np.pi / 180.)
    header['pc1_2'] = -np.sin(roll * np.pi / 180.)
    header['pc2_1'] = np.sin(roll * np.pi / 180.)
    header['pc2_2'] = np.cos(roll * np.pi / 180.)

    if 'summed' in header:

        naxis1 = 2048 / 2 ** (header['summed'] - 1)
        naxis2 = naxis1

    else:
        naxis1 = header['naxis1']
        naxis2 = header['naxis2']

    if naxis1 <= 0:
        naxis1 = 1024.

    if naxis2 <= 0:
        naxis2 = 1024.

    xv = [0.5 * naxis1, naxis1]
    yv = [0.5 * naxis2, 0.5 * naxis2]

    radec = fov2radec(xv, yv, header, 'gei', hi_nominal, extra)

    header['crval1a'] = radec[0, 0]
    header['crval2a'] = radec[1, 0]

    radec = fov2radec(xv, yv, header, 'hpc', hi_nominal, extra)
    header['crval1'] = -radec[0, 0]
    header['crval2'] = radec[1, 0]

    pitch_hi, offset_hi, roll_hi, mu, d = get_hi_params(header, extra, hi_nominal)
    header['pv2_1a'] = mu
    header['pv2_1'] = mu

    fp, fp_mm, plate = fparaxial(d, mu, header['naxis1'], header['naxis2'])

    if header['cunit1a'] == 'deg':
        xsize = plate / 3600.

    if header['cunit2a'] == 'deg':
        ysize = plate / 3600.

    if header['cunit1a'] == 'arcsec':
        xsize = plate

    if header['cunit2a'] == 'arcsec':
        ysize = plate

    header['cdelt1a'] = -xsize
    header['cdelt2a'] = ysize

    if header['cunit1'] == 'deg':
        xsize = plate / 3600.

    if header['cunit2'] == 'deg':
        ysize = plate / 3600.

    if header['cunit1'] == 'arcsec':
        xsize = plate

    if header['cunit2'] == 'arcsec':
        ysize = plate

    header['cdelt1'] = xsize
    header['cdelt2'] = ysize

    header['ins_x0'] = -offset_hi
    header['ins_y0'] = pitch_hi
    header['ins_r0'] = -roll_hi

#######################################################################################################################################

def hi_calib_roll(header, system, extra, post_conj, hi_nominal):
    """
    Conversion of hi_calib_roll.pro for IDL. Calculate the total roll angle of the HI image including
    contributions from the pitch and roll of the spacecraft. The total HI roll is a non-straighforward combination
    of the individual rolls of the s/c and HI, along with the pitch of the s/c and the offsets of HI. This
    routine calculates the total roll by taking 2 test points in the HI fov, transforms them to the
    appropriate frame of reference (given by the system keyword) and calculates the angle they make in this frame.

    @param header: Header of .fits file
    @param system: Which coordinate system to work in 'hpc' or 'gei'
    @param extra: This keyword is pointless, but was present in the original IDL code
    @param post_conj: Is the spacecraft pre- or post conjunction (2014)
    @param hi_nominal: Retrieve nominal pointing values at launch (propagated to get_hi_params)
    @return: Total roll of the spacecraft
    """
    if 'summed' in header:

        naxis1 = 2048 / 2 ** (header['summed'] - 1)
        naxis2 = naxis1

    else:

        naxis1 = header['naxis1']
        naxis2 = header['naxis2']

    if naxis1 <= 0:
        naxis1 = 1024.

    if naxis2 <= 0:
        naxis2 = 1024.

    cpix = np.array([naxis1, naxis2]) / 2. - 0.5
    xv = cpix[0] + [0., 512.]
    yv = np.full_like(cpix, cpix[1])

    xy = fov2pos(xv, yv, header, system, hi_nominal, extra)

    z = xy[2, 1] - xy[2, 0]
    x = -(xy[1, 1] - xy[1, 0])
    y = xy[0, 1] - xy[0, 0]

    tx = xy[0, 0]
    ty = xy[1, 0]
    tz = 0.0

    a = np.sqrt(tx ** 2 + ty ** 2)
    b = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    ab = x * tx + y * ty

    val = np.nanmin([np.nanmax([ab / (a * b), -1.0]), 1.0])

    if z >= 0.0:
        oroll = np.arccos(val) * 180. / np.pi

    else:
        oroll = -np.arccos(val) * 180. / np.pi

    if post_conj:
        oroll = oroll - 180.

    return oroll


#######################################################################################################################################

def fov2pos(xv, yv, header, system, hi_nominal, extra):
    """
    Conversion of fov2pos.pro for IDL. To convert HI pixel positions to solar plane of sky
    coordinates in units of AU (actually, not quite, 1 is defined as the distance from the S/C to the Sun) and
    the Sun at (0,0). HI pixel position is converted to general AZP, then to Cartesian coordinates in the HI frame of
    reference. This is then rotated to be in the S/C frame and finally to the end reference frame. The final frame
    is a left-handed Cartesian system with x into the screen (towards the reference point), and z pointing up.

    @param xv: Array of x-pixel positions to be converted
    @param yv: Array of y-pixel positions to be converted
    @param header: Header of .fits file
    @param system: Which coordinate system to work in 'hpc' or 'gei'
    @param hi_nominal: Retrieve nominal pointing values at launch (propagated to get_hi_params)
    @param extra: This keyword is pointless, but was present in the original IDL code
    @return: An array of (x,y,z,w) quadruplets, where x,y,z have the meaning described above and w is a scale
    factor (see '3D computer graphics' by Alan Watt for further details). The wcolumn can be discounted
    for almost all user applications.
    """
    if system == 'hpc':

        yaw = header['sc_yaw']
        pitch = header['sc_pitch']
        roll = -header['sc_roll']

    else:

        yaw = header['sc_yawa']
        pitch = header['sc_pita']
        roll = header['sc_rolla']

    ccdosx = 0.
    ccdosy = 0.

    naxis = np.array([header['naxis1'], header['naxis2']])

    if naxis.any() == 0:
        naxis[:] = 1024

    pmult = naxis / 2.0 - 0.5

    pitch_hi, offset_hi, roll_hi, mu, d = get_hi_params(header, extra, hi_nominal)

    ang = (90. - 0.5 * d) * np.pi / 180.
    rng = (1.0 + mu) * np.cos(ang) / (np.sin(ang) + mu)

    vfov = np.array([xv, yv])
    nst = len(vfov[0])

    vv4 = np.zeros((4, nst))

    vv4[0][:] = ((vfov[0] - ccdosx) / pmult[0] - 1.0) * rng
    vv4[1][:] = ((vfov[1] - ccdosy) / pmult[1] - 1.0) * rng
    vv4[2][:] = 1.0
    vv4[3][:] = 1.0

    vv3 = azp2cart(vv4, mu)
    vv3[2, :] = 1.0

    vv2 = hi2sc(vv3, roll_hi, pitch_hi, offset_hi)
    vv = sc2cart(vv2, roll, pitch, yaw)

    return vv


#######################################################################################################################################

def get_hi_params(header, extra, hi_nominal):
    """
    Conversion of get_hi_params.pro for IDL. To detect which HI telescope is being used and return the
    instrument offsets relative to the spacecraft along with the mu parameter and the fov in degrees.
    As 'best pointings' may change as further calibration is done, it was thought more useful if there was one
    central routine to provide this data, rather than having to make the same changes in many different
    codes. Note, if you set one of the output variables to some value, then it will retain that value.

    @param header: Header of .fits file
    @param extra: This keyword is pointless, but was present in the original IDL code
    @param hi_nominal: Retrieve nominal pointing values at launch (propagated to get_hi_params)
    @return: HI yaw offset, HI pitch offfset, HI roll, HI distortion parameter, HI fov
    """
    if hi_nominal:

        if ((header['obsrvtry'] == 'STEREO_A') and (header['detector'] == 'HI1')):
            pitch_hi = 0.0
            offset_hi = -13.98
            roll_hi = 0.0
            mu = 0.16677
            d = 20.2663

        if ((header['obsrvtry'] == 'STEREO_A') and (header['detector'] == 'HI2')):
            offset_hi = -53.68
            pitch_hi = 0.
            roll_hi = 0.
            mu = 0.83329
            d = 70.8002

        if ((header['obsrvtry'] == 'STEREO_B') and (header['detector'] == 'HI1')):
            offset_hi = 13.98
            pitch_hi = 0.
            roll_hi = 0.
            mu = 0.10001
            d = 20.2201

        if ((header['obsrvtry'] == 'STEREO_B') and (header['detector'] == 'HI2')):
            offset_hi = 53.68
            pitch_hi = 0.
            roll_hi = 0.
            mu = 0.65062
            d = 69.8352

    else:

        if ((header['obsrvtry'] == 'STEREO_A') and (header['detector'] == 'HI1')):
            pitch_hi = 0.1159
            offset_hi = -14.0037
            roll_hi = 1.0215
            mu = 0.102422
            d = 20.27528

        if ((header['obsrvtry'] == 'STEREO_A') and (header['detector'] == 'HI2')):
            offset_hi = -53.4075
            pitch_hi = 0.0662
            roll_hi = 0.1175
            mu = 0.785486
            d = 70.73507

        if ((header['obsrvtry'] == 'STEREO_B') and (header['detector'] == 'HI1')):
            offset_hi = 14.10
            pitch_hi = 0.022
            roll_hi = 0.37
            mu = 0.09509
            d = 20.23791

        if ((header['obsrvtry'] == 'STEREO_B') and (header['detector'] == 'HI2')):
            offset_hi = 53.690
            pitch_hi = 0.213
            roll_hi = -0.052
            mu = 0.68886
            d = 70.20152

    return pitch_hi, offset_hi, roll_hi, mu, d


#######################################################################################################################################

def azp2cart(vec, mu):
    """
    Conversion of azp2cart.pro for IDL. To convert points seen with an AZP projection with
    parameter, mu to a Cartesian frame. Note, this is a low level code, and would usually not be called directly.
    For details, see "Coordinate systems for solar image data", W.T. Thompson, A&A

    @param vec: Array of vector postions to transform
    @param mu: HI distortion parameter
    @return: An array of transformed vector positions
    """
    nstars = np.shape(vec)[1]
    vout = vec.copy()

    for i in range(nstars):

        rth = np.sqrt(vec[0, i] ** 2 + vec[1, i] ** 2)
        rho = rth / (mu + 1.0)
        cc = np.sqrt(1.0 + rho ** 2)
        th = np.arccos(1.0 / cc) + np.arcsin(mu * rho / cc)
        zz = np.cos(th)
        rr = np.sin(th)

        if rth < 1.0e-6:
            vout[0:1, i] = rr * vec[0:1, i]
        else:
            vout[0:1, i] = rr * vec[0:1, i] / rth

        vout[2, i] = zz

    return vout


#######################################################################################################################################

def hi2sc(vec, roll_hi_deg, pitch_hi_deg, offset_hi_deg):
    """
    Conversion of hi2sec.pro for IDL. To transform the given position from the HI frame of
    reference to the spacecraft frame of reference. Note, this is a low level code, and would usually not be
    called directly. For the transformation we use 4x4 transformation
    matrices discussed in e.g. '3D computer graphics' by Alan Watt

    @param vec: Array of vector postions to transform
    @param roll_hi_deg: HI roll angle relative to spacecraft (in degrees)
    @param pitch_hi_deg: HI pitch angle relative to spacecraft (in degrees)
    @param offset_hi_deg: HI yaw angle relative to spacecraft (in degrees)
    @return: An array of transformed vector positions
    """
    npts = len(vec[0, :])

    theta = (90 - pitch_hi_deg) * np.pi / 180.
    phi = offset_hi_deg * np.pi / 180.
    roll = roll_hi_deg * np.pi / 180.

    normz = np.sin(theta) * np.cos(phi)
    normx = np.sin(theta) * np.sin(phi)
    normy = np.cos(theta)

    vdx = 0.
    vdy = 1.
    vdz = 0.

    vd_norm = vdx * normx + vdy * normy + vdz * normz

    vxtmp = vdx - vd_norm * normx
    vytmp = vdy - vd_norm * normy
    vztmp = vdz - vd_norm * normz

    ndiv = np.sqrt(vxtmp ** 2 + vytmp ** 2 + vztmp ** 2)
    vx = vxtmp / ndiv
    vy = vytmp / ndiv
    vz = vztmp / ndiv

    ux = -(normy * vz - normz * vy)
    uy = -(normz * vx - normx * vz)
    uz = -(normx * vy - normy * vx)

    cx = 0.
    cy = 0.
    cz = 0.

    tmat = np.array([[1., 0., 0., -cx], [0., 1., 0., -cy], [0., 0., 1., -cz], [0., 0., 0., 1.]])
    rmat = np.array([[ux, uy, uz, 0.], [vx, vy, vz, 0.], [normx, normy, normz, 0.], [0., 0., 0., 1.]])

    rollmat = np.array(
        [[np.cos(roll), -np.sin(roll), 0., 0.], [np.sin(roll), np.cos(roll), 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    tview = rollmat @ (rmat @ tmat)

    itview = np.linalg.inv(tview)

    vout = np.zeros((4, npts))

    for i in range(npts):
        vout[:, i] = np.transpose(itview @ np.transpose(vec[:, i]))

    return vout


#######################################################################################################################################

def sc2cart(vec, roll_deg, pitch_deg, yaw_deg):
    """
    Conversion of sc2cart.pro for IDL. To convert spacecraft pointing to Cartesian points in a known reference frame.
    Note, this is a low level code, and would usually not be called directly. For the transformation we use 4x4 transformation
    matrices discussed in e.g. '3D computer graphics' by Alan Watt.

    @param vec: An array of vector postions to transform
    @param roll_deg: Spacecraft roll angle (in degrees)
    @param pitch_deg: Spacecraft pitch angle (in degrees)
    @param yaw_deg: Spacecraft yaw angle (in degrees)
    @return: An array of transformed vector positions
    """
    npts = len(vec[0, :])

    theta = (90 - pitch_deg) * np.pi / 180.
    phi = yaw_deg * np.pi / 180.
    roll = roll_deg * np.pi / 180.

    normx = np.sin(theta) * np.cos(phi)
    normy = np.sin(theta) * np.sin(phi)
    normz = np.cos(theta)

    vdx = 0.
    vdy = 0.
    vdz = 1.

    vd_norm = vdx * normx + vdy * normy + vdz * normz

    vxtmp = vdx - vd_norm * normx
    vytmp = vdy - vd_norm * normy
    vztmp = vdz - vd_norm * normz

    ndiv = np.sqrt(vxtmp ** 2 + vytmp ** 2 + vztmp ** 2)
    vx = vxtmp / ndiv
    vy = vytmp / ndiv
    vz = vztmp / ndiv

    ux = normy * vz - normz * vy
    uy = normz * vx - normx * vz
    uz = normx * vy - normy * vx

    cx = 0.
    cy = 0.
    cz = 0.

    tmat = np.array([[1., 0., 0., -cx], [0., 1., 0., -cy], [0., 0., 1., -cz], [0., 0., 0., 1.]])
    rmat = np.array([[ux, uy, uz, 0.], [vx, vy, vz, 0.], [normx, normy, normz, 0.], [0., 0., 0., 1.]])

    rollmat = np.array(
        [[np.cos(roll), -np.sin(roll), 0., 0.], [np.sin(roll), np.cos(roll), 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    tview = rollmat @ (rmat @ tmat)

    itview = np.linalg.inv(tview)

    vout = np.zeros((4, npts))

    for i in range(npts):
        vout[:, i] = np.transpose(itview @ np.transpose(vec[:, i]))

    return vout


#######################################################################################################################################

def fov2radec(xv, yv, header, system, hi_nominal, extra):
    """
    Conversion of fov2radec for IDL. To convert HI pixel positions to RA-Dec pairs.
    HI pixel positions are converted into a general AZP form, which is converted to cartesian coordintes.
    These are then rotated from HI pointing, to S/C pointing then finally aligned with the RA-Dec frame.
    The Cartesian coordinates are finally converted to RA-Dec.

    @param xv: An array of x pixel positions
    @param yv: An array of y pixel positions
    @param header: Header of .fits file
    @param system: Which coordinate system to work in 'hpc' or 'gei'
    @param hi_nominal: Retrieve nominal pointing values at launch (propagated to get_hi_params)
    @param extra: This keyword is pointless, but was present in the original IDL code
    @return: An array of transformed vector positions
    """
    if system == 'gei':
        yaw = header['sc_yawa']
        pitch = header['sc_pita']
        roll = header['sc_rolla']

    else:
        yaw = header['sc_yaw']
        pitch = header['sc_pitch']
        roll = -header['sc_roll']

    ccdosx = 0.
    ccdosy = 0.
    pmult = header['naxis1'] / 2.0

    pitch_hi, offset_hi, roll_hi, mu, d = get_hi_params(header, extra, hi_nominal)

    ang = (90. - 0.5 * d) * np.pi / 180.
    rng = (1.0 + mu) * np.cos(ang) / (np.sin(ang) + mu)

    vfov = np.array([xv, yv])

    nst = len(vfov[0])

    vv4 = np.zeros((4, nst))

    vv4[0, :] = ((vfov[0] - ccdosx) / pmult - 1.0) * rng
    vv4[1, :] = ((vfov[1] - ccdosy) / pmult - 1.0) * rng
    vv4[3, :] = 1.0

    vv3 = azp2cart(vv4, mu)
    vv2 = hi2sc(vv3, roll_hi, pitch_hi, offset_hi)

    vv = sc2cart(vv2, roll, pitch, yaw)

    radec = np.zeros((2, nst))

    rd = 180 / np.pi

    for i in range(nst):

        th = np.arccos(vv[2, i])
        radec[1, i] = 90. - th * rd

        cphi = vv[0, i] / np.sin(th)
        sphi = vv[1, i] / np.sin(th)

        th1 = np.arccos(cphi) * rd
        th2 = np.arcsin(sphi) * rd

        if (th2 > 0):
            ra = th1

        else:
            ra = -th1

        radec[0, i] = ra

    return radec


#######################################################################################################################################

def fparaxial(fov, mu, naxis1, naxis2):
    """
    Conversion of fparaxial.pro for IDL. Calculate paraxial platescale of HI.
    This routine calculates the paraxial platescale from the calibrated fov and distortion parameter.

    @param fov: Calibrated field of view
    @param mu: HI distortion parameter
    @param naxis1: NAXIS1 from .fits header
    @param naxis2: NAXIS2 from .fits header
    @return: Paraxial platescale in mm
    """
    theta = (90. - (fov / 2.)) * np.pi / 180.

    tmp1 = (np.sin(theta) + mu) / ((1 + mu) * np.cos(theta))

    fp = (naxis1 / 2.) * tmp1

    widthmm = 0.0135 * 2048.
    fp_mm = widthmm * tmp1 / 2.

    plate = (0.0135 / fp_mm) * (180 / np.pi) * 3600.

    plate = plate * 2048. / naxis1

    return fp, fp_mm, plate


#######################################################################################################################################

def data_reduction(start, path, datpath, ftpsc, instrument, bflag, silent, save_path, path_flg):
    """
    Data reduciton routine calling upon various functions converted from IDL. The default correction procedure involves;
    correcting for shutterless exposure, multiplying by the flatfield, subtracting the bias,
    accounting for seb-ip binning, and desaturating the data. This process implicity performs conversion to DN/s.
    Procedure also adds best calibrated pointing into HI header.

    @param start: Start date of Jplot
    @param path: The path where all reduced images, running difference images and J-Maps are saved
    @param datpath: Path to STEREO-HI calibration files
    @param ftpsc: Spacecraft (A/B)
    @param instrument: STEREO-HI instrument (HI-1/HI-2)
    @param bflag: Science or beacon data
    @param silent: Run in silent mode
    @param save_path: Path pointing towards downloaded STEREO .fits files
    @param path_flg: Specifies path for downloaded files, depending on wether science or beacon data is used
    """
    if not silent:
        print('----------------')
        print('DATA REDUCTION')
        print('----------------')

    date = datetime.datetime.strptime(start, '%Y%m%d')

    if ftpsc == 'A':
        sc = 'ahead'

    if ftpsc == 'B':
        sc = 'behind'

    savepath = path + 'reduced/data/' + ftpsc + '/' + start + '/' + bflag + '/'
    calpath = datpath + 'calibration/'
    pointpath = datpath + 'data' + '/' + 'hi/'

    f = 0

    if instrument == 'hi1hi2':
        instrument = ['hi_1', 'hi_2']

    if instrument == 'hi_1':
        instrument = ['hi_1']

    if instrument == 'hi_2':
        instrument = ['hi_2']
        
    for ins in instrument:
        fitsfiles = []

        if bflag == 'science':
            
            for file in sorted(glob.glob(save_path + 'stereo' + sc[0] + '/secchi/' + path_flg + '/img/'+ins+'/' + str(start) + '/*s4*.fts')):
                fitsfiles.append(file)
    
        if bflag == 'beacon':
    
            for file in sorted(glob.glob(save_path + 'stereo' + sc[0] + '/' + path_flg + '/secchi/' + '/img/'+ins+'/' + str(start) + '/*s7*.fts')):
                fitsfiles.append(file)
            
        if not silent:
            print('----------------------------------------')
            print('Starting data reduction for', ins, '...')
            print('----------------------------------------')

        # correct for on-board sebip modifications to image (division by 2, 4, 16 etc.)
        # calls function scc_sebip

        hdul = [fits.open(fitsfiles[i]) for i in range(len(fitsfiles))]
        n_images = [hdul[i][0].header['n_images'] for i in range(len(fitsfiles))]

        if bflag == 'science':
            if ins == 'hi_1':
                norm_img = 30
            else:
                norm_img = 99

        if bflag == 'beacon':
            norm_img = 1
            
        indices = []
        bad_img = []
        
        if not all(val == norm_img for val in n_images):

            bad_ind = [i for i in range(len(n_images)) if n_images[i] != norm_img]
            good_ind = [i for i in range(len(n_images)) if n_images[i] == norm_img]
            bad_img.extend(bad_ind)
            indices.extend(good_ind)

        else:
            indices = [i for i in range(len(fitsfiles))]

        crval1_test = [int(np.sign(hdul[i][0].header['crval1'])) for i in indices]
        
        if len(set(crval1_test)) > 1:

            common_crval = Counter(crval1_test)
            com_val, count = common_crval.most_common()[0]
            
            bad_ind = [i for i in range(len(crval1_test)) if crval1_test[i] != com_val]
            
            for i in sorted(bad_ind, reverse=True):
                bad_img.extend([indices[i]])
                del indices[i]   
                
            if len(bad_ind) >= len(indices):
                print('Too many corrupted images - can\'t determine correct CRVAL1. Exiting...')
                sys.exit()

        if bflag == 'science':
            #Must find way to do this for beacon also
            datamin_test = [hdul[i][0].header['DATAMIN'] for i in indices]
            
            if not all(val == norm_img for val in datamin_test):
                
                bad_ind = [i for i in range(len(datamin_test)) if datamin_test[i] != norm_img]

                for i in sorted(bad_ind, reverse=True):
                    bad_img.extend([indices[i]])
                    del indices[i]

        if bflag == 'beacon':
            test_data = np.array([hdul[i][0].data for i in indices])
            test_data = np.where(test_data == 0, np.nan, test_data)
            
            bad_ind = [i for i in range(len(test_data)) if np.isnan(test_data[i]).all() == True]
            
            for i in sorted(bad_ind, reverse=True):
                bad_img.extend([indices[i]])
                del indices[i]
                    
        missing_ind = np.array([hdul[i][0].header['NMISSING'] for i in indices])
            
        bad_ind = [i for i in range(len(missing_ind)) if missing_ind[i] > 0]
        for i in sorted(bad_ind, reverse=True):
            bad_img.extend([indices[i]])
            del indices[i]     
        
        clean_data = []
        clean_header = []
        
        for i in range(len(fitsfiles)):
            if i in indices:
                clean_data.append(hdul[i][0].data)
                clean_header.append(hdul[i][0].header)
                hdul[i].close()
            else:
                hdul[i].close()
        
        clean_data = np.array(clean_data)
                
        crval1 = [clean_header[i]['crval1'] for i in range(len(clean_header))]

        if ftpsc == 'A':    
            post_conj = [int(np.sign(crval1[i])) for i in range(len(crval1))]
    
        if ftpsc == 'B':    
            post_conj = [int(-1*np.sign(crval1[i])) for i in range(len(crval1))]
            
        if len(set(post_conj)) == 1:

            post_conj = post_conj[0]
    
            if post_conj == -1:
                post_conj = False
            if post_conj == 1:
                post_conj = True

        else:
            print('Corrupted CRVAL1 in header. Exiting...')
            sys.exit()
            
        if not silent:
            print('Correcting for binning...')

        name = np.array([fitsfiles[i].rpartition('/')[2] for i in indices])
        dateavg = [clean_header[i]['date-avg'] for i in range(len(clean_header))]

        timeavg = [datetime.datetime.strptime(dateavg[i], '%Y-%m-%dT%H:%M:%S.%f') for i in range(len(dateavg))]
        
        data_trim = np.array([scc_img_trim(clean_data[i], clean_header[i]) for i in range(len(clean_data))])
        data_sebip = [scc_sebip(data_trim[i], clean_header[i], True) for i in range(len(data_trim))]

        if not silent:
            print('Getting bias...')

        # maps are created from corrected data
        # header is saved into separate list

        biasmean = [get_biasmean(clean_header[i]) for i in range(len(clean_header))]
        biasmean = np.array(biasmean)

        for i in range(len(biasmean)):

            if biasmean[i] != 0:
                clean_header[i].header['OFFSETCR'] = biasmean[i]

        data_sebip = data_sebip - biasmean[:, None, None]

        if not silent:
            print('Removing saturated pixels...')

        # saturated pixels are removed
        # calls function hi_remove_saturation from functions.py

        data_desat = np.array([hi_remove_saturation(data_sebip[i, :, :], clean_header[i]) for i in range(len(data_sebip))])
        # data_desat = data_sebip.copy()

        if not silent:
            print('Desmearing image...')

        dstart1 = [clean_header[i]['dstart1'] for i in range(len(clean_header))]
        dstart2 = [clean_header[i]['dstart2'] for i in range(len(clean_header))]
        dstop1 = [clean_header[i]['dstop1'] for i in range(len(clean_header))]
        dstop2 = [clean_header[i]['dstop2'] for i in range(len(clean_header))]

        naxis1 = [clean_header[i]['naxis1'] for i in range(len(clean_header))]
        naxis2 = [clean_header[i]['naxis2'] for i in range(len(clean_header))]

        exptime = [clean_header[i]['exptime'] for i in range(len(clean_header))]
        n_images = [clean_header[i]['n_images'] for i in range(len(clean_header))]
        cleartim = [clean_header[i]['cleartim'] for i in range(len(clean_header))]
        ro_delay = [clean_header[i]['ro_delay'] for i in range(len(clean_header))]
        ipsum = [clean_header[i]['ipsum'] for i in range(len(clean_header))]

        rectify = [clean_header[i]['rectify'] for i in range(len(clean_header))]
        obsrvtry = [clean_header[i]['obsrvtry'] for i in range(len(clean_header))]

        for i in range(len(obsrvtry)):

            if obsrvtry[i] == 'STEREO_A':
                obsrvtry[i] = True

            else:
                obsrvtry[i] = False

        line_ro = [clean_header[i]['line_ro'] for i in range(len(clean_header))]
        line_clr = [clean_header[i]['line_clr'] for i in range(len(clean_header))]

        header_int = np.array(
            [[dstart1[i], dstart2[i], dstop1[i], dstop2[i], naxis1[i], naxis2[i], n_images[i], post_conj] for i in
             range(len(dstart1))])

        header_flt = np.array(
            [[exptime[i], cleartim[i], ro_delay[i], ipsum[i], line_ro[i], line_clr[i]] for i in range(len(exptime))])

        header_str = np.array([[rectify[i], obsrvtry[i]] for i in range(len(rectify))])

        data_desm = [hi_desmear(data_desat[i, :, :], header_int[i], header_flt[i], header_str[i]) for i in
                     range(len(data_desat))]

        data_desm = np.array(data_desm)

        if not silent:
            print('Calibrating image...')

        ipkeep = [clean_header[k]['IPSUM'] for k in range(len(clean_header))]

        calimg = [get_calimg(ins, ftpsc, clean_header[k], calpath, post_conj, silent) for k in range(len(clean_header))]
        calimg = np.array(calimg)

        calfac = [get_calfac(clean_header[k], timeavg[k]) for k in range(len(clean_header))]
        calfac = np.array(calfac)

        diffuse = [scc_hi_diffuse(clean_header[k], ipkeep[k]) for k in range(len(clean_header))]
        diffuse = np.array(diffuse)

        data_red = calimg * data_desm * calfac[:, None, None] * diffuse

        if not silent:
            print('Calibrating pointing...')

        for i in range(len(clean_header)):
            hi_fix_pointing(clean_header[i], pointpath, ftpsc, ins, post_conj, silent_point=True)

        if not silent:
            print('Saving .fts files...')

        if not os.path.exists(savepath + ins + '/'):
            os.makedirs(savepath + ins + '/')

        else:
        
            oldfiles = glob.glob(os.path.join(savepath + ins + '/', "*.fts"))
            for fil in oldfiles:
                os.remove(fil)
                
        for i in range(len(clean_header)):

            if bflag == 'science':

                newname = datetime.datetime.strptime(clean_header[i]['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S') + '_1b' + ins.replace('i_', '') + ftpsc + '.fts'

            if bflag == 'beacon':
                newname = datetime.datetime.strptime(clean_header[i]['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S') + '_17' + ins.replace('i_', '') + ftpsc + '.fts'

            fits.writeto(savepath + ins + '/' + newname, data_red[i, :, :], clean_header[i], output_verify='silentfix', overwrite=True)

        f = f + 1
            
#######################################################################################################################################

def new_cmap(basemap, up_lim, low_lim):
    """
    Create a new colormap by specifying a pre-existing map and changing its upper and lower limits.

    @param basemap: Pre-existing matplotlib colormap object
    @param up_lim: New upper limit
    @param low_lim: New lower limit
    @return: Modified colormap with new limits
    """
    basemap = cm.get_cmap(basemap, 256)

    newcolors = basemap(np.linspace(0, 1, 256))

    newcolors[0:35, :] = newcolors[0:35, :] - [up_lim, up_lim, up_lim, 0]
    newcolors[35:, :] = newcolors[35:, :] + [low_lim, low_lim, low_lim, 0]
    newcolors = np.where(newcolors < 0, 0, newcolors)
    newcolors = np.where(newcolors > 1, 1, newcolors)

    newcmp = ListedColormap(newcolors)

    return newcmp


#######################################################################################################################################

def clean_hi(data, save_path, date, name, my_cmap):
    """
    Create clean looking .pngs of reduced HI images.

    @param data: HI image data
    @param date: Date for which to produce .pngs
    @param name: Names under which to save .png files
    @param my_cmap: New color map to use on HI data
    """
    newcmp = new_cmap(my_cmap, 0.08, 0)

    med = medfilter(data, 25)
    filt_img = np.where(np.abs(data) > 1.5 * med, med, data)

    img_med = np.nanmedian(filt_img)

    for i in range(len(filt_img)):

        if np.sum(filt_img[:, i]) / len(filt_img) > 50 * img_med:

            cnt = 0

            for j in range(len(filt_img)):
                if filt_img[j, i] > 40 * img_med:
                    cnt = cnt + 1

            if cnt > 400:
                filt_img[:, i] = img_med

    # filt_img = medfilter(filt_img, 5)

    fig, ax = plt.subplots(figsize=(1.024, 1.024), dpi=1000, frameon=False)
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    ax.axis('off')

    ax.imshow(filt_img, cmap=newcmp, aspect='auto', origin='lower', vmin=-5e-14, vmax=5e-13)

    savepath = save_path + 'reduced/pngs/' + date + '/'

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    print('Saving...')
    plt.savefig(savepath + name + '.png')
    plt.close()


#######################################################################################################################################

@numba.njit(parallel=True)
def medfilter(img, kernel):
    """
    Implementation of a median filter.

    @param img: Image data
    @param kernel: Kernel size for median filter
    @return: Image with median filter applied
    """
    imgshape = np.shape(img)
    medimg = np.zeros(imgshape)

    kern = np.round(kernel / 2) - 1

    for i in range(len(img)):
        for j in range(len(img)):

            x_b = int(j - kern)
            x_a = int(j + kern)
            y_b = int(i - kern)
            y_a = int(i + kern)

            if x_b < 0:
                x_b = 0

            if x_a > len(img):
                x_a = len(img)

            if y_b < 0:
                y_b = 0

            if y_a > len(img):
                y_a = len(img)

            medimg[i, j] = np.nanmedian(img[y_b:y_a, x_b:x_a])

    return medimg


#######################################################################################################################################

def reduced_pngs(start, path, bflag, silent):
    """
    Create clean looking .pngs of reduced HI images. Calls clean_hi function.

    @param start: Date for which to produce .pngs
    @param path: Path to Events folder
    @param bflag: Science or beacon data
    @param silent: Run in silent mode
    """
    redpath = path + 'reduced/data/' + start + '/' + bflag + '/hi_1/'

    files = []

    for file in sorted(glob.glob(redpath + '*.fts')):
        files.append(file)

    hdul = [fits.open(files[i]) for i in range(len(files))]

    data = np.array([hdul[i][0].data for i in range(len(files))])

    names = [files[i].split('/')[-1] for i in range(len(files))]
    names = [names[i].split('.')[0] for i in range(len(files))]

    min_arr = np.quantile(data, 0.05, axis=0)

    data_nobg = data - min_arr

    if not silent:
        print(start)
    for i in range(len(data_nobg)):
        clean_hi(data_nobg[i], path, start, names[i], 'afmhot')
