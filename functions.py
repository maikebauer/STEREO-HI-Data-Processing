import numpy as np
import numba
from astropy.time import Time
from astropy.io import fits
import glob
import matplotlib.dates as mdates
from matplotlib import cm
from matplotlib.colors import ListedColormap
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
from urllib3.exceptions import InsecureRequestWarning
from urllib3 import disable_warnings
from requests.adapters import HTTPAdapter, Retry
import psutil
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import traceback
import logging
from skimage import exposure
from matplotlib.ticker import MultipleLocator
from collections import Counter
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.coordinates import Helioprojective
from sunpy.coordinates.ephemeris import get_horizons_coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from skimage.transform import resize, rotate
import yaml
from skimage import morphology
from scipy.ndimage import zoom
from numpy.lib.stride_tricks import sliding_window_view
import time
import multiprocessing as mp
from itertools import repeat
warnings.filterwarnings("ignore")

#######################################################################################################################################

def fix_secchi_hdr(hdr):
    
    # Initialize default values
    int_val = 0
    lon_val = 0
    flt_val = 0.0
    fltd_val = 0.0
    uint8_val = 0
    uint16_val = 0
    uint32_val = 0
    int8_val = 0
    int16_val = 0
    int32_val = 0
    str_val = ''
    sta_val = [str_val] * 20

    secchi_hdr = {
        # SIMPLE: 'T', 
        'EXTEND': 'F', 

        # Most-used keywords
        'BITPIX': int_val, 
        'NAXIS': int_val, 
        'NAXIS1': int_val, 
        'NAXIS2': int_val,
        
        'DATE_OBS': str_val, 
        'TIME_OBS': str_val, 
        'FILEORIG': str_val, 
        'SEB_PROG': str_val, 
        'SYNC': str_val, 
        'SPWX': 'F', 
        'EXPCMD': -1.0, 
        'EXPTIME': -1.0, 
        'DSTART1': int_val, 
        'DSTOP1': int_val, 
        'DSTART2': int_val, 
        'DSTOP2': int_val,
        'P1COL': int16_val, 
        'P2COL': int16_val, 
        'P1ROW': int16_val, 
        'P2ROW': int16_val, 
        'R1COL': int16_val, 
        'R2COL': int16_val, 
        'R1ROW': int16_val, 
        'R2ROW': int16_val, 
        'RECTIFY': 'F', 
        'RECTROTA': int_val, 
        'LEDCOLOR': str_val, 
        'LEDPULSE': uint32_val, 
        'OFFSET': 9999, 
        'BIASMEAN': flt_val, 
        'BIASSDEV': -1.0, 
        'GAINCMD': -1, 
        'GAINMODE': str_val, 
        'SUMMED': flt_val, 
        'SUMROW': 1, 
        'SUMCOL': 1, 
        'CEB_T': 999, 
        'TEMP_CCD': 9999.0, 
        'POLAR': -1.0, 
        'ENCODERP': -1, 
        'WAVELNTH': int_val, 
        'ENCODERQ': -1, 
        'FILTER': str_val, 
        'ENCODERF': -1, 
        'FPS_ON': str_val, 
        'OBS_PROG': 'schedule', 
        'DOORSTAT': -1, 
        'SHUTTDIR': str_val, 
        'READ_TBL': -1, 
        'CLR_TBL': -1, 
        'READFILE': str_val, 
        'DATE_CLR': str_val, 
        'DATE_RO': str_val, 
        'READTIME': -1.0, 
        'CLEARTIM': fltd_val, 
        'IP_TIME': -1, 
        'COMPRSSN': int_val, 
        'COMPFACT': flt_val, 
        'NMISSING': -1.0, 
        'MISSLIST': str_val, 
        'SETUPTBL': str_val, 
        'EXPOSTBL': str_val, 
        'MASK_TBL': str_val, 
        'IP_TBL': str_val, 
        'COMMENT': sta_val, 
        'HISTORY': sta_val, 
        'DIV2CORR': 'F', 
        'DISTCORR': 'F', 

        # Less-used keywords
        'TEMPAFT1': 9999.0, 
        'TEMPAFT2': 9999.0, 
        'TEMPMID1': 9999.0, 
        'TEMPMID2': 9999.0, 
        'TEMPFWD1': 9999.0, 
        'TEMPFWD2': 9999.0, 
        'TEMPTHRM': 9999.0, 
        'TEMP_CEB': 9999.0, 
        'ORIGIN': str_val, 
        'DETECTOR': str_val, 
        'IMGCTR': uint16_val, 
        'TIMGCTR': uint16_val, 
        'OBJECT': str_val, 
        'FILENAME': str_val, 
        'DATE': str_val, 
        'INSTRUME': 'SECCHI', 
        'OBSRVTRY': str_val, 
        'TELESCOP': 'STEREO',
        "DATE__OBS": " ",
        "TELESCOP": "STEREO",
        "WAVEFILE": str_val,   # name of waveform table file used by fsw
        "CCDSUM": flt_val,   # (sumrow + sumcol) / 2.0
        "IPSUM": flt_val,    # (sebxsum + sebysum) / 2.0
        "DATE_CMD": str_val,   # originally scheduled observation time
        "DATE_AVG": str_val,   # date of midpoint of the exposure(s) (UTC standard)
        "DATE_END": str_val,   # Date/time of end of (last) exposure
        "OBT_TIME": flt_val, # value of STEREO on-board-time since epoch ???
        "APID": int_val,       # application identifier / how downlinked
        "OBS_ID": int_val,     # observing sequence ID from planniing tool
        "OBSSETID": int_val,   # observing set (=campaign) ID from planning tool
        "IP_PROG0": int_val,   # description of onboard image processing sequence used
        "IP_PROG1": int_val,   # description of onboard image processing sequence used
        "IP_PROG2": int_val,   # description of onboard image processing sequence used
        "IP_PROG3": int_val,   # description of onboard image processing sequence used
        "IP_PROG4": int_val,   # description of onboard image processing sequence used
        "IP_PROG5": int_val,   # description of onboard image processing sequence used
        "IP_PROG6": int_val,   # description of onboard image processing sequence used
        "IP_PROG7": int_val,   # description of onboard image processing sequence used
        "IP_PROG8": int_val,   # description of onboard image processing sequence used
        "IP_PROG9": int_val,   # description of onboard image processing sequence used
        "IP_00_19": str_val,   # numeral char representation of values 0 - 19 in ip.Cmds
        "IMGSEQ": -1,      # number of image in current sequence (usually 0)
        "OBSERVER": str_val,   # Name of operator
        "BUNIT": str_val,      # unit of values in array
        "BLANK": int_val,      # value in array which means no data
        "FPS_CMD": str_val,    # T/F: from useFPS
        "VERSION": str_val,    # Identifier of FSW header version plus (EUVI only) pointing version
        "CEB_STAT": -1,    # CEB-Link-status (enum CAMERA_INTERFACE_STATUS)
        "CAM_STAT": -1,    # CCD-Interface-status (enum CAMERA_PROGRAM_STATE)
        "READPORT": str_val,   # CCD readout port
        "CMDOFFSE": flt_val, # lightTravelOffsetTime/1000.
        "RO_DELAY": -1.0,  # time (sec) between issuing ro command to the CEB and the start of the ro operation
        "LINE_CLR": -1.0,  # time (sec) per line for clear operation
        "LINE_RO": -1.0,   # time (sec) per line for readout operation
        "RAVG": -999.0,    # average error in star position (pixels)
        "BSCALE": 1.0,     # scale factor for FITS
        "BZERO": flt_val,    # value corresponding to zero in array for FITS
        "SCSTATUS": -1,    # spacecraft status message before exposure
        "SCANT_ON": str_val,   # T/F: derived from s/c status before and after
        "SCFP_ON": str_val,    # T/F: from actualSCFinePointMode
        "CADENCE": int_val,    # Number of seconds between exposures/sequences for the current observing program
        "CRITEVT": str_val,    # 0xHHHH (uppercase hex word)
        "EVENT": 'F',      # A flare IP event has (not) been triggered
        "EVCOUNT": str_val,    # count of number of times evtDetect has run ('0'..'127') ... remains a string
        "EVROW": int_val,      # X-coordinate of centroid of triggered event
        "EVCOL": int_val,      # Y-coordinate of centroid of triggered event
        "COSMICS": int_val,    # Number of pixels removed from image by cosmic ray removal algorithm in FSW
        "N_IMAGES": int_val,   # Number of CCD readouts used to compute the image
        "VCHANNEL": int_val,   # Virtual channel of telemetry downlink
        "OFFSETCR": flt_val, # Offset bias subtracted from image.
        "DOWNLINK": str_val,   # How the image came down
        "DATAMIN": -1.0,   # Minimum value of the image, including the bias derived
        "DATAMAX":-1.0,    # Maximum value of the image, including the bias derived
        "DATAZER": -1,     # Number of zero pixels in the image derived
        "DATASAT": -1,     # Number of saturated values in the image derived
        "DSATVAL": -1.0,   # Value used as saturated constant
        "DATAAVG": -1.0,   # Average value of the image derived
        "DATASIG": -1.0,   # Standard deviation in computing the average derived
        "DATAP01": -1.0,   # Intensity of 1st percentile of image derived
        "DATAP10": -1.0,   # Intensity of 10th percentile image derived
        "DATAP25": -1.0,   # Intensity of 25th percentile of image derived
        "DATAP50": -1.0,   # Intensity of 50th percentile of image derived (median)
        "DATAP75": -1.0,   # Intensity of 75th percentile of image derived
        "DATAP90": -1.0,   # Intensity of 90th percentile of image derived
        "DATAP95": -1.0,   # Intensity of 95th percentile of image derived
        "DATAP98": -1.0,   # Intensity of 98th percentile of image derived
        "DATAP99": -1.0,   # Intensity of 99th percentile of image derived
        "CALFAC": 0.0,     # Calibration factor applied, NOT including binning correction
        "CRPIX1": flt_val,   
        "CRPIX2": flt_val,
        "CRPIX1A": flt_val,  
        "CRPIX2A": flt_val,
        "RSUN": flt_val,     
        "CTYPE1": 'HPLN-TAN',
        "CTYPE2": 'HPLT-TAN',
        "CRVAL1": flt_val,
        "CRVAL2": flt_val,
        "CROTA": flt_val,    
        "PC1_1": 1.0,      
        "PC1_2": flt_val,    
        "PC2_1": flt_val,    
        "PC2_2": 1.0,      
        "CUNIT1": str_val,     # ARCSEC or DEG for HI
        "CUNIT2": str_val,     # ARCSEC or DEG for HI
        "CDELT1": flt_val,
        "CDELT2": flt_val,
        "PV2_1": flt_val,    # parameter for AZP projection (HI only)
        "PV2_1A": flt_val,   # parameter for AZP projection (HI only)
        "SC_ROLL": 9999.0, # values from get_stereo_hpc_point: (deg) - HI from scc_sunvec (GT)
        "SC_PITCH": 9999.0,# arcsec, HI deg
        "SC_YAW": 9999.0,  # arcsec, HI deg
        "SC_ROLLA": 9999.0,# RA/Dec values: (deg)
        "SC_PITA": 9999.0, # degrees
        "SC_YAWA": 9999.0, # degrees
        "INS_R0": 0.0,     # applied instrument offset in roll
        "INS_Y0": 0.0,     # applied instrument offset in pitch (Y-axis)
        "INS_X0": 0.0,     # applied instrument offset in yaw (X-axis) from
        "CTYPE1A": 'RA---TAN',
        "CTYPE2A": 'DEC--TAN',
        "CUNIT1A": 'deg',  # DEG
        "CUNIT2A": 'deg',  # DEG
        "CRVAL1A": flt_val,
        "CRVAL2A": flt_val,
        "PC1_1A": 1.0,     
        "PC1_2A": flt_val,   
        "PC2_1A": flt_val,   
        "PC2_2A": 1.0,     
        "CDELT1A": flt_val,
        "CDELT2A": flt_val,
        "CRLN_OBS": flt_val,
        "CRLT_OBS": flt_val,
        "XCEN": 9999.0,
        "YCEN": 9999.0,        
        "EPHEMFIL": str_val,   # ephemeris SPICE kernel
        "ATT_FILE": str_val,   # attitude SPICE kernel
        "DSUN_OBS": flt_val,
        "HCIX_OBS": flt_val,
        "HCIY_OBS": flt_val,
        "HCIZ_OBS": flt_val,
        "HAEX_OBS": flt_val,
        "HAEY_OBS": flt_val,
        "HAEZ_OBS": flt_val,
        "HEEX_OBS": flt_val,
        "HEEY_OBS": flt_val,
        "HEEZ_OBS": flt_val,
        "HEQX_OBS": flt_val,
        "HEQY_OBS": flt_val,
        "HEQZ_OBS": flt_val,
        "LONPOLE": 180,
        "HGLN_OBS": flt_val,
        "HGLT_OBS": flt_val,
        "EAR_TIME": flt_val,
        "SUN_TIME": flt_val,
        # "JITRSDEV": flt_val,   # std deviation of jitter from FPS or GT values
        # "FPSNUMS": 99999,    # Number of FPS samples
        # "FPSOFFY": 0,        # Y offset
        # "FPSOFFZ": 0,        # Z offset
        # "FPSGTSY": 0,        # FPS Y sum
        # "FPSGTSZ": 0,        # FPS Z sum
        # "FPSGTQY": 0,        # FPS Y square
        # "FPSGTQZ": 0,        # FPS Z square
        # "FPSERS1": 0,        # PZT Error sum [0]
        # "FPSERS2": 0,        # PZT Error sum [1]
        # "FPSERS3": 0,        # PZT Error sum [2]
        # "FPSERQ1": 0,        # PZT Error square [0]
        # "FPSERQ2": 0,        # PZT Error square [1]
        # "FPSERQ3": 0,        # PZT Error square [2]
        # "FPSDAS1": 0,        # PZT DAC sum [0]
        # "FPSDAS2": 0,        # PZT DAC sum [1]
        # "FPSDAS3": 0,        # PZT DAC sum [2]
        # "FPSDAQ1": 0,        # PZT DAC square [0]
        # "FPSDAQ2": 0,        # PZT DAC square [1]
        # "FPSDAQ3": 0         # PZT DAC square [2]
}
    
    for key in secchi_hdr.keys():
        if not key in hdr:
            hdr[key] = secchi_hdr[key]

    return hdr

#######################################################################################################################################

def parse_yml(config_path):
    """
    Parses configuration file.

    @return: Configuration file content
    """
    with open(config_path) as stream:
        try:
            content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return content

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
def download_files(datelist, save_path, ftpsc, instrument, bflag, silent):
    """
    Downloads STEREO images from NASA pub directory

    @param datelist: List of dates for which to download files
    @param save_path: Path for saving downloaded files
    @param ftpsc: Spacecraft (STEREO-A/STEREO-B) for which to download files
    @param instrument: Instrument (HI-1/HI-2) for which to download files
    @param bflag: Data type (science/beacon) for which to download files
    @param silent: Run in silent mode
    """
    
    if ftpsc == 'A':
        sc = 'ahead'

    if ftpsc == 'B':
        sc = 'behind'

    if not silent:
        print('Fetching files...')

    for ins in instrument:
        for date in datelist:

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
              
            num_cpus = cpu_count()

            pool = Pool(int(num_cpus/2), limit_cpu)

            urls = listfd(url, ext)
            inputs = zip(repeat(path_dir), urls)

            try:
                results = pool.starmap(fetch_url, inputs, chunksize=5)

            except ValueError:
                continue
                    
            pool.close()
            pool.join()
      
#######################################################################################################################################

def hi_remove_saturation(data, header, saturation_limit=14000, nsaturated=5):
    """Direct conversion of hi_remove_saturation.pro for IDL.
    Detects and masks saturated pixels with nan. Takes image data and header as input. Returns fixed image.
    @param data: Data of .fits file
    @param header: Header of .fits file
    @param saturation_limit: Threshold value before pixel is considered saturated
    @param nsaturated: Number of pixels in a column before column is considered saturated
    @return: Data with oversaturated columns removed"""

    info="$Id: hi_remove_saturation.pro,v 1.3 2009/06/08 11:03:38 crothers Exp $"
    
    n_im = header['imgseq'] + 1
    imsum = header['summed']

    dsatval = saturation_limit * n_im * (2 ** (imsum - 1)) ** 2

    ind = np.where(data > dsatval)

    # if a pixel has a value greater than the dsatval, begin check to test if column is saturated

    if ind[0].size > 0:

        # all pixels are set to zero, except ones exceeding the saturation limit

        mask = np.zeros(np.shape(data))
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

    # IP_00_19= '41128 31114 37113121  7 41 37120129  7 40 87 50  3 53  1100'
    # IP_00_19= '41128 31 34 37113121  7 41 37120129  7  0  0  0  0  0  0  0' /


    ip_raw = header['IP_00_19']

    while len(ip_raw) < 60:
        ip_raw = ' ' + ip_raw

        header['IP_00_19'] = ip_raw

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
        ind = np.where(ip == '117')
        ip = ip[3 * ind:]
        while len(ip) < 60:
            ip = ip.append('  0')
            header['IP_00_19'] = ''.join(ip)

    ## CHANGE updated header, added count16 + count17 to match IDL behaviour (was xor before)
    

    cnt1   = ip.count('  1')
    cnt2   = ip.count('  2')
    cntspw = ip.count(' 16') + ip.count(' 17')
    cnt50  = ip.count(' 50')
    cnt53  = ip.count(' 53')
    cnt82  = ip.count(' 82')
    cnt83  = ip.count(' 83')
    cnt84  = ip.count(' 84')
    cnt85  = ip.count(' 85')
    cnt86  = ip.count(' 86')
    cnt87  = ip.count(' 87')
    cnt88  = ip.count(' 88')
    cnt118 = ip.count('118')
    # print(cnt1,cnt2,cntspw,cnt50,cnt53,cnt82,cnt83,cnt84,cnt85,cnt86,cnt87,cnt88,cnt118)



    if header['DIV2CORR']:
        cnt1 = cnt1 - 1

    if cnt1 < 0:
        cnt1 = 0

    if cnt1 > 0:
        data = data * (2.0 ** cnt1)
        if not silent:
            print('Corrected for divide by 2 x {}'.format(cnt1))

        header['HISTORY'] = 'seb_ip Corrected for divide by 2 x {}'.format(cnt1)

    if cnt2 > 0:
        ## CHANGE to square instead of multiply
        data = data ** (2.0 ** cnt2)
        if not silent:
            print('Corrected for square root x {}'.format(cnt2))
        
        header['HISTORY'] = 'seb_ip Corrected for square root x {}'.format(cnt2)

    if cntspw > 0:
        data = data * (64.0 ** cntspw)
        if not silent:
            print('Corrected for HI SPW divide by 64 x {}'.format(cntspw))
        
        header['HISTORY'] = 'seb_ip Corrected for HI SPW divide by 64 x {}'.format(cntspw)

    if cnt50 > 0:
        data = data * (4.0 ** cnt50)
        if not silent:
            print('Corrected for divide by 4 x {}'.format(cnt50))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 4 x {}'.format(cnt50)

    if cnt53 > 0 and header['ipsum'] > 0:
        data = data * (4.0 ** cnt53)
        if not silent:
            print('Corrected for divide by 4 x {}'.format(cnt53))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 4 x {}'.format(cnt53)

    if cnt82 > 0:
        data = data * (2.0 ** cnt82)
        if not silent:
            print('Corrected for divide by 2 x {}'.format(cnt82))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 2 x {}'.format(cnt82)

    if cnt83 > 0:
        data = data * (4.0 ** cnt83)
        if not silent:
            print('Corrected for divide by 4 x {}'.format(cnt83))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 4 x {}'.format(cnt83)

    if cnt84 > 0:
        data = data * (8.0 ** cnt84)
        if not silent:
            print('Corrected for divide by 8 x {}'.format(cnt84))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 8 x {}'.format(cnt84)

    if cnt85 > 0:
        data = data * (16.0 ** cnt85)
        if not silent:
            print('Corrected for divide by 16 x {}'.format(cnt85))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 16 x {}'.format(cnt85)

    if cnt86 > 0:
        data = data * (32.0 ** cnt86)
        if not silent:
            print('Corrected for divide by 32 x {}'.format(cnt86))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 32 x {}'.format(cnt86)

    if cnt87 > 0:
        data = data * (64.0 ** cnt87)
        if not silent:
            print('Corrected for divide by 64 x {}'.format(cnt87))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 64 x {}'.format(cnt87)

    if cnt88 > 0:
        data = data * (128.0 ** cnt88)
        if not silent:
            print('Corrected for divide by 128 x {}'.format(cnt88))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 128 x {}'.format(cnt88)

    if cnt118 > 0:
        data = data * (3.0 ** cnt118)
        if not silent:
            print('Corrected for divide by 3 x {}'.format(cnt118))
        
        header['HISTORY'] = 'seb_ip Corrected for divide by 3 x {}'.format(cnt118)

    if not silent:
        print('------------------------------------------------------')

    return data, header

#######################################################################################################################################

def scc_get_missing(hdr, silent=True):
    """
    This function returns the index of the missing pixels.
    
    Args:
        hdr: Image header, either FITS or SECCHI structure.
        silent: Boolean flag to suppress output.
    
    Returns:
        missing: 1D array of longword vector containing the subscripts of the missing pixels.
    """

    # Convert MISSLIST to Superpixel 1D index

    base = 34
    misslist_str = hdr['MISSLIST']
    len_misslist = len(misslist_str)
    
    if len_misslist % 2 != 0:
        misslist_str = ' ' + misslist_str
        len_misslist += 1

    dex = np.arange(0, len_misslist, 2)
    misslist = np.asarray([int(misslist_str[i:i+2].strip(), base) for i in dex])

    n = len(misslist)

    if n != hdr['NMISSING']:
        if not silent:
            print('MISSLIST does not equal NMISSING')

        return np.array([])
    
    if hdr['COMPRSSN'] < 89:
        # Rice Compression and H-compress
        blksz = 64
        blklen = blksz ** 2
        missing = np.zeros(n * blklen, dtype=np.int64)

        ax1 = hdr['naxis1'] // blksz
        ax2 = hdr['naxis2'] // blksz
        blocks = np.vstack((misslist % ax1, misslist // ax2)).T

        dot = np.ones(blksz)
        plus = np.arange(blksz)

        x = np.outer(dot, plus)
        y = np.outer(plus, dot)

        for i in range(n):
            xx = x + blocks[i, 0] * blksz
            yy = y + blocks[i, 1] * blksz
            missing[i * blklen:(i + 1) * blklen] = yy.flatten() * hdr['naxis1'] + xx.flatten()

    elif hdr['COMPRSSN'] in [96, 97]:
        # 16 Segment ICER Compression
        ax1 = 4
        ax2 = 4
        blksz = hdr['naxis1'] // ax1

        blocks = np.vstack((misslist % ax1, misslist // ax2)).T

        if hdr['RECTIFY'] == True:
            if hdr['OBSRVTRY'] == 'STEREO_A':
                if hdr['DETECTOR'] == 'EUVI':
                    blocks = np.column_stack((ax1 - blocks[:, 1] - 1, ax1 - blocks[:, 0] - 1))
                elif hdr['DETECTOR'] == 'COR1':
                    blocks = np.column_stack((blocks[:, 1], ax1 - blocks[:, 0] - 1))
                elif hdr['DETECTOR'] == 'COR2':
                    blocks = np.column_stack((ax1 - blocks[:, 1] - 1, blocks[:, 0]))
            elif hdr['OBSRVTRY'] == 'STEREO_B':
                if hdr['DETECTOR'] == 'EUVI':
                    blocks = np.column_stack((blocks[:, 1], ax1 - blocks[:, 0] - 1))
                elif hdr['DETECTOR'] == 'COR1':
                    blocks = np.column_stack((ax1 - blocks[:, 1] - 1, blocks[:, 0]))
                elif hdr['DETECTOR'] == 'COR2':
                    blocks = np.column_stack((blocks[:, 1], ax1 - blocks[:, 0] - 1))
                elif hdr['DETECTOR'] in ['HI1', 'HI2']:
                    blocks = np.column_stack((ax1 - blocks[:, 0] - 1, ax1 - blocks[:, 1] - 1))

        t = np.zeros((4, 4), dtype=int)
        t[blocks[:, 0], blocks[:, 1]] = 1

        buffer = np.zeros((4, 4, 4), dtype=int)

        buffer[0:3, :, 0] = np.where(t[0:3, :] - t[1:, :] < 0, 0, t[0:3, :] - t[1:, :])
        buffer[1:, :, 1] = np.where(t[1:, :] - t[0:3, :] < 0, 0, t[1:, :] - t[0:3, :])
        buffer[:, 0:3, 2] = np.where(t[:, 0:3] - t[:, 1:] < 0, 0, t[:, 0:3] - t[:, 1:])
        buffer[:, 1:, 3] = np.where(t[:, 1:] - t[:, 0:3] < 0, 0, t[:, 1:] - t[:, 0:3])

        buffer = buffer.reshape(16, 4)
        buffer = buffer[blocks[:, 1] * ax1 + blocks[:, 0], :]

        blklen = np.tile(blksz, n)
        blklen = (blklen + np.sum(buffer[:, 0:2], axis=1) * 20) * (blklen + np.sum(buffer[:, 2:4], axis=1) * 20)
        missing = np.zeros(np.sum(blklen), dtype=np.int64)

        dot = np.ones(blksz + 40)
        plus = np.arange(blksz + 40) - 20

        x = np.outer(dot, plus)
        y = np.outer(plus, dot)

        for i in range(n):
            xx = x.copy()
            yy = y.copy()
            if not buffer[i, 0]:
                xx = xx[0:blksz + 20, :]
                yy = yy[0:blksz + 20, :]
            if not buffer[i, 1]:
                xx = xx[20:, :]
                yy = yy[20:, :]
            if not buffer[i, 2]:
                xx = xx[:, 0:blksz + 20]
                yy = yy[:, 0:blksz + 20]
            if not buffer[i, 3]:
                xx = xx[:, 20:]
                yy = yy[:, 20:]

            xx = xx + blocks[i, 0] * blksz
            yy = yy + blocks[i, 1] * blksz

            missing_slice = slice(np.sum(blklen[:i+1])-blklen[i], np.sum(blklen[:i+1]))
            missing[missing_slice] = yy.flatten() * hdr['naxis1'] + xx.flatten()

    elif 90 <= hdr['COMPRSSN'] <= 95:
        # 32 Segment ICER Compression
        sg = np.zeros(n, dtype=int)

        blksz = np.array([[400, 416, 336, 352], [320, 320, 384, 384]]) // 2**(hdr['summed'] - 1)
        ax1 = [5, 6]
        ax2 = [4, 2]

        s = np.array([[0, -32, 0, -64], [0, 0, -256, -256]]) // 2**(hdr['summed'] - 1)

        blocks = np.vstack((misslist, misslist)).T

        bot = np.where(misslist <= 19)[0]
        top = np.where(misslist >= 20)[0]

        if top.size > 0:
            blocks[top, 0] = (blocks[top, 0] - 20) % ax1[1]
            blocks[top, 1] = ((blocks[top, 1] - 20) // ax1[1]) + ax2[0]

            three = np.where(blocks[top, 0] >= 4)[0]
            if three.size > 0:
                sg[top[three]] = 3
            two = np.where(blocks[top, 0] <= 3)[0]
            if two.size > 0:
                sg[top[two]] = 2

        if bot.size > 0:
            blocks[bot, 0] = blocks[bot, 0] % ax1[0]
            blocks[bot, 1] = blocks[bot, 1] // ax1[0]

            one = np.where(blocks[bot, 0] >= 2)[0]
            if one.size > 0:
                sg[bot[one]] = 1
            zero = np.where(blocks[bot, 0] <= 1)[0]
            if zero.size > 0:
                sg[bot[zero]] = 0

        t = np.zeros((6, 6), dtype=int)
        t[blocks[:, 0], blocks[:, 1]] = 1
        t[[5, 11, 17, 23]] = 2

        buffer = np.zeros((6, 6, 4), dtype=int)
        buffer[0:5, :, 0] = np.where(t[0:5, :] - t[1:, :] < 0, 0, t[0:5, :] - t[1:, :])
        buffer[1:, :, 1] = np.where(t[1:, :] - t[0:5, :] < 0, 0, t[1:, :] - t[0:5, :])
        buffer[:, 0:5, 2] = np.where(t[:, 0:5] - t[:, 1:] < 0, 0, t[:, 0:5] - t[:, 1:])
        buffer[:, 1:, 3] = np.where(t[:, 1:] - t[:, 0:5] < 0, 0, t[:, 1:] - t[:, 0:5])

        c = np.where(t != 2)[0]

        # 2. Reform buffer
        buffer = buffer.reshape(36, 4)
        buffer = buffer[c, :]
        buffer = buffer[misslist, :]

        # 3. Define length of each block
        blklen = np.array([[blksz[sg, 0]], [blksz[sg, 1]]], dtype=np.int64).flatten()
        blklen = (blklen[::2] + np.sum(buffer[:, 0:2], axis=1).astype(np.int64) * 20) * \
                (blklen[1::2] + np.sum(buffer[:, 2:4], axis=1).astype(np.int64) * 20)
        missing = np.zeros((np.sum(blklen), 2), dtype=np.int64)

        # 4. Math Cheats
        dot = np.ones(416 + 40)
        plus = np.arange(416 + 40) - 20

        # 5. Expanded Superpixel index
        x = np.outer(dot, plus)
        y = np.outer(plus, dot)

        # 6. Loop over each Super-Superpixel
        n = len(sg)
        for i in range(n):
            xx = x[0:blksz[sg[i], 0] + 40, 0:blksz[sg[i], 1] + 40]
            yy = y[0:blksz[sg[i], 0] + 40, 0:blksz[sg[i], 1] + 40]

            if not buffer[i, 0]:
                xx = xx[0:blksz[sg[i], 0] + 20, :]
                yy = yy[0:blksz[sg[i], 0] + 20, :]
            if not buffer[i, 1]:
                xx = xx[20:, :]
                yy = yy[20:, :]
            if not buffer[i, 2]:
                xx = xx[:, 0:blksz[sg[i], 1] + 20]
                yy = yy[:, 0:blksz[sg[i], 1] + 20]
            if not buffer[i, 3]:
                xx = xx[:, 20:]
                yy = yy[:, 20:]

            xx = (xx + blocks[i, 0] * blksz[sg[i], 0] + s[sg[i], 0]).reshape(-1)
            yy = (yy + blocks[i, 1] * blksz[sg[i], 1] + s[sg[i], 1]).reshape(-1)

            start_idx = np.sum(blklen[:i+1]) - blklen[i]
            end_idx = np.sum(blklen[:i+1])
            missing[start_idx:end_idx, :] = np.column_stack((xx, yy))

        # 7. Calculate the Rectified 2D index
        if hdr['RECTIFY'] == True:
            if hdr['OBSRVTRY'] == 'STEREO_A':
                if hdr['DETECTOR'] == 'EUVI':
                    missing = np.column_stack((hdr['NAXIS1'] - missing[:, 1] - 1, hdr['NAXIS1'] - missing[:, 0] - 1))
                elif hdr['DETECTOR'] == 'COR1':
                    missing = np.column_stack((missing[:, 1], hdr['NAXIS1'] - missing[:, 0] - 1))
                elif hdr['DETECTOR'] == 'COR2':
                    missing = np.column_stack((hdr['NAXIS1'] - missing[:, 1] - 1, missing[:, 0]))
                # HI1 and HI2 detectors do not require changes

            elif hdr['OBSRVTRY'] == 'STEREO_B':
                if hdr['DETECTOR'] == 'EUVI':
                    missing = np.column_stack((missing[:, 1], hdr['NAXIS1'] - missing[:, 0] - 1))
                elif hdr['DETECTOR'] == 'COR1':
                    missing = np.column_stack((hdr['NAXIS1'] - missing[:, 1] - 1, missing[:, 0]))
                elif hdr['DETECTOR'] == 'COR2':
                    missing = np.column_stack((missing[:, 1], hdr['NAXIS1'] - missing[:, 0] - 1))
                elif hdr['DETECTOR'] in ['HI1', 'HI2']:
                    missing = np.column_stack((hdr['NAXIS1'] - missing[:, 0] - 1, hdr['NAXIS1'] - missing[:, 1] - 1))

        # 8. Calculate final missing values
        missing = (missing[:, 1] * hdr['NAXIS1'] + missing[:, 0]).astype(np.int64)

        if hdr.comprssn > 98:
            if hdr.nmissing > 0:
                missing = np.arange(float(hdr['NAXIS1']) * hdr['NAXIS2']).astype(np.int64)
            else:
                missing = []
        else:
            if not silent:
                print('ICER8 (8-segment) compression not accommodated; returning -1')
            missing = []

    
    return np.asarray(missing)

#######################################################################################################################################

def hi_cosmics(hdr, img, post_conj, silent=True):
    """
    Extracts cosmic ray scrub reports from HI images.
    
    Args:
        hdr: Image header, either FITS or SECCHI structure
        img: Level 0.5 image in DN (long)
    
    Returns:
        Cosmic ray scrub count
    """

    cosmics = -1

    if 's4h' not in hdr['filename']:
        cosmics = hdr['cosmics']
    elif hdr['n_images'] <= 1 and hdr['imgseq'] <= 1:
        cosmics = hdr['cosmics']
    else:
        count = hdr['imgseq'] + 1

        inverted = 0

        if hdr['RECTIFY'] == True:
            if post_conj:
                if hdr['OBSRVTRY'] == 'STEREO_A':
                    inverted = 1
                if hdr['OBSRVTRY'] == 'STEREO_B':
                    inverted = 0
            else:
                if hdr['OBSRVTRY'] == 'STEREO_A':
                    inverted = 0
                if hdr['OBSRVTRY'] == 'STEREO_B':
                    inverted = 1

        if inverted:
            cosmic_counter = img[0,count]

            if cosmic_counter == count:
                cosmics = np.flip(img[0, :count])
                img[0, :count+1] = img[1, :count+1]

            else:
                seek = np.arange(count)
                q = np.where(seek == img[0, :count])[0]

                ctr = q.size

                if ctr > 0:
                    count = q[ctr-1]

                    if count > 1:
                        cosmics = np.flip(img[0, :count])
                        img[0, :count+1] = img[1, :count+1]
                        
                        if not silent:
                            if ctr == 1:
                                print('cosmic ray counter recovered')
                            else:
                                print('cosmic ray counter possibly recovered')

                    else:
                        if hdr['nmissing'] > 0:
                            try:
                                miss = scc_get_missing(hdr, silent=True)
                            except Exception:
                                miss = []
                            if (miss.size > 0) and np.sum(np.array(miss) == hdr['imgseq'] + 1) > 0:
                                if not silent:
                                    print('cosmic ray report is missing')
                            else:
                                if not silent:
                                    print('cosmic ray report implies no images [missing blks?]')
                        else:
                            if not silent:
                                print('cosmic ray report implies no images')
                        cosmics = -1

                else:
                    if not silent:
                        print('cosmic ray counter not recovered')
                    cosmics = -1
        else:

            naxis1 = hdr['naxis1']
            naxis2 = hdr['naxis2']
            cosmic_counter = img[naxis2 - 1, naxis1 - count - 1]

            if cosmic_counter == count:
                cosmics = img[naxis2 - 1, naxis1 - count:naxis1]
                img[naxis2-1,naxis1-count-1:naxis1] = img[naxis2-2, naxis1-count-1:naxis1]

            else:
                seek = np.flip(np.arange(count))
                q = np.where(seek == img[naxis2 - 1, naxis1 - count:naxis1])[0]

                if q.size > 0:
                    count = seek[q[0]]

                    if count > 1:
                        cosmics = img[naxis2-1,naxis1-count:naxis1]
                        img[naxis2-1,naxis1-count-1:naxis1] = img[naxis2 - 2, naxis1-count-1:naxis1]

                        if q.size == 1:
                            if not silent:
                                print('cosmic ray counter recovered')
                        else:
                            if not silent:
                                print('cosmic ray counter possibly recovered')
                    
                    else:
                        if hdr['nmissing'] > 0:
                            try:
                                miss = scc_get_missing(hdr, silent=True)
                            except Exception:
                                miss = []

                            if (miss.size > 0) and np.sum(naxis1 * naxis2 - 1 - np.array(miss) == hdr['imgseq'] + 1) > 0:
                                if not silent:
                                    print('cosmic ray report is missing')
                            else:
                                if not silent:
                                    print('cosmic ray report implies no images [missing blks?]')
                        else:
                            if not silent:
                                print('cosmic ray report implies no images')
                        cosmics = -1

                else:
                    if not silent:
                        print('cosmic ray counter not recovered')
                    cosmics = -1

    return cosmics

#######################################################################################################################################

def sccrorigin(hdr):

    if hdr['RECTIFY'] == True:
        
        if hdr['OBSRVTRY'] == 'STEREO_A':
            if hdr['detector'] == 'EUVI':
                r1col = 129
                r1row = 79
            elif hdr['detector'] == 'COR1':
                r1col = 1
                r1row = 79
            elif hdr['detector'] == 'COR2':
                r1col = 129
                r1row = 51
            elif hdr['detector'] == 'HI1':
                r1col = 51
                r1row = 1
            elif hdr['detector'] == 'HI2':
                r1col = 51
                r1row = 1

        elif hdr['OBSRVTRY'] == 'STEREO_B':
            if hdr['detector'] == 'EUVI':
                r1col = 1
                r1row = 79
            elif hdr['detector'] == 'COR1':
                r1col = 129
                r1row = 51
            elif hdr['detector'] == 'COR2':
                r1col = 1
                r1row = 79
            elif hdr['detector'] == 'HI1':
                r1col = 79
                r1row = 129
            elif hdr['detector'] == 'HI2':
                r1col = 79
                r1row = 129

        else:
            # LASCO/EIT
            r1col = 20
            r1row = 1
    else:
        r1col = 51
        r1row = 1

    return [r1col, r1row]

#######################################################################################################################################

def get_smask(hdr, calpath, post_conj, silent=True):

    if hdr['DETECTOR'] == 'HI1':
        raise NotImplementedError('Not implemented for H1')
        return

    if hdr['DETECTOR'] == 'EUVI':
        filename = 'euvi_mask.fts'
    elif hdr['DETECTOR'] == 'COR1':
        filename = 'cor1_mask.fts'
    elif hdr['DETECTOR'] == 'COR2':
        if hdr['OBSRVTRY'] == 'STEREO_A':
            filename = 'cor2A_mask.fts'
        elif hdr['OBSRVTRY'] == 'STEREO_B':
            filename = 'cor2B_mask.fts'
    elif hdr['DETECTOR'] == 'HI2':
        if hdr['OBSRVTRY'] == 'STEREO_A':
            filename = 'hi2A_mask.fts'
        elif hdr['OBSRVTRY'] == 'STEREO_B':
            filename = 'hi2B_mask.fts'
    
    filename = calpath + filename

    try:
        hdul_smask = fits.open(filename)
        smask = hdul_smask[0].data

    except Exception:
        print('Error reading {}'.format(filename))
        sys.exit()

    xy = sccrorigin(hdr)
    fullm = np.zeros((2176, 2176), dtype=np.uint8)

    x1 = 2048 - np.shape(fullm[xy[0] - 1:, xy[1] - 1:])[0]
    y1 = 2048 - np.shape(fullm[xy[0] - 1:, xy[1] - 1:])[1]

    fullm[xy[1] - 1:y1, xy[0] - 1:x1] = smask

    if post_conj and hdr['DETECTOR'] != 'EUVI':
        fullm = np.rot90(fullm, 2)

    mask = rebin(fullm[hdr['R1ROW']-1:hdr['R2ROW'],hdr['R1COL']-1:hdr['R2COL']], (hdr['NAXIS1'], hdr['NAXIS2']))

    return mask

#######################################################################################################################################


def rebin(array, new_shape):
    """
    Rebin an array to a new shape by interpolation.
    
    Parameters:
    array (numpy.ndarray): Input array to be rebinned.
    new_shape (tuple): New shape (rows, columns) for the output array.
    
    Returns:
    numpy.ndarray: Rebinned array.
    """
    ## CHANGE added this function
    shape = array.shape
    zoom_factors = [n / o for n, o in zip(new_shape, shape)]

    return zoom(array, zoom_factors, order=1)

from scipy.ndimage import rotate
#######################################################################################################################################

def sc_inverse(n, diag, below, above):

    wt_above = float(above) / diag
    wt_below = float(below) / diag

    wt_above_1 = wt_above - 1
    wt_below_1 = wt_below - 1
    power_above = np.zeros(n-1, dtype=float)
    power_below = np.zeros(n-1, dtype=float)

    power_above[0] = 1
    power_below[0] = 1

    for row in range(1, n-1):
        power_above[row] = power_above[row-1] * wt_above_1
        power_below[row] = power_below[row-1] * wt_below_1
    

    v = np.concatenate(([0], wt_below * (power_below * power_above[::-1])))
    u = np.concatenate(([0], wt_above * (power_above * power_below[::-1])))
    

    d = -u[1] / wt_above - (np.sum(v) - v[-1])
    f = 1 / (diag * (d + wt_above * np.sum(v)))

    u[0] = d
    v[0] = d
    u = u * f
    v = v[::-1] * f

    p = np.zeros((n, n), dtype=np.float64)
   
    

    p[0, 0] = u[0]

    for row in range(1, n-1):
        p[row, 0:row] = v[n-row-1:n-1]
        p[row, row:] = u[:n-row]

    p[-1,:] = v

   

   
    return p

#######################################################################################################################################

#@numba.njit()
def hi_desmear(im, hdr, post_conj, silent=True):
    """
    Conversion of hi_desmear.pro for IDL. Removes smear caused by no shutter. First compute the effective exposure time
    [time the ccd was static, generate matrix with this in diagonal, the clear time below and the read time above;
    invert and multiply by the image.

    @param im: Data of .fits file
    @param hdr: Header of .fits file
    @param post_conj: Indicates whether spacecraft is pre or post conjecture
    @param silent: Run in silent mode
    @return: Array corrected for shutterless camera
    """

    version='Applied hi_desmear.pro,v 1.11 2023/08/15 16:22:32'
    hdr['HISTORY'] = version

    # Check valid values in header
    if hdr['CLEARTIM'] < 0:
        raise ValueError('CLEARTIM invalid')
    if hdr['RO_DELAY'] < 0:
        raise ValueError('RO_DELAY invalid')
    if hdr['LINE_CLR'] < 0:
        raise ValueError('LINE_CLR invalid')
    if hdr['LINE_RO'] < 0:
        raise ValueError('LINE_RO invalid')
    
    img = im.astype(float)

    # Extract image array if underscan present
    ## CHANGE fixed messed up indexing
    if hdr['dstart1'] <= 1 or hdr['naxis1'] == hdr['naxis2']:
        image = img
    else:
        image = img[hdr['dstart2']-1:hdr['dstop2'],hdr['dstart1']-1:hdr['dstop1']]

    clearest = 0.70
    exp_eff = hdr['EXPTIME'] + hdr['n_images'] * (clearest - hdr['CLEARTIM'] + hdr['RO_DELAY'])

    dataWeight = hdr['n_images'] * (2 ** (hdr['ipsum'] - 1))

    inverted = 0

    if hdr['RECTIFY'] == True:
        if hdr['OBSRVTRY'] == 'STEREO_B':
            if post_conj == 0:
                inverted = 1
            else:
                print('hi_desmear not implemented for STEREO-B with post_conj=True.')
                sys.exit()
        if hdr['OBSRVTRY'] == 'STEREO_A':
            if post_conj == 1:
                inverted = 1
            else:
                inverted = 0
    

    
    if inverted == 1:
        fixup = sc_inverse(hdr['naxis2'], exp_eff, dataWeight*hdr['line_clr'], dataWeight*hdr['line_ro'])
        
    
    else:
        fixup = sc_inverse(hdr['naxis2'], exp_eff, dataWeight*hdr['line_ro'], dataWeight*hdr['line_clr'])


    image =  fixup @ image
    

    if hdr['dstart1'] <= 1 or (hdr['naxis1'] == hdr['naxis2']):
        img = image.copy()

    else:
        img = image[hdr['dstart2'] - 1:hdr['dstop2'], hdr['dstart1'] - 1:hdr['dstop1']]

    return img


#######################################################################################################################################

def get_calimg(header, calpath, post_conj, silent=True):
    """
    Conversion of get_calimg.pro for IDL. Returns calibration correction array. Checks common block before opening
    calibration file. Saves calibration file to common block. Trims calibration array for under/over scan.
    Re-scales calibration array for summing.

    @param header: Header of .fits file
    @param calpath: Path to calibration files
    @param post_conj: Indicates whether spacecraft is pre or post conjecture
    @param silent: Run on silent mode (True or False)
    @return: Array to correct for calibration
    """


    if header['DETECTOR'] == 'HI1':

        if header['summed'] == 1:
            cal_version = '20061129_flatfld_raw_h1' + header['OBSRVTRY'][7].lower() + '.fts'
            sumflg = 0
        else:
            cal_version = '20100421_flatfld_sum_h1' + header['OBSRVTRY'][7].lower() + '.fts'
            sumflg = 1

    elif header['DETECTOR'] == 'HI2':

        if header['summed'] == 1:
            cal_version = '20150701_flatfld_raw_h2' + header['OBSRVTRY'][7].lower() + '.fts'
            sumflg = 0
        else:
            cal_version = '20150701_flatfld_sum_h2' + header['OBSRVTRY'][7].lower() + '.fts'
            sumflg = 1

    else:
        ## TODO Implement get_calimg for other detectors
        print('get_calimg not implemented for detectors other than HI-1, HI-2.')
        exit()


    calpath = calpath + cal_version

    try:
        hdul_cal = fits.open(calpath)

    except FileNotFoundError:
        print(f'Calibration file {calpath} not found')
        sys.exit()
    
    # if header['NAXIS1'] < 1024:
    #     print('get_calimg does not work with beacon data.')
    #     return np.ones((1,1)),1

    try:
        p1col = hdul_cal[0].header['P1COL']
    except KeyError:
        hdul_cal[0].header = fix_secchi_hdr(hdul_cal[0].header)
        p1col = hdul_cal[0].header['P1COL']

    if (p1col <= 1):
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

    
    if (header['RECTIFY'] == True) and (hdul_cal[0].header['RECTIFY'] == 'F'):
        cal, _ = secchi_rectify(cal.copy(), hdul_cal[0].header, silent=True)

        if not silent:
            print('Rectified calibration image')

    if sumflg:
        if header['summed'] <= 2:
            hdr_sum = 1

        else:
            hdr_sum = 2 ** (header['summed'] - 2)

    else:
        hdr_sum = 2 ** (header['summed'] - 1)

    s = np.shape(cal)

    cal = rebin(cal, (int(s[1] / hdr_sum), int(s[0] / hdr_sum)))

    if post_conj:
        cal = np.rot90(cal, k=2)

    hdul_cal.close()
  
    return cal, cal_version


#######################################################################################################################################

def hi_exposure_wt(hdr, silent=True):

    if 'DETECTOR' not in hdr or hdr['DETECTOR'] not in ['HI1', 'HI2']:
        raise ValueError('for HI DETECTOR only')

    clearest = 0.70
    exp_eff = hdr['EXPTIME'] + hdr['n_images'] * (clearest - hdr['CLEARTIM'] + hdr['RO_DELAY'])

    dataWeight = hdr['n_images'] * (2 ** (hdr['ipsum'] - 1))

    wt0 = np.arange(hdr['naxis2'])
    wt1 = np.reshape(wt0, (1, hdr['naxis2']))
    wt2 = np.reshape(wt0[::-1], (1, hdr['naxis2']))

    if hdr['RECTIFY'] == True and hdr['OBSRVTRY'] == 'STEREO_B':
        if not silent:
            print("rectified")
        wt = exp_eff + wt2 * hdr['line_ro'] + wt1 * hdr['line_clr']
    else:
        if not silent:
            print("normal")
        wt = exp_eff + wt1 * hdr['line_clr'] + wt2 * hdr['line_ro']

    wt =rebin(wt, (hdr['naxis1'], hdr['naxis2']))

    return wt

#######################################################################################################################################

def get_calfac(hdr, conv='MBS', silent=True):

    try:
        hdr_dateavg = datetime.datetime.strptime(hdr['date_avg'], '%Y-%m-%dT%H:%M:%S.%f')
        
    except KeyError:
        hdr.rename_keyword('DATE-AVG', 'DATE_AVG')
        hdr_dateavg = datetime.datetime.strptime(hdr['date_avg'], '%Y-%m-%dT%H:%M:%S.%f')

    if hdr['DETECTOR'] == 'COR1':

        if hdr['OBSRVTRY'] == 'STEREO_A':
            calfac = 6.578E-11
            tai0 = datetime.datetime.strptime('2007-12-01T03:41:48.174', '%Y-%m-%dT%H:%M:%S.%f')
            rate = 0.00648

        elif hdr['OBSRVTRY'] == 'STEREO_B':
            calfac = 7.080E-11
            tai0 = datetime.datetime.strptime('2008-01-17T02:20:15.717', '%Y-%m-%dT%H:%M:%S.%f')
            rate = 0.00258
        
        years = (hdr_dateavg - tai0).total_seconds() / (3600. * 24 * 365.25)
        calfac = calfac / (1 - rate * years)
    
    elif hdr['DETECTOR'] == 'COR2':

        if hdr['OBSRVTRY'] == 'STEREO_A':
            calfac = 2.7E-12 * 0.5

        elif hdr['OBSRVTRY'] == 'STEREO_B':
            calfac = 2.8E-12 * 0.5
    
    elif hdr['DETECTOR'] == 'EUVI':
        gain = 15.0
        calfac = gain * (3.65 * hdr['WAVELNTH']) / (13.6 * 911)
    
    elif hdr['DETECTOR'] == 'HI1':

        if hdr['OBSRVTRY'] == 'STEREO_A':
            years = (hdr_dateavg - datetime.datetime.strptime('2011-06-27T00:00:00.000', '%Y-%m-%dT%H:%M:%S.%f')).total_seconds() / (3600 * 24 * 365.25)
            
            ## commented this, otherwise why would it be used in paper 
            if years < 0:
                years = 0

            if conv == 's10':
                # calfac = 763.2 + 1.315 * years
                calfac = (3.453e-13 + 5.914e-16 * years )
            else:
                calfac = 3.453e-13 + 5.914e-16 * years
                # calfac = 3.453E-13 * 2.223e15

            hdr['HISTORY'] = 'revised calibration Tappin et al Solar Physics 2022 DOI 10.1007/s11207-022-01966-x'

        elif hdr['OBSRVTRY'] == 'STEREO_B':
            years = (hdr_dateavg - datetime.datetime.strptime('2007-01-01T00:00:00.000', '%Y-%m-%dT%H:%M:%S.%f')).total_seconds() / (3600 * 24 * 365.25)

            if years < 0:
                years = 0

            if conv == 's10':
                calfac = 790.0 + 0.001503*years
                ## TODO where is the year factor for HI1 STEREO B?
            else:
                annualchange = 0.001503
                calfac = 3.55E-13
                calfac = calfac / (1 - annualchange * years)


            hdr['HISTORY'] = 'revised calibration Tappin et al Solar Physics 2017 DOI 10.1007/s11207-017-1052-0'
    
    elif hdr['DETECTOR'] == 'HI2':

        if hdr['OBSRVTRY'] == 'STEREO_A':

            years = (hdr_dateavg - datetime.datetime.strptime('2015-01-01T00:00:00.000', '%Y-%m-%dT%H:%M:%S.%f')).total_seconds() / (3600 * 24 * 365.25)
            
            if years < 0:
                if conv == 's10':
                    calfac = 99.5 + 0.1225 * years
                else:
                    calfac = 4.476E-14 + 5.511E-17 * years
            else:
                if conv == 's10':
                    calfac = 100.3 + 0.1580 * years
                else:
                    calfac = 4.512E-14 + 7.107E-17 * years

            hdr['HISTORY'] = 'revised calibration Tappin et al Solar Physics 2022 DOI 10.1007/s11207-022-01966-x'

        elif hdr['OBSRVTRY'] == 'STEREO_B':
            years = (hdr_dateavg - datetime.datetime.strptime('2000-12-31T00:00:00.000', '%Y-%m-%dT%H:%M:%S.%f')).total_seconds() / (3600 * 24 * 365.25)
            if conv == 's10':
                calfac = 95.424 + 0.067 * years
            else:
                calfac = 4.293E-14 + 3.014E-17 * years
    
    # calfac = 799.391
    hdr['calfac'] = calfac
    if 'ipsum' in hdr and hdr['ipsum'] > 1 and calfac != 1.0:
        divfactor = (2 ** (hdr['ipsum'] - 1)) ** 2
        hdr['ipsum'] = 1
        calfac = calfac / divfactor

        if not silent:
            print(f'Divided calfac by {divfactor} to account for IPSUM')
            print('IPSUM changed to 1 in header.')

        hdr['HISTORY'] =  f'get_calfac Divided calfac by {divfactor} to account for IPSUM'

    if 'polar' in hdr and hdr['polar'] == 1001 and hdr.get('seb_prog') != 'DOUBLE':
        calfac *= 2

        if not silent:
            print('Applied factor of 2 for total brightness')

        hdr['HISTORY'] =  'get_calfac Applied factor of 2 for total brightness'

    return calfac, hdr


#######################################################################################################################################

def scc_hi_diffuse(header, ipsum=None):
    """
    Conversion of scc_hi_diffuse.pro for IDL. Compute correction for diffuse sources arrising from changes
    in the solid angle in the optics. In the mapping of the optics the area of sky viewed is not equal off axis.

    @param header: Header of .fits file
    @param ipsum: Allows override of header ipsum value for use in L1 and beyond images
    @return: Correction factor for given image
    """

    if ipsum is None:
        ipsum = header['ipsum']

    summing = 2 ** (ipsum - 1)

    ##CHANGE changed if-else logic
    
    try:
        ravg = header['ravg']

    except KeyError:
        ravg = 0

    if ravg >= 0:
        mu = header['pv2_1']
        cdelt = header['cdelt1'] * np.pi / 180

    else:
        
        if header['detector'] == 'HI1':

            if header['OBSRVTRY'] == 'STEREO_A':
                mu = 0.102422
                cdelt = 35.96382 / 3600 * np.pi / 180 * summing

            elif header['OBSRVTRY'] == 'STEREO_B':
                mu = 0.095092
                cdelt = 35.89977 / 3600 * np.pi / 180 * summing

        elif header['detector'] == 'HI2':

            if header['OBSRVTRY'] == 'STEREO_A':
                mu = 0.785486
                cdelt = 130.03175 / 3600 * np.pi / 180 * summing

            if header['OBSRVTRY'] == 'STEREO_B':
                mu = 0.68886
                cdelt = 129.80319 / 3600 * np.pi / 180 * summing

    pixelSize = 0.0135 * summing
    fp = pixelSize / cdelt

    x = np.arange(header['naxis1']) - header['crpix1'] + header['dstart1']
    x = x[:,None].repeat(header['naxis2'],1)
    # x = np.array([x for i in range(header['naxis1'])])

    y = np.arange(header['naxis2']) - header['crpix2'] + header['dstart2']
    # y = np.transpose(y)
    # y = np.array([y for i in range(header['naxis1'])])
    y = y[None,:].repeat(header['naxis1'],0)

    r = np.sqrt(x * x + y * y) * pixelSize

    gamma = fp * (mu + 1.0) / r
    cosalpha1 = (-1.0 * mu + gamma * np.sqrt(1.0 - mu * mu + gamma * gamma)) / (1.0 + gamma * gamma)

    correct = ((mu + 1.0) ** 2 * (mu * cosalpha1 + 1.0)) / ((mu + cosalpha1) ** 3)

    return correct


#######################################################################################################################################

def secchi_rectify(a, scch, hdr=None, norotate=False, silent=True):

    ## CHANGE Added history, changed dstart1, dstart2 (< function behaves differently in IDL)

    info = "$Id: secchi_rectify.pro,v 1.29 2023/08/14 17:50:07 secchia Exp $"
    histinfo = info[1:-2]

    if scch['rectify'] == True:
        if not silent:
            print('RECTIFY=T -- Returning with no changes')
        return a

    crval1 = scch['crval1']

    if scch['OBSRVTRY'] == 'STEREO_A':    
        post_conj = int(np.sign(crval1))

    if scch['OBSRVTRY'] == 'STEREO_B':    
        post_conj = int(-1*np.sign(crval1))

    if post_conj == -1:
        post_conj = False
    if post_conj == 1:
        post_conj = True
        
    stch = scch.copy()
    
    ## TODO implement other detectors

    if not norotate:
        stch['rectify'] = True

        if scch['OBSRVTRY'] == 'STEREO_A' and post_conj == 0:
            if scch['detector'] == 'EUVI':
                # b = np.rot90(a.T, 2)
                # stch['r1row'] = 2176 - scch['p2col'] + 1
                # stch['r2row'] = 2176 - scch['p1col'] + 1
                # stch['r1col'] = 2176 - scch['p2row'] + 1
                # stch['r2col'] = 2176 - scch['p1row'] + 1
                # stch['crpix1'] = scch['naxis2'] - scch['crpix2'] + 1
                # stch['crpix2'] = scch['naxis1'] - scch['crpix1'] + 1
                # stch['naxis1'], stch['naxis2'] = scch['naxis2'], scch['naxis1']
                # stch['sumrow'] = scch['sumcol']
                # stch['sumcol'] = scch['sumrow']
                # stch['rectrota'] = 6
                # rotcmt = 'transpose and rotate 180 deg CCW'
                # stch['dstart1'] = max(1, 129 - stch['r1col'] + 1)
                # stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                # stch['dstart2'] = max(1, 79 - stch['r1row'] + 1)
                # stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)
                print('Rectify not implemented for EUVI')
                sys.exit()

            elif scch['detector'] == 'COR1':
                # b = rotate(a, 3)
                # stch['r1row'] = 2176 - scch['p2col'] + 1
                # stch['r2row'] = 2176 - scch['p1col'] + 1
                # stch['r1col'] = scch['p1row']
                # stch['r2col'] = scch['p2row']
                # stch['crpix1'] = scch['crpix2']
                # stch['crpix2'] = scch['naxis1'] - scch['crpix1'] + 1
                # stch['naxis1'], stch['naxis2'] = scch['naxis2'], scch['naxis1']
                # stch['sumrow'] = scch['sumcol']
                # stch['sumcol'] = scch['sumrow']
                # stch['rectrota'] = 3
                # rotcmt = 'rotate 270 deg CCW'
                # stch['dstart1'] = 1
                # stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                # stch['dstart2'] = max(1, 79 - stch['r1row'] + 1)
                # stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)
                print('Rectify not implemented for COR1')
                sys.exit()

            elif scch['detector'] == 'COR2':
                # b = rotate(a, 1)
                # stch['r1row'] = scch['p1col']
                # stch['r2row'] = scch['p2col']
                # stch['r1col'] = 2176 - scch['p2row'] + 1
                # stch['r2col'] = 2176 - scch['p1row'] + 1
                # stch['crpix1'] = scch['naxis2'] - scch['crpix2'] + 1
                # stch['crpix2'] = scch['crpix1']
                # stch['naxis1'], stch['naxis2'] = scch['naxis2'], scch['naxis1']
                # stch['sumrow'] = scch['sumcol']
                # stch['sumcol'] = scch['sumrow']
                # stch['rectrota'] = 1
                # rotcmt = 'rotate 90 deg CCW'
                # stch['dstart1'] = max(1, 129 - stch['r1col'] + 1)
                # stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                # stch['dstart2'] = max(1, 51 - stch['r1row'] + 1)
                # stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)

                print('Rectify not implemented for COR2')
                sys.exit()

            elif scch['detector'] in ['HI1', 'HI2']:
                b = a  # no change
                stch['r1row'] = scch['p1row']
                stch['r2row'] = scch['p2row']
                stch['r1col'] = scch['p1col']
                stch['r2col'] = scch['p2col']
                stch['rectrota'] = 0
                rotcmt = 'no rotation necessary'

        elif scch['OBSRVTRY'] == 'STEREO_B' and post_conj == 0:
            if scch['detector'] == 'EUVI':
                # b = rotate(a, 3)
                # stch['r1row'] = 2176 - scch['p2col'] + 1
                # stch['r2row'] = 2176 - scch['p1col'] + 1
                # stch['r1col'] = scch['p1row']
                # stch['r2col'] = scch['p2row']
                # stch['crpix1'] = scch['crpix2']
                # stch['crpix2'] = scch['naxis1'] - scch['crpix1'] + 1
                # stch['naxis1'], stch['naxis2'] = scch['naxis2'], scch['naxis1']
                # stch['sumrow'] = scch['sumcol']
                # stch['sumcol'] = scch['sumrow']
                # stch['rectrota'] = 3
                # rotcmt = 'rotate 270 deg CCW'
                # stch['dstart1'] = 1
                # stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                # stch['dstart2'] = max(1, 79 - stch['r1row'] + 1)
                # stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)
                print('Rectify not implemented for EUVI')
                sys.exit()

            elif scch['detector'] == 'COR1':
                # b = rotate(a, 1)
                # stch['r1row'] = scch['p1col']
                # stch['r2row'] = scch['p2col']
                # stch['r1col'] = 2176 - scch['p2row'] + 1
                # stch['r2col'] = 2176 - scch['p1row'] + 1
                # stch['crpix1'] = scch['naxis2'] - scch['crpix2'] + 1
                # stch['crpix2'] = scch['crpix1']
                # stch['naxis1'], stch['naxis2'] = scch['naxis2'], scch['naxis1']
                # stch['sumrow'] = scch['sumcol']
                # stch['sumcol'] = scch['sumrow']
                # stch['rectrota'] = 1
                # rotcmt = 'rotate 90 deg CCW'
                # stch['dstart1'] = max(1, 51 - stch['r1col'] + 1)
                # stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                # stch['dstart2'] = max(1, 129 - stch['r1row'] + 1)
                # stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)

                print('Rectify not implemented for COR1')
                sys.exit()

            elif scch['detector'] == 'COR2':
                # b = rotate(a, 3)
                # stch['r1row'] = 2176 - scch['p2col'] + 1
                # stch['r2row'] = 2176 - scch['p1col'] + 1
                # stch['r1col'] = 2176 - scch['p2row'] + 1
                # stch['r2col'] = 2176 - scch['p1row'] + 1
                # stch['crpix1'] = scch['naxis2'] - scch['crpix2'] + 1
                # stch['crpix2'] = scch['naxis1'] - scch['crpix1'] + 1
                # stch['naxis1'], stch['naxis2'] = scch['naxis2'], scch['naxis1']
                # stch['sumrow'] = scch['sumcol']
                # stch['sumcol'] = scch['sumrow']
                # stch['rectrota'] = 3
                # rotcmt = 'rotate 270 deg CCW'
                # stch['dstart1'] = max(1, 129 - stch['r1col'] + 1)
                # stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                # stch['dstart2'] = max(1, 79 - stch['r1row'] + 1)
                # stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)

                print('Rectify not implemented for COR2')
                sys.exit()

            elif scch['detector'] in ['HI1', 'HI2']:

                b = np.rot90(a, 2)
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

                stch['dstart1'] = max(1, 79 - stch['r1col'] + 1)
                stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                stch['dstart2'] = max(1, 129 - stch['r1row'] + 1)
                stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)
        
        elif scch['OBSRVTRY'] == 'STEREO_A' and post_conj == 1:

            if scch['detector'] == 'EUVI':

                # b = a.T
                # stch['r1row'] = scch['p1col']
                # stch['r2row'] = scch['p2col']
                # stch['r1col'] = scch['p1row']
                # stch['r2col'] = scch['p2row']
                # stch['crpix1'] = scch['naxis1'] - scch['crpix2'] + 1
                # stch['crpix2'] = scch['naxis2'] - scch['crpix1'] + 1
                # stch['naxis1'] = scch['naxis2']
                # stch['naxis2'] = scch['naxis1']
                # stch['sumrow'] = scch['sumcol']
                # stch['sumcol'] = scch['sumrow']
                # stch['rectrota'] = 4
                # rotcmt = 'transpose'
                # stch['dstart1'] = max(1, 129 - stch['r1col'] + 1)
                # stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                # stch['dstart2'] = max(1, 79 - stch['r1row'] + 1)
                # stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)

                print('Rectify not implemented for EUVI')
                sys.exit()

            elif scch['detector'] == 'COR1':

                # b = np.rot90(a, 1)
                # stch['r1row'] = scch['p1col']
                # stch['r2row'] = scch['p2col']
                # stch['r1col'] = 2176 - scch['p2row'] + 1
                # stch['r2col'] = 2176 - scch['p1row'] + 1
                # stch['crpix1'] = scch['naxis2'] - scch['crpix2'] + 1
                # stch['crpix2'] = stch['crpix1']
                # stch['naxis1'] = scch['naxis2']
                # stch['naxis2'] = scch['naxis1']
                # stch['sumrow'] = scch['sumcol']
                # stch['sumcol'] = scch['sumrow']
                # stch['rectrota'] = 1
                # rotcmt = 'rotate 90 deg CCW'
                # stch['dstart1'] = max(1, 129 - stch['r1col'] + 1)
                # stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                # stch['dstart2'] = max(1, 51 - stch['r1row'] + 1)
                # stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)

                print('Rectify not implemented for COR1')
                sys.exit()

            elif scch['detector'] == 'COR2':
                
                # b = rotate(a, 3)
                # stch['r1row'] = 2176 - scch['p2col'] + 1
                # stch['r2row'] = 2176 - scch['p1col'] + 1
                # stch['r1col'] = scch['p1row']
                # stch['r2col'] = scch['p2row']
                # stch['crpix1'] = scch['crpix2']
                # stch['crpix2'] = scch['naxis1'] - scch['crpix1'] + 1
                # stch['naxis1'] = scch['naxis2']
                # stch['naxis2'] = scch['naxis1']
                # stch['sumrow'] = scch['sumcol']
                # stch['sumcol'] = scch['sumrow']
                # stch['rectrota'] = 3
                # rotcmt = 'rotate 270 deg CCW'
                # stch['dstart1'] = 1
                # stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                # stch['dstart2'] = max(1, 79 - stch['r1row'] + 1)
                # stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)

                print('Rectify not implemented for COR2')
                sys.exit()

            elif scch['detector'] in ['HI1', 'HI2']:

                b = rotate(a, 2)
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
                stch['dstart1'] = max(1, 79 - stch['r1col'] + 1)
                stch['dstop1'] = stch['dstart1'] - 1 + min((stch['r2col'] - stch['r1col'] + 1), 2048)
                stch['dstart2'] = max(1, 129 - stch['r1row'] + 1)
                stch['dstop2'] = stch['dstart2'] - 1 + min((stch['r2row'] - stch['r1row'] + 1), 2048)

            else:
                b = a  # If detector is not recognized, return the original image
                rotcmt = None

        elif scch['OBSRVTRY'] == 'STEREO_B' and post_conj == 1:
            print('Case of ST-B with post_conj = True not implemented. Exiting...')
            sys.exit()

        
    else:

        stch['rectify'] = False
        b = a  # no rotation performed

        stch['r1row'] = scch['p1row']
        stch['r2row'] = scch['p2row']
        stch['r1col'] = scch['p1col']
        stch['r2col'] = scch['p2col']
        stch['rectrota'] = 0
        rotcmt = 'no rotation necessary'

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


    if stch['NAXIS1'] > 0 and stch['NAXIS2'] > 0:
         
        try:
            wcoord = wcs.WCS(stch)
            xycen = wcoord.all_pix2world((stch['naxis1'] - 1.) / 2., (stch['naxis2'] - 1.) / 2., 0)

            stch['xcen'] = float(xycen[0])
            stch['ycen'] = float(xycen[1])

        except wcs.SingularMatrixError:

            stch['xcen'] = 9999.0
            stch['ycen'] = 9999.0

    if hdr is not None:

        hdr['NAXIS1'] = stch['naxis1']
        hdr['NAXIS2'] = stch['naxis2']
        hdr['R1COL'] = stch['r1col']
        hdr['R2COL'] = stch['r2col']
        hdr['R1ROW'] = stch['r1row']
        hdr['R2ROW'] = stch['r2row']
        hdr['SUMROW'] = stch['sumrow']
        hdr['SUMCOL'] = stch['sumcol']
        hdr['RECTIFY'] = stch['rectify']
        hdr['CRPIX1'] = stch['crpix1']
        hdr['CRPIX2'] = stch['crpix2']
        hdr['XCEN'] = stch['xcen']
        hdr['YCEN'] = stch['ycen']
        hdr['CRPIX1A'] = stch['crpix1']
        hdr['CRPIX2A'] = stch['crpix2']
        hdr['DSTART1'] = stch['dstart1']
        hdr['DSTART2'] = stch['dstart2']
        hdr['DSTOP1'] = stch['dstop1']
        hdr['DSTOP2'] = stch['dstop2']
        hdr['HISTORY'] = histinfo  # Assuming histinfo is defined
        hdr['RECTROTA'] = f"{stch['rectrota']} {rotcmt}"  # Assuming rotcmt is defined

    scch = stch

    if norotate:
        if not silent:
            print('norotate set -- Image returned unchanged')
        
        return a, scch
    
    else:
        if not silent:
            print(f'Rectification applied to {scch["filename"]}: {rotcmt}')

        return b, scch

#######################################################################################################################################
    
def get_biasmean(header, silent=True):
    """
    Conversion of get_biasmean.pro for IDL. Returns mean bias for a give image.

    @param header: Header of .fits file
    @return: Bias to be subtracted from the image
    """
    bias = header['BIASMEAN']
    ipsum = header['IPSUM']

    if ('103' in header['IP_00_19']) or (' 37' in header['IP_00_19']) or (' 38' in header['IP_00_19']):

        if not silent:
            print('Biasmean subtracted onboard in seb ip.')
            
        bias = 0
        return bias
    
    if header['DETECTOR'][0:2] == 'HI':
        bias = bias-(bias/header['N_IMAGES'])

    if header['OFFSETCR'] > 0:
        bias = 0
        return bias

    if ipsum > 1:
        bias = bias*((2**(ipsum-1))**2)

    return bias

#######################################################################################################################################

def hi_fill_missing(data, header, silent=True):
    """
    Conversion of fill_missing.pro for IDL. Set missing block values sensibly.

    @param data: Data from .fits file
    @param header:Header of .fits file
    @return: Corrected image
    """
    if header['NMISSING'] > 0:
        if len(header['MISSLIST']) < 1:
            if not silent:
                print('Mismatch between nmissing and misslist.')
        else:
            fields = scc_get_missing(header)
            shp = np.shape(data)
            data = data.flatten()
            data[fields] = np.nan
            data = data.reshape(shp)

    return data


#######################################################################################################################################

def scc_img_trim(im, header, silent=True):
    """
    Conversion of scc_img_trim.pro for IDL. Returns rectified images with under/over scan areas removed.
    The program returns the imaging area of the CCD. If the image has not been rectified such that ecliptic north
    is up then the image is rectified.

    @param im: Selected image
    @param header: Header of .fits file
    @param silent: Suppress print statements
    @return: Rectified image with under-/overscan removed
    """
    info = "$Id: scc_img_trim.pro,v 2.4 2007/12/13 17:01:13 colaninn Exp $"
    histinfo = info[1:-1]

    if (header['DSTOP1'] < 1) or (header['DSTOP1'] > header['NAXIS1']) or (header['DSTOP2'] > header['NAXIS2']):
        precommcorrect(im, header, silent)

    x1 = header['DSTART1'] - 1
    x2 = header['DSTOP1'] - 1
    y1 = header['DSTART2'] - 1
    y2 = header['DSTOP2'] - 1

    img = im[x1:x2 + 1, y1:y2 + 1]

    s = np.shape(img)

    if (header['NAXIS1'] != s[0]) or (header['NAXIS2'] != s[1]):

        if not silent:
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

        header['xcen'] = float(xycen[0])
        header['ycen'] = float(xycen[1])

        header['HISTORY'] = histinfo

    return img, header

#######################################################################################################################################

def scc_putin_array(im, hdr, outsize=2048, trim_off=False, silent=False, full=False, new=True):
    """
    Places input image in FFV array, binned to match outsize.

    Parameters:
    im (ndarray): Input SECCHI image.
    hdr (dict): Image header (FITS or SECCHI header structure).
    outsize (int, optional): Output size, must be a multiple of 256; implies full.
    trim_off (bool, optional): If set, outsize is set to 2176.
    silent (bool, optional): If set, suppress messages.
    full (bool, optional): Place subfield in FFV array (default behavior).
    new (bool, optional): Re-initialize common output_array; implies full.

    Returns:
    ndarray: Output image placed in FFV array, binned to match outsize.
    """
    
    info = "$Id: scc_putin_array.pro,v 2.15 2020/09/03 20:25:10 nathan Exp $"
    histinfo = info[5:-20]

    if trim_off:
        outsize = 2176

    ccdfac = 1.0
    ccdsizeh = 2048

    if hdr['INSTRUME'] == 'SECCHI' and trim_off:
        ccdsizeh = 2176
    elif hdr['INSTRUME'] != 'SECCHI':
        ccdfac = 0.5  # relative to SECCHI 2048x2048

    if hdr['INSTRUME'] == 'MK3':
        ccdfac = 0.25
    elif hdr['INSTRUME'] == 'AIA':
        ccdfac = 2.0  # full-res AIA

    binfac = (2048.0 * ccdfac / outsize)
    output_img = im.copy()

    if new or full:
        start = sccrorigin(hdr)
        offsetarr = [start[1], start[1] + (2048 * ccdfac-1), start[0], start[0] + (2048 * ccdfac-1)]

        if trim_off:
            offsetarr = [1, 2176 * ccdfac, 1, 2176 * ccdfac]
        if hdr['INSTRUME'] != 'SECCHI':
            offsetarr = [1, 2048 * ccdfac, 1, 2048 * ccdfac]

        out = {
            'outsize': [ccdsizeh * ccdfac / binfac, outsize],
            'offset': offsetarr,
            'readsize': np.array([max(ccdsizeh, outsize), max(2048, outsize)]) * ccdfac,
            'binned': binfac
        }

    if hdr['NAXIS1'] != out['outsize'][0] or hdr['NAXIS2'] != out['outsize'][1]:
        if (hdr['R2COL'] - hdr['R1COL'], hdr['R2ROW'] - hdr['R1ROW']) != tuple(out['readsize']-1):
            sfac = 1.0 / 2 ** (hdr['SUMMED'] - 1)
            output_img = np.zeros((int(out['readsize'][0] * sfac), int(out['readsize'][1] * sfac)), dtype=np.float32)
            x1 = int(max(0, hdr['R1COL'] - out['offset'][0]) * sfac)
            x2 = int((hdr['R2COL'] - out['offset'][0]) * sfac)
            y1 = int(max(0, hdr['R1ROW'] - out['offset'][2]) * sfac)
            y2 = int((hdr['R2ROW'] - out['offset'][2]) * sfac)

            x1i = max(0, out['offset'][0] - hdr['R1COL']) * sfac
            y1i = max(0, out['offset'][2] - hdr['R1ROW']) * sfac
            x2i = (hdr['R2COL'] - hdr['R1COL'] - max(0, hdr['R2COL'] - out['offset'][1])) * sfac
            y2i = (hdr['R2ROW'] - hdr['R1ROW'] - max(0, hdr['R2ROW'] - out['offset'][3])) * sfac

            output_img[y1:y2+1,x1:x2+1] = im[int(y1i):int(y2i)+1,int(x1i):int(x2i)+1]

            if not silent:
                print("SUB-FIELD PUT IN FULL FIELD")

            hdr.update({
                'DSTART1': x1 + 1,
                'DSTART2': y1 + 1,
                'DSTOP1': x2 + 1,
                'DSTOP2': y2 + 1,
                'CRPIX1': hdr['CRPIX1'] + x1,
                'CRPIX1A': hdr['CRPIX1A'] + x1,
                'CRPIX2': hdr['CRPIX2'] + y1,
                'CRPIX2A': hdr['CRPIX2A'] + y1,
                'R1COL': out['offset'][0],
                'R2COL': out['offset'][1],
                'R1ROW': out['offset'][2],
                'R2ROW': out['offset'][3]
            })

        if out['binned'] != 2 ** (hdr['SUMMED'] - 1):
            bindif = max(output_img.shape) / max(out['outsize'])
            output_img = rebin(output_img, (out['outsize'][1], out['outsize'][0]))

            if not silent:
                print("IMAGE REBINNED FOR OUTPUT")

            hdr.update({
                'SUMMED': int(np.log(out['binned'])/np.log(2) + 1),
                'DSTOP1': int(hdr['DSTOP1'] / bindif),
                'DSTOP2': int(hdr['DSTOP2'] / bindif),
                'CRPIX1': 0.5 + (hdr['CRPIX1'] - 0.5) / bindif,
                'CRPIX1A': 0.5 + (hdr['CRPIX1A'] - 0.5) / bindif,
                'CRPIX2': 0.5 + (hdr['CRPIX2'] - 0.5) / bindif,
                'CRPIX2A': 0.5 + (hdr['CRPIX2A'] - 0.5) / bindif,
                'CDELT1': hdr['CDELT1'] * bindif,
                'CDELT2': hdr['CDELT2'] * bindif,
                'CDELT1A': hdr['CDELT1A'] * bindif,
                'CDELT2A': hdr['CDELT2A'] * bindif
            })

        if not silent:
            print("HEADER UPDATED")

        hdr.update({
            'NAXIS1': output_img.shape[1],
            'NAXIS2': output_img.shape[0],
            'DSTOP1': min(hdr['DSTOP1'], output_img.shape[0]),
            'DSTOP2': min(hdr['DSTOP2'], output_img.shape[1])
        })

        wcoord = wcs.WCS(hdr)
        xycen = wcoord.wcs_pix2world((hdr['naxis1'] - 1.) / 2., (hdr['naxis2'] - 1.) / 2., 0)

        hdr['xcen'] = float(xycen[0])
        hdr['ycen'] = float(xycen[1])

        hdr['HISTORY'] = histinfo + ',CHANGED IMAGE SIZE'

    return output_img, hdr

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
        #RA and DEC of center pixel
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
            if (np.round(time_i_j) <= -maxgap * cadence) & (np.round(time_i_j) >= (cadence-5)):

                j_ind.append(i - j)

        # if no adequate preceding image is found, append array of np.nan to the running difference list
        # criteria: time diffference is larger than -maxgap * cadence or time difference is smaller than cadence

        if len(j_ind) == 0:
            r_dif.append(nandata)

        # for appropriate time differences, create running differene images and apply median filter
        else:
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

def get_bins(a, bin_edges):
    """
    Creates weekly minimum background image for STEREO-HI data.

    @param a: 3D array of reduced images
    @param bin_edges: 3D array of bin valuess for estiamting the median
    @return: bin_arr: Approximate median weekly background
    """

    sum_left = np.zeros(np.shape(bin_edges))

    for i in range(sum_left.shape[0]):
        bin_cnt = a < bin_edges[i]
        sum_left[i] = bin_cnt.sum(axis=0)
        
    bin_ind = np.zeros((bin_edges.shape[1], bin_edges.shape[2]))
    bin_half = int((np.shape(a)[0]+1)/2)

    bin_ind = sum_left >= bin_half
    
    bin_ind = np.argmax(bin_ind, axis=0)

    if np.shape(a)[0]//2 == 0:
        bin_arr_left = np.take_along_axis(bin_edges, bin_ind[np.newaxis], axis=0)[0]
        bin_arr_right = np.take_along_axis(bin_edges, bin_ind[np.newaxis]+1, axis=0)[0]

        bin_arr = (bin_arr_left + bin_arr_right)/2

    else:
        bin_arr = np.take_along_axis(bin_edges, bin_ind[np.newaxis], axis=0)[0]    

    return bin_arr

#######################################################################################################################################

def get_bkgd(path, ftpsc, start, bflag, ins, bg_dur, rolling=False):
    """
    Creates weekly median background image for STEREO-HI data.

    @param path: The path where all reduced images, running difference images and J-Maps are saved
    @param ftpsc: Spacecraft (A/B)
    @param start: First date (DDMMYYYY) for which to create running difference images
    @param bflag: Science or beacon data
    @param ins: STEREO-HI instrument (HI-1/HI-2)
    @return: bkgd: Median weekly background
    """
        
    date = datetime.datetime.strptime(start, '%Y%m%d') - datetime.timedelta(days=bg_dur) 
    
    if rolling:
        interv = np.arange(bg_dur+1)
    else:
        interv = np.arange(bg_dur)

    print(interv)
    datelist = [datetime.datetime.strftime(date + datetime.timedelta(days=int(i)), '%Y%m%d') for i in interv]  
    red_path = path + 'reduced/data/' + ftpsc + '/'

    red_paths = []
    red_files = []

    for k, dates in enumerate(datelist):
        red_paths.append(red_path + str(dates) + '/' + bflag + '/' + ins + '/*.fts')
        red_files.extend(sorted(glob.glob(red_path + str(dates) + '/' + bflag + '/' + ins + '/*.fts')))

    if len(red_files) == 0:
        return np.nan

    data = []

    for i in range(len(red_files)):
        file = fits.open(red_files[i])
        
        d = file[0].data.copy()
        data.append(d)
        file.close()

    data = np.array(data)

    nan_mask = np.array([np.isnan(data[i]) for i in range(len(data))])
    
    for i in range(len(data)):
        data[i][nan_mask[i]] = np.array(np.interp(np.flatnonzero(nan_mask[i]), np.flatnonzero(~nan_mask[i]), data[i][~nan_mask[i]]))

    if rolling:
        print('Rolling background not implemented yet...')
        sys.exit()
    else:            
        bkgd = np.median(data, axis=0)

    return data

#######################################################################################################################################
def minmax_scaler(arr, *, vmin=0, vmax=1):
    arr_min, arr_max = arr.min(), arr.max()
    return ((arr - arr_min) / (arr_max - arr_min)) * (vmax - vmin) + vmin
    
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

    if np.isnan(bkgd).all():
        return

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
        
        mask = np.array([get_smask(hdul[i][0].header, calpath, post_conj) for i in range(0, len(data))])
    
        data[mask == 0] = np.nanmedian(data)

    if not silent:
        print('Creating running difference images...')

    # if bflag == 'science':
    #     footprint = morphology.disk(7)
    
    # if bflag == 'beacon':
    #     footprint = morphology.disk(2)

    # data_nostars = np.zeros(data.shape)

    # for i in range(len(data)):
    #     data_scaled = minmax_scaler(data[i], vmin=0, vmax=1)
    #     tophat_img = morphology.white_tophat(data_scaled, footprint)

    #     data_nostars[i] = data_scaled - tophat_img

    # Creation of running difference images
    
    r_dif, ind = create_rdif(time_obj, maxgap, cadence, data, hdul, wcoord, bflag, ins)
    r_dif = np.array(r_dif)
    
    if bflag == 'science':
        vmin = np.nanmedian(r_dif) - np.std(r_dif) # -1e-13
        vmax = np.nanmedian(r_dif) + np.std(r_dif) # 1e-13

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

def reduced_nobg(start, bkgd, path, datpath, ftpsc, ins, bflag, silent):
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

    bkgd_arr = bkgd
    
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

    crval = [hdul[i][0].header['crval1'] for i in range(len(hdul))]

    if ftpsc == 'A':    
        post_conj = [int(np.sign(crval[i])) for i in range(len(crval))]

    if ftpsc == 'B':    
        post_conj = [int(-1*np.sign(crval[i])) for i in range(len(crval))]

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
    time = [hdul[i][0].header['DATE-END'] for i in range(len(hdul))]

    if ins == 'hi_2':
        
        mask = np.array([get_smask(hdul[i][0].header, calpath, post_conj) for i in range(0, len(data))])
    
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

#######################################################################################################################################

def rotate_via_numpy(xy, radians,center):
    """Use numpy to build a rotation matrix and take the dot product."""
    x, y = xy
    xc,yc = center
    x= x-xc
    y= y-yc

    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, [x, y])

    x= float(m.T[0])+xc
    y= float(m.T[1])+yc

    return x, y

#######################################################################################################################################

def ecliptic_cut(data, header, bflag, ftpsc, post_conj, datetime_data, datetime_series, mode):
    
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
    
    width_cut = 1
    date_steps = len(datetime_series)

    for i in range(len(wcoord)):
                
        thetax, thetay = wcoord[i].all_pix2world(xv, yv, 0)

        tx = thetax*np.pi/180
        ty = thetay*np.pi/180
        
        pa_reg = np.arctan2(-np.cos(ty)*np.sin(tx), np.sin(ty))
        elon_reg = np.arctan2(np.sqrt((np.cos(ty)**2)*(np.sin(tx)**2)+(np.sin(ty)**2)), np.cos(ty)*np.cos(tx))
        
        delta_pa = e_pa[i]

        e_val = [(delta_pa)-1*np.pi/180, (delta_pa)+1*np.pi/180]

        if mode == 'median':
            
            if i == 0:
                dif_cut = np.zeros((date_steps, xsize))
                dif_cut[:] = np.nan
                arr_ind = 0

            else: 
                arr_ind = (np.abs(datetime_series - datetime_data[i])).argmin()

            data_mask = np.where((pa_reg > min(e_val)) & (pa_reg < max(e_val)), data[i], np.nan)

            data_med = np.nanmedian(data_mask, 0)

            dif_cut[arr_ind] = data_med

            elon_mask = np.where((pa_reg > min(e_val)) & (pa_reg < max(e_val)), elon_reg, np.nan)
            
            elongation_max = np.nanmax(elon_mask*180/np.pi)
            elongation_min = np.nanmin(elon_mask*180/np.pi)
            elongation.append(elongation_min)
            elongation.append(elongation_max)

        elif mode == 'no_median':

            if ftpsc == 'A':
                farside = -1 if post_conj else 0

            if ftpsc == 'B':
                farside = 0 if post_conj else -1

            data_rot = rotate(data[i], -delta_pa, preserve_range=True, mode='constant', cval=np.median(data[i]))
            elon_rot = rotate(elon_reg, -delta_pa, preserve_range=True, mode='constant', cval=np.nan)
            pa_rot = rotate(pa_reg, -delta_pa, preserve_range=True, mode='constant', cval=np.nan)

            farside_ids = np.array(np.where((pa_rot[:, farside].flatten() >= min(e_val)) & (pa_rot[:,farside].flatten() <= max(e_val))))
            farside_ids = farside_ids.flatten()
            min_id_farside = min(farside_ids)

            if i == 0:
                max_id_farside = max(farside_ids)            
                width_cut = max_id_farside - min_id_farside
                dif_cut = np.zeros((date_steps, width_cut, xsize))
                dif_cut[:] = np.nan
                arr_ind = 0

            else:
                max_id_farside = min_id_farside + width_cut
                arr_ind = (np.abs(datetime_series - datetime_data[i])).argmin()
                
            diff_slice = data_rot[min_id_farside:max_id_farside, :]

            dif_cut[arr_ind] = diff_slice

            elongation_max = np.nanmax(elon_rot[min_id_farside:max_id_farside+1, :]*180/np.pi)
            elongation_min = np.nanmin(elon_rot[min_id_farside:max_id_farside+1, :]*180/np.pi)
            elongation.append(elongation_min)
            elongation.append(elongation_max)

        else:
            print('Invalid mode. Exiting...')
            sys.exit()

    if mode == 'no_median':
        dif_cut = np.reshape(dif_cut, (width_cut*date_steps, xsize))

    elongation = np.array(elongation)

    return dif_cut, elongation

#######################################################################################################################################

def process_jplot(savepaths, ftpsc, ins, bflag, silent, jplot_type):
    """
    Creates Jplot from running difference images. Method similar to create_jplot_tam.pro written in IDL by Tanja Amerstorfer.
    Middle slice of each running difference is cut out, strips are aligned, time-gaps are filled with nan.

    @param savepaths: Path pointing towards running difference files of respective instrument
    @param ftpsc: Spacecraft (A/B)
    @param ins: STEREO-HI instrument (HI-1/HI-2)
    @param bflag: Science or beacon data
    @param silent: Run in silent mode
    """ 

    files = []

    for savepath in savepaths:
        for file in sorted(glob.glob(savepath + '*.fts')):
            files.append(file)

    # get times and headers from .fits files
    rdif = []
    header = []

    for i in range(len(files)):
        file = fits.open(files[i])
        rdif.append(file[0].data.copy())
        header.append(file[0].header.copy())
        file.close()

    rdif = np.array(rdif)

    if (bflag == 'science') & (ins == 'hi_1'):
        cadence = 40.0
    else:
        cadence = 120.0
        
    maxgap = 3.5

    if not silent:
        print('Getting data...')

    time_arr = [header[i]['DATE-END'] for i in range(len(header))]
    datetime_data = [datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f') for t in time_arr]

    # Determine if STEREO spacecraft are pre- or post-conjunction
    sc = 'ahead' if ftpsc == 'A' else 'behind'

    crval1 = [header[i]['crval1'] for i in range(len(header))]

    post_conj = [int(np.sign(crval1[i])) if ftpsc == 'A' else int(-1 * np.sign(crval1[i])) for i in range(len(crval1))]

    if len(set(post_conj)) == 1:
        post_conj = post_conj[0]

        if post_conj == -1:
            post_conj = False
        elif post_conj == 1:
            post_conj = True
    else:
        print('Invalid dates. Exiting...')
        sys.exit()

    if not silent:
        print('Making ecliptic cut...')

    datetime_series = np.arange(np.min(datetime_data), np.max(datetime_data) + datetime.timedelta(minutes=cadence), datetime.timedelta(minutes=cadence)).astype(datetime.datetime)

    dif_med, elongation = ecliptic_cut(rdif, header, bflag, ftpsc, post_conj, datetime_data, datetime_series, mode=jplot_type)

    dif_med = np.where(np.isnan(dif_med), np.nanmedian(dif_med), dif_med)
    elongation = np.abs(elongation)

    if not silent:
        print('Calculating elongation...')

    if not post_conj:
        tflag = 0 if ftpsc == 'A' else 1
    else:
        tflag = 1 if ftpsc == 'A' else 0

    orig = 'lower' if tflag else 'upper'

    jmap_interp = np.array(dif_med).transpose()

    # Contrast stretching
    p2, p98 = np.nanpercentile(jmap_interp, (2, 98))
    img_rescale = exposure.rescale_intensity(jmap_interp, in_range=(p2, p98))

    img_rescale = np.where(np.isnan(img_rescale), np.nanmedian(img_rescale), img_rescale)

    return jmap_interp, orig, elongation, datetime_data

#######################################################################################################################################

def plot_jplot(img_rescale, elongation, datetime_data, cadence, ftpsc, save_path, instrument, bflag, jplot_type, orig):
    """
    Plot and save a jplot image.

    Parameters:
    img_rescale (numpy.ndarray): The rescaled image data.
    elongation (numpy.ndarray): The elongation data.
    datetime_data (list): The list of datetime objects.
    cadence (int): The cadence in minutes.
    ftpsc (str): Spacecraft (A/B).
    save_path (str): The path to save the plot.
    instrument (str): STEREO-HI instrument (HI-1/HI-2).
    bflag (str): Science or beacon data.
    jplot_type (str): The jplot type.
    orig (str): The image origin (lower or upper, depends on date).

    Returns:
    None
    """

    if instrument == 'hi_1':
        vmin = np.nanmedian(img_rescale) - 1 * np.nanstd(img_rescale)
        vmax = np.nanmedian(img_rescale) + 1 * np.nanstd(img_rescale)

    if instrument == 'hi_2':
        vmin = np.nanmedian(img_rescale) - 2 * np.nanstd(img_rescale)
        vmax = np.nanmedian(img_rescale) + 2 * np.nanstd(img_rescale)

    elongations = [np.nanmin(elongation), np.nanmax(elongation)]

    time_mdates = [mdates.date2num(datetime_data[0] - datetime.timedelta(minutes=cadence / 2)),
                   mdates.date2num(datetime_data[-1] + datetime.timedelta(minutes=cadence / 2))]
    
    loc_ticks= int(np.ceil(((datetime_data[-1]-datetime_data[0]).total_seconds()/(60*60*24)*1/7)))

    fig, ax = plt.subplots(figsize=(10,5), sharex=True, sharey=True)

    plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 24), interval=loc_ticks))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
    plt.gca().xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6), interval=loc_ticks))

    plt.gca().yaxis.set_minor_locator(MultipleLocator(2))

    ax.xaxis_date()

    ax.imshow(img_rescale, cmap='gray', aspect='auto', interpolation='none', vmin=vmin, vmax=vmax, origin=orig,
              extent=[time_mdates[0], time_mdates[-1], elongations[0], elongations[-1]])

    ax.set_title(datetime.datetime.strftime(datetime_data[0], '%Y%m%d')  + ' STEREO-' + ftpsc)

    plt.ylim(elongations[0], elongations[-1])

    plt.xlabel('Date (d/m/y)')
    plt.ylabel('Elongation (°)')

    if not os.path.exists(save_path + 'pub/'):
        os.makedirs(save_path + 'pub/')

    bbi = 'tight'
    pi = 0.5

    plt.ylim(elongations[0], elongations[-1])
    
    plt.savefig(save_path + 'pub/' + 'jplot_' + instrument + '_' + datetime.datetime.strftime(datetime_data[0], '%Y%m%d') + '_' + datetime.datetime.strftime(datetime_data[-1], '%Y%m%d') + '_' + ftpsc + '_' + bflag[0] + '_' + jplot_type + '.png', bbox_inches=bbi, pad_inches=pi, dpi=300)

#######################################################################################################################################

def save_jplot_data(path, ftpsc, bflag, datetime_data, img_rescale, orig, elongation, ins, jp_name):
    """
    Save jplot data and parameters to pickle files.

    Args:
        path (str): The base path where the data will be saved.
        ftpsc (str): Spacecraft (A/B).
        bflag (str): Science or beacon data.
        datetime_data (numpy.ndarray): The array of datetime values.
        img_rescale (numpy.ndarray): The rescaled image data.
        orig (numpy.ndarray): The image origin (lower or upper, depends on date).
        elongation (numpy.ndarray): The elongation data.
        ins (str): STEREO-HI instrument (HI-1/HI-2).
        jp_name (str): The type of the jplot.

    Returns:
        None
    """

    if ins == 'hi_1':
        ins_name = 'hi1'
    
    elif ins == 'hi_2':
        ins_name = 'hi2'

    # Save path for data
    start = datetime.datetime.strftime(datetime_data[0], '%Y%m%d')
    savepath_jplot = os.path.join(path, 'jplot', ftpsc, bflag, ins, start)

    time_file_comb = datetime.datetime.strftime(np.nanmin(datetime_data), '%Y%m%d_%H%M%S')

    if not os.path.exists(savepath_jplot):
        os.makedirs(savepath_jplot)

    with open(os.path.join(savepath_jplot, f'jplot_{ins_name}_{start}_{time_file_comb}_UT_{ftpsc}_{bflag[0]}{jp_name}.pkl'), 'wb') as f:
        pickle.dump([img_rescale, orig], f)

    # Save path for params
    savepath_param = os.path.join(path, 'jplot', ftpsc, bflag, ins, str(start[:4]), 'params')

    if not os.path.exists(savepath_param):
        os.makedirs(savepath_param)

    with open(os.path.join(savepath_param, f'jplot_{ins_name}_{start}_{time_file_comb}UT_{ftpsc}_{bflag[0]}{jp_name}_params.pkl'), 'wb') as f:
        pickle.dump([datetime_data[0], datetime_data[-1], np.nanmin(elongation), np.nanmax(elongation)], f)

#######################################################################################################################################

def make_jplot(datelst, path, ftpsc, instrument, bflag, save_path, silent, jplot_type):
    """
    Creates Jplot from running difference images using process_jplot function. Saves Jplot as .png and parameters as .pkl file.

    @param datelst: List of dates for which to create Jplot
    @param path: The path where all reduced images, running difference images and J-Maps are saved
    @param ftpsc: Spacecraft (A/B)
    @param instrument: STEREO-HI instrument (HI-1/HI-2)
    @param bflag: Science or beacon data
    @param save_path: Path pointing towards downloaded STEREO .fits files
    @param silent: Run in silent mode
    """
    if not silent:
        print('-------------------')
        print('JPLOT')
        print('-------------------')
    
    jp_name = '' if jplot_type == 'median' else '_no_median'

    if bflag == 'science':
        cadence_h1 = 40.0
        cadence_h2 = 120.0
    else:
        cadence_h1 = 120.0
        cadence_h2 = 120.0

    if (instrument == 'hi_1') or (instrument == 'hi1hi2'):
        
        ins = 'hi_1'

        savepaths_h1 = [path + 'running_difference/data/' + ftpsc + '/' + dat + '/' + bflag + '/hi_1/' for dat in datelst]

        img_rescale_h1, orig_h1, elongation_h1, datetime_h1 = process_jplot(savepaths_h1, ftpsc, ins, bflag, silent, jplot_type)

        plot_jplot(img_rescale_h1, elongation_h1, datetime_h1, cadence_h1, ftpsc, path, ins, bflag, jplot_type, orig_h1)

        save_jplot_data(path, ftpsc, bflag, datetime_h1, img_rescale_h1, orig_h1, elongation_h1, ins, jp_name)

    if (instrument == 'hi_2') or (instrument == 'hi1hi2'):

        ins = 'hi_2'
        
        savepaths_h2 = [path + 'running_difference/data/' + ftpsc + '/' + dat + '/' + bflag + '/hi_2/' for dat in datelst]

        img_rescale_h2, orig_h2, elongation_h2, datetime_h2 = process_jplot(savepaths_h2, ftpsc, ins, bflag, silent, jplot_type)

        plot_jplot(img_rescale_h2, elongation_h2, datetime_h2, cadence_h2, ftpsc, path, ins, bflag, jplot_type, orig_h2)

        save_jplot_data(path, ftpsc, bflag, datetime_h2, img_rescale_h2, orig_h2, elongation_h2, ins, jp_name)

    if instrument == 'hi1hi2':

        ins = 'hi1hi2'

        savepath_h1h2 = path + 'jplot/' + ftpsc + '/' + bflag + '/hi1hi2/' + datelst[0][0:4] + '/'

        vmin_h1 = np.nanmedian(img_rescale_h1) - 1 * np.nanstd(img_rescale_h1)
        vmax_h1 = np.nanmedian(img_rescale_h1) + 1 * np.nanstd(img_rescale_h1)

        vmin_h2 = np.nanmedian(img_rescale_h2) - 2 * np.nanstd(img_rescale_h2)
        vmax_h2 = np.nanmedian(img_rescale_h2) + 2 * np.nanstd(img_rescale_h2)

        time_file_comb = datetime.datetime.strftime(min(np.nanmin(datetime_h1), np.nanmin(datetime_h2)), '%Y%m%d_%H%M%S')

        elongations = [np.nanmin(elongation_h1), np.nanmax(elongation_h1), np.nanmin(elongation_h2), np.nanmax(elongation_h2)]

        time_mdates_h1 = [mdates.date2num(datetime_h1[0] - datetime.timedelta(minutes=cadence_h1/2)), mdates.date2num(datetime_h1[-1] + datetime.timedelta(minutes=cadence_h1/2))]
        time_mdates_h2 = [mdates.date2num(datetime_h2[0] - datetime.timedelta(minutes=cadence_h2/2)), mdates.date2num(datetime_h2[-1] + datetime.timedelta(minutes=cadence_h2/2))]

        loc_ticks= int(np.ceil(((datetime_h1[-1]-datetime_h1[0]).total_seconds()/(60*60*24)*1/7)))

        fig, ax = plt.subplots(figsize=(10,5), sharex=True, sharey=True)

        plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 24), interval=loc_ticks))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
        plt.gca().xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6), interval=loc_ticks))
        
        plt.gca().yaxis.set_minor_locator(MultipleLocator(2))

        ax.xaxis_date()
        
        helcats=False
        
        if helcats:
            helcats_data = []
            
            helcats_data.append(['TRACK_NO', 'DATE', 'ELON', 'PA', 'SC'])
            
            helcats_file = glob.glob("HELCATS/HCME_"+ftpsc+"__"+datelst[0][0:4]+"_*.txt")[0]
            
            with open(helcats_file, mode="r") as f:
                for line in f:
                    helcats_data.append(line.split())
            
            helcats_data = pd.DataFrame(helcats_data[1:], columns=helcats_data[0])
            helcats_time = mdates.datestr2num(helcats_data['DATE'])
            helcats_elon = helcats_data['ELON'].values
            helcats_elon = helcats_elon.astype(np.float)
        
            ax.scatter(helcats_time, helcats_elon, marker='+', facecolor='r', linewidths=.5)

        ax.imshow(img_rescale_h1, cmap='gray', aspect='auto', interpolation='none', vmin=vmin_h1, vmax=vmax_h1, origin=orig_h1, extent=[time_mdates_h1[0], time_mdates_h1[-1], elongations[0], elongations[1]])    
        ax.imshow(img_rescale_h2, cmap='gray', aspect='auto', interpolation='none', vmin=vmin_h2, vmax=vmax_h2, origin=orig_h2, extent=[time_mdates_h2[0], time_mdates_h2[-1], elongations[2], elongations[3]]) 

        ax.set_title(datelst[0] + ' STEREO-' + ftpsc)
        
        plt.ylim(elongations[0], 80)

        plt.xlabel('Date (dd/mm/yyyy)')
        plt.ylabel('Elongation (°)')

        if not os.path.exists(savepath_h1h2 + 'pub/'):
            os.makedirs(savepath_h1h2 + 'pub/')
            
        bbi = 'tight'
        pi = 0.5        

        plt.savefig(savepath_h1h2 + 'pub/' + 'jplot_' + instrument + '_' + datelst[0] + '_' + datelst[-1] + '_' + ftpsc + '_' + bflag[0] + '_' + jplot_type + '.png', bbox_inches=bbi, pad_inches=pi, dpi=300)
        
#######################################################################################################################################

def scc_img_stats(img0):
    """
    This procedure generates image statistics for the header.

    Parameters:
    img0 (np.ndarray): Input image
    satmax (float, optional): Set saturation value of image; default is image maximum
    satmin (float, optional): Set minimum value of image; default is image minimum > 0
    verbose (bool, optional): Flag to print the statistics
    missing (np.ndarray, optional): Index of missing pixels where the statistics should not be calculated

    Returns:
    dict: A dictionary containing image statistics
    """
    
    img1 = img0.astype(float)

    img1[img1 == 0] = np.nan
    finite_mask = np.isfinite(img1)
    img = img1[finite_mask]
    zeros = np.sum(~finite_mask)
    
    # Calculate Minimum and Maximum
    mn = np.nanmin(img)
    mx = np.nanmax(img)
        
    # Calculate Standard Deviation and Mean
    sig = np.nanstd(img)
    men = np.nanmean(img)
    
    # Calculate Image Percentiles
    percentiles = [1, 10, 25, 50, 75, 90, 95, 98, 99]
    percentile = np.percentile(img, percentiles)
    
    return {
        'mn': mn,
        'mx': mx,
        'zeros': zeros,
        'men': men,
        'sig': sig,
        'percentile': percentile
    }

#######################################################################################################################################

def scc_update_hdr(im, hdr0, silent=True):
    """
    This function returns updated header structure for level 1 processing.

    Parameters:
    im (np.ndarray): Calibrated image
    hdr0 (dict): Image header, SECCHI structure
    silent (bool, optional): Flag to suppress messages

    Returns:
    dict: Updated header
    """

    hdr = hdr0.copy()
    
    # Update structure
    hdr['BSCALE'] = 1.0
    hdr['BZERO'] = 0.0
    

    # Calculate Data Dependent Values
    stats = scc_img_stats(im)
    hdr['DATAMIN'] = stats['mn']
    hdr['DATAMAX'] = stats['mx']
    hdr['DATAZER'] = stats['zeros']
    hdr['DATAAVG'] = stats['men']
    hdr['DATASIG'] = stats['sig']
    hdr['DATAP01'] = stats['percentile'][0]
    hdr['DATAP10'] = stats['percentile'][1]
    hdr['DATAP25'] = stats['percentile'][2]
    hdr['DATAP50'] = stats['percentile'][3]
    hdr['DATAP75'] = stats['percentile'][4]
    hdr['DATAP90'] = stats['percentile'][5]
    hdr['DATAP95'] = stats['percentile'][6]
    hdr['DATAP98'] = stats['percentile'][7]
    hdr['DATAP99'] = stats['percentile'][8]
    
    date_mod = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    hdr['date'] = date_mod
    
    return hdr

#######################################################################################################################################

def hi_fix_pointing(header, point_path, post_conj, ravg=5, silent=True):
    """
    Conversion of fix_pointing.pro for IDL. To read in the pointing information from the appropriate  pnt_HI??_yyyy-mm-dd_fix_mu_fov.fts file and update the
    supplied HI index with the best fit pointing information and optical parameters calculated by the minimisation
    process of Brown, Bewsher and Eyles (2008).

    @param header: Header of .fits file
    @param point_path: Path of pointing calibration files
    @param ftpsc: STEREO Spacecraft (A/B)
    @param post_conj: Is the spacecraft pre- or post conjunction (2014)
    @param ravg: Set ravg wuality parameter
    @param silent: Run in silent mode
    """

    ## CHANGE From 1 to 0 to reflect default IDL behaviour
    hi_nominal = 0

    try:
        header.rename_keyword('DATE-AVG', 'DATE_AVG')

    except ValueError:
        if not silent:
            print('Header information already corrected')

    hdr_date = header['DATE_AVG']
    hdr_date = hdr_date[0:10]

    point_file = 'pnt_' + header['DETECTOR'] + header['OBSRVTRY'][7] + '_' + hdr_date + '_' + 'fix_mu_fov.fts'
    fle = point_path + point_file
    
    if os.path.isfile(fle):

        if not silent:
            print(('Reading {}...').format(point_file))
    
        hdul_point = fits.open(fle)

        for i in range(1, len(hdul_point)):
            extdate = hdul_point[i].header['extname']
            fledate = hdul_point[i].header['filename'][0:13]

            if (header['DATE_AVG'] == extdate) or (datetime.datetime.strptime(header['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M') == fledate):
                ec = i
                break

        if (header['DATE_AVG'] == extdate) or (datetime.datetime.strptime(header['DATE-OBS'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M') == fledate):

            stcravg = hdul_point[ec].header['ravg']
            stcnst1 = hdul_point[ec].header['nst1']

            if header['naxis1'] != 0:
                sumdif = np.round(header['cdelt1'] / hdul_point[ec].header['cdelt1'])
            else:
                sumdif = 1

            if stcnst1 < 20:
                if not silent:
                    print('Subfield presumed')
                    print('Using calibrated fixed instrument offsets')

                hi_calib_point(header, post_conj, hi_nominal)
                header['ravg'] = -894.

            else:
                ## CHANGE < to <=, > to >=

                if (stcravg <= ravg) & (stcravg >= 0.):
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
                    if not silent:
                        print('R_avg does not meet criteria')
                        print('Using calibrated fixed instrument offsets')

                    hi_calib_point(header, post_conj, hi_nominal)
                    header['ravg'] = -883.

        else:
            if not silent:
                print(('No pointing calibration file found for file {}').format(point_file))
                print('Using calibrated fixed instrument offsets')

            hi_calib_point(header, post_conj, hi_nominal)
            header['ravg'] = -882.

    if not os.path.isfile(fle):
        if not silent:
            print(('No pointing calibration file found for file {}').format(point_file))
            print('Using calibrated fixed instrument offsets')

        hi_calib_point(header, post_conj, hi_nominal)
        header['ravg'] = -881.
    
    return header
#######################################################################################################################################

def hi_calib_point(header, post_conj, hi_nominal):
    """
    Conversion of hi_calib_point.pro for IDL.

    @param header: Header of .fits file
    @param post_conj: Is the spacecraft pre- or post conjunction (2014)
    @param hi_nominal: Retrieve nominal pointing values at launch (propagated to get_hi_params)
    """

    roll = hi_calib_roll(header, 'gei', post_conj, hi_nominal)

    header['pc1_1a'] = np.cos(roll * np.pi / 180.)
    header['pc1_2a'] = -np.sin(roll * np.pi / 180.)
    header['pc2_1a'] = np.sin(roll * np.pi / 180.)
    header['pc2_2a'] = np.cos(roll * np.pi / 180.)

    roll = hi_calib_roll(header, 'hpc', post_conj, hi_nominal)

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

    radec = fov2radec(xv, yv, header, 'gei', hi_nominal)

    header['crval1a'] = radec[0, 0]
    header['crval2a'] = radec[1, 0]

    radec = fov2radec(xv, yv, header, 'hpc', hi_nominal)
    header['crval1'] = -radec[0, 0]
    header['crval2'] = radec[1, 0]

    pitch_hi, offset_hi, roll_hi, mu, d = get_hi_params(header, hi_nominal)
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

def hi_calib_roll(header, system, post_conj, hi_nominal):
    """
    Conversion of hi_calib_roll.pro for IDL. Calculate the total roll angle of the HI image including
    contributions from the pitch and roll of the spacecraft. The total HI roll is a non-straighforward combination
    of the individual rolls of the s/c and HI, along with the pitch of the s/c and the offsets of HI. This
    routine calculates the total roll by taking 2 test points in the HI fov, transforms them to the
    appropriate frame of reference (given by the system keyword) and calculates the angle they make in this frame.

    @param header: Header of .fits file
    @param system: Which coordinate system to work in 'hpc' or 'gei'
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

    xy = fov2pos(xv, yv, header, system, hi_nominal)

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

def fov2pos(xv, yv, header, system, hi_nominal):
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

    ## CHANGE From if any then set all 1024
    naxis = np.where(naxis == 0, 1024, naxis)

    pmult = naxis / 2.0 - 0.5

    pitch_hi, offset_hi, roll_hi, mu, d = get_hi_params(header, hi_nominal)

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

def get_hi_params(header, hi_nominal):
    """
    Conversion of get_hi_params.pro for IDL. To detect which HI telescope is being used and return the
    instrument offsets relative to the spacecraft along with the mu parameter and the fov in degrees.
    As 'best pointings' may change as further calibration is done, it was thought more useful if there was one
    central routine to provide this data, rather than having to make the same changes in many different
    codes. Note, if you set one of the output variables to some value, then it will retain that value.

    @param header: Header of .fits file
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

    ## CHANGE Indexing changed here

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
            vout[0:2, i] = rr * vec[0:2, i]
        else:
            vout[0:2, i] = rr * vec[0:2, i] / rth

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

def fov2radec(xv, yv, header, system, hi_nominal):
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

    pitch_hi, offset_hi, roll_hi, mu, d = get_hi_params(header, hi_nominal)

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

def hi_fix_beacon_date(header):
    """
    Fix wrong date in STEREO-HI beacon data. Number of summed images in beacon header is incorrect, so dates calculated using exposure times are wrong.
    @param header: Header of .fits file
    """
    
    n_im = header['IMGSEQ'] + 1

    if header['N_IMAGES'] == 1:
        header['EXPTIME'] = header['EXPTIME']*n_im

        if n_im == 30:
            header['DATE-OBS'] = (datetime.datetime.strptime(header['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f') - datetime.timedelta(minutes=29,seconds=40)).strftime('%Y-%m-%dT%H:%M:%S.%f')
            header['DATE-CMD'] = header['DATE-OBS'] 
            header['DATE-AVG'] = (datetime.datetime.strptime(header['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f') - datetime.timedelta(minutes=14,seconds=50)).strftime('%Y-%m-%dT%H:%M:%S.%f')
        if n_im == 99:
            header['DATE-OBS'] = (datetime.datetime.strptime(header['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f') - datetime.timedelta(hours=1,minutes=38,seconds=50)).strftime('%Y-%m-%dT%H:%M:%S.%f')
            header['DATE-CMD'] = header['DATE-OBS'] 
            header['DATE-AVG'] = (datetime.datetime.strptime(header['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f') - datetime.timedelta(minutes=49,seconds=25)).strftime('%Y-%m-%dT%H:%M:%S.%f')


        header['N_IMAGES'] = n_im
#######################################################################################################################################

def scc_icerdiv2(i, d, pipeline=False, silent=True):
    """
    Correct for conditional DIV2 by on-board IP prior to ICER.

    Parameters:
    i (dict): Header index structure containing necessary tags.
    d (np.array): Image data array, replaced by corrected data array.
    pipeline (bool): If True, pipeline is set. Default is False.
    silent (bool): If True, suppress print statements. Default is True.

    Returns:
    dict: Updated header index structure.
    np.array: Updated image data array.
    str: Updated ICER message.
    """
    # Info for logging
    info = "$Id: scc_icerdiv2.pro,v 1.19 2011/09/15 21:57:35 nathan Exp $"
    histinfo = info[1:-2]

    # Get the IP commands
    ip = i['IP_00_19']
    if len(ip) < 60:
        ip = ' ' + ip
    if len(ip) < 60:
        ip = ' ' + ip
    ip = np.array([np.int8(x) for x in ip], dtype=np.int8).reshape(3, 20)

    if not silent:
        print('IP_00_19:', ip)

    w = np.where(ip != 0)
    nip = len(w[0])

    icradiv2 = 0
    idecdiv2 = 0
    icramsg = ''
    datap01 = i['DATAP01']
    biasmean = i['BIASMEAN']

    if pipeline:
        print('Pipeline = True not implemented yet - should not be used on L0.5 data.')
        sys.exit()

    # Calculate various conditions
    icer = 90 <= ip[nip-1] <= 102
    div2 = ip[nip-2] == 1
    noticfilt = ip[nip-2] < 106 or ip[nip-2] > 112
    nosubbias = np.where(ip == 103)[0].size == 0
    biasmp01 = (biasmean / 2) - datap01
    p01ltbias = abs(biasmp01) < 0.02 * (biasmean / 2)

    # Logic to determine whether data was most likely divided by 2
    domul2 = icradiv2 or idecdiv2 or (icer and noticfilt and nosubbias and p01ltbias)

    if not silent:
        print(f'{icradiv2}=icradiv2, {idecdiv2}=idecdiv2, {icer}=icer, {noticfilt}=noticfilt, {nosubbias}=nosubbias, {p01ltbias}=p01ltbias')

    if pipeline:
        print('Pipeline = True not implemented yet - should not be used on L0.5 data.')
        sys.exit()
        
    # Apply correction
    if domul2:
        m2 = np.array(2, dtype=d.dtype)
        d *= m2
        for key in ['DATAP01', 'DATAMIN', 'DATAMAX', 'DATAAVG', 'DATAP10', 'DATAP25', 'DATAP75', 'DATAP90', 'DATAP95', 'DATAP98', 'DATAP99']:
            i[key] *= 2
        i['DIV2CORR'] = 'T'

        if idecdiv2 and icradiv2:
            i['DIV2CORR'] = 'F'

        if not silent:
            print('Image corrected by icerdiv2')

        icramsg = 'Corrected for icerdiv2 because: ' + icramsg

    else:
        icramsg = 'No div2 correction: ' + icramsg

    if not silent:
        print(icramsg)

    if pipeline:
        print('Pipeline = True not implemented yet - should not be used on L0.5 data.')
        sys.exit()

    if datap01 < 0.75 * biasmean:
        if not silent:
            print('datap01:', datap01, 'biasmean:', biasmean)
            print(0.02*(biasmean/2))

    h_dex = 20 - np.sum([x == '' for x in i['HISTORY']])
    i['HISTORY'][h_dex] = histinfo

    return i, d

#######################################################################################################################################

def precommcorrect(im, hdr, extra = None, silent=True):
    """
    Apply corrections to images taken before all commissioning data is reprocessed.
    
    Args:
    im (np.ndarray): level 0.5 image from sccreadfits
    hdr (dict): level 0.5 header from sccreadfits, SECCHI structure
    extra (dict): Extra information for pointing correction (COR, EUVI only)
    silent (bool, optional): Suppress print statements if True
    """
    
    # Apply IcerDiv2 correction (Bug 49)
    if 89 < hdr['comprssn'] < 102:
        if hdr['DIV2CORR'] == 'F':
            im, hdr = scc_icerdiv2(hdr, im)
        else:
            biasmean = hdr['biasmean']
            p01mbias = hdr['datap01'] - biasmean
            if not silent:
                print(f'p01mbias: {p01mbias}, biasmean: {biasmean}')
            if p01mbias > 0.8 * hdr['biasmean']:
                im = im/2
                for key in ['datap01', 'datamin', 'datamax', 'dataavg', 'datap10', 'datap25', 'datap75', 'datap90', 'datap95', 'datap98', 'datap99']:
                    hdr[key] /= 2
                hdr['div2corr'] = 'F'

                if not silent:
                    print('Image corrected for incorrect icerdiv2', info=True)

    hdr['mask_tbl'] = 'NONE'

    # Correct Image Center
    if hdr['DETECTOR'] == 'EUVI':
        print('Precommcorrect not implemented for EUVI')
        sys.exit()
        #euvi_point(hdr, quiet=silent)
    elif hdr['DETECTOR'] == 'COR1':
        print('Precommcorrect not implemented for COR1')
        sys.exit()
        #cor1_point(hdr, SILENT=silent, **ex)
    elif hdr['DETECTOR'] == 'COR2':
        print('Precommcorrect not implemented for COR2')
        sys.exit()
        #cor2_point(hdr, SILENT=silent, **ex)

    # Add DSTART(STOP)1(2)
    if hdr['DSTOP1'] < 1 or hdr['DSTOP1'] > hdr['NAXIS1'] or hdr['DSTOP2'] > hdr['NAXIS2']:
        x1 = max(0, 51 - hdr['P1COL'])
        x2 = min(2048 + x1 - 1, hdr['P2COL'] - hdr['P1COL'])
        y1 = max(0, 1 - hdr['P1ROW'])
        y2 = min(2048 + y1 - 1, hdr['P2ROW'] - hdr['P1ROW'])

        if hdr['P1COL'] < 51:
            hdr['P1COL'] = 51

        hdr['P2COL'] = hdr['P1COL'] + (x2 - x1)

        if hdr['P1ROW'] < 1:
            hdr['P1ROW'] = 1

        hdr['P2ROW'] = hdr['P1ROW'] + (y2 - y1)

        x1 = int(x1 / 2 ** (hdr['summed'] - 1))
        x2 = int((hdr['P2COL'] - hdr['P1COL'] + 1) / 2 ** (hdr['summed'] - 1)) + x1 - 1
        y1 = int(y1 / 2 ** (hdr['summed'] - 1))
        y2 = int((hdr['P2ROW'] - hdr['P1ROW'] + 1) / 2 ** (hdr['summed'] - 1)) + y1 - 1

        if hdr['RECTIFY'] == True:
            if hdr['OBSRVTRY'] == 'STEREO_A':
                if hdr['DETECTOR'] == 'EUVI':
                    rx1 = hdr['naxis1'] - y2 - 1
                    rx2 = hdr['naxis1'] - y1 - 1
                    ry1 = hdr['naxis2'] - x2 - 1
                    ry2 = hdr['naxis2'] - x1 - 1
                    hdr['R1COL'] = 2176 - hdr['P2ROW'] + 1
                    hdr['R2COL'] = 2176 - hdr['P1ROW'] + 1
                    hdr['R1ROW'] = 2176 - hdr['P2COL'] + 1
                    hdr['R2ROW'] = 2176 - hdr['P1COL'] + 1

                    print('Precommcorrect not implemented for EUVI')
                    sys.exit()

                elif hdr['DETECTOR'] == 'COR1':
                    rx1 = y1
                    rx2 = y2
                    ry1 = hdr['naxis2'] - x2 - 1
                    ry2 = hdr['naxis2'] - x1 - 1
                    hdr['R1COL'] = hdr['P1ROW']
                    hdr['R2COL'] = hdr['P2ROW']
                    hdr['R1ROW'] = 2176 - hdr['P2COL'] + 1
                    hdr['R2ROW'] = 2176 - hdr['P1COL'] + 1

                    print('Precommcorrect not implemented for COR1')
                    sys.exit()

                elif hdr['DETECTOR'] == 'COR2':
                    rx1 = hdr['naxis1'] - y2 - 1
                    rx2 = hdr['naxis1'] - y1 - 1
                    ry1 = x1
                    ry2 = x2
                    hdr['R1COL'] = 2176 - hdr['P2ROW'] + 1
                    hdr['R2COL'] = 2176 - hdr['P1ROW'] + 1
                    hdr['R1ROW'] = hdr['P1COL']
                    hdr['R2ROW'] = hdr['P2COL']
                    print('Precommcorrect not implemented for COR2')
                    sys.exit()

                elif hdr['DETECTOR'] == 'HI1':
                    rx1 = x1
                    rx2 = x2
                    ry1 = y1
                    ry2 = y2
                    hdr['R1COL'] = hdr['P1COL']
                    hdr['R2COL'] = hdr['P2COL']
                    hdr['R1ROW'] = hdr['P1ROW']
                    hdr['R2ROW'] = hdr['P2ROW']

                elif hdr['DETECTOR'] == 'HI2':
                    rx1 = x1
                    rx2 = x2
                    ry1 = y1
                    ry2 = y2
                    hdr['R1COL'] = hdr['P1COL']
                    hdr['R2COL'] = hdr['P2COL']
                    hdr['R1ROW'] = hdr['P1ROW']
                    hdr['R2ROW'] = hdr['P2ROW']

            elif hdr['OBSRVTRY'] == 'STEREO_B':
                if hdr['DETECTOR'] == 'EUVI':
                    rx1 = y1
                    rx2 = y2
                    ry1 = hdr['naxis2'] - x2 - 1
                    ry2 = hdr['naxis2'] - x1 - 1
                    hdr['R1COL'] = hdr['P1ROW']
                    hdr['R2COL'] = hdr['P2ROW']
                    hdr['R1ROW'] = 2176 - hdr['P2COL'] + 1
                    hdr['R2ROW'] = 2176 - hdr['P1COL'] + 1
                    print('Precommcorrect not implemented for EUVI')
                    sys.exit()

                elif hdr['DETECTOR'] == 'COR1':
                    rx1 = hdr['naxis1'] - y2 - 1
                    rx2 = hdr['naxis1'] - y1 - 1
                    ry1 = x1
                    ry2 = x2
                    hdr['R1COL'] = 2176 - hdr['P2ROW'] + 1
                    hdr['R2COL'] = 2176 - hdr['P1ROW'] + 1
                    hdr['R1ROW'] = hdr['P1COL']
                    hdr['R2ROW'] = hdr['P2COL']
                    print('Precommcorrect not implemented for COR1')
                    sys.exit()

                elif hdr['DETECTOR'] == 'COR2':
                    rx1 = y1
                    rx2 = y2
                    ry1 = hdr['naxis2'] - x2 - 1
                    ry2 = hdr['naxis2'] - x1 - 1
                    hdr['R1COL'] = hdr['P1ROW']
                    hdr['R2COL'] = hdr['P2ROW']
                    hdr['R1ROW'] = 2176 - hdr['P2COL'] + 1
                    hdr['R2ROW'] = 2176 - hdr['P1COL'] + 1
                    print('Precommcorrect not implemented for COR2')
                    sys.exit()

                elif hdr['DETECTOR'] == 'HI1':
                    rx1 = hdr['naxis1'] - x2 - 1
                    rx2 = hdr['naxis1'] - x1 - 1
                    ry1 = hdr['naxis2'] - y2 - 1
                    ry2 = hdr['naxis2'] - y1 - 1
                    hdr['R1COL'] = 2176 - hdr['P2ROW']
                    hdr['R2COL'] = 2176 - hdr['P1ROW']
                    hdr['R1ROW'] = 2176 - hdr['P2COL']
                    hdr['R2ROW'] = 2176 - hdr['P1COL']

                elif hdr['DETECTOR'] == 'HI2':
                    rx1 = hdr['naxis1'] - x2 - 1
                    rx2 = hdr['naxis1'] - x1 - 1
                    ry1 = hdr['naxis2'] - y2 - 1
                    ry2 = hdr['naxis2'] - y1 - 1
                    hdr['R1COL'] = 2176 - hdr['P2ROW']
                    hdr['R2COL'] = 2176 - hdr['P1ROW']
                    hdr['R1ROW'] = 2176 - hdr['P2COL']
                    hdr['R2ROW'] = 2176 - hdr['P1COL']

            x1 = rx1
            x2 = rx2
            y1 = ry1
            y2 = ry2

        hdr['DSTART1'] = x1+1
        hdr['DSTART2'] = y1+1
        hdr['DSTOP1'] = x2+1
        hdr['DSTOP2'] = y2+1

    return im, hdr

#######################################################################################################################################

def hi_correction(im, hdr, post_conj, calpath, sebip_off=False, calimg_off=False, desmear_off=False,
                  calfac_off=False, exptime_off=False, silent=True,
                  saturation_limit=None, nsaturated=None, bias_off=False, **kw_args):
    
    version = "Applied hi_correction.pro,v 1.20 2015/02/09 14:43:14 crothers"
    
    hdr['HISTORY'] = version
    
    # Correct for SEB IP (ON)
    if not sebip_off:
        im, hdr = scc_sebip(im, hdr, silent=silent)

    # Bias Subtraction (ON)
    if bias_off:
        biasmean = 0.0
        
    else:
        biasmean = get_biasmean(hdr, silent=silent)

        if biasmean != 0.0:
            hdr['HISTORY'] = 'Bias Subtracted ' + str(biasmean)
            hdr['OFFSETCR'] = biasmean
            im -= biasmean

            if not silent:
                print(f"Subtracted BIAS={biasmean}")

    # Extract and correct for cosmic ray reports

    ### hi_cosmics modifies the images as reference! 
    cosmics = hi_cosmics(hdr, im, post_conj, silent=silent)
    im = hi_remove_saturation(im, hdr)

    if not exptime_off:
        if desmear_off:
            im /= hi_exposure_wt(hdr)

            if hdr['NMISSING'] > 0:
                im = hi_fill_missing(im, hdr, silent=silent)

            hdr['HISTORY'] = 'Applied exposure weighting'
            hdr['BUNIT'] = 'DN/s'

            if not silent:
                print("Exposure Normalized to 1 Second, exposure weighting method")

        else:
            im = hi_desmear(im, hdr, post_conj, silent=silent)
            
            if hdr['NMISSING'] > 0:
                im = hi_fill_missing(im, hdr, silent=silent)

            hdr['BUNIT'] = 'DN/s'

            if not silent:
                print("Exposure Normalized to 1 Second, desmearing method")
    
    ipkeep = hdr['IPSUM']
    # # Apply calibration factor
    if calfac_off:
        calfac = 1.0
    else:
        calfac, hdr = get_calfac(hdr, silent=silent)
    
    calfac = 1.0
    diffuse = 1.0
    
    if calfac != 1.0:
        hdr['HISTORY'] = 'Applied calibration factor ' + str(calfac)

        if not silent:
            print(f"Applied calibration factor {calfac}")

        if not calimg_off:
            diffuse = scc_hi_diffuse(hdr, ipsum=ipkeep)
            hdr['HISTORY'] = 'Applied diffuse source correction'

            if not silent:
                print("Applied diffuse source correction")
    else:
        calfac_off = True

    calimg = 1.0
    # Correction for flat field and vignetting (ON)
    if calimg_off:
        calimg = 1.0
    else:
        calimg, fn = get_calimg(hdr, calpath, post_conj)
        
        if calimg.shape[0] > 1:
            hdr['HISTORY'] = f'Applied Flat Field {fn}'
    
    # Apply Correction
    im = im * calfac * diffuse * calimg
    
    return im, hdr

#######################################################################################################################################

def hi_prep(im, hdr, post_conj, calpath, pointpath, calibrate_on=True, smask_on=False, fill_mean=True, fill_value=None, update_hdr_on=True, silent=True, **kw_args):
    """
    Conversion of hi_prep.pro for IDL. Processes the image with various corrections and updates based on the header information and flags.

    Parameters:
    -----------
    im : numpy.ndarray
        Image data to be processed.
    hdr : dict
        Header information associated with the image.
    calibrate_on : bool, optional
        If False, disables calibration corrections.
    smask_on : bool, optional
        If True, apply smoothing mask (only for HI2 detector).
    fill_mean : bool, optional
        If True, fill mask regions with mean image value.
    fill_value : float, optional
        Specific value to fill mask regions.
    update_hdr_on : bool, optional
        If False, disables updating header to Level 1 values.
    silent : bool, optional
        If True, suppress informational messages.
    corr_kw : dict, optional
        Dictionary of correction keywords passed to hi_correction().
    """

    # Update IMGSEQ for hi-res images if imgseq is not 0
    if hdr['NAXIS1'] > 1024 and hdr['IMGSEQ'] != 0 and hdr['N_IMAGES'] == 1:
        hdr['imgseq'] = 0

    # Calibration corrections
    if calibrate_on:
        im, hdr = hi_correction(im, hdr, post_conj, calpath, **kw_args)
        hdr = hi_fix_pointing(hdr, pointpath, post_conj, silent=silent)
    else:
        cosmics = -1

    # Smooth Mask (only for HI2 detector)
    if smask_on and calibrate_on and hdr['DETECTOR'] == 'HI2':
        mask = get_smask(hdr, calpath, post_conj, silent=True)
        m_dex = np.where(mask == 0)
        if fill_mean:
            im[m_dex] = np.mean(im)
        elif fill_value is not None:
            im[m_dex] = fill_value
        else:
            im *= mask
        if not silent:
            print('Mask applied to HI2 image')


    if kw_args['calfac_off'] and kw_args['nocalfac_butcorrforipsum']:
        sumcount = hdr['ipsum'] - 1
        divfactor = (2 ** sumcount) ** 2
        im = im / divfactor

        if hdr['ipsum'] > 1:
            if not silent:
                print(f'Divided image by {divfactor} to account for IPSUM')
                print('IPSUM changed to 1 in header.')
            
            hdr['history'] = f'image Divided by {divfactor} to account for IPSUM'
        
        hdr['ipsum'] = 1
        hdr['bunit'] = hdr['bunit'] + '/CCDPIX'

    # Update Header to Level 1 values
    if update_hdr_on:
        hdr = scc_update_hdr(im, hdr)

    calfac,hdr = get_calfac(hdr,'MBS')
    calfac = calfac*2.223e15

    cdelt=35.96382/3600
    summing=int(np.log(hdr["CDELT1"]/cdelt)/np.log(2.))+1
    diffuse = scc_hi_diffuse(hdr,summing)

    im = im * calfac * diffuse 

    #if we want to modify it there
    # hdr['bunit'] = 'S10'


    return im, hdr


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

    if ftpsc == 'A':
        sc = 'ahead'

    if ftpsc == 'B':
        sc = 'behind'

    savepath = path + 'reduced/data/' + ftpsc + '/' + start + '/' + bflag + '/'
    calpath = datpath + 'calibration/'
    pointpath = datpath + 'data' + '/' + 'hi/'

    for ins in instrument:
        fitsfiles = []

        start_time = time.time()
        if bflag == 'science':
            for file in sorted(glob.glob(save_path + 'stereo' + sc[0] + '/secchi/' + path_flg + '/img/'+ins+'/' + str(start) + '/*s4*.fts')):
                fitsfiles.append(file)
    
        if bflag == 'beacon':
            for file in sorted(glob.glob(save_path + 'stereo' + sc[0] + '/beacon/' + path_flg + '/img/'+ins+'/' + str(start) + '/*s7*.fts')):
                fitsfiles.append(file)


        if len(fitsfiles) == 0:
            print(save_path + 'stereo' + sc[0] + '/secchi/' + path_flg + '/img/'+ins+'/' + str(start) + '/*s4*.fts')
            print(save_path + 'stereo' + sc[0] + '/secchi/' + path_flg + '/img/'+ins+'/' + str(start) + '/*s7*.fts')
            print('No files found for ', ins, ' on ', start)
            continue

        if not os.path.exists(savepath + ins + '/'):
            os.makedirs(savepath + ins + '/')

        if not silent:
            print('----------------------------------------')
            print('Starting data reduction for', ins, '...')
            print('----------------------------------------')

        # correct for on-board sebip modifications to image (division by 2, 4, 16 etc.)
        # calls function scc_sebip

        hdul = [fits.open(fitsfiles[i]) for i in range(len(fitsfiles))]
        

        hdul_data = np.array([hdul[i][0].data for i in range(len(hdul))])
        hdul_header = [hdul[i][0].header for i in range(len(hdul))]

        # rectify = [hdul_header[i]['rectify'] for i in range(len(hdul))]

        ## CHANGE rectify inserted here
        for i in range(len(hdul)):
            if hdul_header[i]['rectify'] != True:
                hdul_header[i]['r1col'] = hdul_header[i]['p1col']
                hdul_header[i]['r2col'] = hdul_header[i]['p2col']
                hdul_header[i]['r1row'] = hdul_header[i]['p1row']
                hdul_header[i]['r2row'] = hdul_header[i]['p2row']

                hdul_data[i], hdul_header[i] = secchi_rectify(hdul_data[i], hdul_header[i])

        rectify_on =  True
        precomcorrect_on = False

        # if rectify_on == True:                    
        #     for i in range(len(hdul)):
        #         if rectify[i] != True:
        #             hdul_data[i], hdul_header[i] = secchi_rectify(hdul_data[i], hdul_header[i])

        ## CHANGE implemented precommcorrect here, is necessary for COR1, optional for HI

        

        if precomcorrect_on == False:

            if ftpsc == 'A':
                date_cutoff = datetime.datetime.strptime('2007-02-03T13:15', '%Y-%m-%dT%H:%M')
            else:
                date_cutoff = datetime.datetime.strptime('2007-02-21T21:00', '%Y-%m-%dT%H:%M')

            precomcorrect_on = (ins == 'cor1') and (hdul_header[0]['date_obs'] < date_cutoff) and (hdul_header[0]['date'] < datetime.datetime.strptime('2008-01-17', '%Y-%m-%d'))

        if precomcorrect_on == True:

            for i in range(len(hdul)):
                xh = hdul[i][1].header
                cnt_exp = np.where(xh['EXPTIME'] == 0)[0]

                if len(cnt_exp) <= 0:
                    hdul_header[i]['EXPTIME'] = np.sum(xh['EXPTIME'])

                hdul_data[i], hdul_header[i] = precommcorrect(hdul_data[i], hdul_header[i], silent=silent)
        
        if bflag == 'science':
            if ins == 'hi_1':
                norm_img = 30
                acc_img = 15

            else:
                norm_img = 99
                acc_img = 99

        if bflag == 'beacon':
            norm_img = 1
            acc_img = 1
            
        indices = []
        bad_img = []
    
        n_images = [hdul_header[i]['n_images'] for i in range(len(hdul_header))]

        if not all(val == norm_img for val in n_images):

            bad_ind = [i for i in range(len(n_images)) if (n_images[i] != norm_img) and (n_images[i] != acc_img)]
            good_ind = [i for i in range(len(n_images)) if (n_images[i] == norm_img) or (n_images[i] == acc_img)]
            bad_img+=bad_ind
            indices+=good_ind

        else:
            indices = [i for i in range(len(fitsfiles))]

        crval1_test = [int(np.sign(hdul_header[i]['crval1'])) for i in indices]
        
        if len(set(crval1_test)) > 1:

            common_crval = Counter(crval1_test)
            com_val, count = common_crval.most_common()[0]
            
            bad_ind = [i for i in range(len(crval1_test)) if crval1_test[i] != com_val]
            
            bad_img += bad_ind
            indices = list(np.setdiff1d(indices, np.array(bad_img)))

            # for i in sorted(bad_ind, reverse=True):
            #     bad_img.extend([indices[i]])
            #     del indices[i]   
                
            if len(bad_ind) >= len(indices):
                print('Too many corrupted images - can\'t determine correct CRVAL1. Exiting...')
                sys.exit()


        if bflag == 'science':
            #Must find way to do this for beacon also
            datamin_test = [hdul_header[i]['DATAMIN'] for i in indices]
            
            if not all(val == norm_img for val in datamin_test):
                
                bad_ind = [i for i in range(len(datamin_test)) if datamin_test[i] != norm_img]
                bad_img += bad_ind
                indices = list(np.setdiff1d(indices, np.array(bad_img)))

                # for i in sorted(bad_ind, reverse=True):
                #     bad_img.extend([indices[i]])
                #     del indices[i]

        if bflag == 'beacon':
            test_data = np.array([hdul_data[i] for i in indices])
            test_data = np.where(test_data == 0, np.nan, test_data)
            
            bad_ind = [i for i in range(len(test_data)) if np.isnan(test_data[i]).all() == True]
            bad_img += bad_ind
            indices = list(np.setdiff1d(indices, np.array(bad_img)))

            # for i in sorted(bad_ind, reverse=True):
            #     bad_img.extend([indices[i]])
            #     del indices[i]
                    
        missing_ind = np.array([hdul_header[i]['NMISSING'] for i in indices])

        bad_ind = [i for i in range(len(missing_ind)) if missing_ind[i] > 0]
        bad_img += bad_ind
        indices = list(np.setdiff1d(indices, np.array(bad_img)))
    
        # for i in sorted(bad_ind, reverse=True):
        #     bad_img.extend([indices[i]])
        #     del indices[i]     
        
        clean_data = []
        clean_header = []
        
        for i in range(len(fitsfiles)):
            if i in indices:
                clean_data.append(hdul_data[i])
                clean_header.append(hdul_header[i])
                hdul[i].close()
            else:
                hdul[i].close()
        
        clean_data = np.array(clean_data)

        if bflag == 'beacon':
            for i in range(len(clean_header)):
                hi_fix_beacon_date(clean_header[i])

        crval1 = [clean_header[i]['crval1'] for i in range(len(clean_header))]

        if ftpsc == 'A':    
            post_conj = [int(np.sign(crval1[i])) for i in range(len(crval1))]
    
        if ftpsc == 'B':    
            post_conj = [int(-1*np.sign(crval1[i])) for i in range(len(crval1))]
        
        if len(clean_header) == 0:
            print('No clean files found for ', ins, ' on ', start)
            return

        if len(set(post_conj)) == 1:

            post_conj = post_conj[0]
    
            if post_conj == -1:
                post_conj = False
            if post_conj == 1:
                post_conj = True

        else:
            print('Corrupted CRVAL1 in header. Exiting...')
            sys.exit()
        
        trim_off = False
        
        
        if trim_off == False:
            for i in range(len(clean_data)):
                clean_data[i], clean_header[i]= scc_img_trim(clean_data[i], clean_header[i], silent=silent)
                
        # print("time sorting bad and good data",time.time()-start_time)
        ### is it really  unecessary ? 
        # for i in range(len(clean_data)):
        #     clean_data[i], clean_header[i] = scc_putin_array(clean_data[i], clean_header[i], 1024,trim_off=trim_off, silent=silent)

        ## TODO: Implement discri_pobj.pro

        ## TODO: Implement COR_PREP.pro

        ## TODO: Implement COR_POLARIZ.pro

        ## TODO: Implement EUVI_PREP.pro

        if ins == 'hi_1':
            
            nocalfac_butcorrforipsum = True

            kw_args = {
                'rectify_on' : rectify_on,
                'precomcorrect_on' : precomcorrect_on,
                'trim_off' : trim_off,
                'nocalfac_butcorrforipsum': nocalfac_butcorrforipsum,
                'calibrate_on': True,
                'smask_on': False,
                'fill_mean': True,
                'fill_value': None,
                'update_hdr_on': True,
                'sebip_off': False,
                'calimg_off': False,
                'desmear_off': False,
                'calfac_off': nocalfac_butcorrforipsum,
                'exptime_off': False,
                'bias_off': False,
                'silent': silent,
            }

            for i in range(len(clean_data)):
                clean_data[i], clean_header[i]  = hi_prep(clean_data[i], clean_header[i], post_conj, calpath, pointpath, **kw_args)
                if bflag == 'science':
                    newname = datetime.datetime.strptime(clean_header[i]['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S') + '_1b' + ins.replace('i_', '') + ftpsc + '.fts'
                if bflag == 'beacon':
                    newname = datetime.datetime.strptime(clean_header[i]['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S') + '_17' + ins.replace('i_', '') + ftpsc + '.fts'

                fits.writeto(savepath + ins + '/' + newname, clean_data[i, :, :].astype(np.float32), clean_header[i], output_verify='silentfix', overwrite=True)

        elif ins == 'hi_2':

            nocalfac_butcorrforipsum = True

            kw_args = {
                'rectify_on' : rectify_on,
                'precomcorrect_on' : precomcorrect_on,
                'trim_off' : trim_off,
                'nocalfac_butcorrforipsum': True,
                'calibrate_on': True,
                'smask_on': True,
                'fill_mean': True,
                'fill_value': None,
                'update_hdr_on': True,
                'sebip_off': False,
                'calimg_off': False,
                'desmear_off': False,
                'calfac_off': nocalfac_butcorrforipsum,
                'exptime_off': False,
                'bias_off': False,
                'silent': silent,
            }

            for i in range(len(clean_data)):
                clean_data[i], clean_header[i] = hi_prep(clean_data[i], clean_header[i].astype(np.float32), post_conj, calpath, pointpath, **kw_args)
                if bflag == 'science':
                    newname = datetime.datetime.strptime(clean_header[i]['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S') + '_1b' + ins.replace('i_', '') + ftpsc + '.fts'
                if bflag == 'beacon':
                    newname = datetime.datetime.strptime(clean_header[i]['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S') + '_17' + ins.replace('i_', '') + ftpsc + '.fts'

                fits.writeto(savepath + ins + '/' + newname, clean_data[i, :, :].astype(np.float32), clean_header[i], output_verify='silentfix', overwrite=True)

        if not silent:
            print('Saving .fts files...')

        # if not os.path.exists(savepath + ins + '/'):
        #     os.makedirs(savepath + ins + '/')

        # else:
        #     oldfiles = glob.glob(os.path.join(savepath + ins + '/', "*.fts"))
        #     for fil in oldfiles:
        #         os.remove(fil)
                
        # for i in range(len(clean_header)):
        #     if bflag == 'science':

        #         newname = datetime.datetime.strptime(clean_header[i]['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S') + '_1b' + ins.replace('i_', '') + ftpsc + '.fts'

        #     if bflag == 'beacon':
        #         newname = datetime.datetime.strptime(clean_header[i]['DATE-END'], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d_%H%M%S') + '_17' + ins.replace('i_', '') + ftpsc + '.fts'

        #     fits.writeto(savepath + ins + '/' + newname, clean_data[i, :, :], clean_header[i], output_verify='silentfix', overwrite=True)
            
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
