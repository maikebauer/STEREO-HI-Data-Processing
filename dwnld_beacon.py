import requests
from bs4 import BeautifulSoup
from multiprocessing.pool import ThreadPool
from time import time as timer
import os, glob
import wget
import pandas as pd
import datetime
# downloads STEREO beacon images from NASA server

fitsfil = []
ftpsc = str(input('Enter the spacecraft (ahead/behind):'))
instrument = str(input('Enter the instrument (hi_1/hi_2/both):'))
start = str(input('Enter the start date (YYYYMMDD/today):'))

if instrument == 'both':
    instrument = ['hi_1', 'hi_2']

elif instrument == 'hi_1':
    instrument = ['hi_1']

elif instrument == 'hi_2':
    instrument = ['hi_2']

else:
    instrument = ['hi_1', 'hi_2']
    print('Invalid instrument specification, downloading HI-1 and 2 data.')

if start == 'today':
    start = datetime.datetime.today() - datetime.timedelta(days=7)
    start = start.strftime('%Y%m%d')

date = datetime.datetime.strptime(start, '%Y%m%d')

datelist = pd.date_range(date, periods=7).tolist()

datelist_int = [str(datelist[i].year)+datelist[i].strftime('%m')+datelist[i].strftime('%d') for i in range(len(datelist))]
# listfd makes makes list of urls and corresponding file names to download


def listfd(input_url, extension):
    output_urls = []
    page = requests.get(input_url).text

    soup = BeautifulSoup(page, 'html.parser')
    url_found = [input_url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(extension)]
    filename = [node.get('href') for node in soup.find_all('a') if node.get('href').endswith(extension)]

    for i in range(len(filename)):
        output_urls.append((filename[i], url_found[i]))

    return output_urls

# mk_dir creteas directory for downloades files


def mk_dir(d):

    if not os.path.exists(path):
        try:
            os.mkdir(path)

        except OSError:
            print('Creation of the directory %s failed' % path)
        else:
            print('Directory successfuly created!\n')

    if glob.glob(path + d + '*.fts'):
        flg = False

    if not glob.glob(path + d + '*.fts'):
        flg = True

    return flg

# fetch_url downloads the urls specified by listfd


def fetch_url(entry):
    filename, uri = entry

    if not os.path.exists(path + filename):
        wget.download(uri, path)

    return path + filename


start_t = timer()

# creates list of urls using listfd -> passes list of urls on to fetch_url which starts the download

print('Fetching files...')

for ins in instrument:
    for date in datelist_int:

        url = 'https://stereo-ssc.nascom.nasa.gov/pub/beacon/' + ftpsc + '/secchi/img/' + ins + '/' + str(date)
        ext = 'fts'
        file = open('config.txt', 'r')
        path_dir = file.readlines()
        path_dir = path_dir[0].splitlines()[0]
        path = path_dir + str(datelist_int[0]) + '_' + ftpsc + '_' + ins + '/'

        flag = mk_dir(date)
        if flag:
            urls = listfd(url, ext)
            print(date)
            fitsfil.extend(ThreadPool(8).map(fetch_url, urls))
        if not flag:
            print('Files have already been downloaded')
            for file in glob.glob(path+date+'*.fts'):
                fitsfil.append(file)

if ftpsc == 'ahead':
    SC = 'A'
else:
    SC = 'B'

print('Done!')
#print(f"Elapsed Time: {timer() - start_t}")