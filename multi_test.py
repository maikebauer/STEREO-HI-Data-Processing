from functions import data_reduction, running_difference, make_jplot, download_files
from multiprocessing import Pool
from itertools import repeat
import numpy as np
from multiprocessing import set_start_method
import datetime
import sys
from time import time as timer

start_t = timer()

def main():

  silent = False

  config_path = 'config.txt'
  file = open(config_path, 'r')

  config = file.readlines()

  path = config[0].splitlines()[0]
  datpath = config[1].splitlines()[0]
  ftpsc = config[2].splitlines()[0]
  instrument = config[3].splitlines()[0]
  bflag = config[4].splitlines()[0]
  start = config[5].splitlines()[0]
  mode = config[6].splitlines()[0]
  task = config[7].splitlines()[0]
  save_jpeg = config[8].splitlines()[0]
  silent = config[9].splitlines()[0]

  if save_jpeg == 'save_rdif_jpeg':
    save_jpeg = True

  else:
    save_jpeg = False

  if silent == 'silent':
    silent = True

  else:
    silent = False

  if bflag == 'science':
    path_flg = 'L0'

  if bflag == 'beacon':
    path_flg = bflag

  if ftpsc == 'A':
    sc = 'ahead'

  if ftpsc == 'B':
    sc = 'behind'

  date = datetime.datetime.strptime(start, '%Y%m%d')


  if mode == 'week':
    ran = 8
    interv = np.arange(ran)

  if mode == 'month':
    ran = 40
    interv = np.arange(ran)
    interv_down = np.linspace(0, ran-8, 5)
    interv_jplot = np.arange(ran-8)

    datelist_down = [datetime.datetime.strftime(date + datetime.timedelta(days=int(i)), '%Y%m%d') for i in interv_down]
    datelist_jplot = [datetime.datetime.strftime(date + datetime.timedelta(days=int(i)), '%Y%m%d') for i in interv_jplot]

  datelist = [datetime.datetime.strftime(date + datetime.timedelta(days=int(i)), '%Y%m%d') for i in interv]

  p = Pool(len(datelist))

  if task == 'download':

    if mode == 'week':
      download_files(start, path, ftpsc, instrument, bflag, silent)

    if mode == 'month':
      p.starmap(download_files, zip(datelist_down, repeat(path), repeat(ftpsc), repeat(instrument), repeat(bflag), repeat(silent)))

    print('\n')

    print('Files saved to:', '/nas/helio/data/STEREO/secchi/' + path_flg + '/' + sc[0] + '/img/hi_1/')

    print('--------------------------------------------------------------------------------')

    print('Files saved to:', '/nas/helio/data/STEREO/secchi/' + path_flg + '/' + sc[0] + '/img/hi_2/')

  if task == 'reduction':
    p.starmap(data_reduction, zip(datelist, repeat(path), repeat(datpath), repeat(ftpsc), repeat(instrument), repeat(bflag), repeat(silent)))

    print('\n')

    print('Files saved to:', path + 'reduced/chosen_dates/' + bflag + '/hi_1/')

    print('--------------------------------------------------------------------------------')

    print('Files saved to:', path + 'reduced/chosen_dates/' + bflag + '/hi_2/')

  if task == 'difference':
    p.starmap(running_difference, zip(datelist, repeat(path), repeat(datpath), repeat(ftpsc), repeat(instrument), repeat(bflag), repeat(silent), repeat(save_jpeg)))

    #running_difference(start, path, datpath, ftpsc, instrument, bflag, silent, save_jpeg)
    print('\n')

    print('Pickle files saved to:', path + 'running_difference/data/'+ bflag + '/hi_1/chosen_dates/')

    print('--------------------------------------------------------------------------------')

    print('Pickle files saved to:', path + 'running_difference/data/'+ bflag + '/hi_2/chosen_dates/')

    print('--------------------------------------------------------------------------------')

    if save_jpeg:

      print('jpeg/png files saved to:', path + 'running_difference/pngs/'+ bflag + '/hi_1/chosen_dates/')

      print('--------------------------------------------------------------------------------')

      print('jpeg/png files saved to:', path + 'running_difference/pngs/'+ bflag + '/hi_2/chosen_dates/')

  if task == 'jplot':

    if mode == 'week':
      make_jplot(start, path, datpath, ftpsc, instrument, bflag, silent)

    if mode == 'month':
      p.starmap(make_jplot, zip(datelist_jplot, repeat(path), repeat(datpath), repeat(ftpsc), repeat(instrument), repeat(bflag), repeat(silent),))

    print('\n')

    print('Jplots saved to:', path + 'jplot/' + bflag + '/hi_1/' + str(start[0:4]) + '/')

    print('--------------------------------------------------------------------------------')

    print('Jplots saved to:', path + 'jplot/' + bflag + '/hi_2/' + str(start[0:4]) + '/')

    print('--------------------------------------------------------------------------------')

  if task == 'all':

    if mode == 'week':
      download_files(start, path, ftpsc, instrument, bflag, silent)

    if mode == 'month':
      p.starmap(download_files, zip(datelist_down, repeat(path), repeat(ftpsc), repeat(instrument), repeat(bflag), repeat(silent)))

    p.starmap(data_reduction, zip(datelist, repeat(path), repeat(datpath), repeat(ftpsc), repeat(instrument), repeat(bflag), repeat(silent)))

    p.starmap(running_difference, zip(datelist, repeat(path), repeat(datpath), repeat(ftpsc), repeat(instrument), repeat(bflag), repeat(silent), repeat(save_jpeg)))

    if mode == 'week':
      make_jplot(start, path, datpath, ftpsc, instrument, bflag, silent)

    if mode == 'month':
      p.starmap(make_jplot, zip(datelist_jplot, repeat(path), repeat(datpath), repeat(ftpsc), repeat(instrument), repeat(bflag), repeat(silent),))

  print('\n')
  print('Done.')

  hours, rem = divmod(timer() - start_t, 3600)
  minutes, seconds = divmod(rem, 60)
  print("Elapsed Time: {} minutes {} seconds".format(int(minutes),int(seconds)))

if __name__ == '__main__':
    main()
