from functions import data_reduction, running_difference, make_jplot, run_all, download_files
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

  file = open('config.txt', 'r')
  config = file.readlines()
  start = config[5].splitlines()[0]
  mode = config[6].splitlines()[0]
  task = config[7].splitlines()[0]
  save_jpeg = config[8].splitlines()[0]

  if save_jpeg == 'save_rdif_jpeg':
    save_jpeg = True

  else:
    save_jpeg = False

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
      download_files(start, silent)

    if mode == 'month':
      p.starmap(download_files, zip(datelist_down, repeat(silent)))

  if task == 'data_reduction':
    p.starmap(data_reduction, zip(datelist, repeat(silent)))

  if task == 'running_difference':
    p.starmap(running_difference, zip(datelist, repeat(silent), repeat(save_jpeg)))

  if task == 'jplot':

    if mode == 'week':
      make_jplot(start, silent)

    if mode == 'month':
      p.starmap(make_jplot, zip(datelist_jplot, repeat(silent)))

  print('Done.')
  print(f"Elapsed Time: {timer() - start_t}")

if __name__ == '__main__':
    main()
