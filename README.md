# STEREO-HI-Data-Processing
Code for data reduction, creation of J-Maps and tracking of STEREO-HI beacon and science images.

## Setup

All programs in this repository work under the ‘helio_hi’ environment. It can be recreated by using the provided environment.yml and requirements.txt files using the following commands. A working installation of Anaconda is required.

This command will create the helio_hi environment:

`conda create --name helio_hi`

This command will install all required packages to the environment (this might take some time):

`pip install -r requirements.txt`

To activate the environment, simply enter:

`conda activate helio_hi`

The data reduction done in these programs is based on the IDL [SECCHI_PREP](https://hesperia.gsfc.nasa.gov/ssw/stereo/secchi/doc/secchi_prep.html) routine developed by NASA for working with STEREO image data.
The programs do not require IDL to run, but do use some calibration files distributed by NASA. More specifically, the files containing the pointing information for
STEREO as well as the files providing flatfields for the spacecraft must be obtained before data reduction is possible.

Configuration of the programs is done via the config.txt file. The config.txt file must be updatet with your own paths before running the program for the first time. The file has 10 lines in total. Each line’s function will be described here:

1. The path where all reduced images, running difference images and J-Maps are saved.
2. The path where all downloaded raw HI images will be saved.
3. The path to the aforementioned calibration files. The flatfields must be located in your_path + ‘calibration’;
   the pointing files in your_path + ‘data/hi/’.
4. Which if the two STEREO spacecraft should be used (‘A’ or ‘B’).
5. Whether to use only data from the HI-1 camera (‘hi_1’) / the HI-2 camera (‘hi_2’) or bth cameras (‘hi1hi2’).
   It is recommended that you use the ‘hi1hi2’ option as Jplots can not be generated if there is no HI-2 data at all.
6. Whether to use ‘science’ or ‘beacon’ data.
7. Which date to use as the starting date. The format is YYYYMMDD.
8. Which timeframe to use for the data. ‘week’ or ‘month’ are possible.
9. Which operation to carry out. Possibilities are ‘download’ (downloads data), ‘reduction’ (performs data reduction),
   ‘difference’ (produces running difference images), ‘jplot’ (produces J-map) and ‘all’
   (executes all previously listed operations one after the other).
10. Whether or not to save .pngs of the running difference images. Usually, running difference images are saved as .pkl files.
    Use ‘save_rdif_img’ if you wish to save the running difference images as .pngs.
11. Whether to run the program without or without output to the console. Use ‘silent’ if you wish no console output to be generated.

After filling out the config.txt file, activate the helio environment and start the program via the following command:

`python3 hi_data_processing.py`

## Tracking a CME

Once the desired J-Maps have been created, the CME can be tracked via the track_cme.py program. It takes all of its input from the config.txt file. The date, spacecraft and data type (science or beacon) in this file will determine which J-Map is displayed for tracking. Tracking is done simply by left-clicking on the desired point. Points can be removed again by right-clicking. The program must be terminated via a click on the mouse wheel (middle-click). The resulting time-elongation pairs will be saved as a .csv file. To start the tracking program, type the following command into the console:

`python3 track_cme.py`
