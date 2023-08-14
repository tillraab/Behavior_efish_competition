# How to competition experiment
__Workflow__ (Python-scripts/applications):
1) wavetracker.trackingGUI
2) wavetracker.EODsorter
3) LED_detect.py
4) eval_LED.py
5) trial_analysis.py
6) event_videos.py (optional)

## Raw data analysis using the wavetracker-modul

### trackingGUI.py
__Frequency extraction and tracking__
- open Raw-file (traces-grid1.raw)
- 'Spectrogram'-settings:
    - overlap fraction: 0.8
    - frequency resolution: 1
- check 'Auto-save'; press 'Run'

__Fine spectrogram__
- repeat steps above but press 'Calc. fine spec' instead of Run
  - fine spec data saved in /home/"user"/analysis/"filename"

### EODsorter.py

- load dataset/folder 
- correct tracked EOD traces
- fill EOD traces
  - fine spec data needs to be manually added to the dataset-folder

## Competition trial analysis

### trail_analysis.py

- Detection of winners, their EODf traces, rises, etc. Results stored in "data-path"/analysis.
- (optional) Meta.csv file in base-path of analyzed data. Creates entries for each 
 analyzed recording (index = file names) and stores Meta-data. Manual competation suggested.

## Video analysis

### LED_detect.py
- Detect blinking LED (emitted by electric recording setup). Used for synchronization.
- "-c" argument to identify correct detection area for LED
- '-x' (tuple) borders of LED detection window on X-axis (in pixels)
- '-y' (tuple) borders of LED detection window on Y-axis (in pixels)

### eval_LED.py
- creates time vector to synchronize electric and video recording
- for each frame contains a time-point (in s) that corresponds to the electric recordings.

## Rise videos (optional)
- generates for each detected rise a short video showing the fish's behavior around the rise event.
- sorted in 'base-path'/rise_video.