# Pupil-Core EyeTracker

## Introduction:

* The tools under this project are used for processing and visualizing the data retrieved from the application named `Pupil-Core`.
* As a first step, one needs to export data from PupilCore application.
* There are two versions for visualizing the data:
    * Real-time visualization on Video (via Command Terminal)
    * Visualization on Dashboard( Jupyter Notebook)

## How to get the exported files from Pupil-core:
* Make sure to activate the plugins given below:
    * Blink Detector
    * Head Pose Tracker
    * Raw Data Exporter
    * Surface Tracker
    * World Video Exporter
* Then check certain blink threshold on the `Blink Detector`. Check the values of 1 which suggests occurance of a blink, tighten the range till you get a subset of blinks correctly, false positives excluded.
* Then enter `Head Pose Tracker`, press calculate.
* Then press the `Download` icon on the left to export the files you need.
* You will find the files needed scattered around inside the path like `000/exports/#Number_Of_Export`.

## Real-time visualization on Video:
### Description
* This tool processes the data exported from Pupil-core and visualizes the data in the form of:
    * Head-pose and gaze plotting on a screen : Combining both features we are estimating approximately where the user is looking at the screen
    * Blink-rate : To interpret certain moments the user is experiencing, such as excitement, anxiety, we are checking blink-rate
    * Eye-Dilation : To interpret driver's focus.
    * Heatmap of gaze: Where user gazed in the screen

### How to run through Command Terminal:
    $ python main.py /
                --headpose_tracker resources/000/head_pose_tracker_poses.csv /
                --marker_detections resources/000/marker_detections.csv /
                --pupil_positions resources/000/pupil_positions.csv /
                --gaze_positions resources/000/gaze_positions.csv /
                --blinks resources/000/blinks.csv /
                --video resources/000/world.mp4 /
                --world_timestamps resources/000/world_timestamps.csv
* The scripts has lots of arguments. So it would be easy to use for users to stick with the convention of putting in the same folder for their sake.

## Visualization on Dashboard:
### Description
* This tool also processes the data exported from Pupil-core and visualizes the data in the form of:
    * Head-pose and gaze plotting on a screen : Combining both features we are estimating approximately where the user is looking at the screen
    * Eye-Dilation : To interpret driver's focus.
    * Heatmap of gaze: Where user gazed in the screen  
    * Blinks
    * Marker Counts
* But this tools, allows the cumulative graphs to be created between specified indexes. So allow the user to do comparison much easier.

### How to run Visualization on Dashboard:
* Export files as descibed above
    *  You will find the necessary files as described from an example below:
        * `head_pose_tracker_poses.csv` file would be stored in `000/exports/000/head_pose_tracker_poses.csv` which should be set as `headpose_tracker_poses_path`

        * `marker_detections.csv` file would be stored in `000/exports/000/marker_detections.csv` which should be set as `marker_detections_path`

        * `world_timestamps.csv` file would be stored in `000/exports/000/world_timestamps.csv` which should be set as `world_timestamps_path`

        * `pupil_positions.csv` file would be stored in `000/exports/000/pupil_positions.csv` which should be set as `pupil_positions_path`

        * `gaze_positions.csv` file would be stored in `000/exports/000/gaze_positions.csv` which should be set as `gaze_positions_path`

        * `blinks.csv` file would be stored in `000/exports/000/blinks.csv` which should be set as `blinks_path`

* Install jupyter
* Start jupyter lab with the following code in any terminal `jupyter lab`
* Put the file-paths inside the `Configure Filepaths` cell. As shown in the example below:
![Alt text](Configure_File_Paths.png)
* Run the code in any jupyter lab and the code will run.
