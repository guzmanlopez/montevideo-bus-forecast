#!/bin/bash
#SBATCH --job-name=mvd_bus_forecast_processs_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384
#SBATCH --time=3:00:00
#SBATCH --tmp=9G
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --mail-type=ALL

source /etc/profile.d/modules.sh

poetry shell
export PYTHONPATH=$PWD
python src/preparation/download_stm_bus_data.py
python src/preparation/download_bus_stops.py
python src/preparation/download_bus_tracks.py
python src/processing/process_stm_bus_data.py
python src/processing/build_bus_line_tracks_and_stops.py
python src/processing/sort_bus_stops_along_bus_track.py
python src/processing/build_adyacency_matrix.py
python src/processing/build_features_matrix.py
python src/processing/build_graph.py
