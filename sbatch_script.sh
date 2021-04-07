#!/bin/bash
#SBATCH --job-name=read_ais
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --partition=cpu
#SBATCH --mem-per-cpu=12G

python -m ifc_data_engineering --path_to_input ../../raw_data/20210225-AIS-006/WTDOX/WTDOX.tar.gz
python -m ifc_data_engineering --path_to_input ../../raw_data/20210225-AIS-006/F91SDOX/F91SDOX.tar.gz
python -m ifc_data_engineering --path_to_input ../../raw_data/20210225-AIS-006/L4QDOX/L4QDOX.tar.gz
python -m ifc_data_engineering --path_to_input ../../raw_data/20210225-AIS-006/R45HDOX/R45HDOX.tar.gz
wait

python -m ifc_data_engineering --path_to_input ../../raw_data/20210225-AIS-006/WTCTRL/WTCTRL.tar.gz
python -m ifc_data_engineering --path_to_input ../../raw_data/20210225-AIS-006/F91SCTRL/F91SCTRL.tar.gz
python -m ifc_data_engineering --path_to_input ../../raw_data/20210225-AIS-006/L4QCTRL/L4QCTRL.tar.gz
python -m ifc_data_engineering --path_to_input ../../raw_data/20210225-AIS-006/R45HCTRL/R45HCTRL.tar.gz
wait
