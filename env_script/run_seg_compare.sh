#source env_script/active_labelme_conda.sh
export PYTHONPATH=$PWD
python3 perception/script/seg_compare.py > log.txt
