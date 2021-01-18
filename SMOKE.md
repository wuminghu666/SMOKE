conda activate SMOKE
python tools/plain_train_net.py --config-file "configs/smoke_gn_vector.yaml"
python tools/plain_train_net.py --eval-only --config-file "configs/smoke_gn_vector.yaml"



MODEL
DETECT_CLASSES
kitti
data