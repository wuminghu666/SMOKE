
$ python kitti_object.py
```
Specific your own folder,
```shell
$ python kitti_object.py -d /path/to/kitti/object
```

Show LiDAR only
```
$ python kitti_object.py --show_lidar_with_depth --img_fov --const_box --vis
```

Show LiDAR and image
```
$ python kitti_object.py --show_lidar_with_depth --img_fov --const_box --vis --show_image_with_boxes
```

Show image with specific index
```
python kitti_object.py  --img_fov --const_box --vis --show_image_with_boxes --ind 100 
```

Show LiDAR with label (5 vector)
```
$ python kitti_object.py --show_lidar_with_depth --img_fov --const_box --vis --pc_label
```



Firstly, map KITTI official formated results into data directory
```
./map_pred.sh /media/wuminghu/work/model/SMOKE/train/inference/kitti_train/data
```

```python
python kitti_object.py -p