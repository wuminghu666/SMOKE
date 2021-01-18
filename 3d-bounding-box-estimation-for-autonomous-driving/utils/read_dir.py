import os

class ReadDir():
    def __init__(self,
                 base_dir,
                 cam,
                 subset='training',
                 tracklet_date='2011_09_26',
                 tracklet_file ='2011_09_26_drive_0084_sync'
                 ):
        # Todo: set local base dir
        # self.base_dir = '/home/user/PycharmProjects/dataset/kitti/data_object_image_2/'
        self.cam=cam
        self.base_dir = base_dir

        # if use kitti training data for train/val evaluation
        if subset == 'training':
            self.label_dir = os.path.join(self.base_dir, subset, self.cam,'label_2/')
            self.image_dir = os.path.join(self.base_dir, subset, self.cam,'image_2/')
            self.calib_dir = os.path.join(self.base_dir, subset, self.cam,'calib/')
            # self.prediction_dir = os.path.join(self.base_dir, subset, 'box_3d/')
            self.prediction_dir = '/media/wuminghu/work/model/SMOKE/train/carla/%s/inference/kitti_train/data/'%(cam)

        # if use raw data
        if subset == 'tracklet':
            self.tracklet_drive = os.path.join(self.base_dir, tracklet_date, tracklet_file)
            self.label_dir = os.path.join(self.tracklet_drive, 'label_02/')
            self.image_dir = os.path.join(self.tracklet_drive, 'image_02/data/')
            self.calib_dir = os.path.join(self.tracklet_drive, 'calib_02/')
            self.prediction_dir = os.path.join(self.tracklet_drive, 'box_3d_mobilenet/')


if __name__ == '__main__':
    dir = ReadDir(subset='training')
    # dir_ = ReadDir(subset='tracklet',
    #                 tracklet_date='2011_09_06',
    #                 tracklet_file='2011_09_26_drive_0084_sync')
    print(dir.image_dir)
    # print(dir_.image_dir)