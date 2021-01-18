import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
from PIL import Image
import csv

import sys
sys.path.append("/home/wuminghu/3D_vision/SMOKE/3d-bounding-box-estimation-for-autonomous-driving/utils")

from read_dir import ReadDir
from config import config as cfg
from correspondece_constraint import *


VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2

def compute_birdviewbox(line, shape, scale):
    npline = [np.float64(line[i]) for i in range(1, len(line))]
    h = npline[7] * scale
    w = npline[8] * scale
    l = npline[9] * scale
    x = npline[10] * scale
    y = npline[11] * scale
    z = npline[12] * scale
    rot_y = npline[13]

    R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                  [np.sin(rot_y), np.cos(rot_y)]])
    t = np.array([x, z]).reshape(1, 2).T

    x_corners = [0, l, l, 0]  # -l/2
    z_corners = [w, w, 0, 0]  # -w/2


    x_corners += -w / 2
    z_corners += -l / 2

    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])
    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = t - corners_2D
    # in camera coordinate
    corners_2D[0] += int(shape/2)
    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T

    return np.vstack((corners_2D, corners_2D[0,:]))

def draw_birdeyes(ax2, line_gt, line_p, shape):
    # shape = 900
    scale = 15

    pred_corners_2d = compute_birdviewbox(line_p, shape, scale)
    gt_corners_2d = compute_birdviewbox(line_gt, shape, scale)

    codes = [Path.LINETO] * gt_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(gt_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='red', label='ground truth')
    ax2.add_patch(p)

    codes = [Path.LINETO] * pred_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='green', label='prediction')
    ax2.add_patch(p)

def draw_birdeyes_gt(ax2, line_gt, shape,scale):
    gt_corners_2d = compute_birdviewbox(line_gt, shape, scale)

    codes = [Path.LINETO] * gt_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(gt_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='red', label='ground truth')
    ax2.add_patch(p)

def draw_birdeyes_pd(ax2, line_p, shape,scale):
    pred_corners_2d = compute_birdviewbox(line_p, shape, scale)

    codes = [Path.LINETO] * pred_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='green', label='prediction')
    ax2.add_patch(p)
def compute_3Dbox(P2, line):
    obj = detectionInfo(line)
    # Draw 2D Bounding Box
    xmin = int(obj.xmin)
    xmax = int(obj.xmax)
    ymin = int(obj.ymin)
    ymax = int(obj.ymax)
    # width = xmax - xmin
    # height = ymax - ymin
    # box_2d = patches.Rectangle((xmin, ymin), width, height, fill=False, color='red', linewidth='3')
    # ax.add_patch(box_2d)

    # Draw 3D Bounding Box

    R = np.array([[np.cos(obj.rot_global), 0, np.sin(obj.rot_global)],
                  [0, 1, 0],
                  [-np.sin(obj.rot_global), 0, np.cos(obj.rot_global)]])

    x_corners = [0, obj.l, obj.l, obj.l, obj.l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, obj.h, obj.h, 0, 0, obj.h, obj.h]  # -h
    z_corners = [0, 0, 0, obj.w, obj.w, obj.w, obj.w, 0]  # -w/2

    x_corners = [i - obj.l / 2 for i in x_corners]
    y_corners = [i - obj.h for i in y_corners]
    z_corners = [i - obj.w / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([obj.tx, obj.ty, obj.tz]).reshape((3, 1))

    # corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(corners_3D)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    return corners_2D

def draw_3Dbox(ax, P2, line, color):

    corners_2D = compute_3Dbox(P2, line)

    width = corners_2D[:, 3][0] - corners_2D[:, 1][0]
    height = corners_2D[:, 2][1] - corners_2D[:, 1][1]

    # draw all lines through path
    # https://matplotlib.org/users/path_tutorial.html
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
    verts = bb3d_on_2d_lines_verts.T
    codes = [Path.LINETO] * verts.shape[0]
    codes[0] = Path.MOVETO
    # codes[-1] = Path.CLOSEPOLYq
    pth = Path(verts, codes)
    p = patches.PathPatch(pth, fill=False, color=color, linewidth=2)


    # put a mask on the front
    front_fill = patches.Rectangle((corners_2D[:, 1]), width, height, fill=True, color=color, alpha=0.4)
    ax.add_patch(p)
    ax.add_patch(front_fill)

def visualization(args, image_path, label_path, calib_path, pred_path,
                  dataset, VEHICLES):

    for index in range(start_frame, end_frame):
        image_file = os.path.join(image_path, dataset[index]+ '.png')
        label_file = os.path.join(label_path, dataset[index] + '.txt')
        prediction_file = os.path.join(pred_path, dataset[index]+ '.txt')
        # calibration_file = os.path.join(calib_path, dataset[index] + '.txt')
        # for line in open(calibration_file):
        #     if 'P2' in line:
        #         P2 = line.split(' ')
        #         P2 = np.asarray([float(i) for i in P2[1:]])
        #         P2 = np.reshape(P2, (3, 4))

        with open(os.path.join(calib_path, "%.6d.txt"%(0)), 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=' ')
            for line, row in enumerate(reader):
                if row[0] == 'P2:':
                    P2 = row[1:]
                    P2 = [float(i) for i in P2]
                    P2 = np.array(P2, dtype=np.float32).reshape(3, 3)
                    P2 = P2[:3, :3]
                    break
        fig = plt.figure(figsize=(20.00, 5.12), dpi=100)

        # fig.tight_layout()
        gs = GridSpec(1, 4)
        gs.update(wspace=0)  # set the spacing between axes.

        ax = fig.add_subplot(gs[0, :3])
        ax2 = fig.add_subplot(gs[0, 2:])

        # with writer.saving(fig, "kitti_30_20fps.mp4", dpi=100):
        image = Image.open(image_file).convert('RGB')
        shape = 900
        birdimage = np.zeros((shape, shape, 3), np.uint8)

        with open(label_file) as f1, open(prediction_file) as f2:
####################################################################################
            for line_gt in f1:
                line_gt = line_gt.strip().split(' ')
                corners_2D = compute_3Dbox(P2, line_gt)
                width_half = (corners_2D[:, 3][0] - corners_2D[:, 1][0])/2
                height_half = (corners_2D[:, 2][1] - corners_2D[:, 1][1])/2
                # if corners_2D[:, 3][0]-width_half*0.9>VIEW_WIDTH or corners_2D[:, 2][1]-height_half>VIEW_HEIGHT or corners_2D[:, 1][0]+width_half*0.9<0 or corners_2D[:, 1][1]+height_half<0:
                #     continue
                if corners_2D[:, 3][0]-width_half*0.9>VIEW_WIDTH or corners_2D[:, 2][1]-height_half>VIEW_HEIGHT:
                    continue
                draw_3Dbox(ax, P2, line_gt, 'red')
                draw_birdeyes_gt(ax2, line_gt, shape,15)
####################################################################################
            for line_p in f2:
                line_p = line_p.strip().split(' ')
                corners_2D = compute_3Dbox(P2, line_p)
                width_half = (corners_2D[:, 3][0] - corners_2D[:, 1][0])/2
                height_half = (corners_2D[:, 2][1] - corners_2D[:, 1][1])/2
                # if corners_2D[:, 3][0]-width_half*0.9>VIEW_WIDTH or corners_2D[:, 2][1]-height_half>VIEW_HEIGHT or corners_2D[:, 1][0]+width_half*0.9<0 or corners_2D[:, 1][1]+height_half<0:
                #     continue
                if corners_2D[:, 3][0]-width_half*0.9>VIEW_WIDTH or corners_2D[:, 2][1]-height_half>VIEW_HEIGHT:
                    continue
                draw_3Dbox(ax, P2, line_p, 'green')
                draw_birdeyes_pd(ax2, line_p, shape,15)
####################################################################################
            # for line_gt, line_p in zip(f1, f2):
            #     import pdb; pdb.set_trace()
            #     line_gt = line_gt.strip().split(' ')
            #     line_p = line_p.strip().split(' ')

            #     truncated = np.abs(float(line_p[1]))
            #     occluded = np.abs(float(line_p[2]))
            #     trunc_level = 1 if args.a == 'training' else 255

            # # truncated object in dataset is not observable
            #     if line_p[0] in VEHICLES  and truncated < trunc_level:
            #         color = 'green'
            #         if line_p[0] == 'Cyclist':
            #             color = 'yellow'
            #         elif line_p[0] == 'Pedestrian':
            #             color = 'cyan'
            #         draw_3Dbox(ax, P2, line_p, color)
            #         # draw_3Dbox(ax, P2, line_gt, 'yellow')
            #         draw_birdeyes(ax2, line_gt, line_p, shape)
############################################################################################

        # visualize 3D bounding box
        ax.imshow(image)
        ax.set_xticks([]) #remove axis value
        ax.set_yticks([])

        # plot camera view range
        x1 = np.linspace(0, shape / 2)
        x2 = np.linspace(shape / 2, shape)
        ax2.plot(x1, shape / 2 - x1, ls='--', color='grey', linewidth=1, alpha=0.5)
        ax2.plot(x2, x2 - shape / 2, ls='--', color='grey', linewidth=1, alpha=0.5)
        ax2.plot(shape / 2, 0, marker='+', markersize=16, markeredgecolor='red')

        # visualize bird eye view
        ax2.imshow(birdimage, origin='lower')
        ax2.set_xticks([])
        ax2.set_yticks([])
        # add legend
        handles, labels = ax2.get_legend_handles_labels()
        legend = ax2.legend([handles[0], handles[-1]], [labels[0], labels[-1]], loc='lower right',
                            fontsize='x-small', framealpha=0.2)
        for text in legend.get_texts():
            plt.setp(text, color='w')

        print(dataset[index])
        if args.save == False:
            plt.show()
        else:
            fig.savefig(os.path.join(args.path, dataset[index]), dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
        # video_writer.write(np.uint8(fig))

def compute_location_loss(args, image_path, label_path, calib_path, pred_path,
                  dataset, VEHICLES):

    sum_dis=0
    count=0
    sum_gt=0
    sum_match=0
    with open(os.path.join(calib_path, "%.6d.txt"%(0)), 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == 'P2:':
                P2 = row[1:]
                P2 = [float(i) for i in P2]
                P2 = np.array(P2, dtype=np.float32).reshape(3, 3)
                P2 = P2[:3, :3]
                break
    for index in range(start_frame, end_frame):
        label_file = os.path.join(label_path, dataset[index] + '.txt')
        prediction_file = os.path.join(pred_path, dataset[index]+ '.txt')

        with open(label_file) as f1, open(prediction_file) as f2:
            line_gts = [line_gt.strip().split(' ') for line_gt in f1]
            line_tem=[]
            for line_gt in line_gts:
                corners_2D = compute_3Dbox(P2, line_gt)
                width_half = (corners_2D[:, 3][0] - corners_2D[:, 1][0])/2
                height_half = (corners_2D[:, 2][1] - corners_2D[:, 1][1])/2
                # if corners_2D[:, 3][0]-width_half>VIEW_WIDTH or corners_2D[:, 2][1]-height_half>VIEW_HEIGHT or corners_2D[:, 1][0]+width_half<0 or corners_2D[:, 1][1]+height_half<0:
                #     continue
                if corners_2D[:, 3][0]-width_half>VIEW_WIDTH or corners_2D[:, 2][1]-height_half>VIEW_HEIGHT :
                    continue
                line_tem.append(line_gt)
            line_gts=line_tem
            obj_gts = [detectionInfo(line_gt) for line_gt in line_gts]

            line_ps = [line_p.strip().split(' ') for line_p in f2]
            line_tem=[]
            for line_p in line_ps:
                corners_2D = compute_3Dbox(P2, line_p)
                width_half = (corners_2D[:, 3][0] - corners_2D[:, 1][0])/2
                height_half = (corners_2D[:, 2][1] - corners_2D[:, 1][1])/2
                # if corners_2D[:, 3][0]-width_half>VIEW_WIDTH or corners_2D[:, 2][1]-height_half>VIEW_HEIGHT or corners_2D[:, 1][0]+width_half<0 or corners_2D[:, 1][1]+height_half<0:
                #     continue
                if corners_2D[:, 3][0]-width_half>VIEW_WIDTH or corners_2D[:, 2][1]-height_half>VIEW_HEIGHT:
                    continue
                line_tem.append(line_p)
            line_ps=line_tem
            obj_ps = [detectionInfo(line_p) for line_p in line_ps]

            # if len(line_ps)!=len(line_gts):
            #     print(len(line_ps),len(line_gts))

            g_xys=[np.array([obj_gt.tz,obj_gt.tx,obj_gt.rot_global]) for obj_gt in obj_gts if np.sqrt(obj_gt.tz**2+obj_gt.tx**2)>0 and np.sqrt(obj_gt.tz**2+obj_gt.tx**2)<20]
            p_xys=[np.array([obj_p.tz,obj_p.tx,obj_p.rot_global]) for obj_p in obj_ps]
            # dis=[[(np.linalg.norm(g_xy[:2]-p_xy[:2])+abs(g_xy[2]-p_xy[2])/10)for p_xy in p_xys] for g_xy in g_xys ]
            dis=[[np.linalg.norm(g_xy[:2]-p_xy[:2]) for p_xy in p_xys] for g_xy in g_xys ]
            dis_sorts=[sorted(miny_dis) for miny_dis in dis]
            sum_gt+=len(dis_sorts)
            # import pdb; pdb.set_trace()
            for dis_sort in dis_sorts:
                if len(dis_sort)==0 :
                    continue
                if  dis_sort[0]<0.5:
                    sum_dis+=dis_sort[0]
                    sum_match+=1
                    count+=1
    if count==0:
        return
    return sum_dis/count,sum_match/sum_gt
    # if count!=0:
    #     print(sum_dis/count)
    #     print(sum_match/sum_gt)

def main(args):
    # cams=['cam5','cam5','cam_mix_1_2']
    cams=['cam7']
    sum_locaton_error,sum_precision,sum_count=0,0,0
    for cam in cams:
        base_dir = '/media/wuminghu/work/output/SMOKE/object1/'
        dir = ReadDir(base_dir=base_dir, subset=args.a,
                    tracklet_date='2011_09_26', tracklet_file='2011_09_26_drive_0093_sync',cam=cam)
        label_path = dir.label_dir
        image_path = dir.image_dir
        calib_path = dir.calib_dir
        pred_path = dir.prediction_dir

        dataset = [name.split('.')[0] for name in sorted(os.listdir(pred_path))]
        # import pdb; pdb.set_trace()

        VEHICLES = cfg().KITTI_cat
        visualization(args, image_path, label_path, calib_path, pred_path,dataset, VEHICLES)
    #     locaton_error,precision=compute_location_loss(args, image_path, label_path, calib_path, pred_path,dataset, VEHICLES)
    #     sum_locaton_error+=locaton_error
    #     sum_precision+=precision
    #     sum_count+=1
    # print(sum_locaton_error/sum_count,sum_precision/sum_count,sum_count)

if __name__ == '__main__':
    start_frame = 0
    end_frame = 5000

    parser = argparse.ArgumentParser(description='Visualize 3D bounding box on images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '-dataset', type=str, default='training', help='training dataset or tracklet')
    parser.add_argument('-s', '--save', type=bool, default=True, help='Save Figure or not')
    parser.add_argument('-p', '--path', type=str, default='/media/wuminghu/work/output/SMOKE/bird_eye', help='Output Image folder')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        os.mkdir(args.path)

    main(args)
