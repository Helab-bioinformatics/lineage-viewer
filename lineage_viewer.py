#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Author  : Yuanhui Liao
# @File    : lineage_viewer.py

from mayavi import mlab
from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm
import cv2
import numpy as np
import scipy.io as sio
import os
import math


class Settings:
    def __init__(self, matFile, colormapFile, imgPath):
        self.matFile = matFile
        self.colormapFile = colormapFile
        self.imgPath = imgPath
        self.colormapPath = './colormap/'
        self.datasetPath = './dataset_mat/'
        self.outputPath = './output/'
        self.zScale = 2 / 0.40625
        self.fps = 15
        self.prefix = 'frame'
        self.padding = 3
        self.ext = '.tiff'
        if not os.path.exists(self.outputPath):
            os.makedirs(self.outputPath)

    def print_info(self):
        print('*****Lineage Viewer*****')
        print('Viewer settings:')
        print('Dataset = \'{}\''.format(self.datasetPath + self.matFile))
        print('Colormap = \'{}\''.format(self.colormapPath + self.colormapFile))
        print('Output path = \'{}\''.format(self.outputPath))
        print('Z-axis scale = {}'.format(self.zScale))
        print('FFMpeg settings:')
        print('fps = {}'.format(self.fps))
        print('prefix = \'{}\''.format(self.prefix))
        print('padding = {}'.format(self.padding))
        print('ext = \'{}\''.format(self.ext))

    def set_fps(self, fps):
        self.fps = fps
        return fps

    def set_ext(self, ext):
        self.ext = ext
        return ext


class Viewer(Settings):
    def __init__(self, settings_obj):
        super().__init__(matFile=settings_obj.matFile,
                         colormapFile=settings_obj.colormapFile,
                         imgPath=settings_obj.imgPath)
        self.print_info()

    def read_files(self, img_type='tif'):
        read_dataset = sio.loadmat(self.datasetPath + self.matFile)
        mat = read_dataset["trackingMatrix"]
        read_colormap = sio.loadmat(self.colormapPath + self.colormapFile)
        colormap = read_colormap["mycmap"]

        # Z-axis rescale
        mat[:, 4] = mat[:, 4] * self.zScale

        img_list = os.listdir(self.imgPath)
        img_list.sort(key=lambda x: int(x.split('.')[0]))
        tif_list = []
        for img in img_list:
            if img.endswith('.' + img_type):
                tmp = os.path.join(os.path.abspath(self.imgPath), img)
                tif_list.append(tmp)

        return mat, colormap, tif_list

    def paint_cell(self, mat, colormap, tp, mat_col_index, low_cutoff, high_cutoff, if_fix_z_postition=True):
        self.adjust_cutoffs(hist=mat[:, mat_col_index], low_cutoff=low_cutoff, high_cutoff=high_cutoff)
        tmp_hist = self.mapping_to_colorbar(hist=mat[:, mat_col_index], rows=colormap.shape[0])
        tmp_tp_index_list = np.argwhere(mat[:, 1] == tp)[:, 0]

        for index in tmp_tp_index_list:
            if math.isnan(mat[index, mat_col_index]):
                tmp_hist[index] = 0
            z_position = 10 if if_fix_z_postition else mat[index, 4]
            r_value = float(colormap[tmp_hist[index], 0])
            g_value = float(colormap[tmp_hist[index], 1])
            b_value = float(colormap[tmp_hist[index], 2])
            mlab.points3d(mat[index, 3], mat[index, 2], z_position, resolution=40, color=(r_value, g_value, b_value),
                          scale_factor=10, name=str(mat[index, 0])).actor.actor.rotate_y(0)

    def paint_trajectory(self, mat, colormap, start_tp, end_tp, mat_col_index, low_cutoff, high_cutoff, if_fix_z_postition=True):
        self.adjust_cutoffs(hist=mat[:, mat_col_index], low_cutoff=low_cutoff, high_cutoff=high_cutoff)
        tmp_hist = self.mapping_to_colorbar(hist=mat[:, mat_col_index], rows=colormap.shape[0])
        start_index = np.argwhere(mat[:, 1] == start_tp)[:, 0][0]
        end_index = np.argwhere(mat[:, 1] == end_tp)[:, 0][-1]

        for t in range(start_tp, end_tp + 1):
            for i in range(start_index, end_index + 1):
                cur_tp = int(mat[i, 1])
                for j in range(i, -1, -1):
                    if mat[j, 1] == cur_tp - 1 and mat[i, 5] == mat[j, 0]:
                        begin_x = mat[j, 3]
                        begin_y = mat[j, 2]
                        begin_z = mat[j, 4] if not if_fix_z_postition else 10
                        end_x = mat[i, 3]
                        end_y = mat[i, 2]
                        end_z = mat[i, 4] if not if_fix_z_postition else 10
                        slope = np.linspace(0, 1, 5)
                        x = begin_x + slope * (end_x - begin_x)
                        y = begin_y + slope * (end_y - begin_y)
                        z = begin_z + slope * (end_z - begin_z)

                        if math.isnan(mat[i, mat_col_index]):
                            tmp_hist[i] = 0
                        r_value = float(colormap[tmp_hist[i], 0])
                        g_value = float(colormap[tmp_hist[i], 1])
                        b_value = float(colormap[tmp_hist[i], 2])
                        mlab.plot3d(x, y, z, color=(r_value, g_value, b_value), tube_radius=0.7, tube_sides=10)

                    if mat[j, 1] < cur_tp - 1:
                        break

    def make_frames(self, mat, colormap, img_list, mat_col_index, low_cutoff=93, high_cutoff=160, img_start_index=1, img_end_index=None, sliding_window=4,
                    actor_position=None, if_paint_cell=True, if_paint_trajectory=True, if_savefig=True):
        if actor_position is None or if_paint_cell | if_paint_trajectory is False:
            actor_position = [0, 0, 0]
        if img_end_index is None:
            img_end_index = len(img_list) if len(img_list) > 0 else 1
        pbar = tqdm(total=img_end_index - img_start_index + 1, desc='Make image frames', position=0)

        for i in range(img_start_index - 1, img_end_index):
            mlab.figure(size=(1560, 1560), fgcolor=(0, 0, 0), bgcolor=(0, 0, 0))
            mlab.view(azimuth=0, elevation=0, distance=50)
            arr = np.array([cv2.imread(img_list[i], 0)])
            tif_obj = mlab.imshow(arr[0][:][:])
            tif_obj.actor.position = actor_position

            if if_paint_cell:
                self.paint_cell(mat, colormap, i, mat_col_index, low_cutoff, high_cutoff)
            if if_paint_trajectory:
                start_tp = i - sliding_window if i - sliding_window > 0 else 0
                end_tp = i
                self.paint_trajectory(mat, colormap, start_tp, end_tp, mat_col_index, low_cutoff, high_cutoff)

            filename = os.path.join(self.outputPath, '{}{}'.format(i + 1, self.ext))
            if not if_savefig:
                mlab.show()
            if if_savefig:
                mlab.savefig(filename=filename)
                mlab.clf()
                mlab.close()
            pbar.update(1)
        pbar.close()

    def make_video(self, frame_path, video_output_path, width, height, verbose=True):
        ffmpeg_filename = os.path.join(frame_path, '{}%0{}d{}'.format(self.prefix, 3, self.ext))
        cmd = 'ffmpeg -f image2 -r {} -i {} -pix_fmt yuv420p -strict experimental -vf scale={}:{} -acodec aac -vcodec libx264 -y {}{}.mp4'.format(
            self.fps, ffmpeg_filename, width, height, video_output_path, 'video')
        os.system(cmd)

        # Remove temp image files
        if not verbose:
            [os.remove(files) for files in os.listdir(frame_path) if files.endswith(self.ext)]

        print('Video has been written to {}\\{}.'.format(os.path.abspath(video_output_path), 'video.mp4'))

    @staticmethod
    def mapping_to_colorbar(hist, rows):
        return [int(round((hist[i] - hist.min()) / (hist.max() - hist.min()) * (rows - 1)))
                for i in range(hist.shape[0])]

    @staticmethod
    def adjust_cutoffs(hist, low_cutoff, high_cutoff):
        for i in range(hist.shape[0]):
            if hist[i] < low_cutoff:
                hist[i] = low_cutoff
            if hist[i] > high_cutoff:
                hist[i] = high_cutoff
        return hist


class Process(Settings):
    def __init__(self, settings_obj):
        super().__init__(matFile=settings_obj.matFile,
                         colormapFile=settings_obj.colormapFile,
                         imgPath=settings_obj.imgPath)
        self.font = ImageFont.truetype('arial.ttf', 32)

    def add_mask(self, img_path, img_start_index, img_end_index, left_top_x, left_top_y, width, height,
                 output_path=None, if_savefig=True):
        if output_path is None:
            output_path = self.outputPath
        pbar = tqdm(total=img_end_index - img_start_index + 1, desc='Add masks', position=0)

        for index in range(img_start_index, img_end_index + 1):
            img = Image.fromarray(cv2.imread(img_path + str(index) + '.tiff'))
            draw = ImageDraw.Draw(img)
            draw.rectangle((left_top_x, left_top_y, left_top_x + width, left_top_y + height), fill=(0, 0, 0),
                           outline="#000000", width=1)
            img_processed = np.array(img)

            if if_savefig:
                cv2.imwrite(output_path + str(index) + '.tiff', img_processed)
            if not if_savefig:
                cv2.imshow("add_mask", img_processed)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            pbar.update(1)
        pbar.close()

    def add_text(self, mat, img_path, img_start_index, img_end_index, output_path=None, if_savefig=True):
        if output_path is None:
            output_path = self.outputPath
        pbar = tqdm(total=img_end_index - img_start_index + 1, desc='Add texts', position=0)

        for index in range(img_start_index, img_end_index + 1):
            img = Image.fromarray(cv2.imread(img_path + str(index) + '.tiff'))
            draw = ImageDraw.Draw(img)
            cell_number_per_tp = self.count_cells(mat, index)
            totalTime = (index - 1) * 5
            hours = totalTime // 60
            minutes = totalTime % 60

            draw.text((1026, 260), "Klf2-H2Bmcherry", font=self.font, fill=(255, 255, 255))
            draw.text((315, 260), str(hours) + " h " + str(minutes) + " min", font=self.font, fill=(255, 255, 255))
            draw.text((315, 290), str(cell_number_per_tp) + " cells", font=self.font, fill=(255, 255, 255))
            # len_bar: 48*1019/400 = 122
            draw.text((1161, 1069), "20 " + chr(956) + "m", font=self.font, fill=(255, 255, 255))
            draw.rectangle((1145, 1110, 1145 + 122, 1110 + 4), fill=(255, 255, 255), outline="#FFFFFF", width=1)
            img_processed = np.array(img.crop((277, 230, 1311, 1169)))

            if if_savefig:
                cv2.imwrite(output_path + str(index) + '.tiff', img_processed)
            if not if_savefig:
                cv2.imshow("add_text", img_processed)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            pbar.update(1)
        pbar.close()

    def add_colorbar(self, colorbar_img, img_path, img_start_index, img_end_index, output_path=None, if_savefig=True):
        if output_path is None:
            output_path = self.outputPath
        pbar = tqdm(total=img_end_index - img_start_index + 1, desc='Add colorbars', position=0)

        for index in range(img_start_index, img_end_index + 1):
            background = Image.open(img_path + str(index) + '.tiff')
            foreground = Image.open(colorbar_img)
            background.paste(foreground.resize((250, 25), Image.ANTIALIAS), (40, 880))
            background.save(output_path + str(index) + ".tiff", subsampling=0)

            img = Image.fromarray(cv2.imread(img_path + str(index) + '.tiff'))
            draw = ImageDraw.Draw(img)
            draw.text((40, 840), "Min", font=self.font, fill=(255, 255, 255))
            draw.text((230, 840), "Max", font=self.font, fill=(255, 255, 255))
            img_processed = np.array(img)

            if if_savefig:
                cv2.imwrite(output_path + str(index) + '.tiff', img_processed)
            if not if_savefig:
                cv2.imshow("add_colorbar", img_processed)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            pbar.update(1)
        pbar.close()

    @staticmethod
    def count_cells(mat, tp):
        return len(np.argwhere(mat[:, 1] == tp)[:, 0])

    @staticmethod
    def rename_for_video(img_path, img_type='tiff'):
        file_list = os.listdir(img_path)
        file_list.sort(key=lambda x: int(x.split('.')[0]))
        index = 0
        for filename in file_list:
            if filename.endswith('.' + img_type):
                # XXX -> frame(XXX-1)
                src = os.path.join(os.path.abspath(img_path), filename)
                dst = os.path.join(os.path.abspath(img_path), 'frame' + str(index).zfill(3) + '.tiff')
                os.rename(src, dst)
                index += 1
        print('Image frames have been renamed for making video.')


if __name__ == '__main__':
    # Class Instantiation
    example_settings = Settings(matFile='example_dataset.mat',
                                colormapFile='white_magenta_colormap.mat',
                                imgPath='./dataset_img/')
    lineage_viewer = Viewer(example_settings)
    process = Process(example_settings)

    # Merge VTK Ojects With Fluorescence Images
    trackingMatrix, mycmap, imglist = lineage_viewer.read_files(img_type='tif')
    lineage_viewer.make_frames(mat=trackingMatrix, colormap=mycmap, img_list=imglist, mat_col_index=6,
                               img_start_index=1, img_end_index=384, actor_position=[200, 200, 0])

    # Post-process
    process.add_mask(img_path='./output/', img_start_index=1, img_end_index=203,
                     left_top_x=200, left_top_y=200, width=700, height=200)
    process.add_mask(img_path='./output/', img_start_index=284, img_end_index=384,
                     left_top_x=200, left_top_y=200, width=700, height=100)
    process.add_text(mat=trackingMatrix, img_path='./output/', img_start_index=1, img_end_index=384)
    process.add_colorbar(colorbar_img='./colormap/insert_colorbar_white_magenta.png',
                         img_path='./output/', img_start_index=1, img_end_index=384)

    # FFmpeg Makes Image Frames Into Video
    process.rename_for_video(img_path='./output/', img_type='tiff')
    lineage_viewer.make_video(frame_path='./output/', video_output_path='./output/', width=1034, height=938)
