# Lineage viewer

<img src='material/viewer_output.gif' width=1034 alt="An error occurs when opening GIF, please refer to './material/viewer_output.gif'.">

We provide our implementation of visualizing cell lineage, as a downstream of our lab's imaging processing pipeline. It provides 3D rendering presentation of  color-coded cell centroids and moving trajectories, which are merged with fluorescence images.

### Prerequisites
- GrapeBio Pre: Bio Image Preprocessing

  [GrapeBio Pre](https://sourceforge.net/projects/grapebio/files/) is our home-built C++ based image preprocessing pipeline to improve the quality of 4D bio images, consisting of adaptive background subtraction, median filter de-nosing, HDR, light balance and spatio-temporal alignment of image stacks. Coupled with TGMM 2.0 below, it can substantially improve the segmenting and tracking accuracies.
  
  The z-projections of  preprocessed image dataset after proper histogram specification are located at `./dataset_img/`.

- TGMM 2.0: Cell Tracking with Gaussian Mixture Models

  [TGMM 2.0](https://bitbucket.org/fernandoamat/tgmm-paper/src/master/) performs cell segmentation and tracking for mining cell lineage, which employs a machine learning approach to division detection utilizing both lineage-based and image-based features.

  An example result of the same image dataset that we reformed is in `./dataset_mat/`. Each of the columns contains the following information:

1. Unique Id from the database to identify the point (a large integer number).
2. Time point of the nucleus.
3. x location of the nucleus centroid.
4. Same as 3 but for y location.
5. Same as 3 but for z location.
6. Id of the cell in the previous time point. It is 0 if there is no linkage. Otherwise it has the unique id of the parent from column 1.
7. Fluorescence intensity of the nucleus.
8. Skeleton id. All cells belonging to the same lineage have the same unique skeleton id.
9. Cell type classified by human observer (an integer, TE=1, ICM=2, Undefined=3). 
10. Cell type classified by intensity prediction (an integer, TE=1, ICM=2, Undefined=3). 
11. Cell cycle rate (a percentage).

  We also recommend Colormap Editor in MATLAB  to design custom colormap for visualization, and save the colormap as a MAT file. An example including MAT file and PNG image is located at `./colormap/`.

### Getting started 
- Clone this repo:
```bash
git clone https://github.com/Helab-bioinformatics/lineage-viewer.git lineage-viewer
cd lineage-viewer
```
- Install Python-OpenCV, Mayavi2 and other dependencies (e.g., VTK, PyQT5).

  For pip users, please type the command `pip install -r requirements.txt`.

  For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.
  
  #Update 2021.7.23:
  
  By itself Mayavi is not a difficult package to install, but its dependencies are unfortunately rather heavy. If `import VTK` doesn't work, try available wheels in this [website](https://www.lfd.uci.edu/~gohlke/pythonlibs/), such as `VTK-8.2.0-cp36-cp36m-win_amd64.whl` and `mayavi-4.7.1+vtk82-cp36-cp36m-win_amd64.whl`.
  
- Install FFmpeg multimedia framework.

  Download [FFmpeg](https://ffmpeg.org/download.html) and add it to Windows path using environment variables.

- Make frames and video using lineage viewer.

  To use lineage viewer for visualizing cell lineage, run `python lineage_viewer.py`. The results containing image frames and video will be stored at `./output/`.
  
  In this instance, VTK objects including centroids and trajectories are both color-coded with intensity information (`mat_col_index=6`).The functions in `Process` class is a template of post-process, which is only applicate for given example dataset.
```python
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

    # FFmpeg Makes Images Frames Into Video
    process.rename_for_video(img_path='./output/', img_type='tiff')
    lineage_viewer.make_video(frame_path='./output/', video_output_path='./output/', width=1034, height=938)
```

### Citation
If you use [GrapeBio Pre](https://sourceforge.net/projects/grapebio/files/) for your research, please cite our [paper](https://doi.org/10.1038/s41556-020-0475-2).
```
@article{yue2020long,
  title={Long-term, in toto live imaging of cardiomyocyte behaviour during mouse ventricle chamber formation at single-cell resolution},
  author={Yue, Yanzhu and Zong, Weijian and Li, Xin and Li, Jinghang and Zhang, Youdong and Wu, Runlong and Liu, Yazui and Cui, Jiahao and Wang, Qianhao and Bian, Yunkun and others},
  journal={Nature cell biology},
  year={2020},
}
```
If you use [TGMM 2.0](https://bitbucket.org/fernandoamat/tgmm-paper/src/master/) mentioned in this repo, please cite the following [paper](https://doi.org/10.1016/j.cell.2018.09.031).
```
@article{mcdole2018toto,
  title={In toto imaging and reconstruction of post-implantation mouse development at the single-cell level},
  author={McDole, Katie and Guignard, L{\'e}o and Amat, Fernando and Berger, Andrew and Malandain, Gr{\'e}goire and Royer, Lo{\"\i}c A and Turaga, Srinivas C and Branson, Kristin and Keller, Philipp J},
  journal={Cell},
  year={2018},
}
```