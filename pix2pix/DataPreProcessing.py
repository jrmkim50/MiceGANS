import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from skimage import measure, morphology
import nibabel as nib
from scipy import ndimage as nd


def print_slices(vol):
    sn = vol.shape[2]
    figDim = np.ceil(sn**0.5).astype(int)
    fig, ax = plt.subplots(figDim, figDim, figsize=[11, 11])
    for i in range(sn):
        ax[int(i / figDim), int(i % figDim)].set_title('slice %d' % i)
        ax[int(i / figDim), int(i % figDim)].imshow(vol[:, :, i], cmap='gray')
        ax[int(i / figDim), int(i % figDim)].axis('off')
    plt.show()


def DicomFolder2Slices(path):
    files = []
    for fname in [_ for _ in os.listdir(path) if _.endswith('dcm')]:
        # print("loading: {}".format(fname))
        files.append(pydicom.dcmread(path + fname))

    print("file count: {}".format(len(files)))

    # skip files with no SliceLocation (eg scout views)
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
        else:
            skipcount = skipcount + 1

    print("skipped, no SliceLocation: {}".format(skipcount))

    # ensure they are in the correct order
    return sorted(slices, key=lambda s: s.SliceLocation)


def slices2arryNoBed(slices, removeBed = False, voxelSize = [3, 3, 3]):
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.append(len(slices))
    img3d = np.zeros(img_shape)
    img3dF = np.zeros(img_shape)

    # fill 3D array with the images from the files
    for i, s in enumerate(slices):
        img3d[:, :, i] = s.pixel_array
        if removeBed:
            img2dHU = s.RescaleSlope * s.pixel_array + s.RescaleIntercept
            img2dHUF = morphology.opening((img2dHU >= -100) * (img2dHU <= 1000), morphology.disk(2))
            img2dHUF = morphology.dilation(img2dHUF, morphology.disk(5))
            img2dHUF = morphology.convex_hull_image(img2dHUF)
            img3dF[:, :, i] =  img2dHU * (img2dHUF == True)-1000.0 * (img2dHUF == False)
    img3dF = img3dF if removeBed else img3d

    factors = [float(slices[0].PixelSpacing[0]) / voxelSize[0],
               float(slices[0].PixelSpacing[1]) / voxelSize[1],
               float(slices[1].SliceLocation - slices[0].SliceLocation) / voxelSize[2]]
    return nd.interpolation.zoom(img3dF, factors)


def preprocessPatient(patientPath, slicesSize = [128,128]):
    processedPath = '/'.join(patientPath.split('/')[:-2]) + '/Propreccessed/' + patientPath.split('/')[-1].split('_')[0]
    print(processedPath)
    try:
        os.mkdir(processedPath)
    except Exception as e:
        print(e)

    CTPath = patientPath + '/CT/'
    PETCPath = patientPath + '/PETCorrected/'
    PETUCPath = patientPath + '/PETUncorrected/'
    print(CTPath)

    img3dFR = slices2arryNoBed(DicomFolder2Slices(CTPath), removeBed=True)
    print_slices(img3dFR)
    sb = input("Enter smallest slice in range: ")
    se = input("Enter highest slice in range: ")
    c1, c2 = int(img3dFR.shape[0] / 2.0), int(img3dFR.shape[1] / 2.0)
    img3dFR = img3dFR[(c1 - int(slicesSize[0] / 2.0)):(c1 + int(slicesSize[0] / 2.0)),
              (c2 - int(slicesSize[1] / 2.0)):(c2 + int(slicesSize[1] / 2.0)), int(sb):int(se)]
    img3dFR = (1000.0*(img3dFR>1000) + -1000.0*(img3dFR<-1000) + img3dFR*(img3dFR>=-1000)*(img3dFR<=1000) + 1000.0)/2000
    NiftiImg = nib.Nifti1Image(img3dFR, np.eye(4))
    niftiName = 'CT.nii.gz'
    print(os.path.join(processedPath, niftiName))
    nib.save(NiftiImg, os.path.join(processedPath, niftiName))

    img3dFR = slices2arryNoBed(DicomFolder2Slices(PETCPath))
    c1, c2 = int(img3dFR.shape[0] / 2.0), int(img3dFR.shape[1] / 2.0)
    img3dFR = img3dFR[(c1 - int(slicesSize[0] / 2.0)):(c1 + int(slicesSize[0] / 2.0)),
              (c2 - int(slicesSize[1] / 2.0)):(c2 + int(slicesSize[1] / 2.0)), int(sb):int(se)]
    img3dFR /= np.max(img3dFR)-np.min(img3dFR)
    NiftiImg = nib.Nifti1Image(img3dFR, np.eye(4))
    niftiName = 'PETCorrected.nii.gz'
    nib.save(NiftiImg, os.path.join(processedPath, niftiName))

    img3dFR = slices2arryNoBed(DicomFolder2Slices(PETUCPath))
    img3dFR = img3dFR[(c1 - int(slicesSize[0] / 2.0)):(c1 + int(slicesSize[0] / 2.0)),
              (c2 - int(slicesSize[1] / 2.0)):(c2 + int(slicesSize[1] / 2.0)), int(sb):int(se)]
    img3dFR /= np.max(img3dFR) - np.min(img3dFR)
    NiftiImg = nib.Nifti1Image(img3dFR, np.eye(4))
    niftiName = 'PETUncorrected.nii.gz'
    nib.save(NiftiImg, os.path.join(processedPath, niftiName))


rawDataPath = '/Volumes/SP PHD U3/Bruker/CS236G_Project/DATASET/raw'
patientPaths = [file for file in os.listdir(rawDataPath) if file[0] != '.']
for p in patientPaths:
    preprocessPatient(os.path.join(rawDataPath, p))

# I did it for all sets of train and test for CT and uncorrected pet
import os, shutil

path = '/Volumes/SP PHD U3/Bruker/CS236G_Project/DATASET/test'
path = [file for file in os.listdir(path) if file[0] != '.']
for i, filename in enumerate(path):
    shutil.copyfile('/Volumes/SP PHD U3/Bruker/CS236G_Project/DATASET/test/' + filename + '/CT.nii.gz',
                    '/Users/jonathanfisher/PycharmProjects/CS236G_EBAC/dataset/PETCT/test/a/p' + str(i) + '.nii.gz')
    shutil.copyfile('/Volumes/SP PHD U3/Bruker/CS236G_Project/DATASET/test/' + filename + '/PETUncorrected.nii.gz',
                    '/Users/jonathanfisher/PycharmProjects/CS236G_EBAC/dataset/PETCT/test/b/p' + str(i) + '.nii.gz')

path = '/Volumes/SP PHD U3/Bruker/CS236G_Project/DATASET/train'
path = [file for file in os.listdir(path) if file[0] != '.']
for i, filename in enumerate(path):
    shutil.copyfile('/Volumes/SP PHD U3/Bruker/CS236G_Project/DATASET/train/' + filename + '/CT.nii.gz',
                    '/Users/jonathanfisher/PycharmProjects/CS236G_EBAC/dataset/PETCT/train/a/p' + str(i) + '.nii.gz')
    shutil.copyfile('/Volumes/SP PHD U3/Bruker/CS236G_Project/DATASET/train/' + filename + '/PETUncorrected.nii.gz',
                    '/Users/jonathanfisher/PycharmProjects/CS236G_EBAC/dataset/PETCT/train/b/p' + str(i) + '.nii.gz')



# load the DICOM files
# APath = '/Users/jonathanfisher/PycharmProjects/CS236G_EBAC/dataset/PETCT/train/a'
# BPath = '/Users/jonathanfisher/PycharmProjects/CS236G_EBAC/dataset/PETCT/train/b'
# patientPaths = [file for file in os.listdir(APath) if file[0] != '.']
# a_sum = 0
# a_slices = 0
# for p in patientPaths:
#     print('a = ', nib.load(os.path.join(APath, p)).get_fdata().mean(), nib.load(os.path.join(APath, p)).get_fdata().std())
# for p in patientPaths:
#     print('b = ', nib.load(os.path.join(BPath, p)).get_fdata().mean(), nib.load(os.path.join(BPath, p)).get_fdata().std())

