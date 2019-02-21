import h5py
import numpy as np
import scipy.misc as sm
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from osgeo import gdal
from osgeo import ogr
from osgeo import gdalconst
# import gdal
from gdalconst import GA_ReadOnly
from subprocess import call
import os
import glob
from sklearn.model_selection import train_test_split
from keras import utils
from keras.callbacks import Callback
from sklearn.feature_extraction.image import extract_patches_2d
import keras.backend as K

import numpy as np


def replace_no_data(data, value, no_data_val=None):
    if no_data_val is None:
        no_data_val = data.min()
    data[data == no_data_val] = value
    return data




def get_biggest_divisor(number, upper_bound=100):
    ret = None
    for i in range(1, upper_bound + 1):
        if number % i == 0:
            ret = i
    return ret


# ~ def get_possible_patch_size(dtm)

def get_dtm_patches(image, matsize, winsize):
    i = 0
    break_now = False
    while (True):
        j = 0
        while (True):
            yield (i, j, image[i:i + matsize[0], j:j + matsize[1]])
            j = j + matsize[1] - winsize[1]
            if image.shape[1] - j < matsize[1]:
                yield (i, j, image[i:i + matsize[0], j:j + matsize[1]])
                break
        i = i + matsize[0] - winsize[0]
        if break_now:
            break
        if image.shape[0] - i < matsize[0]:
            break_now = True


def get_dtm_patches2(image, matsize, winsize):
    counteri = 0
    i = 0
    break_now = False
    while (True):
        j = 0
        counterj = 0
        while (True):
            j = j + matsize[1] - winsize[1]
            if image.shape[1] - j < matsize[1]:
                break
            counterj += 1
        i = i + matsize[0] - winsize[0]
        if break_now:
            break
        if image.shape[0] - i < matsize[0]:
            break_now = True
        counteri += 1
    return counteri, counterj


def get_big_dtm_patches(image, matrix_size, model_window):
    # slide a window across the image
    stepSize = matrix_size - model_window
    for y in range(0, image.shape[0] - stepSize, stepSize):
        for x in range(0, image.shape[1] - stepSize, stepSize):
            # yield the current window
            yield (y, x, image[y:y + matrix_size, x:x + matrix_size])


def polygon_to_raster(tif_file, shp_file, output_file, field_name, nodata_val):
    ndsm = tif_file
    shp = shp_file
    data = gdal.Open(ndsm, gdalconst.GA_ReadOnly)
    geo_transform = data.GetGeoTransform()
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * data.RasterXSize
    y_min = y_max + geo_transform[5] * data.RasterYSize
    x_res = data.RasterXSize
    y_res = data.RasterYSize
    mb_v = ogr.Open(shp)
    mb_l = mb_v.GetLayer()
    pixel_width = geo_transform[1]

    output = output_file
    target_ds = gdal.GetDriverByName('GTiff').Create(output, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, pixel_width))
    band = target_ds.GetRasterBand(1)

    NoData_value = nodata_val
    band.SetNoDataValue(NoData_value)
    band.FlushCache()
    gdal.RasterizeLayer(target_ds, [1], mb_l, options=["ATTRIBUTE=" + field_name])
    target_ds = None


def polygon_to_raster_new(tif_file, shp_file, output_file, nodata_val=-3.40282346639e+38, burn_value=1,
                          field_name=None):
    ndsm = tif_file
    shp = shp_file
    data = gdal.Open(ndsm, gdalconst.GA_ReadOnly)
    geo_transform = data.GetGeoTransform()
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * data.RasterXSize
    y_min = y_max + geo_transform[5] * data.RasterYSize
    x_res = data.RasterXSize
    y_res = data.RasterYSize
    mb_v = ogr.Open(shp)
    mb_l = mb_v.GetLayer()
    pixel_width = geo_transform[1]

    output = output_file
    target_ds = gdal.GetDriverByName('GTiff').Create(output, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geo_transform)
    band = target_ds.GetRasterBand(1)

    NoData_value = nodata_val
    band.SetNoDataValue(NoData_value)
    band.FlushCache()
    if field_name is not None:
        gdal.RasterizeLayer(target_ds, [1], mb_l, options=["ATTRIBUTE=" + field_name])
    else:
        gdal.RasterizeLayer(target_ds, [1], mb_l, None, None, [burn_value], ['ALL_TOUCHED=TRUE'])
    target_ds = None




def clip_raster_by_raster(small_raster,
                          output_folder=r'O:\forschung\MWK_NLD_proNI\dat\Segmentation\similartogis\marslynormalized_data',
                          big_raster=r'O:\forschung\MWK_NLD_proNI\dat\Segmentation\marsly_normalized_dtm1.tif'):
    data = gdal.Open(small_raster, GA_ReadOnly)
    geoTransform = data.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * data.RasterXSize
    miny = maxy + geoTransform[5] * data.RasterYSize
    _, name = os.path.split(small_raster)
    output_raster = os.path.join(output_folder, name)
    call('gdal_translate -projwin ' + ' '.join(
        [str(x) for x in [minx, maxy, maxx, miny]]) + ' -of GTiff ' + big_raster + ' ' + output_raster, shell=True)



def clip_raster_by_extent(raster, output, extent):
    """
	clips a raster (tif file) based on the extent given, writes results to output file
	:param raster: raster to clip
	:param output: file to save the clipped raster
	:param extent: [minx,maxy,maxx,miny] the bounding box
	:return:
	"""
    call('gdal_translate -projwin ' + ' '.join(
        [str(x) for x in extent]) + ' -of GTiff ' + raster + ' ' + output, shell=True)


def clip_raster_by_shp_file(raster, shp_file, output):
	import ogr
	driver = ogr.GetDriverByName('ESRI Shapefile')
	shp_source = driver.Open(shp_file, 0)
	shp_layer = shp_source.GetLayer()
	extent = shp_layer.GetExtent()
	minx,maxx,miny,maxy = list(extent)
	new_extent = [minx, maxy,maxx, miny]
	clip_raster_by_extent(raster, output, new_extent)


def get_desired_extent(height, width, extent):
    minx, maxx, miny, maxy = extent
    difx = maxx - minx
    dify = maxy - miny
    extrax = (height - difx) / 2
    extray = (width - dify) / 2

    minx = minx - extrax
    maxx = maxx + extrax
    miny = miny - extray
    maxy = maxy + extray
    return minx, maxy,maxx, miny


def create_training_patches(shapefile, raster_file, data_folder, label_folder, height=80, width=80,
                            raster_resolution=0.5, fieldname=None):
    import ogr, gdal, os
    height = int(height * raster_resolution)
    width = int(width * raster_resolution)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp_source = driver.Open(shapefile, 0)
    shp_layer = shp_source.GetLayer()
    path, name = os.path.split(raster_file)
    just_name, _ = os.path.splitext(name)
    rasterized_shp = os.path.join(path, r'masked_' + name)
    polygon_to_raster_new(raster_file, shapefile, rasterized_shp, field_name=fieldname)
    # if not os.path.exists(rasterized_shp):
    #     polygon_to_raster_new(raster_file, shapefile, rasterized_shp, field_name=fieldname)
    for i, feature in enumerate(shp_layer):
        geom = feature.GetGeometryRef()
        extent = geom.GetEnvelope()
        real_extent = get_desired_extent(height, width, extent)
        data_file = os.path.join(data_folder, just_name + str(i) + r'.tif')
        label_file = os.path.join(label_folder, just_name + str(i) + r'.tif')
        clip_raster_by_extent(raster_file, data_file, real_extent)
        clip_raster_by_extent(rasterized_shp, label_file, real_extent)


def get_big_dtm_patches2(image, matrix_size, matrix_size2, model_window):
    # slide a window across the image
    stepSize = matrix_size - model_window
    stepSize2 = matrix_size2 - model_window
    for y in range(0, image.shape[0] - stepSize, stepSize):
        for x in range(0, image.shape[1] - stepSize2, stepSize2):
            # yield the current window
            yield (y, x, image[y:y + matrix_size, x:x + matrix_size2])




def write_h5f(variables, variable_names, file_name):
    """
    Writes variables with variable_names to file named file_name
    :param variables: list of variables
    :param variable_names: list of names for each variable
    :param file_name: name of the file to save to
    :return:
    """
    h5f = h5py.File(file_name, 'w')
    for (v, vn) in zip(variables, variable_names):
        h5f.create_dataset(vn, data=v)
    h5f.close()


def create_training_data(image_folder=r'O:\forschung\MWK_NLD_proNI\dat\Segmentation\similartogis\marslynormalized_data',
                         label_folder=r'O:\forschung\MWK_NLD_proNI\dat\Segmentation\similartogis\gdal_files',
                         output_npy_name='marslynormed_seg_data.npy'):
    tif_files = glob.glob(image_folder + '\*.tif')
    num_examples = len(tif_files)
    images = np.empty((num_examples, 100, 100))
    labels = np.empty((num_examples, 100, 100))
    i = 0
    not_counter = 0
    for el in tif_files:
        raster = gdal.Open(el)
        imarray = np.array(raster.ReadAsArray())
        if imarray.shape == (100, 100):
            images[i] = imarray
            _, label_file = os.path.split(el)
            label_file = os.path.join(label_folder, label_file)
            label_raster = gdal.Open(label_file)
            label_array = np.array(label_raster.ReadAsArray())
            labels[i] = label_array
            i = i + 1
        else:
            not_counter = not_counter + 1
    images = images[:i]
    labels = labels[:i]
    xtrain, xtest, ytrain, ytest = train_test_split(images, labels, test_size=0.20, random_state=42)
    ytrain = utils.to_categorical(ytrain, num_classes=4)
    ytest = utils.to_categorical(ytest, num_classes=4)
    np.save(output_npy_name, [xtrain, ytrain, xtest, ytest])


def read_h5f(file_name):
    """
    reads data from file_name
    :param file_name: name of the file to read data from
    :return: data in the h5 file
    """
    h5f = h5py.File(file_name, 'r')
    ret = [h5f[vn][:] for vn in list(h5f.keys())]
    h5f.close()
    return ret


def read_tif(filename):
    # print('filename: {}'.format(filename))
    raster = gdal.Open(filename, GA_ReadOnly)
    imarray = np.array(raster.ReadAsArray())
    # geotransform = raster.GetGeoTransform()
    return imarray  # , geotransform


def write_hm_to_tif(tif, labels, hm_file=None, hm=None, output_dir='', hm_name=''):
    if hm is None:
        hm = read_h5f(hm_file)[0]
    for i in range(len(labels)):
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(output_dir + hm_name + labels[i] + '.tif', hm[i].shape[0], hm[i].shape[1], 1,
                                gdal.GDT_Float32)
        dataset.GetRasterBand(1).WriteArray(hm[i])
        original_data = gdal.Open(tif)
        geotrans = original_data.GetGeoTransform()
        proj = original_data.GetProjection()
        dataset.SetGeoTransform(geotrans)
        dataset.SetProjection(proj)
        dataset.FlushCache()
        dataset = None
        # ~ original_data.flushCache()
        original_data = None


def crop_tif_and_save(original_tif, newRow, newCol, output_tif, xOff, yOff):
    # the axes are reversed when calling driver.create, for some weird reason.
    # xOff is the offset from original array at xAxis
    # yOff is the offset from original array yAxis
    # driver = gdal.GetDriverByName('GTiff')
    # dataset = driver.Create(output_tif, newCol, newRow, 1, gdal.GDT_Float32)
    # original_data = gdal.Open(original_tif)
    # geotrans = original_data.GetGeoTransform()
    # proj = original_data.GetProjection()
    # original_array = np.array(original_data.ReadAsArray())
    # new_array = original_array[:newRow,:newCol]
    # print(new_array.shape)
    # print(original_array.shape)
    # dataset.GetRasterBand(1).WriteArray(new_array)
    # dataset.SetGeoTransform(geotrans)
    # dataset.SetProjection(proj)
    # dataset.FlushCache()
    # dataset = None
    # original_data = None
    call('gdal_translate -srcwin ' + ' '.join(
        [str(x) for x in [xOff, yOff, newCol, newRow]]) + ' -of GTiff ' + original_tif + ' ' + output_tif, shell=True)


def write_numpy_array_to_tif(numpyarray, tifname, original_tif=None):
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(tifname, numpyarray.shape[1], numpyarray.shape[0], 1, gdal.GDT_Float32)
    dataset.GetRasterBand(1).WriteArray(numpyarray)
    if original_tif is not None:
        original_data = gdal.Open(original_tif, gdalconst.GA_ReadOnly)
        geotrans = original_data.GetGeoTransform()
        proj = original_data.GetProjection()
        dataset.SetGeoTransform(geotrans)
        dataset.SetProjection(proj)
        original_data = None
    dataset.FlushCache()
    dataset = None


def plot_heatmap(hm, labels=["rest", "waterway", "way"], model_name=None):
    if model_name is None:
        model_name = "model"
    cmap = clr.LinearSegmentedColormap.from_list('custom blue', ['#244162', '#DCE6F1'], N=256)
    i = 0
    for l in labels:
        fig, ax = plt.subplots()
        cax = ax.imshow(hm[i], interpolation='nearest', cmap=cmap)
        ax.set_title(model_name + ' detecting the ' + l + ' objects')
        cbar = fig.colorbar(cax, ticks=[hm.min(), hm[i].max()])
        cbar.ax.set_yticklabels(['Certain that the object does not exist here', 'Cetain that the object exists here'])
        fig.savefig(model_name + '_' + l + '_plots.png')
        plt.show()
        i = i + 1


def write_heatmap_to_png(hm, labels=["rest", "waterway", "way"],
                         directory="/home/kazimi/PycharmProjects/ikg-nld/data/output/", model_name=None):
    """
    takes a heatmap produced by a classifier, the labels, and saves it in the directory as png file.
    :param hm:
    :param labels:
    :param directory:
    :param model_name:
    :return:
    """
    if model_name != None:
        directory = directory + model_name
    i = 0
    for l in labels:
        sm.toimage(hm[i]).save(directory + l + ".png")
        i += 1


def normalize_like_deepsat(data):
    """
    Takes data set in shape of (n, row, col, channel) and normalizes it with
    subtracting the min difference between max and min. like in DeepSat
    :param data: input data
    :return: normalized input data
    """
    minimum = data.min()
    maximum = data.max()
    diff = maximum - minimum
    return (data - minimum) / diff


def normalize_like_marsnet(data):
    """
    Takes data set in shape of (n, row, col, channel) and normalizes it with
    subtracting the mean and dividing by std.
    :param data: input data
    :return: normalized input data
    """
    norm_data = np.empty([data.shape[0], data.shape[1], data.shape[2]])  # a better initialization?
    for i in range(data.shape[0]):
        std = data[i].std()
        mu = data[i].mean()
        norm_data[i] = (data[i] - mu) / std
    return norm_data


def normalize_like_marsnet2(data):
    """
    Takes data set in shape of (n, row, col, channel) and normalizes it with
    subtracting the mean and dividing by std.
    :param data: input data
    :return: normalized input data
    """
    std = data.std()
    mu = data.mean()
    normalized_data = (data - mu) / std
    return normalized_data, mu, std


def normalize_test_set(data, std, mu):
    normalized_data = (data - mu) / std
    return normalized_data


def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[1]//2, x.shape[2]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[:, centerw-halfw:centerw+halfw,centerh-halfh:centerh+halfh]

def random_crop(x, random_crop_size, sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    w, h = x.shape[1], x.shape[2]
    rangew = (w - random_crop_size[0]) // 2
    rangeh = (h - random_crop_size[1]) // 2
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    return x[:, offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1]]


def data_generator3(original_image_ids,shuffle=True,batch_size=16,original_input_shape=[256,256,1],input_shape=[256,256,1],
                   output_shape=[256,256,2],valid=False,read_func=None,data_dirx=None,data_diry=None,
                   augment=False,normed=False):
    b = 0 # keep track of batch_item
    index=-1
    image_ids = original_image_ids.copy()
    num_images = len(image_ids)
    num_batches_per_epoch = int(np.floor(len(image_ids) / batch_size))
    num_batch_counter = 0
    errors = 0
    seed_counter=0
    while(True):
        try:

            index = index+1 % num_images
            # print('index: {}'.format(index))
            # print('index: {}'.format(index))
            if shuffle and index == 0:
                np.random.shuffle(image_ids)
                # print(image_ids)
            image_id = image_ids[index]
            # print('image_id: {}'.format(image_id))
            if valid:
                # current_image = np.load('../data/256gen/xtest/' + str(image_id) + '.npy')
                # current_target = np.load('../data/256gen/ytest/' + str(image_id) + '.npy')
                current_image = read_tif(os.path.join(data_dirx,image_id))
                current_image = np.expand_dims(current_image, axis=-1)
                current_target = read_tif(os.path.join(data_diry,image_id))
                current_target = np.expand_dims(current_target, axis=-1)
                if normed:
                    current_image,_,_ = normalize_like_marsnet2(current_image)

            else:
                current_image = read_tif(os.path.join(data_dirx,image_id))
                current_image = np.expand_dims(current_image, axis=-1)
                current_target = read_tif(os.path.join(data_diry,image_id))
                current_target = np.expand_dims(current_target, axis=-1)
               	if normed:
                    current_image,_,_ = normalize_like_marsnet2(current_image)
                # current_image = np.load('../data/256gen/xtrain/'+str(image_id)+'.npy')
                # current_image = np.expand_dims(current_image,axis=-1)
                # current_target = np.load('../data/256gen/ytrain/'+str(image_id)+'.npy')
            if b==0:
                images = np.zeros((batch_size,original_input_shape[0],original_input_shape[1],original_input_shape[2]))
                targets = np.zeros((batch_size,original_input_shape[0],original_input_shape[1],original_input_shape[2]))
            images[b] = current_image
            targets[b] = current_target
            seed_counter += 1
            b+=1
            # print('b: {}'.format(b))
            if b==batch_size:
                if augment:
                    if valid:
                        # print('here!')
                        images = center_crop(images, input_shape[:2])
                        targets = center_crop(targets, input_shape[:2])
                    else:
                        images_random = random_crop(images,input_shape[:2],sync_seed=seed_counter)
                        targets_random = random_crop(targets,input_shape[:2],sync_seed=seed_counter)
                        center_images = center_crop(images,input_shape[:2])
                        center_targets = center_crop(targets,input_shape[:2])
                        rot1_images = np.rot90(images_random,1,(1,2))
                        rot2_images = np.rot90(images_random,2,(1,2))
                        # rot3_images = np.rot90(images_random,3,(1,2))
                        rot1_targets = np.rot90(targets_random,1,(1,2))
                        rot2_targets = np.rot90(targets_random,2,(1,2))
                        # rot3_targets = np.rot90(targets_random,3,(1,2))
                        # print("random: {}, center: {}, ro1: {}, rot2: {}, ro3: {}".format(images_random.shape, center_images.shape, rot1_images.shape,
                        #                                                                   rot2_images.shape, rot3_images.shape))
                        # images = np.concatenate((images_random,center_images,rot1_images,rot2_images,rot3_images))
                        # targets = np.concatenate((targets_random,center_targets,rot1_targets,rot2_targets,rot3_targets))
                        images = np.concatenate((images_random,center_images,rot1_images,rot2_images))
                        targets = np.concatenate((targets_random,center_targets,rot1_targets,rot2_targets))
                # print('here')
                yield images, targets
                num_batch_counter += 1
                b = 0
                # print('there: {}'.format(b))
                if num_batch_counter >= num_batches_per_epoch:
                    # index = -1
                    num_batch_counter = 0
            if index >= len(image_ids)-1:
                index = -1
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            print('Something went wrong')
            errors+=1
            if errors>=1:
                raise


def data_generator2(original_image_ids,shuffle=True,batch_size=16,original_input_shape=[256,256,1],input_shape=[256,256,1],
                   output_shape=[256,256,2],data_dirx=None,data_diry=None,only_once=True,normed=False):
	b = 0 # keep track of batch_item
	index=-1
	image_ids = original_image_ids.copy()
	num_images = len(image_ids)
	num_batches_per_epoch = int(np.ceil((num_images) / batch_size))
	num_batch_counter = 0
	errors = 0
	seed_counter=0
	over=False
	while(True):
		try:
			if over:
				if only_once:
					return
				else:
					over=False
					index=-1
					b=0
					num_batch_counter=0
					seed_counter=0
			index = index+1 % num_images
			if shuffle and index == 0:
				np.random.shuffle(image_ids)
			image_id = image_ids[index]
			# print('image_id: {}'.format(image_id))
			current_image = read_tif(os.path.join(data_dirx,image_id))
			current_image = np.expand_dims(current_image, axis=-1)
			current_target = read_tif(os.path.join(data_diry,image_id))
			current_target = np.expand_dims(current_target, axis=-1)
			if normed:
				current_image,_,_ = normalize_like_marsnet2(current_image)
			# current_image = np.load('../data/256gen/xtrain/'+str(image_id)+'.npy')
			# current_image = np.expand_dims(current_image,axis=-1)
			# current_target = np.load('../data/256gen/ytrain/'+str(image_id)+'.npy')
			if b==0:
				images = np.zeros((batch_size,original_input_shape[0],original_input_shape[1],original_input_shape[2]))
				targets = np.zeros((batch_size,original_input_shape[0],original_input_shape[1],original_input_shape[2]))
			images[b] = current_image
			targets[b] = current_target
			seed_counter += 1
			b+=1
			# print('b: {}'.format(b))
			if num_batch_counter == (num_batches_per_epoch-1) and b==(num_images%batch_size):
				images = center_crop(images, input_shape[:2])
				targets = center_crop(targets, input_shape[:2])
				over=True
				yield images[:b],targets[:b]
			if b==batch_size:
				images = center_crop(images, input_shape[:2])
				targets = center_crop(targets, input_shape[:2])
				num_batch_counter += 1
				b = 0
				if num_batch_counter >= num_batches_per_epoch:
					over=True
				yield images, targets 
			# if index >= len(image_ids)-1:
			# 	print(3)
			# 	index = -1
		except (GeneratorExit, KeyboardInterrupt):
			raise
		except:
			print('Something went wrong')
			errors+=1
			if errors>=1:
			    raise


class IoU(Callback):
    def __init__(self,num_classes=6,loss_type='sparse_categorical'):
        super().__init__()
        self.loss_type=loss_type
        self.num_classes=num_classes
    def on_train_begin(self, logs=None):
            self._data=[]

    def on_epoch_end(self, epoch, logs=None):
        # Compute the confusion matrix to get the number of true positives,
        # false positives, and false negatives
        conf = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)
        self.x, self.y_true = self.validation_data[0], self.validation_data[1]
        for idx in range(len(self.x)):
            # Get a training sample and make a prediction using the current model
            sample = self.x[[idx]]
            target = self.y_true[[idx]]
            predicted = np.asarray(self.model.predict_on_batch(sample))

            # Convert predictions and target from categorical to integer format
            if self.loss_type == 'sparse_categorical':
                target = target.ravel()
            else:
                target = np.argmax(target, axis=-1).ravel()
            predicted = np.argmax(predicted, axis=-1).ravel()

            # Trick from torchnet for bincounting 2 arrays together
            # https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
            x = predicted + self.num_classes * target
            bincount_2d = np.bincount(x.astype(np.int32), minlength=self.num_classes**2)
            assert bincount_2d.size == self.num_classes**2
            conf += bincount_2d.reshape((self.num_classes, self.num_classes))

        # Compute the IoU and mean IoU from the confusion matrix
        true_positive = np.diag(conf)
        false_positive = np.sum(conf, 0) - true_positive
        false_negative = np.sum(conf, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error and set the value to 0
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
        iou[np.isnan(iou)] = 0

        print("IoU: {}\n".format(iou))




def iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(y_true, label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    y_true = K.squeeze(y_true,axis=-1)
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1]
    # initialize a variable to store total IoU in
    total_iou = K.variable(0)
    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        total_iou = total_iou + iou(y_true, y_pred, label)
    # divide total IoU by number of labels to get mean IoU
    return total_iou / num_labels


def custom_loss(y_true,y_pred,w=2):
    print(w)
    w_init = tf.ones(K.shape(y_true),tf.int32)
    weights = tf.where(tf.equal(y_true,1),w_init*w, w_init)
    y_true = tf.cast(y_true,tf.int32)
    loss = tf.losses.sparse_softmax_cross_entropy(
        y_true,
        y_pred,
        weights=weights
    )
    return loss

def simple_custom_loss(y_true,y_pred):
    y_true = tf.cast(y_true,tf.int64)
    loss = tf.losses.sparse_softmax_cross_entropy(
        y_true,
        y_pred
    )
    return loss

def make_loss_func(w):
    def custom_loss(y_true,y_pred,w=w):
        w_init = tf.ones(K.shape(y_true),tf.int64)
        weights = tf.where(tf.not_equal(y_true,0),w_init*w, w_init)
        y_true = tf.cast(y_true,tf.int64)
        loss = tf.losses.sparse_softmax_cross_entropy(
            y_true,
            y_pred,
            weights=weights
        )
        return loss
    return custom_loss

def sliding_window(image, step_size, windowSize):
    """
    Scan image with a stride of step_size and patches of (windowSize X windowSize)
    return only those that have exact (windowSize X windowSize) shape
    """
    for i in range(0, image.shape[0], step_size):
        for j in range(0, image.shape[1], step_size):
            im = image[i:i + windowSize, j:j + windowSize]
            if im.shape == (windowSize,windowSize):
                yield (i, j, im)


def save_result(test_result, 
                step_size, 
                test_result_rows,
                test_result_cols, 
                test_data_output_tif,
                original_tif,
                windowSize
               ):
    
    # get from the test region only the pixels for which we have predictions!
    t = test_result[int((windowSize-step_size)/2):int((windowSize-step_size)/2)+test_result_rows,\
                    int((windowSize-step_size)/2):int((windowSize-step_size)/2)+test_result_cols]
    
    # Read original input region and get geotransformation and projection info!
    
    od = gdal.Open(original_tif, gdalconst.GA_ReadOnly)
    gt = od.GetGeoTransform()
    proj = od.GetProjection()
    
    # For an input of 128x128 for example, the model predicts label for
    # the inner 64x64 pixels. therefore, the predictions for the input
    # region only starts with an offset of 64/2 = 32 from top left corner
    # Here, we adjust and calculate the geotransformation for the prediction
    # gt = list(gt)
    # gt[3] = gt[3] - (step_size/2 * gt[1])
    # gt[0] = gt[0] + (step_size/2 * gt[1])
    # gt = tuple(gt)

    gt = list(gt)
    gt[3] = gt[3] - ((windowSize-step_size)/2 * gt[1])
    gt[0] = gt[0] + ((windowSize-step_size)/2 * gt[1])
    gt = tuple(gt)
        
    # Crate and save a tif file for the result, with the calculated geotransformation,
    # and projection from the original input
    output_raster = gdal.GetDriverByName('GTiff').Create(test_data_output_tif,
                                                     test_result_cols, 
                                                     test_result_rows, 
                                                     1 ,gdal.GDT_Float32)  
    output_raster.GetRasterBand(1).WriteArray(t)
    output_raster.SetGeoTransform(gt)
    output_raster.SetProjection(proj)
    output_raster.FlushCache()
    output_raster=None

    # return trimmed result
    return t


def run_detection(batch_size, windowSize, output_patch, step_size, model, original_tif, test_data_output_tif):
    """
    runs detection using patches from data generator gen
    batch_size: how many examples to process at a time
    windowSize: size of the input patch to the model
    model: trained model
    step_size: strides while sliding window on the region
    output_patch: shape of the prediction for each input
    original_tif: test region tif file
    test_data_output_tif: save the prediction into this file
    return: prediction for the test region
    """
    counter = 0 # keep track of number of examples
    batch_input_index = 0 # keep track of examples added to current batch
    top_indices = [] # keep track of top left indices for each input

    # read test region to numpy array
    test_data = read_tif(original_tif)

    gen = sliding_window(test_data, step_size, windowSize)

    # number of rows and columns for the prediction is smaller due to window 
    # sliding and taking only patches of (windwSize X windowSize)

    test_result_rows = test_data.shape[0] - (test_data.shape[0] % windowSize) - int((windowSize-step_size)/2)
    test_result_cols = test_data.shape[1] - (test_data.shape[1] % windowSize) - int((windowSize-step_size)/2)

    # Create an empty numpy array of shape similar to input test region
    test_result = np.zeros((test_data.shape))

    
    for i,j,im in gen:
        top_indices.append((i,j))
        if counter % batch_size == 0:
            batch_input = np.empty((batch_size, windowSize, windowSize))
        im = (im - im.min()) / (im.max() - im.min())
        batch_input[batch_input_index] = im
        batch_input_index += 1
        counter += 1
        if batch_input_index == batch_size:
            predictions = np.argmax(model.predict(np.expand_dims(batch_input,-1)),axis=-1)
            p = 0
            for i,j in top_indices:
                test_result[i+int((windowSize-step_size)/2):i+int((windowSize-step_size)/2)+output_patch[0],\
                            j+int((windowSize-step_size)/2):j+int((windowSize-step_size)/2)+output_patch[1]]\
                = predictions[p]                
                p+=1
            top_indices = []
            batch_input_index = 0

    prediction = save_result(test_result, 
                step_size, 
                test_result_rows,
                test_result_cols, 
                test_data_output_tif,
                original_tif,
                windowSize
               )
    return prediction


    


def gen_helper(xfiles, yfiles, index, input_shape, output_shape,rs=0,max_patches=5):
    xf = read_tif(xfiles[index])
    xf = (xf - xf.min()) / (xf.max() - xf.min())
    yf = read_tif(yfiles[index])
    ret_x = extract_patches_2d(xf, (input_shape[0],input_shape[1]), max_patches=max_patches, random_state=rs)
    ret_y = extract_patches_2d(yf, (input_shape[0],input_shape[1]), max_patches=max_patches, random_state=rs)
    lower_limit = int(input_shape[0]/2 - output_shape[0]/2)
    upper_limit = int(input_shape[0]/2 + output_shape[0]/2)
    ret_y = ret_y[:,lower_limit:upper_limit,lower_limit:upper_limit]  
    return ret_x, ret_y

def data_generator(xfiles, yfiles, batch_size, input_shape, output_shape,shuffle=True,max_patches=5):
    rs_counter = 0
    image_ids = [i for i in range(len(xfiles))]
    num_batches_per_epoch = int(np.floor(len(xfiles) / batch_size)) 
    num_batch_counter = 0
    errors = 0
    seed_counter = 0
    index = -1
    b = 0
    num_images = len(xfiles)
    while(True):
        try:
            index = index+1 % num_images
            if shuffle and index == 0:
                np.random.shuffle(image_ids)
            image_id = image_ids[index]
            current_x,current_y = gen_helper(xfiles, yfiles, image_id, input_shape, output_shape,rs=rs_counter,max_patches=max_patches)
            rs_counter+=1
            if b == 0:
                x = np.empty((batch_size,max_patches, input_shape[0], input_shape[1]))
                y = np.empty((batch_size,max_patches, output_shape[0], output_shape[1]))
            x[b] = current_x
            y[b] = current_y
            seed_counter+=1
            b+=1
            if b==batch_size:
                x = x.reshape((batch_size*max_patches,input_shape[0],input_shape[1]))
                y = y.reshape((batch_size*max_patches,output_shape[0],output_shape[1]))
                yield np.expand_dims(x,-1),np.expand_dims(y,-1)
                num_batch_counter += 1
                b = 0
                if num_batch_counter >= num_batches_per_epoch:
                    num_batch_counter = 0
            if index >= len(image_ids) - 1:
                index = -1
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            print('Something went wrong')
            errors+=1
            if errors>=1:
                raise


def gen_helper_std(xfiles, yfiles, index, input_shape, output_shape,rs=0,max_patches=5):
    xf = read_tif(xfiles[index])
    xf = (xf - xf.mean()) / ((xf.std())+1)
    yf = read_tif(yfiles[index])
    ret_x = extract_patches_2d(xf, (input_shape[0],input_shape[1]), max_patches=max_patches, random_state=rs)
    ret_y = extract_patches_2d(yf, (input_shape[0],input_shape[1]), max_patches=max_patches, random_state=rs)
    lower_limit = int(input_shape[0]/2 - output_shape[0]/2)
    upper_limit = int(input_shape[0]/2 + output_shape[0]/2)
    ret_y = ret_y[:,lower_limit:upper_limit,lower_limit:upper_limit]  
    return ret_x, ret_y

def data_generator_std(xfiles, yfiles, batch_size, input_shape, output_shape,shuffle=True,max_patches=5):
    rs_counter = 0
    image_ids = [i for i in range(len(xfiles))]
    num_batches_per_epoch = int(np.floor(len(xfiles) / batch_size)) 
    num_batch_counter = 0
    errors = 0
    seed_counter = 0
    index = -1
    b = 0
    num_images = len(xfiles)
    while(True):
        try:
            index = index+1 % num_images
            if shuffle and index == 0:
                np.random.shuffle(image_ids)
            image_id = image_ids[index]
            current_x,current_y = gen_helper_std(xfiles, yfiles, image_id, input_shape, output_shape,rs=rs_counter,max_patches=max_patches)
            rs_counter+=1
            if b == 0:
                x = np.empty((batch_size,max_patches, input_shape[0], input_shape[1]))
                y = np.empty((batch_size,max_patches, output_shape[0], output_shape[1]))
            x[b] = current_x
            y[b] = current_y
            seed_counter+=1
            b+=1
            if b==batch_size:
                x = x.reshape((batch_size*max_patches,input_shape[0],input_shape[1]))
                y = y.reshape((batch_size*max_patches,output_shape[0],output_shape[1]))
                yield np.expand_dims(x,-1),np.expand_dims(y,-1)
                num_batch_counter += 1
                b = 0
                if num_batch_counter >= num_batches_per_epoch:
                    num_batch_counter = 0
            if index >= len(image_ids) - 1:
                index = -1
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            print('Something went wrong')
            errors+=1
            if errors>=1:
                raise


def plot_truth_prediction(y,preds,num_classes=2, figsize=(15,15)):

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    for i in range(y.shape[0]):
        fig = plt.figure(figsize=figsize)

        ax1 = fig.add_subplot(1,2,1)
        ax1.set_title('Ground truth {}'.format(i))
        plt.setp(ax1.get_xticklabels(),visible=False)
        plt.setp(ax1.get_yticklabels(),visible=False)
        cmap = plt.get_cmap('RdBu', num_classes)
        b = ax1.imshow(np.squeeze(y[i],axis=-1),cmap=cmap)
        ax2 = fig.add_subplot(1,2,2)
        ax2.set_title('Prediction {}'.format(i))
        plt.setp(ax2.get_xticklabels(),visible=False)
        plt.setp(ax2.get_yticklabels(),visible=False)
        im = ax2.imshow(preds[i],cmap=cmap)

def interpolate_nan(tif):
    array = read_tif(tif)
    nodata = array.min()
    array[array == nodata] = np.nan
    nans, x = np.isnan(array), lambda z:z.nonzero()[0]
    array[nans] = np.interp(x(nans), x(~nans), array[~nans])
    output,_ = os.path.splitext(tif)
    output=output+'_interped.tif'
    write_numpy_array_to_tif(array, output, original_tif=tif)