IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

from torchvision.transforms import Normalize
imagenet_normalizer = Normalize(IMAGENET_MEAN, IMAGENET_STD)

from osgeo import gdal

def write_geotiff(output_tif, ncols, nrows,
                  xmin, xres,ymax, yres,
                 raster_srs, label_arr):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_tif, ncols, nrows, len(label_arr), gdal.GDT_Byte)
    out_ds.SetGeoTransform((xmin, xres, 0, ymax, 0, yres))
    out_ds.SetProjection(raster_srs.ExportToWkt())
    for i in range(len(label_arr)):
        outband = out_ds.GetRasterBand(i+1)
        outband.WriteArray(label_arr[i])
        #outband.SetNoDataValue(0)
        outband.FlushCache()
    out_ds = None


def freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
        
        
def unfreeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)