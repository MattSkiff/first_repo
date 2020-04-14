library("rayshader")
library("raster")
library("gdalUtils")
library("devtools")
library("tiff")
source_url("https://raw.githubusercontent.com/wcmbishop/rayshader-demo/master/R/rayshader-gif.R")

#data from https://data.linz.govt.nz/layer/53621-wellington-lidar-1m-dem-2013/
#cropped to central wellington area to meet download limit of 2.5 GB
img <- "C:\\Users\\skiff\\Desktop\\rayshading stuff\\mt_t-2layers-GTiff\\nz-10m-satellite-imagery-2017\\nz-10m-satellite-imagery-2017.tif"

img_overlay <- readTIFF(img)

DEM_directory <- "C:\\Users\\skiff\\Desktop\\rayshading stuff\\mt_t-2layers-GTiff\\nz-8m-digital-elevation-model-2012\\"

setwd(DEM_directory) # switch directory to tiff files

gdalbuildvrt(gdalfile = "*.tif", # uses all tiffs in the current folder
						 output.vrt = "dem.vrt")

gdal_translate(src_dataset = "dem.vrt", 
							 dst_dataset = "dem.tif", 
							 output_Raster = TRUE, # returns the raster as Raster*Object
							 # if TRUE, you should consider to assign 
							 # the whole function to an object like dem <- gddal_tr..
							 options = c("BIGTIFF=YES", "COMPRESSION=LZW"))

localtif = raster("dem.tif")

elmat = matrix(raster::extract(localtif, raster::extent(localtif), buffer = 1000),
							 nrow = ncol(localtif), ncol = nrow(localtif))


#elmat <- elmat[1:dim(img_overlay)[1],1:dim(img_overlay)[2]] # resizing DEM matrix to aerial imagery matrix

qcshadow = ray_shade(elmat, zscale = 5, lambert = FALSE)
qcamb = ambient_shade(elmat, zscale = 5)
elmat %>%
	sphere_shade(zscale = 5, texture = "imhof3") %>%
	add_shadow(qcshadow, 0.5) %>%
	add_shadow(qcamb) %>%
	add_overlay(overlay = img_overlay,alphalayer = 0.5) %>%
	plot_3d(heightmap = elmat, zscale = 5, fov = 0, theta = 45, phi = 45, windowsize = c(1000, 600), baseshape = "circle",zoom = 0.75)
render_depth(focus = 0.6, focallength = 200, clear = TRUE)

#elmat <- elmat[1:dim(img_overlay)[1],1:dim(img_overlay)[2]]

# ambmat = ambient_shade(elmat)

# elmat %>%
# 	sphere_shade(texture = "imhof1") %>%
# 	add_water(detect_water(elmat), color = "imhof1") %>%
# 	add_shadow(ray_shade(elmat, zscale = 3, maxsearch = 200), 0.5) %>%
# 	add_shadow(ambmat, 0.5) %>%
# 	plot_3d(elmat, zscale = 5, fov = 0, theta = 135, zoom = 0.75, phi = 45, windowsize = c(1920, 1080),water = TRUE, waterdepth = 0, wateralpha = 0.5, watercolor = "lightblue",
# 					waterlinecolor = "white", waterlinealpha = 0.5)
# render_snapshot("qc1")

#### -- UNCOMMENT HERE FOR IMAGE -- ####

# change wd to gif location
setwd("C:\\Users\\skiff\\Desktop\\rayshading stuff\\mt_taranaki_gif\\")




# build gif
magick::image_write_gif(magick::image_read(img_frames), 
												path = "C:\\Users\\skiff\\Desktop\\rayshading stuff\\mt_taranaki_gif\\mt_gif_trial_2", 
												delay = 6/n_frames)

#render_snapshot("MT_t2")
#### -- END UNCOMMENT -- ####


# ,water = TRUE, waterdepth = 0, wateralpha = 0.5, watercolor = "lightblue",
#waterlinecolor = "white", waterlinealpha = 0.5

#### -- UNCOMMENT HERE FOR GIF -- ####
# calculate input vectors for gif frames
n_frames <- 360
thetas <- transition_values(from = 0, to = 360, steps = n_frames,one_way = T,type = 'cos')
#generate gif
zscale <- 5
qcshadow = ray_shade(elmat, zscale = 5, lambert = FALSE)
qcamb = ambient_shade(elmat, zscale = 5)
elmat %>%
	sphere_shade(zscale = 5, texture = "imhof3") %>%
	add_shadow(qcshadow, 0.5) %>%
	add_shadow(qcamb) %>%
	add_overlay(overlay = img_overlay,alphalayer = 0.5) %>%
	save_3d_gif(heightmap = elmat, file = "mt_taranaki.gif", duration = 12,zscale = zscale,fov = 0,
							theta = thetas, phi = 45,windowsize = c(1920, 1080))
#### -- END UNCOMMENT -- ####