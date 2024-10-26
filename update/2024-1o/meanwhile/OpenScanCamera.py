from OpenScanCommon import load_bool, load_int, load_str
from picamera2 import Picamera2, Preview
from time import sleep, time
from PIL import Image, ImageDraw, ImageOps, ImageFilter, ImageEnhance, ImageChops, ImageFont
from skimage import feature, color, transform
import numpy as np
from scipy import ndimage
from math import sqrt
import math

class OpenCamera:
    def __init__(self) -> None:
        self.camera = load_str('camera')
        self.cam_mode = 0
        self.arducams = ['imx519', 'arducam_64mp']
        if self.camera in self.arducams:
           self.arducam_init()

    def arducam_init(self):
        self.picam2 = Picamera2()
        img_size: dict = {}
        if self.camera == 'arducam_64mp':
            #img_size: dict = {"size": (4656, 3496)}
            #img_size: dict = {"size": (6864, 5208)}
            img_size: dict = {"size": (8064, 6048)}

        self.preview_config = self.picam2.create_preview_configuration(
            main={"size": (2028, 1520)},
            controls={"FrameDurationLimits": (1, 1000000)}
        )
        self.capture_config = self.picam2.create_still_configuration(
            main=img_size,
            controls={"FrameDurationLimits": (1, 1000000)}
        )
        self.picam2.configure(self.preview_config)
        self.picam2.controls.AnalogueGain = 1.0
        self.picam2.start()

    def camera_highlight_sharpest_areas(self, image, threshold=load_int('cam_sharpness'), dilation_size=5):
        # Convert PIL image to grayscale
        image_gray = image.convert('L')
        # Convert grayscale image to numpy array
        image_array = np.array(image_gray)

        # Calculate the gradient using a Sobel filter
        dx = ndimage.sobel(image_array, 0)  # horizontal derivative
        dy = ndimage.sobel(image_array, 1)  # vertical derivative
        mag = np.hypot(dx, dy)  # magnitude

        # Threshold the gradient to create a mask of the sharpest areas
        mask = np.where(mag > threshold, 255, 0).astype(np.uint8)

        dilated_mask = ndimage.binary_dilation(mask, structure=np.ones((dilation_size,dilation_size)))
        # Create a PIL image from the mask
        mask_image = Image.fromarray(dilated_mask)

        return mask_image

    def camera_create_mask(self, image: Image, scale: float = 0.1, threshold: int = 45) -> Image:
        threshold = load_int("cam_mask_threshold")
        if threshold <= 1:
            return image
        orig = image
        image = image.resize((int(image.width*scale),int(image.height*scale)))
        image = image.convert("L")
        reduced = image
        image = image.filter(ImageFilter.EDGE_ENHANCE)
        image = image.filter(ImageFilter.BLUR)
        reduced = reduced.filter(ImageFilter.EDGE_ENHANCE_MORE)
        mask = ImageChops.difference(image, reduced)
        mask = ImageEnhance.Brightness(mask).enhance(2.5)
        mask = mask.filter(ImageFilter.MaxFilter(9))
        mask = mask.filter(ImageFilter.MinFilter(5))
        mask = mask.point(lambda x: 255 if x <threshold else 0)
        mask = mask.filter(ImageFilter.MaxFilter(5))
        mask = mask.convert(orig.mode)
        mask = mask.resize((orig.width,orig.height), resample=Image.BOX)
        result = ImageChops.subtract(orig, mask)
        return result

    def camera_overlay_mask(self, image, mask_image):
        # Ensure image is in RGB mode
        image_rgb = image.convert('RGB')
        # Create an empty image with RGBA channels
        overlay = Image.new('RGBA', image_rgb.size)

        # Prepare a red image of the same size
        red_image = Image.new('RGB', image_rgb.size, (255, 0, 0))
        # Prepare a mask where the condition is met (mask_image pixels == 255)
        mask_condition = np.array(mask_image) > 0
        overlay_mask = Image.fromarray(np.uint8(mask_condition) * 255)
        # Paste the red image onto the overlay using the condition mask
        overlay.paste(red_image, mask=overlay_mask)
        # Combine the original image with the overlay
        combined = Image.alpha_composite(image_rgb.convert('RGBA'), overlay)
        # Convert the final image to RGB
        combined_rgb = combined.convert('RGB')
        return combined_rgb

    def camera_add_histo(self, img):
        histo_size = 241

        img_gray = ImageOps.grayscale(img)
        histogram = img_gray.histogram()
        histogram_log = [math.log10(h + 1) for h in histogram]
        histogram_max = max(histogram_log)
        histogram_normalized = [float(h) / histogram_max for h in histogram_log]
        hist_image = Image.new("RGBA", (histo_size, histo_size), (255, 255, 255, 0))
        draw = ImageDraw.Draw(hist_image)

        for i in range(0, 256):
            x = i
            y = 256 - int(histogram_normalized[i] * 256)
            draw.line((x, 256, x, y), fill=(0, 0, 0, 255))

        text = ""
        if min(histogram[235:238])>0:
            text = "overexposed"
        if sum(histogram[190:192])<8:
            text = "underexposed"
        font = ImageFont.truetype("DejaVuSans.ttf", 30)

        bbox = draw.textbbox((0, 0), text, font=font)

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]


        x = (hist_image.width - text_width )/2
        y = hist_image.height - text_height - 10
        draw.text((x, y), text, font=font, fill=(255,0,0))

        scale = 0.25
        width1, height1 = hist_image.size
        width2 = img.size[0]
        new_width1 = int(width2 * scale)
        new_height1 = int((height1 / width1) * new_width1)
        hist_image = hist_image.convert('RGB')

        hist_image = hist_image.resize((new_width1, new_height1))
        x = hist_image.width - text_width - 10
        y = hist_image.height - text_height - 10


        img.paste(hist_image, (img.size[0]-new_width1-int(0.01*img.size[0]),img.size[1]-new_height1-int(0.01*img.size[0])))

        return img


    def camera_take_photo(self):
        print(self.camera)
        if self.camera in self.arducams:
            starttime = time()

            cropx = load_int('cam_cropx')/200
            cropy = load_int('cam_cropy')/200
            rotation = load_int('cam_rotation')
            img = self.picam2.capture_image()

            if self.cam_mode != 1:
                img = img.convert('RGB')
            w, h = img.size

            if cropx != 0 or cropy != 0:
                img = img.crop((w*cropx, h*cropy, w * (1-cropx), h * (1-cropy)))

            if rotation == 90:
                img = img.transpose(Image.ROTATE_90)
            elif rotation == 180:
                img = img.transpose(Image.ROTATE_180)
            elif rotation == 270:
                img = img.transpose(Image.ROTATE_270)

            if load_bool("cam_mask"):
                downscale = 0.045*1.4 if self.cam_mode == 1 else 0.1*1.4
                img = self.camera_create_mask(img, downscale)

            if load_bool("cam_features") and not load_bool("cam_sharparea"):
                img = self.camera_plot_orb_keypoints(img)

            if load_bool("cam_sharparea") and not load_bool("cam_features"):
                img2 = self.camera_highlight_sharpest_areas(img)
                img = self.camera_overlay_mask(img, img2)

            if self.cam_mode != 1 and not load_bool("cam_sharparea") and not load_bool("cam_features"):
                img = self.camera_add_histo(img)

            img.save("/home/pi/OpenScan/tmp2/preview.jpg", quality=load_int('cam_jpeg_quality'))
            print("total " + str(int(1000*(time()-starttime))) + "ms")


    def camera_plot_orb_keypoints(self, pil_image):
        downscale = 2
        # Read the image from the given image path
        image = np.array(pil_image)
        #image = io.imread(image_path)
        image = transform.resize(image, (image.shape[0] // downscale, image.shape[1] // downscale), anti_aliasing=True)

        # Convert the image to grayscale
        gray_image = color.rgb2gray(image)

        try:
            orb = feature.ORB(n_keypoints=10000, downscale=1.2, fast_n=2, fast_threshold=0.2 , n_scales=3, harris_k=0.001)
            orb.detect_and_extract(gray_image)
            keypoints = orb.keypoints
        except:
            return pil_image

        # Convert the image back to the range [0, 255]
        display_image = (image * 255).astype(np.uint8)

        # Draw the keypoints on the image
        draw = ImageDraw.Draw(pil_image)
        size = max(2,int(image.shape[0]*downscale*0.005))
        for i, (y, x) in enumerate(keypoints):
            draw.ellipse([(downscale*x-size, downscale*y-size), (downscale*x+size, downscale*y+size)], fill = (0,255,0))
        # Save the image with keypoints to the given output path
        return pil_image


    def camera_focus(self, focus):
        if self.camera in self.arducams:
            self.picam2.set_controls({"AfMode": 0, "LensPosition": focus})
            print("focus:" + str(focus))

    def camera_exposure(self, exposure):
        '''Set camera exposure time'''
        if self.camera in self.arducams:
            exposure = int(exposure)
            self.picam2.controls.AnalogueGain = 1.0
            self.picam2.controls.ExposureTime = exposure

    def camera_contrast(self, contrast):
        '''Set camera contrast'''
        if self.camera in self.arducams:
            contrast = float(contrast)
            self.picam2.controls.Contrast = contrast

    def camera_saturation(self, saturation):
        '''Set camera saturation'''
        if self.camera in self.arducams:
            saturation = float(saturation)
            self.picam2.controls.Saturation = saturation

    def switch_mode(self, mode):
        '''Switch camera mode'''
        if self.camera in self.arducams:
            print(mode)
            self.cam_mode = int(mode)
            if self.cam_mode == 1:
                self.picam2.switch_mode(self.capture_config)
            else:
                self.picam2.switch_mode(self.preview_config)

    def show_mode(self):
        if self.camera in self.arducams:
            return self.cam_mode

    def focus_af(self):
        if self.camera in self.arducams:
            '''Trigger auto focus'''
            self.picam2.set_controls({"AfMode": 1, "AfTrigger": 0})  # --> wait 3-5s
