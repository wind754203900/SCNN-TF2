import cv2
import numpy as np

class Postprocess(object):
    def __init__(self):
        self.h_samples = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260,
             270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420,
             430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580,
             590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]

    @staticmethod
    def _morphological_process(image, kernel_size=5):
        """
        morphological process to fill the hole in the binary segmentation result
        :param image:
        :param kernel_size:
        :return:
        """
        if len(image.shape) == 3:
            raise ValueError('Binary segmentation result image should be a single channel image')

        if image.dtype is not np.uint8:
            image = np.array(image, np.uint8)

        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

        # close operation fille hole
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

        return closing

    def _convert_pts_to_json(self,lane_pts):
        pty = lane_pts[:, 1]
        pt = []
        for h in self.h_samples:
            idx = np.where(pty == h)
            if idx[0].shape[0] == 0:
                pt.append(-2)
            else:
                ptx = int(round(np.mean(lane_pts[idx,][0, :, 0]) - 0.1))
                pt.append(ptx)
        pt = np.squeeze(np.vstack(pt))
        return pt


    def postprocess_tensor(self,binary_image,img_name,lane_exist):
        # binary_image:H,W,C(4 lanes)
        lane_json = []
        for lane_i in range(binary_image.shape[-1]):
            lane_image = binary_image[:,:,lane_i]
            lant_pts = self._get_one_lane_pts(lane_image)
            if lant_pts.size!=0 and lane_exist[lane_i]>=0.5:
                lane_json.append(lant_pts)
        #         print(lant_pts)
        # print("process")
        lane_json = [self._convert_pts_to_json(lane_pt).tolist() for lane_pt in lane_json]

        dict = {}
        dict['lanes'] = lane_json
        dict['h_samples'] = self.h_samples
        dict['raw_file'] = bytes.decode(img_name)
        dict['run_time'] = 10
        return dict

    def _get_one_lane_pts(self,binary_seg_result, min_area_threshold=100):
        """
        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param data_source:
        :return:
        """
        # convert binary_seg_result

        binary_seg_result = np.array(binary_seg_result*255, dtype=np.uint8)

        # apply image morphology operation to fill in the hold and reduce the small area
        morphological_ret = self._morphological_process(binary_seg_result, kernel_size=5)

        idx = np.where(morphological_ret == 255)
        lane_pts = np.vstack((idx[1], idx[0])).transpose()
        return lane_pts