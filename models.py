import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import re

endpoint = "https://core-ocr.cognitiveservices.azure.com/"
key = "33947f900faf438c953abcd092db2be5"

client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)

def infer_ocr(image_data):
    result = client.analyze(
        image_data,
        visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ],
        gender_neutral_caption=True,  # Optional (default is False)
    )
    return result

def resize_img(img, size=1024):
    h, w = img.shape[:-1]
    scale = size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    new_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR )
    return new_img, scale

def intersection_area_cal(box1, box2):
    # Calculate the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of the intersection rectangle
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    return intersection_area

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
        
    return mask_image

def merge_rocks(nrow, row_points, row_points_coord, row_blocks, filter_type = 0):
    for row in range(nrow):
        point_need_merge = []
        skip_indices = set()
        for i in range(len(row_points[row]) - 1):
            for j in range(i+1, len(row_points[row])):
                if i in skip_indices or j in skip_indices:
                    continue
                if row_blocks[row][i] != None or row_blocks[row][j] != None:
                    continue
                x1, x2 = row_points[row][i]
                x3, x4 = row_points[row][j]

                intersection = max(0, min(x2, x4) - max(x1, x3))
                merge = False
                if filter_type == 0:
                    union = (x2 - x1) + (x4 - x3) - intersection
                    iou = intersection / union
                    if iou > 0.95:
                        merge = True
                else:
                    area1 = x2 - x1
                    area2 = x4 - x3
                    if intersection / area1 > 0.6 or intersection / area2 > 0.6:
                        merge = True
                if merge:
                    point_need_merge.append([i, j])
                    skip_indices.add(i)
                    skip_indices.add(j)

        new_points = []
        new_points_coord = []
        new_blocks = []
        skip_indices = set()

        for i, j in point_need_merge:
            if i in skip_indices or j in skip_indices:
                continue
            x1, x2 = row_points[row][i]
            x3, x4 = row_points[row][j]
            minx = min(x1, x3)
            maxx = max(x2, x4)
            new_points.append([minx, maxx])

            x1, x2 = row_points_coord[row][i]
            x3, x4 = row_points_coord[row][j]
            minx = min(x1[0], x2[0], x3[0], x4[0])
            maxx = max(x1[0], x2[0], x3[0], x4[0])
            y = x1[1]
            new_points_coord.append([[minx, y], [maxx, y]])

            skip_indices.add(i)
            skip_indices.add(j)
            new_blocks.append(None)

        for i in range(len(row_points[row])):
            if i not in skip_indices:
                new_points.append(row_points[row][i])
                new_points_coord.append(row_points_coord[row][i])
                new_blocks.append(row_blocks[row][i])

        # Sort new_points and new_points_coord based on the first value of new_points
        sorted_indices = sorted(range(len(new_points)), key=lambda k: new_points[k][0])
        new_points = [new_points[i] for i in sorted_indices]
        new_points_coord = [new_points_coord[i] for i in sorted_indices]
        new_blocks = [new_blocks[i] for i in sorted_indices]

        row_points[row] = new_points
        row_points_coord[row] = new_points_coord
        row_blocks[row] = new_blocks

    for row in range(nrow):
        for i in range(len(row_points[row]) - 1):
            for j in range(i+1, len(row_points[row])):
                if i in skip_indices or j in skip_indices:
                    continue
                if row_blocks[row][i] != None or row_blocks[row][j] != None:
                    continue
                x1, x2 = row_points[row][i]
                x3, x4 = row_points[row][j]

                intersection = max(0, min(x2, x4) - max(x1, x3))
                area1 = x2 - x1
                area2 = x4 - x3
                if intersection / area1 > 0.6 or intersection / area2 > 0.6:
                    return True
    return False

class Processor:
    def __init__(self, model_path, model_cfg):
        # sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        sam2_model = build_sam2(model_cfg, model_path, device=device)
        self.mask_generator = SAM2AutomaticMaskGenerator(sam2_model, points_per_batch=16, points_per_side=32, crop_n_layers=1, \
                                                         stability_score_thresh = 0.8)

    def process(self, image, nrow):
        org = image.copy()
        auto_masks = self.mask_generator.generate(image)
        auto_masks = sorted(auto_masks, key=lambda x: x['area'], reverse=True)

        try:
            jpeg_quality = 100
            size=1024
            ocr_inp, scale = resize_img(image, size)
            ret_code, jpg_buffer = cv2.imencode(".jpg", ocr_inp, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            image_data = np.array(jpg_buffer).tobytes()

            result = infer_ocr(image_data)

            ocr = result.as_dict()
            text_match = {}
            if len(ocr['readResult']['blocks']) > 0:
                block = ocr['readResult']['blocks'][0]
                for line in block['lines']:
                    box = np.array([(data['x'], data['y']) for data in line['boundingPolygon']])
                    xmin = np.min(box[:, 0]) / scale
                    ymin = np.min(box[:, 1]) / scale
                    xmax = np.max(box[:, 0]) / scale
                    ymax = np.max(box[:, 1]) / scale

                    regex = r"[^\d.,]"
                    subst = ""
                    try:
                        depth = float(re.sub(regex, subst, line['text'], 0, re.MULTILINE).replace(",", "."))
                        text_match[depth] = [xmin, ymin, xmax, ymax]
                    except:
                        pass
        except:
            text_match = {}
            
        h, w = image.shape[:2]
        interval = h // nrow

        row_limits = []
        for i in range(nrow):
            row_limits.append([i*interval, (i+1)*interval])

        # full_mask = np.zeros_like(image)
        # for mask in auto_masks:
        #     vis_mask = mask['segmentation'].astype(np.uint8)
        #     mask = show_mask(vis_mask, plt.gca(), random_color=True, borders=False)
        #     mask *= 255
        #     mask = mask.astype(np.uint8)
        #     mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2BGR)
        #     full_mask += mask
        # image = cv2.addWeighted(image, 0.7, full_mask, 0.4, 0)
        # cv2.imwrite("tmp.png", image)

        kernel = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.uint8)

        row_polygons = {}
        row_blocks = {}
        lines = {}
        row_lengths = {}

        for i in range(nrow):
            row_start, row_end = row_limits[i]
            row_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            rocks = []
            for mask in auto_masks:
                mask = mask['segmentation']
                h, w = mask.shape[-2:]
                mask = mask.astype(np.uint8)
                contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                # Try to smooth contours
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                max_contour = max(contours, key=cv2.contourArea)
                x,y,w,h = cv2.boundingRect(max_contour)

                if w < 10 or h < 10:
                    continue

                yc = y + h // 2
                if yc >= row_start and yc <= row_end:
                    new_mask = np.zeros_like(image)
                    new_mask = cv2.drawContours(new_mask, [max_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                    new_mask = new_mask[:,:,0]
                    if np.count_nonzero(np.bitwise_and(row_mask, new_mask)) > 100:
                        continue
                    row_mask += new_mask
                    rocks.append(max_contour)
                    # cv2.drawContours(image, [max_contour], -1, (0, 255, 0), thickness=2)

            #         tmp = cv2.imread(path)
            #         tmp = cv2.drawContours(tmp, [max_contour], -1, (0, 255, 0), thickness=2)
            #         cv2.imshow("tmp", tmp)
            #         cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # break
            
            row_mask = row_mask.astype(np.uint8)
            dilation = cv2.dilate(row_mask,kernel,iterations = 5)
            contours, _ = cv2.findContours(dilation,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) == 0:
                continue
            max_contour = max(contours, key=cv2.contourArea)

            vx,vy,x,y = cv2.fitLine(max_contour, cv2.DIST_L2,0,0.01,0.01)
            vx = float(vx[0])
            vy = float(vy[0])
            x = float(x[0])
            y = float(y[0])
            rows,cols = image.shape[:2]
            lefty = int((-x*vy/vx) + y)
            righty = int(((cols-x)*vy/vx)+y)
            # cv2.line(image,(cols-1,righty),(0,lefty),(0,255,0),2)
            lines[i] = [vx,vy,x,y]
            row_polygons[i] = rocks
            row_lengths[i] = float(np.linalg.norm(np.array([cols-1,righty]) - np.array([(0,lefty)])))

        row_points = {}
        row_points_coord = {}

        for row in range(nrow):
            vx,vy,x,y = lines[row]

            direction_vector = (vx,vy)
            direction_vector = direction_vector / np.linalg.norm(direction_vector) 
            first_point = np.array([0, int(((0-x)*vy/vx) + y)])

            points = []
            coords = []

            # Sort the contours based on the x value of their center
            row_polygons[row].sort(key=lambda cnt: cv2.boundingRect(cnt)[0])

            for i in range(len(row_polygons[row])):
                projections = [np.dot(point, direction_vector) for point in row_polygons[row][i]]
                min_proj, max_proj = min(projections), max(projections)
                ymin = int(((min_proj[0]-x)*vy/vx) + y)
                ymax = int(((max_proj[0]-x)*vy/vx) + y)
                p1 = np.array([int(min_proj[0]), ymin])
                p2 = np.array([int(max_proj[0]), ymax])

                # min_point = row_polygons[row][i][projections.index(min_proj)][0]
                # max_point = row_polygons[row][i][projections.index(max_proj)][0]
                # random_color = np.random.randint(0, 256, size=3, dtype=np.uint8)
                # random_color = [int(x_) for x_ in random_color]
                # cv2.line(image, tuple(min_point), tuple(max_point), (0, 0, 255), 2)
                # cv2.line(image, (int(min_proj[0]), ymin), (int(max_proj[0]), ymax), tuple(random_color), 2)

                l1 = np.linalg.norm(first_point - p1)
                l2 = np.linalg.norm(first_point - p2)

                points.append([l1, l2])
                coords.append([[int(p1[0]), int(p1[1])], [int(p2[0]), int(p2[1])]])

            row_points[row] = points
            row_points_coord[row] = coords

        for row in range(nrow):
            blocks = []
            for i in range(len(row_polygons[row])):
                x1, x2 = row_points[row][i]
                len_percent = (x2 - x1) / row_lengths[row]
                bbx,bby,bbw,bbh = cv2.boundingRect(row_polygons[row][i])
                # x, y = bbx, bby
                rock_bbox = [bbx, bby, bbx+bbw, bby+bbh]
                is_block = False
                for text, bbox in text_match.items():
                    # box1_area = (rock_bbox[2] - rock_bbox[0]) * (rock_bbox[3] - rock_bbox[1])
                    text_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    intersect = intersection_area_cal(bbox, rock_bbox)
                    percent = (intersect/text_area) * 100
                    # percent = (intersect/box1_area) * 100
                    if percent > 80 and len_percent < 0.12:
                        is_block = text
                        break
                    else:
                        is_block = False
                if is_block != False:
                    blocks.append(is_block)
                else:
                    blocks.append(None)
            row_blocks[row] = blocks

        merge_rocks(nrow, row_points, row_points_coord, row_blocks, filter_type=0)

        # is_continue = merge_rocks(nrow, row_points, row_points_coord, row_blocks, filter_type=1)
        while True:
            is_continue = merge_rocks(nrow, row_points, row_points_coord, row_blocks, filter_type=1)
            if not is_continue:
                break

        row_segments = {}
        row_categories = {}
        new_row_points_coord = {}

        for row in range(nrow):
            segments = []
            categories = []
            points_coord = []
            for i, p in enumerate(row_points[row]):
                coord = row_points_coord[row][i]
                if row_blocks[row][i] != None:
                    segments.append(p)
                    categories.append(row_blocks[row][i])
                    points_coord.append(coord)
                    continue
                elif row_blocks[row][i-1] != None:
                    segments.append(p)
                    categories.append("empty")
                    points_coord.append(coord)
                    continue

                if len(segments) != 0 and p[0] < segments[-1][1]:
                    p[0] = segments[-1][1]
                    coord[0] = points_coord[-1][1]

                if p[1] > p[0]:
                    segments.append(p)
                    points_coord.append(coord)
                categories.append("empty")
            
            row_segments[row] = segments
            row_categories[row] = categories
            new_row_points_coord[row] = points_coord

        row_rects = []
        for row in range(nrow):
            new_points_coord = new_row_points_coord[row]
            vx,vy,x,y = lines[row]
            perp_vx = -vy
            perp_vy = vx
            # for i in range(len(new_points_coord)):
            #     x1, x2 = new_points_coord[i]
            #     random_color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            #     random_color = [int(x) for x in random_color]
            #     cv2.line(image, tuple(x1), tuple(x2), random_color, 2)
            #     tmp = org.copy()
            #     cv2.line(tmp, tuple(x1), tuple(x2), random_color, 2)
            #     cv2.imshow("tmp", tmp)
            #     cv2.waitKey(0)
            # cv2.destroyAllWindows()

            vx,vy,x,y = lines[row]
            half_height = interval // 2
            start_offset = 5
            end_offset = 5
            perp_vx = -vy
            perp_vy = vx
            
            p1 = (int(start_offset - half_height * perp_vx), int(y - half_height * perp_vy))
            p2 = (int(image.shape[1] - end_offset - half_height * perp_vx), int(y - half_height * perp_vy))
            p3 = (int(image.shape[1] - end_offset + half_height * perp_vx), int(y + half_height * perp_vy))
            p4 = (int(start_offset + half_height * perp_vx), int(y + half_height * perp_vy))

            rect = np.array([p1, p2, p3, p4], dtype=np.int32)
            # mask = np.zeros_like(image, dtype=np.uint8)
            # cv2.fillPoly(mask, [rect], (255, 255, 255))
            # cropped_image = cv2.bitwise_and(image, mask)
            x, y, w, h = cv2.boundingRect(rect)
            # cropped_image = cropped_image[y:y+h, x:x+w]
            # row_images.append(cropped_image)
            row_rects.append([x, y, w, h])


            new_points_coord_cropped = []
            for coord in new_points_coord:
                x1, y1 = coord[0]
                x2, y2 = coord[1]
                x1 -= x
                y1 -= y
                x2 -= x
                y2 -= y
                new_points_coord_cropped.append([[x1, y1], [x2, y2]])
            new_row_points_coord[row] = new_points_coord_cropped

        # for row in range(nrow):
        #     row_img = row_images[row]
        #     row_points = new_row_points_coord[row]
        #     for i in range(len(row_points)):
        #         x1, y1 = row_points[i][0]
        #         x2, y2 = row_points[i][1]
        #         random_color = np.random.randint(0, 256, size=3, dtype=np.uint8)
        #         random_color = [int(x) for x in random_color]
        #         cv2.line(row_img, tuple([x1, y1]), tuple([x2, y2]), random_color, 2)
            # row_images[row] = row_img

        output_data = {}
        for row in range(nrow):
            rock_lengths = [np.linalg.norm(p[1] - p[0]) for p in row_segments[row]]
            rock_percents = [l / row_lengths[row] for l in rock_lengths]
            categories = row_categories[row]
            output_data[row] = (rock_percents, categories, new_row_points_coord[row])

        return output_data, row_rects, row_polygons