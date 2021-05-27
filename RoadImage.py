from PIL import Image, ImageDraw, ImagePath
import numpy as np

class RoadImage():
    def __init__(self, win_width, win_height, bg_color = (79, 95, 240)):
        self.win_width = win_width
        self.win_height = win_height
        self.horizon = 50
        self.road_width = 300
        self.shoulder = 20
        self.offset = 40
        self.rotation = 0
        self.paint_width = 5
        self.right_lane = self.road_width/4
        self.file_name = ''
        
        self.img = Image.new('RGB', (self.win_width, self.win_height), bg_color)

    # creates the image and returns the truth values as an array: [target_base, vanishing_point]
    def create(self, horizon, road_width, shoulder, offset, rotation, paint_width, directory, truth_line):
        self.horizon = horizon
        self.road_width = road_width
        self.shoulder = shoulder
        self.offset = offset
        self.rotation = rotation
        self.paint_width = paint_width

        img1 = ImageDraw.Draw(self.img)

        land_point1 = (0, horizon)
        land_point2 = (self.win_width,self.win_height)

        img1.rectangle([land_point1, land_point2], fill ="#4fc251")

        # create road
        vanishing_point = (self.win_width/2 + rotation, horizon)
        road_left  = (self.win_width/2 - road_width/2 + offset - self.right_lane, self.win_height)
        road_right = (self.win_width/2 + road_width/2 + offset - self.right_lane, self.win_height)

        img1.polygon([road_left, road_right, vanishing_point], fill='#363636')

        # center line
        center_left  =  (self.win_width / 2  - paint_width / 2 + offset - self.right_lane, self.win_height)
        center_right =  (self.win_width / 2  + paint_width / 2 + offset - self.right_lane, self.win_height)
        img1.polygon([center_left, center_right, vanishing_point], fill = 'yellow')

        # create shoulders
        left_shoulder_left = (self.win_width/2 - road_width/2 + offset - self.right_lane + shoulder, self.win_height)
        left_shoulder_right = (self.win_width/2 - road_width/2 + offset - self.right_lane + shoulder + paint_width, self.win_height)
        img1.polygon([left_shoulder_left, left_shoulder_right, vanishing_point], fill='white')

        right_shoulder_left = (self.win_width/2 + road_width/2 + offset - self.right_lane - shoulder, self.win_height)
        right_shoulder_right = (self.win_width/2 + road_width/2 + offset - self.right_lane - shoulder + paint_width, self.win_height)
        img1.polygon([right_shoulder_left, right_shoulder_right, vanishing_point], fill='white')

        target_base = (self.win_width / 2  - self.paint_width / 2 + self.offset - self.right_lane + self.road_width / 4 - self.shoulder/2, self.win_height)

        base_val = self.win_width / 2  - self.paint_width / 2 + self.offset - self.right_lane + self.road_width / 4


        target = (target_base[0], vanishing_point[0])
        
        # optionally add the line marking the true center
        if truth_line:
            img1.line(target, fill = 'red')

        return [target_base[0][0], vanishing_point[0][0]]

    def getIMGObj(self):
        return self.img

    def set_file_name(self, file_name):
        self.file_name = file_name

