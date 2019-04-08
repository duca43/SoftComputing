import itertools


class NumberObject:
    newid = next(itertools.count())
    max_frames_disappeared = 50

    def __init__(self, center, radius, value):
        self.id = NumberObject.newid
        self.center = center
        self.radius = radius
        self.values = [value]
        self.collided_green = False
        self.collided_blue = False
        self.frames_disappeared = 0
        self.updated = False
