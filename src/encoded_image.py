class EncodedImage(object):

    def __init__(self, shape, position_x, position_y, radius, rotation):
        # shape is of type string
        self.shape = shape
        self.position_x = position_x
        self.position_y = position_y
        self.radius = radius
        self.rotation = rotation

    def to_dict(self):
        encoded_image_dict = {}
        encoded_image_dict['s'] = self.shape
        encoded_image_dict['px'] = self.position_x
        encoded_image_dict['py'] = self.position_y
        encoded_image_dict['ra'] = self.radius
        encoded_image_dict['ro'] = self.rotation
        return encoded_image_dict

    def clone(self):
        return EncodedImage(self.shape,
            self.position_x,
            self.position_y,
            self.radius,
            self.rotation)

    @staticmethod
    def from_dict(encoded_image_json):
        return EncodedImage(encoded_image_json['s'],
            encoded_image_json['px'],
            encoded_image_json['py'],
            encoded_image_json['ra'],
            encoded_image_json['ro'])
