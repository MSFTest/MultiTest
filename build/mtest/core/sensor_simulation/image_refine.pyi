from _typeshed import Incomplete

class Image_Harmonization:
    size: Incomplete
    lr_trans: Incomplete
    model: Incomplete
    def __init__(self) -> None: ...
    def process_data(self, obj_rgb, obj_mask, bg_rgb, pos): ...
    def convert_data2tensor(self, img, mask): ...
    def run(self, obj_rgb, obj_mask, bg_rgb, pos): ...
