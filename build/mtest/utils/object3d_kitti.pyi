from _typeshed import Incomplete

def get_objects_from_label(label_file): ...
def cls_type_to_id(cls_type): ...

class Object3d:
    src: Incomplete
    cls_type: Incomplete
    type: Incomplete
    cls_id: Incomplete
    truncation: Incomplete
    occlusion: Incomplete
    alpha: Incomplete
    box2d: Incomplete
    h: Incomplete
    w: Incomplete
    l: Incomplete
    loc: Incomplete
    dis_to_cam: Incomplete
    ry: Incomplete
    score: Incomplete
    level_str: Incomplete
    level: Incomplete
    def __init__(self, line: Incomplete | None = ..., label: Incomplete | None = ...) -> None: ...
    def get_kitti_obj_level(self): ...
    def generate_corners3d(self): ...
    def to_str(self): ...
    def to_kitti_format(self): ...
