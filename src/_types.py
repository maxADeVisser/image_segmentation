from dataclasses import dataclass

IMAGE_HEIGHT = 720 
IMAGE_WIDTH = 960

@dataclass(slots=True)
class RGB:
    r: int
    g: int
    b: int

    def __str__(self) -> str:
        return f"RGB({self.r}, {self.g}, {self.b})"

    def get_label(self) -> str:
        """Return the label that matches the RGB values in CLASS_COLOR_LABELS"""
        for key, value in CLASS_COLOR_LABELS.items():
            if value == self:
                return key
        else:
            raise ValueError(f"No label exists for RGB-value {self}")


# RGB values for each class label in the dataset
CLASS_COLOR_LABELS = {
    "Animal": RGB(64, 128, 64),
    "Archway": RGB(192, 0, 128),
    "Bicyclist": RGB(0, 128, 192),
    "Bridge": RGB(0, 128, 64),
    "Building": RGB(128, 0, 0),
    "Car": RGB(64, 0, 128),
    "CartLuggagePram": RGB(64, 0, 192),
    "Child": RGB(192, 128, 64),
    "Column_Pole": RGB(192, 192, 128),
    "Fence": RGB(64, 64, 128),
    "LaneMkgsDriv": RGB(128, 0, 192),
    "LaneMkgsNonDriv": RGB(192, 0, 64),
    "Misc_Text": RGB(128, 128, 64),
    "MotorcycleScooter": RGB(192, 0, 192),
    "OtherMoving": RGB(128, 64, 64),
    "ParkingBlock": RGB(64, 192, 128),
    "Pedestrian": RGB(64, 64, 0),
    "Road": RGB(128, 64, 128),
    "RoadShoulder": RGB(128, 128, 192),
    "Sidewalk": RGB(0, 0, 192),
    "SignSymbol": RGB(192, 128, 128),
    "Sky": RGB(128, 128, 128),
    "SUVPickupTruck": RGB(64, 128, 192),
    "TrafficCone": RGB(0, 0, 64),
    "TrafficLight": RGB(0, 64, 64),
    "Train": RGB(192, 64, 128),
    "Tree": RGB(128, 128, 0),
    "Truck_Bus": RGB(192, 128, 192),
    "Tunnel": RGB(64, 0, 64),
    "VegetationMisc": RGB(192, 192, 0),
    "Void": RGB(0, 0, 0),
    "Wall": RGB(64, 192, 0),
}

CLASS_COUNT = len(CLASS_COLOR_LABELS)
