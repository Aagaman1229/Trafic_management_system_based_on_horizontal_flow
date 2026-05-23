import random

# Window setup
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 900
FPS = 60

# Road geometry (6-lane 2-way road for all approaches)
LANE_WIDTH = 30
ROAD_WIDTH = LANE_WIDTH * 6  # 180 pixels (3 lanes each way)
INTERSECTION_SIZE = 180       # Size of central box

# Premium dark theme color palette
COLOR_BG = (15, 23, 42)          # Deep slate blue
COLOR_ROAD = (30, 41, 59)        # Asphalt slate
COLOR_ISLAND = (15, 23, 42)      # Center island dark slate
COLOR_STOP_LINE = (241, 245, 249) # Bright off-white
COLOR_MARKING = (148, 163, 184)  # Muted slate gray for dashes
COLOR_ZEBRA = (226, 232, 240)    # Soft white

# Signal colors (glowing premium look)
COLOR_RED = (239, 68, 68)        # Vibrant Red
COLOR_YELLOW = (245, 158, 11)    # Warm Amber/Yellow
COLOR_GREEN = (16, 185, 129)     # Emerald Green
COLOR_DARK_RED = (127, 29, 29)
COLOR_DARK_YELLOW = (120, 53, 4)
COLOR_DARK_GREEN = (6, 78, 59)

# HUD colors
COLOR_HUD_BG = (30, 41, 59, 190)  # Glassmorphic transcluent Slate
COLOR_HUD_BORDER = (71, 85, 105)  # Slate border

# Vehicle colors fallback (if image missing)
VEHICLE_COLORS = {
    'car': (239, 68, 68),        # Emerald red
    'motorcycle': (245, 158, 11), # Yellow/Gold
    'truck': (148, 163, 184),     # Slate gray
    'bus': (59, 130, 246)         # Indigo blue
}

# Vehicle sizes (width, height)
VEHICLE_SIZES = {
    'car': (20, 42),
    'motorcycle': (12, 26),
    'truck': (24, 60),
    'bus': (26, 70)
}

# Physics (values in pixels and pixels/sec using delta time)
MAX_SPEED = 180.0         # Pixels per second
ACCELERATION = 140.0      # Pixels per second^2
DECELERATION = 340.0      # Pixels per second^2 (increased responsiveness)
SAFE_DISTANCE = 80.0      # Pixels of safe following gap (greater than braking distance of 57px)

# Direction configurations: D=North (top→bottom), A=East (right→left), B=South (bottom→top), C=West (left→right)
DIRECTIONS = ['D', 'A', 'B', 'C']

# Real-world road names (Prithivi Chowk, Pokhara — highest-traffic intersection)
ROAD_NAMES = {
    'A': 'China Pull Road',
    'B': 'Airport Road',
    'C': 'Sabhagriha Chowk Road',
    'D': 'Naya Bazar Road',
}

ROAD_NAMES_SHORT = {
    'A': 'China Pull',
    'B': 'Airport',
    'C': 'Sabhagriha Chowk',
    'D': 'Naya Bazar',
}

# Per-road straight-going vehicle ratio (replaces universal 2/3)
STRAIGHT_RATIOS = {
    'A': 0.72,
    'B': 0.72,
    'C': 0.78,
    'D': 0.6,
}

# All roads use 3 inbound lanes (left/straight/right) in simulation to prevent blocking
APPROACH_LANES = {
    'A': 3,
    'B': 3,
    'C': 3,
    'D': 3,
}

def get_lane_offset(direction_idx, movement):
    """Return lane center offset (px from road center line) for a given approach and movement type.
    
    Each road has 3 inbound lanes:
      Lane 0 (closest to center)  → Right Turn + Straight   → offset = LANE_WIDTH * 0.5 = 15
      Lane 1                      → Straight                → offset = LANE_WIDTH * 1.5 = 45
      Lane 2 (farthest from center)→ Left Turn               → offset = LANE_WIDTH * 2.5 = 75
    """
    dir_char = DIRECTIONS[direction_idx]
    n_lanes = APPROACH_LANES[dir_char]
    
    if movement == 'right':
        lane_idx = 0
    elif movement == 'straight':
        lane_idx = random.choice([0, 1])
    else:  # left
        lane_idx = min(2, n_lanes - 1)
    
    return LANE_WIDTH * (lane_idx + 0.5)


def max_lane_offset(direction_idx):
    """Return the largest lane offset (outermost lane edge) for a given approach."""
    dir_char = DIRECTIONS[direction_idx]
    n_lanes = APPROACH_LANES[dir_char]
    return LANE_WIDTH * n_lanes