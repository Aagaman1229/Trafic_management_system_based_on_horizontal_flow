import math
import pygame

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def rotate_center(image, angle):
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image

def draw_rotated_rect(surface, color, rect, angle):
    shape_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    shape_surf.fill(color)
    rotated = pygame.transform.rotate(shape_surf, angle)
    rot_rect = rotated.get_rect(center=rect.center)
    surface.blit(rotated, rot_rect)