import math

import pygame

WIDTH, HEIGHT = 1200, 800
TRACK_WIDTH = 900
FPS = 60

GRASS = (60, 140, 60)
ROAD = (90, 90, 90)
PANEL = (30, 30, 30)
RAY = (255, 220, 0)

ROAD_WIDTH = 80
TRACK_POINTS = [
    (200, 700), (100, 550), (120, 250), (250, 120), (450, 150), (520, 300),
    (650, 330), (780, 180), (850, 350), (800, 600), (650, 700), (450, 620),
]

TURN = 5
ACCEL = 0.2
MAX_SPEED = 6
RADAR_MAX = 200
RADAR_ANGLES = (-90, -45, 0, 45, 90)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT Racer")
clock = pygame.time.Clock()

track = pygame.Surface((TRACK_WIDTH, HEIGHT))
track.fill(GRASS)
pygame.draw.lines(track, ROAD, True, TRACK_POINTS, ROAD_WIDTH)
for point in TRACK_POINTS:
    pygame.draw.circle(track, ROAD, point, ROAD_WIDTH // 2)

car_image = pygame.image.load("assets/car.png").convert_alpha()
car_image = pygame.transform.scale_by(car_image, 0.5)


def on_road(x, y):
    if not (0 <= x < TRACK_WIDTH and 0 <= y < HEIGHT):
        return False
    return track.get_at((int(x), int(y))) == ROAD


class Car:
    def __init__(self):
        self.x, self.y = TRACK_POINTS[0]
        next_x, next_y = TRACK_POINTS[1]
        self.angle = math.degrees(math.atan2(self.y - next_y, next_x - self.x))
        self.speed = 2
        self.alive = True
        self.distance = 0

    def act(self, choice):
        if choice == 0:
            self.angle += TURN
        elif choice == 1:
            self.angle -= TURN
        elif choice == 2:
            self.speed = min(self.speed + ACCEL, MAX_SPEED)
        else:
            self.speed = max(self.speed - ACCEL, 0)

    def update(self):
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y -= math.sin(rad) * self.speed
        self.distance += self.speed

        for x, y in self.corners():
            if not on_road(x, y):
                self.alive = False

    def corners(self):
        points = []
        for offset in (30, 150, 210, 330):
            rad = math.radians(self.angle + offset)
            points.append((self.x + math.cos(rad) * 20, self.y - math.sin(rad) * 20))
        return points


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.blit(track, (0, 0))
    pygame.draw.rect(screen, PANEL, (TRACK_WIDTH, 0, WIDTH - TRACK_WIDTH, HEIGHT))
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
