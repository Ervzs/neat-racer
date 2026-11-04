import pygame

WIDTH, HEIGHT = 1200, 800
TRACK_WIDTH = 900
FPS = 60

GRASS = (60, 140, 60)
ROAD = (90, 90, 90)
PANEL = (30, 30, 30)

ROAD_WIDTH = 80
TRACK_POINTS = [
    (200, 700), (100, 550), (120, 250), (250, 120), (450, 150), (520, 300),
    (650, 330), (780, 180), (850, 350), (800, 600), (650, 700), (450, 620),
]

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT Racer")
clock = pygame.time.Clock()

track = pygame.Surface((TRACK_WIDTH, HEIGHT))
track.fill(GRASS)
pygame.draw.lines(track, ROAD, True, TRACK_POINTS, ROAD_WIDTH)
for point in TRACK_POINTS:
    pygame.draw.circle(track, ROAD, point, ROAD_WIDTH // 2)

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
