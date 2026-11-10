import math
import sys

import neat
import pygame

WIDTH, HEIGHT = 1200, 800
TRACK_WIDTH = 900
FPS = 60

GRASS = (60, 140, 60)
ROAD = (90, 90, 90)
PANEL = (30, 30, 30)
WHITE = (255, 255, 255)
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
TIME_LIMIT = 30 * FPS
STUCK_CHECK = 3 * FPS
STUCK_DISTANCE = 50

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT Racer")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 28)

track = pygame.Surface((TRACK_WIDTH, HEIGHT))
track.fill(GRASS)
for i in range(len(TRACK_POINTS)):
    start = pygame.Vector2(TRACK_POINTS[i - 1])
    end = pygame.Vector2(TRACK_POINTS[i])
    side = (end - start).normalize().rotate(90) * (ROAD_WIDTH / 2)
    pygame.draw.polygon(track, ROAD, [start + side, end + side, end - side, start - side])
for point in TRACK_POINTS:
    pygame.draw.circle(track, ROAD, point, ROAD_WIDTH // 2)

car_images = []
for color in ("black", "blue", "green", "orange", "yellow"):
    image = pygame.image.load(f"assets/car-{color}.png").convert_alpha()
    car_images.append(pygame.transform.scale_by(image, 0.5))

pause_button = pygame.Rect(TRACK_WIDTH + 20, 20, 260, 40)
slower_button = pygame.Rect(TRACK_WIDTH + 20, 70, 125, 40)
faster_button = pygame.Rect(TRACK_WIDTH + 155, 70, 125, 40)

paused = False
speed = 1
generation = 0
best_history = []


def on_road(x, y):
    if not (0 <= x < TRACK_WIDTH and 0 <= y < HEIGHT):
        return False
    return track.get_at((int(x), int(y))) == ROAD


class Car:
    def __init__(self, image):
        self.image = image
        self.x, self.y = TRACK_POINTS[0]
        next_x, next_y = TRACK_POINTS[1]
        self.angle = math.degrees(math.atan2(self.y - next_y, next_x - self.x))
        self.speed = 2
        self.alive = True
        self.distance = 0
        self.frames = 0
        self.check_x, self.check_y = self.x, self.y

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

        self.frames += 1
        if self.frames % STUCK_CHECK == 0:
            if math.dist((self.x, self.y), (self.check_x, self.check_y)) < STUCK_DISTANCE:
                self.alive = False
            self.check_x, self.check_y = self.x, self.y

    def corners(self):
        points = []
        for offset in (30, 150, 210, 330):
            rad = math.radians(self.angle + offset)
            points.append((self.x + math.cos(rad) * 20, self.y - math.sin(rad) * 20))
        return points

    def radar(self, offset):
        rad = math.radians(self.angle + offset)
        length = 0
        x, y = self.x, self.y
        while length < RADAR_MAX and on_road(x, y):
            length += 1
            x = self.x + math.cos(rad) * length
            y = self.y - math.sin(rad) * length
        return x, y, length

    def inputs(self):
        return [self.radar(offset)[2] / RADAR_MAX for offset in RADAR_ANGLES]

    def draw(self):
        for offset in RADAR_ANGLES:
            x, y, _ = self.radar(offset)
            pygame.draw.line(screen, RAY, (self.x, self.y), (x, y), 1)
        image = pygame.transform.rotate(self.image, self.angle - 90)
        screen.blit(image, image.get_rect(center=(self.x, self.y)))


def draw_text(text, x, y):
    screen.blit(font.render(text, True, WHITE), (x, y))


def draw_button(rect, text):
    pygame.draw.rect(screen, (70, 70, 70), rect, border_radius=6)
    label = font.render(text, True, WHITE)
    screen.blit(label, label.get_rect(center=rect.center))


def draw_panel(cars):
    pygame.draw.rect(screen, PANEL, (TRACK_WIDTH, 0, WIDTH - TRACK_WIDTH, HEIGHT))
    draw_button(pause_button, "Resume" if paused else "Pause")
    draw_button(slower_button, "Speed -")
    draw_button(faster_button, "Speed +")
    draw_text(f"Speed: {speed:g}x", TRACK_WIDTH + 20, 130)

    alive = sum(car.alive for car in cars)
    best_now = max(car.distance for car in cars)
    best_ever = max(best_history + [best_now])
    draw_text(f"Generation: {generation}", TRACK_WIDTH + 20, 200)
    draw_text(f"Cars alive: {alive}", TRACK_WIDTH + 20, 235)
    draw_text(f"Best (this gen): {best_now:.0f}", TRACK_WIDTH + 20, 270)
    draw_text(f"Best (all-time): {best_ever:.0f}", TRACK_WIDTH + 20, 305)


def handle_click(pos):
    global paused, speed
    if pause_button.collidepoint(pos):
        paused = not paused
    elif slower_button.collidepoint(pos):
        speed = max(speed / 2, 0.5)
    elif faster_button.collidepoint(pos):
        speed = min(speed * 2, 2)


def eval_genomes(genomes, config):
    global generation
    generation += 1

    nets = []
    cars = []
    for i, (_, genome) in enumerate(genomes):
        genome.fitness = 0
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        cars.append(Car(car_images[i % len(car_images)]))

    frames = 0
    while frames < TIME_LIMIT and any(car.alive for car in cars):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                handle_click(event.pos)

        if not paused:
            frames += 1
            for car, net, (_, genome) in zip(cars, nets, genomes):
                if car.alive:
                    output = net.activate(car.inputs())
                    car.act(output.index(max(output)))
                    car.update()
                    genome.fitness = car.distance

        screen.blit(track, (0, 0))
        for car in cars:
            if car.alive:
                car.draw()
        draw_panel(cars)
        pygame.display.flip()
        clock.tick(FPS * speed)

    best_history.append(max(genome.fitness for _, genome in genomes))


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     "config-feedforward.txt")
population = neat.Population(config)
population.add_reporter(neat.StdOutReporter(True))
population.run(eval_genomes, 1000)
