# NEAT Racer: neuroevolution car driver.
# Watch the cars learn, generation after generation, until you close the window.
import math
import sys

import neat
import pygame

# --- Window ---
WIDTH, HEIGHT = 1200, 800
TRACK_WIDTH = 900          # left part of the window is the track, the rest is the info panel
FPS = 60                   # frames per second; every car makes one move per frame (playback rate only)

# --- Colors ---
GRASS = (60, 140, 60)
ROAD = (90, 90, 90)
PANEL = (30, 30, 30)
WHITE = (255, 255, 255)
RAY = (255, 220, 0)

# --- Track ---
ROAD_WIDTH = 80
TRACK_POINTS = [           # corners of the closed loop; the first point is the start line
    (200, 700), (100, 550), (120, 250), (250, 120), (450, 150), (520, 300),
    (650, 330), (780, 180), (850, 350), (800, 600), (650, 700), (450, 620),
]

# --- Car Parameters ---
TURN = 5                   # degrees turned per move
ACCEL = 0.2                # speed gained or lost per move
MAX_SPEED = 6              # speed limit in pixels per frame
RADAR_MAX = 200            # longest distance a radar ray measures, in pixels
RADAR_ANGLES = (-90, -45, 0, 45, 90)  # ray directions relative to the car's heading
TIME_LIMIT = 30 * 60       # frames per generation (1800); fixed so training does not depend on FPS
STUCK_CHECK = 3 * 60       # frames between stuck checks (180); fixed so training does not depend on FPS
STUCK_DISTANCE = 50        # a car that moves less than this many pixels between checks is stuck

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT Racer")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 28)

# Draw the track once: grass background, then a road rectangle per segment and a circle per corner.
track = pygame.Surface((TRACK_WIDTH, HEIGHT))
track.fill(GRASS)
for i in range(len(TRACK_POINTS)):
    start = pygame.Vector2(TRACK_POINTS[i - 1])
    end = pygame.Vector2(TRACK_POINTS[i])
    side = (end - start).normalize().rotate(90) * (ROAD_WIDTH / 2)  # offset to the road's edges
    pygame.draw.polygon(track, ROAD, [start + side, end + side, end - side, start - side])
for point in TRACK_POINTS:
    pygame.draw.circle(track, ROAD, point, ROAD_WIDTH // 2)  # rounds off each corner

car_images = []
for color in ("black", "blue", "green", "orange", "yellow"):
    image = pygame.image.load(f"assets/car-{color}.png").convert_alpha()
    car_images.append(pygame.transform.scale_by(image, 0.5))

generation = 0
best_history = []          # best fitness of every finished generation


def on_road(x, y):
    """True if the pixel is road. Anything off the track area or on grass is not."""
    if not (0 <= x < TRACK_WIDTH and 0 <= y < HEIGHT):
        return False
    return track.get_at((int(x), int(y))) == ROAD


class Car:
    def __init__(self, image):
        self.image = image
        self.x, self.y = TRACK_POINTS[0]
        next_x, next_y = TRACK_POINTS[1]
        self.angle = math.degrees(math.atan2(self.y - next_y, next_x - self.x))  # face the next point
        self.speed = 2
        self.alive = True
        self.distance = 0                          # total distance driven; this is the fitness
        self.frames = 0
        self.check_x, self.check_y = self.x, self.y  # position at the last stuck check

    def act(self, choice):
        """Applies the network's choice: 0 left, 1 right, 2 speed up, 3 slow down."""
        if choice == 0:
            self.angle += TURN
        elif choice == 1:
            self.angle -= TURN
        elif choice == 2:
            self.speed = min(self.speed + ACCEL, MAX_SPEED)
        else:
            self.speed = max(self.speed - ACCEL, 0)

    def update(self):
        """Moves one step, then kills the car if it left the road or is stuck."""
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y -= math.sin(rad) * self.speed  # screen y grows downward, so subtract
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
        """The four corners of the car's body, used for crash detection."""
        points = []
        for offset in (30, 150, 210, 330):
            rad = math.radians(self.angle + offset)
            points.append((self.x + math.cos(rad) * 20, self.y - math.sin(rad) * 20))
        return points

    def radar(self, offset):
        """Casts a ray until it hits grass or reaches RADAR_MAX. Returns (end_x, end_y, length)."""
        rad = math.radians(self.angle + offset)
        length = 0
        x, y = self.x, self.y
        while length < RADAR_MAX and on_road(x, y):
            length += 1
            x = self.x + math.cos(rad) * length
            y = self.y - math.sin(rad) * length
        return x, y, length

    def inputs(self):
        """Network inputs: each ray length scaled to 0-1."""
        return [self.radar(offset)[2] / RADAR_MAX for offset in RADAR_ANGLES]

    def draw(self):
        for offset in RADAR_ANGLES:
            x, y, _ = self.radar(offset)
            pygame.draw.line(screen, RAY, (self.x, self.y), (x, y), 1)
        image = pygame.transform.rotate(self.image, self.angle - 90)  # image points up, angle 0 points right
        screen.blit(image, image.get_rect(center=(self.x, self.y)))


def draw_text(text, x, y):
    screen.blit(font.render(text, True, WHITE), (x, y))


def draw_panel(cars):
    """Draws the info panel on the right side of the window."""
    pygame.draw.rect(screen, PANEL, (TRACK_WIDTH, 0, WIDTH - TRACK_WIDTH, HEIGHT))

    alive = sum(car.alive for car in cars)
    best_now = max(car.distance for car in cars)
    best_ever = max(best_history + [best_now])
    draw_text(f"Generation: {generation}", TRACK_WIDTH + 20, 70)
    draw_text(f"Cars alive: {alive}", TRACK_WIDTH + 20, 100)
    draw_text(f"Best (this gen): {best_now:.0f}", TRACK_WIDTH + 20, 140)
    draw_text(f"Best (all-time): {best_ever:.0f}", TRACK_WIDTH + 20, 175)


def eval_genomes(genomes, config):
    """Runs one generation: every genome drives a car, and its fitness is the distance driven."""
    global generation
    generation += 1

    nets = []
    cars = []
    for i, (_, genome) in enumerate(genomes):
        genome.fitness = 0
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        cars.append(Car(car_images[i % len(car_images)]))  # cycle through the 5 car colors

    # The generation ends at the time limit or when every car is dead.
    frames = 0
    while frames < TIME_LIMIT and any(car.alive for car in cars):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        frames += 1
        for car, net, (_, genome) in zip(cars, nets, genomes):
            if car.alive:
                output = net.activate(car.inputs())
                car.act(output.index(max(output)))  # the strongest output neuron picks the action
                car.update()
                genome.fitness = car.distance

        screen.blit(track, (0, 0))
        for car in cars:
            if car.alive:
                car.draw()
        draw_panel(cars)
        pygame.display.flip()
        clock.tick(FPS)

    best_history.append(max(genome.fitness for _, genome in genomes))


# --- NEAT Setup ---
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     "config-feedforward.txt")
population = neat.Population(config)
population.add_reporter(neat.StdOutReporter(True))  # prints each generation's stats to the console
population.run(eval_genomes, 1000)                  # at most 1000 generations