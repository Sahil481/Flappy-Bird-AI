import pygame
import neat
import os
import pickle
from constants import *
from classes.base import Base
from classes.bird import Bird
from classes.pipe import Pipe

pygame.font.init()

limit = 200

def draw_window(win, birds, pipes, base, score):
    win.blit(BG_IMG, (0, 0))
    
    for pipe in pipes:
        pipe.draw(win)

    text = STAT_FONT.render(f"Score: {score}", True, (255, 255, 255))

    base.draw(win)
    for bird in birds:
        bird.draw(win)
    win.blit(text, (WIN_WIDTH - 5 - text.get_width(), 5))
    pygame.display.update()

def main(genomes, config,):
    nets = []
    ge = []
    birds = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        g.fitness = 0
        ge.append(g)


    base = Base(730)
    pipes = [Pipe(600)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Flappy Bird - AI")
    pygame.display.set_icon(pygame.image.load(os.path.join("imgs", "bird1.png")))
    clock = pygame.time.Clock()

    score = 0

    run = True
    while run:
        clock.tick(120)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        pipe_index = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_index = 1
        else:
            run = False
            break    

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.01

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_index].height), abs(bird.y - pipes[pipe_index].bottom)))

            if output[0] > 0.5:
                bird.jump()

        add_pipe = False
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 0.1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True
            
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 1
            pipes.append(Pipe(600))

        for pipe in rem:
            pipes.remove(pipe)
            
        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() >= 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)
        
        if score >= limit:
            run = False
            break

        base.move()
        draw_window(win, birds, pipes, base, score)


def run(config_path, LIMIT: int = 200):
    global limit
    # Initializing the configuration file with default values
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    limit = LIMIT
    winner = p.run(main, 50)

    currentModel = pickle.load(open("model.pkl", "rb"))

    while True:
        input = input(f"The fitness of current model : {currentModel.fitness}\nDo you want to overide the current model with this? (y/n) : ")
        input.lower()
        if input == "y":
            break
        elif input == "n":
            return
        else:
            print("Invalid input, please try again")

    if input == "y":
        if currentModel.fitness < winner.fitness:
            pickle.dump(winner, open("model.pkl", "wb"))
