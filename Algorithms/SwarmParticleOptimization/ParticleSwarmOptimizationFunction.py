from random import uniform
import numpy as np


def f(v):
    x, y = v[0], v[1]
    return - np.sin(5*np.sqrt(x**2+y**2))/(5*np.sqrt(x**2+y**2))


class Particle:

    def __init__(self, v_min, v_max):
        # We consider the volume [v_min[0], v_max[0]] x ... x [v_min[N], v_max[N]],
        # where N is a number of dimensions.
        self.v_min = v_min
        self.v_max = v_max
        self.dimension = len(v_min)

        self.position = self.initialize_position()
        self.velocity = self.initialize_velocity()

        self.best_known_position = self.position
        # The algorithm seeks minimum in given volume, so in the beginning, we start with very large value
        self.best_known_value = float("inf")

    def update_position(self):
        new_position = self.position + self.velocity
        # the particle position must be in the considered domain
        for i in range(self.dimension):
            if new_position[i] < self.v_min[i]:
                new_position[i] = self.v_min[i]
            if new_position[i] > self.v_max[i]:
                new_position[i] = self.v_max[i]

        self.position = new_position

    def initialize_position(self):
        return np.array([uniform(self.v_min[i], self.v_max[i]) for i in range(self.dimension)])

    def initialize_velocity(self):
        v = [uniform(-self.v_max[i] - self.v_min[i], self.v_max[i] - self.v_min[i]) for i in range(self.dimension)]
        return np.array(v)

    def update_velocity(self, w, c1, c2, best_known_swarm_position):
        self.velocity = w * self.velocity + c1 * uniform(0, 1) * (self.best_known_position - self.position) + \
                       c2 * uniform(0, 1) * (best_known_swarm_position - self.position)
        return


class Swarm:
    def __init__(self, v_min, v_max, n_particles):
        self.particles = [Particle(v_min, v_max) for _ in range(n_particles)]
        self.best_known_value = float("inf")
        self.best_known_position = self.particles[0].position

    def set_particles_best_parameters(self):
        for particle in self.particles:
            cost_function_value = f(particle.position)

            if particle.best_known_value > cost_function_value:
                particle.best_known_value = cost_function_value
                particle.best_known_position = particle.position

    def set_swarm_best_parameters(self):
        for particle in self.particles:
            cost_function_value = f(particle.position)

            if self.best_known_value > cost_function_value:
                self.best_known_value = cost_function_value
                self.best_known_position = particle.position


class ParticleSwarmOptimization:
    def __init__(self, v_min, v_max, n_particles=100, max_iteration=30, w=0.7, c1=1.0, c2=1.0):
        self.n_particles = n_particles
        self.max_iteration = max_iteration
        self.swarm = Swarm(v_min, v_max, n_particles)

        # inertia weight, i.e., exploration and exploitation trade-off
        self.w = w
        # cognitive coefficient and social coefficient
        self.c1 = c1
        self.c2 = c2

    def run(self):

        for _ in range(self.max_iteration):
            self.move_particles()
            self.swarm.set_swarm_best_parameters()
            self.swarm.set_particles_best_parameters()

        print(f"Global minimum: {self.swarm.best_known_position} with the value: {self.swarm.best_known_value}")

    def move_particles(self):
        for particle in self.swarm.particles:
            particle.update_velocity(self.w, self.c1, self.c2, self.swarm.best_known_position)
            particle.update_position()


if __name__ == '__main__':
    algorithm = ParticleSwarmOptimization(v_min=[-5, -5], v_max=[5, 5])
    algorithm.run()
