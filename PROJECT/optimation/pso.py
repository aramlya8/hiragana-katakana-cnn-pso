import numpy as np

class PSO:
    def __init__(self, n_particles, w=0.5, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.velocities = []
        self.pbest = []
        self.pbest_score = []
        self.gbest = None
        self.gbest_score = np.inf

    def initialize(self, particles):
        # velocities: list of list (sesuai struktur weights CNN)
        self.velocities = [
            [np.zeros_like(w) for w in particle]
            for particle in particles
        ]

        self.pbest = [particle.copy() for particle in particles]
        self.pbest_score = [np.inf] * len(particles)

    def update(self, particles, fitness_scores):
        # Update pbest dan gbest
        for i in range(self.n_particles):
            if fitness_scores[i] < self.pbest_score[i]:
                self.pbest_score[i] = fitness_scores[i]
                self.pbest[i] = [w.copy() for w in particles[i]]

            if fitness_scores[i] < self.gbest_score:
                self.gbest_score = fitness_scores[i]
                self.gbest = [w.copy() for w in particles[i]]

        # Update velocity & position per layer
        for i in range(self.n_particles):
            r1, r2 = np.random.rand(), np.random.rand()

            for j in range(len(particles[i])):
                self.velocities[i][j] = (
                    self.w * self.velocities[i][j]
                    + self.c1 * r1 * (self.pbest[i][j] - particles[i][j])
                    + self.c2 * r2 * (self.gbest[j] - particles[i][j])
                )

                particles[i][j] = particles[i][j] + self.velocities[i][j]

        return particles
